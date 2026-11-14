"""ByT5-era dual-hemisphere core: sequential-residency, both sides really learn.

架构映射（旧 legacy 语义 → 本机 3GB 显存物理约束下的实现）：
- 双半球：左右各自独立完整 byt5-small 权重、独立 Adafactor、独立回放/债务账本。
  两侧都真实在线学习——没有任何冻结或适配器间接层。
- 顺序驻留：fp32 主副本常驻 CPU（2×1.2GB 内存），GPU 同一时刻只驻留活动侧
  工作副本；被动巩固时换入被动侧训练再换回（PCIe 换入换出 ~0.3s）。
- 曲线自适应切换：紧迫度 = smoothstep(净压力增长 / 学到的偿还容量)，
  累加器 ≥1 触发切换——时机是连续曲线，无固定阈值；切换含债务交换
  与 CPU 上 2% 对称插值。
- 字节级统计与质量闸门沿用 chat_cli 的同一套（正确字词字节才入损失）。
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.optimization import Adafactor

from life_agent.memory import DebtLedger, ReplayBuffer, Seq2SeqSample

BASE = "google/byt5-small"
BASE_LR = 1e-4
ENC_MAX = 512
DEC_CAP = 192
SYNC_MIX = 0.01          # 每侧 1%，合计 2% 对称插值（legacy sync_mix=0.02）
CONSOLIDATION_STEPS = 2
DEBT_RETENTION = 0.35
SWITCH_RATIO_SCALE = 2.0  # 紧迫度曲线：净压力增长 / max(容量,0.5) / 2.0 → smoothstep


class ByteHemisphere:
    """单半球：fp32 主副本驻 CPU，GPU 工作副本按需换入换出，独立优化器/回放/债务。"""

    def __init__(self, name: str, source: str, state_dir: Path) -> None:
        self.name = name
        self.source = source
        self.state_dir = state_dir
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        # 单对象原地迁移：权重只有一份，驻留 = 整体 .to(device)。
        self.model = AutoModelForSeq2SeqLM.from_pretrained(source, dtype=torch.float32)
        self.model.config.tie_word_embeddings = False
        for name_, param in self.model.named_parameters():
            param.requires_grad = "block" in name_ and (".q." in name_ or ".v." in name_)
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        self.device = "cpu"
        trainable = [p for n, p in self.model.named_parameters() if p.requires_grad]
        self.optimizer = Adafactor(trainable, lr=BASE_LR, scale_parameter=False, relative_step=False)
        self.replay = ReplayBuffer(256)
        self.debt = DebtLedger(128, 0.75, 0.005)
        vocab = self.tokenizer.get_vocab()
        self.byte_to_id = {b: vocab[chr(b)] for b in range(256)}

    # ---------- 驻留管理 ----------

    @property
    def gpu_model(self):
        """兼容旧调用点：仅当半球驻留 GPU 时返回模型，否则 None。"""
        return self.model if self.device == "cuda" else None

    def ensure_gpu(self) -> None:
        if self.device == "cuda":
            return
        self.model.to("cuda")
        self.device = "cuda"
        self._move_optimizer_state("cuda")

    def to_cpu(self) -> None:
        if self.device == "cpu":
            return
        self.model.to("cpu")
        self.device = "cpu"
        self._move_optimizer_state("cpu")
        torch.cuda.empty_cache()

    def _move_optimizer_state(self, device: str) -> None:
        # Adafactor 的状态张量不会跟随 module.to() 自动迁移，手动跟随
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    # ---------- 在线学习（活动侧，每轮） ----------

    def learn(self, enc_bytes: list[int], resp_bytes: list[int], intensity: float, max_loss: float = 2.0) -> dict:
        """活动侧真权重更新：入参为字节值（0..255），内部映射到 token id。

        等效学习率 = BASE_LR × intensity（曲线缩放）。
        """
        self.ensure_gpu()
        enc_tokens = [self.byte_to_id[b] for b in enc_bytes[-ENC_MAX:] if b in self.byte_to_id]
        labels = [self.byte_to_id[b] for b in resp_bytes[-DEC_CAP:] if b in self.byte_to_id]
        labels += [self.tokenizer.eos_token_id]
        inputs = torch.tensor([enc_tokens], device="cuda")
        label_tensor = torch.tensor([labels], device="cuda")
        for group in self.optimizer.param_groups:
            group["lr"] = BASE_LR * intensity
        self.model.train()
        loss = self.model(input_ids=inputs, labels=label_tensor).loss
        loss_value = float(loss.item())
        if not torch.isfinite(loss):
            return {"loss": loss_value, "learned": False, "reason": "non-finite"}
        if loss_value > max_loss:
            # 质量闸门：模型对正确字节毫无把握（多半是自己的退化输出），
            # 把垃圾当教材会自我污染权重——跳过本轮更新。
            return {"loss": loss_value, "learned": False, "reason": "quality-gate"}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return {"loss": loss_value, "learned": True, "intensity": intensity}

    # ---------- 被动巩固（换入 GPU 训练，债务账本记账） ----------

    def consolidation_samples(self, limit: int = 4) -> list[Sample]:
        samples = self.debt.due(limit)
        if self.replay.samples:
            samples.extend(self.replay.choose(max(1, limit // 4), random.Random()))
        return samples

    def consolidate(self, samples: list[Sample], intensity: float) -> dict:
        self.ensure_gpu()
        if not samples:
            return {"steps": 0}
        self.model.train()
        lr = BASE_LR * max(0.25, intensity)
        steps_done = 0
        last_loss = None
        for step in range(CONSOLIDATION_STEPS):
            sample = samples[step % len(samples)]
            # seq2seq 语义：encoder = 用户字节，decoder = 回应字节（账本以字节为单位）
            enc_tokens = [self.byte_to_id[b] for b in sample.ids[-ENC_MAX:] if b in self.byte_to_id]
            dec_tokens = [self.byte_to_id[b] for b in sample.labels[-DEC_CAP:] if b in self.byte_to_id]
            inputs = torch.tensor([enc_tokens], device="cuda")
            targets = torch.tensor([dec_tokens], device="cuda")
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(input_ids=inputs, labels=targets)
            loss = out.loss
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0
            )
            self.optimizer.step()
            steps_done += 1
            last_loss = float(loss.item())
        if steps_done:
            self.debt.commit_repayment(samples, trained_steps=steps_done)
            self.debt.decay_after_training()
            stored_sample = samples[0]
            self.replay.add(Seq2SeqSample(
                ids=stored_sample.ids.copy(), labels=stored_sample.labels.copy(),
                weight=1.0, source=f"consolidate:{self.name}",
            ))
        return {"steps": steps_done, "loss": last_loss}

    # ---------- 压力与债务 ----------

    def admit_exchange(self, sample: Sample) -> None:
        """切换交换：对侧已学材料入本侧债务（不构成新净压力）。"""
        self.debt.add(sample, DEBT_RETENTION, count_pressure=False)

    def admit_pressure(self, sample: Sample) -> None:
        """活动侧每轮吸收的新数据进入债务账本，构成切换压力。"""
        self.debt.add(sample, 1.0, count_pressure=True)
        self.replay.add(sample)

    # ---------- 持久化 ----------

    def save(self) -> None:
        out = self.state_dir / "hemispheres" / self.name
        out.mkdir(parents=True, exist_ok=True)
        was_gpu = self.device == "cuda"
        self.to_cpu()   # 权重落盘前回到 CPU
        self.model.save_pretrained(out, safe_serialization=True)
        self.tokenizer.save_pretrained(out)
        torch.save(self.optimizer.state_dict(), out / "optimizer_state.pt")
        memory = {"replay": self.replay.to_dict(), "debt": self.debt.to_dict()}
        (out / "memory.json").write_text(json.dumps(memory, ensure_ascii=False), encoding="utf-8")

    def load(self) -> bool:
        out = self.state_dir / "hemispheres" / self.name
        if not (out / "config.json").exists():
            return False
        self.model = AutoModelForSeq2SeqLM.from_pretrained(out, dtype=torch.float32)
        self.model.config.tie_word_embeddings = False
        for name_, param in self.model.named_parameters():
            param.requires_grad = "block" in name_ and (".q." in name_ or ".v." in name_)
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(out)
        opt_path = out / "optimizer_state.pt"
        if opt_path.exists():
            self.optimizer.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=True))
        memory_path = out / "memory.json"
        if memory_path.exists():
            memory = json.loads(memory_path.read_text(encoding="utf-8"))
            self.replay.load_dict(memory.get("replay", {}))
            self.debt.load_dict(memory.get("debt", {}))
        return True


class Byt5DualBrain:
    """双半球大脑：活动侧 GPU 在线生成+学习，被动侧巩固，曲线自适应切换。"""

    def __init__(self, source: str, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.hemispheres = {
            name: ByteHemisphere(name, source, state_dir) for name in ("left", "right")
        }
        self.active = "left"
        self.switch_urgency_acc = 0.0
        self.pressure_baseline = {"left": 0.0, "right": 0.0}
        self.switch_count = 0
        self.switch_log: list[dict] = []

    def load(self) -> bool:
        loaded = [h.load() for h in self.hemispheres.values()]
        meta_path = self.state_dir / "brain.json"
        if meta_path.exists() and all(loaded):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.active = meta.get("active", "left")
            self.switch_urgency_acc = float(meta.get("switch_urgency_acc", 0.0))
            self.pressure_baseline = meta.get("pressure_baseline", self.pressure_baseline)
            self.switch_count = int(meta.get("switch_count", 0))
            self.switch_log = meta.get("switch_log", [])
            return True
        return False

    def save(self) -> None:
        for hemisphere in self.hemispheres.values():
            hemisphere.save()
        meta = {
            "active": self.active,
            "switch_urgency_acc": self.switch_urgency_acc,
            "pressure_baseline": self.pressure_baseline,
            "switch_count": self.switch_count,
            "switch_log": self.switch_log[-64:],
        }
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "brain.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def active_h(self) -> ByteHemisphere:
        return self.hemispheres[self.active]

    def passive_h(self) -> ByteHemisphere:
        return self.hemispheres["right" if self.active == "left" else "left"]

    def ensure_active_gpu(self) -> None:
        # 驻留不变式：GPU 上只允许活动侧存在——被动侧一律清回 CPU
        for name, hemisphere in self.hemispheres.items():
            if name != self.active and hemisphere.device == "cuda":
                hemisphere.to_cpu()
        self.active_h().ensure_gpu()

    # ---------- 切换时机：曲线自适应 ----------

    def switch_urgency(self) -> float:
        """紧迫度 = smoothstep(活动侧净压力增长 / 被动侧容量 / SCALE) ∈ [0,1]。

        语义：活动侧吸收的新数据（被动侧尚未巩固的量）超过被动侧
        学到的单批偿还容量到什么程度——比率连续映射到紧迫度，累加器
        攒满 1.0 即切换。无固定阈值：涨得快切得快，涨得慢切得慢。
        """
        active = self.active_h()
        passive = self.passive_h()
        growth = max(0.0, active.debt.total_clean_added - self.pressure_baseline[active.name])
        capacity = max(passive.debt.repayment_capacity, 0.5)
        ratio = growth / capacity / SWITCH_RATIO_SCALE
        t = max(0.0, min(1.0, ratio))
        return t * t * (3.0 - 2.0 * t)

    def maybe_switch(self) -> bool:
        """紧迫度积分满 1 触发切换。两条曲线约束：
        - 积分上限封顶 1.0：陈年积压不会在零紧迫度时级联触发切换；
        - 触发时要求当前紧迫度 > 0：切换必须由当下的曲线状态正名。
        """
        urgency = self.switch_urgency()
        if urgency <= 0.0:
            return False
        self.switch_urgency_acc = min(1.0, self.switch_urgency_acc + urgency)
        if self.switch_urgency_acc < 1.0:
            return False
        self.switch_urgency_acc = 0.0
        self.switch()
        return True

    def switch(self) -> None:
        old = self.active
        new = self.passive_h().name
        old_h, new_h = self.hemispheres[old], self.hemispheres[new]
        # 1) 债务交换：双方最新 replay 克隆入对方债务（已学材料，非新压力）
        for source_h, target_h in ((old_h, new_h), (new_h, old_h)):
            for item in source_h.replay.latest(2):
                target_h.admit_exchange(
                    Seq2SeqSample(
                        ids=item.ids.copy(), labels=item.labels.copy(),
                        weight=1.0, source=f"exchange:{item.source}",
                    )
                )
        # 2) CPU 主副本上 2% 对称插值（双方都已同步到 CPU 最新状态）
        old_h.to_cpu()
        mix = SYNC_MIX / 2
        with torch.no_grad():
            for left_p, right_p in zip(old_h.model.parameters(), new_h.model.parameters()):
                snapshot = left_p.detach().clone()
                left_p.lerp_(right_p, mix)
                right_p.lerp_(snapshot, mix)
        # 3) 换驻留：旧活动侧落回 CPU，新活动侧上 GPU
        new_h.ensure_gpu()
        self.active = new
        self.switch_count += 1
        # 4) 压力基线重置（切换后压力从零起算）
        for name, hemisphere in self.hemispheres.items():
            self.pressure_baseline[name] = hemisphere.debt.total_clean_added
        self.switch_log.append({
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "from": old,
            "to": new,
            "index": self.switch_count,
        })

    # ---------- 被动巩固（换入-训练-换回） ----------

    def consolidate_passive(self, intensity: float) -> dict | None:
        passive = self.passive_h()
        samples = passive.consolidation_samples()
        if not samples:
            return None
        self.active_h().to_cpu()
        result = passive.consolidate(samples, intensity)
        passive.to_cpu()          # 巩固完必须换出，否则两侧同时驻留 GPU
        self.ensure_active_gpu()
        return result

    # ---------- 每轮数据流 ----------

    def admit_exchange_sample(self, sample: Sample) -> None:
        self.active_h().admit_pressure(sample)
