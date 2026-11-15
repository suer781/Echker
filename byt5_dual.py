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
import shutil
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.optimization import Adafactor

from life_agent.memory import DebtLedger, ReplayBuffer, Seq2SeqSample

BASE = "google/byt5-small"
BASE_LR = 1e-4
ENC_MAX = 512
DEC_CAP = 192
SYNC_MIX = 0.02          # 合计 2% 对称插值；执行时每侧 lerp SYNC_MIX/2=1%
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

    def _sync_device_field(self) -> None:
        """以第一个参数的真实所在设备为准回写 device 字段（P1-3）。"""
        try:
            first = next(self.model.parameters())
            self.device = "cuda" if first.is_cuda else "cpu"
        except StopIteration:
            self.device = "cpu"

    def ensure_gpu(self) -> None:
        if self.device == "cuda":
            return
        try:
            self.model.to("cuda")
        except torch.cuda.OutOfMemoryError:
            self._sync_device_field()
            # P2-1：部分迁移后尽力整体回到 CPU，避免撕裂驻留让下轮以
            # RuntimeError（而非 OOM）崩溃
            try:
                self.model.to("cpu")
            except Exception:
                pass
            self._sync_device_field()
            raise
        self._sync_device_field()
        self._move_optimizer_state("cuda")

    def to_cpu(self) -> None:
        if self.device == "cpu":
            return
        try:
            self.model.to("cpu")
        except Exception:
            self._sync_device_field()
            raise
        self._sync_device_field()
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
            # fp32 T5 的已知现象：个别批前向会出 NaN。降半强度重试一次，
            # 仍非有限则跳过（权重不动，无污染）。
            for group in self.optimizer.param_groups:
                group["lr"] = BASE_LR * intensity * 0.5
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.model(input_ids=inputs, labels=label_tensor).loss
            loss_value = float(loss.item())
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                return {"loss": loss_value, "learned": False, "reason": "non-finite-retry"}
        if loss_value > max_loss:
            # 质量闸门：模型对正确字节毫无把握（多半是自己的退化输出），
            # 把垃圾当教材会自我污染权重——跳过本轮更新。
            return {"loss": loss_value, "learned": False, "reason": "quality-gate"}
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        if not torch.isfinite(total_norm):
            # P2-7：反向 NaN 会让 step 把权重整体毒化为 NaN 并静默停学
            self.optimizer.zero_grad(set_to_none=True)
            return {"loss": loss_value, "learned": False, "reason": "non-finite-grad"}
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
        trained_indices: list[int] = []
        try:
            for step in range(CONSOLIDATION_STEPS):
                index = step % len(samples)
                sample = samples[index]
                # seq2seq 语义：encoder = 用户字节，decoder = 回应字节（账本以字节为单位）
                enc_tokens = [self.byte_to_id[b] for b in sample.ids[-ENC_MAX:] if b in self.byte_to_id]
                # TTT 语义：巩固目标 = 账本样本的观测输入自身（ids 的后段），
                # 不再使用 labels（旧回应标签语义已废弃）
                dec_source = sample.ids[-DEC_CAP:]
                dec_tokens = [self.byte_to_id[b] for b in dec_source if b in self.byte_to_id]
                dec_tokens += [self.tokenizer.eos_token_id]
                inputs = torch.tensor([enc_tokens], device="cuda")
                targets = torch.tensor([dec_tokens], device="cuda")
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                self.optimizer.zero_grad(set_to_none=True)
                out = self.model(input_ids=inputs, labels=targets)
                loss = out.loss
                loss_value = float(loss.item())
                # P0-1：巩固侧同款质量闸门——活动侧被拒的垃圾不得经债务
                # →巩固→交换的路径烧进被动侧真权重。
                if not torch.isfinite(loss) or loss_value > 2.0:
                    continue
                loss.backward()
                # P2-7：反向 NaN 卫兵——clip 范数非有限时 step 会把权重整体
                # 变 NaN 并从此静默停学，必须在 step 前拦截。
                total_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                if not torch.isfinite(total_norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.optimizer.step()
                steps_done += 1
                trained_indices.append(index)
                last_loss = loss_value
        except Exception:
            # P1-4：巩固失败必须释放 due() 的预留，否则这批债务本会话
            # 永远不会被再次选中（legacy B4 修复的对应移植）。
            self.debt.release_reservation(samples)
            raise
        if trained_indices:
            # P1-1：只对**实际训练过的**样本记账/入回放/计入容量——
            # 位置前缀切片会把被闸门拒绝的样本错记为已训练。
            committed = [samples[i] for i in trained_indices]
            self.debt.commit_repayment(committed, trained_steps=len(trained_indices))
            self.debt.decay_after_training()
            self.replay.add(Seq2SeqSample(
                ids=committed[0].ids.copy(), labels=committed[0].labels.copy(),
                weight=1.0, source=f"consolidate:{self.name}",
            ))
        # 全部剩余样本（含被拒/未训练的）释放预留，下轮可重选
        self.debt.release_reservation(samples)
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
        import os
        import time as _time
        out = self.state_dir / "hemispheres" / self.name
        tmp = out.with_suffix(".tmp")
        # 用户实测发现的 bug：重复保存时残留 .tmp 会污染写入，且 Windows
        # 目录 replace 对已存在目标/被杀软锁定的新档抛 PermissionError。
        # 修复：残留 tmp 先清；替换失败按 1s/2s/4s 退避重试三次。
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        was_gpu = self.device == "cuda"
        self.to_cpu()   # 权重落盘前回到 CPU
        corrupt = out.with_suffix(".corrupt")
        if corrupt.exists():
            shutil.rmtree(corrupt)   # N4：保存成功后清理历史隔离产物
        self.model.save_pretrained(tmp, safe_serialization=True)
        self.tokenizer.save_pretrained(tmp)
        torch.save(self.optimizer.state_dict(), tmp / "optimizer_state.pt")
        memory = {"replay": self.replay.to_dict(), "debt": self.debt.to_dict()}
        memory_tmp = tmp / "memory.json.tmp"
        memory_tmp.write_text(json.dumps(memory, ensure_ascii=False), encoding="utf-8")
        os.replace(memory_tmp, tmp / "memory.json")
        # P1-5：绝不先删旧档——旧档改名备份，新档就位后再删备份，
        # 消除"旧档已毁、新档未就位"的断电窗口。
        bak = out.with_suffix(".bak")
        if out.exists():
            if bak.exists():
                shutil.rmtree(bak)      # 上一代备份此时才清理（新档已就位）
            os.replace(out, bak)        # 现档退位为备份（新档就位失败时仍可恢复）
        for attempt in range(3):
            try:
                os.replace(tmp, out)
                break
            except PermissionError as exc:
                # Windows：杀软/索引器会锁定刚写完 1.2GB 的新档目录
                if attempt == 2:
                    raise PermissionError(
                        f"{self.name} 保存三次均被锁定（{exc}）。上一代权重在 {bak}，"
                        f"可手动恢复；或稍后重试 /save。") from exc
                _time.sleep(1.0 * (attempt + 1))
        # bak 保留至下一次保存（提供上一代回退路径），不立即删除
        if was_gpu:
            self.ensure_gpu()   # 恢复原驻留

    def load(self) -> bool:
        import os
        out = self.state_dir / "hemispheres" / self.name
        # N2：主档不完整（config 或 weights 缺失）即判定需要恢复
        main_ok = (out / "config.json").exists() and (out / "model.safetensors").exists()
        if not main_ok:
            # P2-4：上一代备份存活 → 先恢复备份再加载
            bak = out.with_suffix(".bak")
            bak_ok = (bak / "config.json").exists() and (bak / "model.safetensors").exists()
            if bak_ok:
                print(f"[{self.name}] 主档缺失/不完整，从备份恢复。", flush=True)
                # N1：Windows 的 replace 不能覆盖已存在目录——残档先挪走
                if out.exists():
                    shutil.rmtree(out)
                os.replace(bak, out)
            elif not out.exists():
                return False
            # 残档存在但无备份：交给后续加载路径隔离（.corrupt），不在此崩溃
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(out, dtype=torch.float32)
            model.config.tie_word_embeddings = False
            for name_, param in model.named_parameters():
                param.requires_grad = "block" in name_ and (".q." in name_ or ".v." in name_)
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            tokenizer = AutoTokenizer.from_pretrained(out)
            # P0-1：load() 产生全新 Parameter——优化器必须在**新参数**上重建
            # 后再载入状态，否则 step() 永久空操作（日志照常打印学习而权重
            # 冻结），且旧参数被引用无法回收（每次重启泄漏 2.4GB RAM）。
            trainable = [p for n, p in model.named_parameters() if p.requires_grad]
            optimizer = Adafactor(trainable, lr=BASE_LR, scale_parameter=False, relative_step=False)
            opt_path = out / "optimizer_state.pt"
            if opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=True))
            memory_path = out / "memory.json"
            if memory_path.exists():
                memory = json.loads(memory_path.read_text(encoding="utf-8"))
                self.replay.load_dict(memory.get("replay", {}))
                self.debt.load_dict(memory.get("debt", {}))
        except Exception as exc:
            print(f"[{self.name}] 检查点损坏，已隔离该半球档案：{exc}", flush=True)
            corrupt = out.with_suffix(".corrupt")
            if out.exists():
                if corrupt.exists():
                    shutil.rmtree(corrupt)   # P1-2：Windows 不能 replace 覆盖已存在目录
                os.replace(out, corrupt)
            return False
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = "cpu"
        return True


class Byt5DualBrain:
    """双半球大脑：活动侧 GPU 在线生成+学习，被动侧巩固，曲线自适应切换。"""

    def __init__(self, source: str, state_dir: Path) -> None:
        self.source = source
        self.state_dir = state_dir
        self.hemispheres = {
            name: ByteHemisphere(name, source, state_dir) for name in ("left", "right")
        }
        self.active = "left"
        self.switch_urgency_acc = 0.0
        self.pressure_baseline = {"left": 0.0, "right": 0.0}
        self.switch_count = 0
        self._pending_switch = None   # P2-2：待完成切换的重入保护
        self.switch_log: list[dict] = []

    def load(self) -> bool:
        loaded = [h.load() for h in self.hemispheres.values()]
        # P1-5：部分加载（一侧完整一侧损坏）时整体判为新档，绝不混血——
        # 混血 + 压力基线归零会让首轮出现假切换。
        if not all(loaded):
            # P2-3：绝不混血——成功侧保留但基线对齐当前账本（否则旧累计
            # 压力在首轮饱和触发假切换）；失败侧按全新重建。
            any_exists = False
            for ok, (name, hemisphere) in zip(loaded, self.hemispheres.items()):
                if ok:
                    hemisphere.to_cpu()
                    self.pressure_baseline[name] = hemisphere.debt.total_clean_added
                    any_exists = True
                else:
                    self.hemispheres[name] = ByteHemisphere(name, self.source, self.state_dir)
            if any_exists:
                print("[双半球] 检查点不完整：成功侧保留（基线已校准），失败侧全新重建。", flush=True)
            else:
                print("[双半球] 无历史检查点，按全新档启动。", flush=True)
            return False
        meta_path = self.state_dir / "brain.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                active = meta.get("active")
                self.active = active if active in ("left", "right") else "left"
                self.switch_urgency_acc = max(0.0, min(1.0, float(meta.get("switch_urgency_acc", 0.0))))
                baselines = meta.get("pressure_baseline", {})
                self.pressure_baseline = {
                    name: max(0.0, float(baselines.get(name, 0.0))) for name in ("left", "right")
                }
                self.switch_count = int(meta.get("switch_count", 0))
                log = meta.get("switch_log")
                self.switch_log = log if isinstance(log, list) else []
                return True
            except (json.JSONDecodeError, TypeError, ValueError, UnicodeDecodeError) as exc:
                print(f"[双半球] brain.json 元数据损坏/类型异常（{exc}），按默认处理并校准基线。", flush=True)
                for name, hemisphere in self.hemispheres.items():
                    self.pressure_baseline[name] = hemisphere.debt.total_clean_added
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
        import os
        self.state_dir.mkdir(parents=True, exist_ok=True)
        meta_tmp = self.state_dir / "brain.json.tmp"
        meta_tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(meta_tmp, self.state_dir / "brain.json")

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
        # 容量下限提到满批巩固量级（4.0）：冷启动期被动侧尚未形成真实容量，
        # 0.5 会让单轮数据就饱和紧迫度，退化成每 2-3 轮一次的病态轮换。
        capacity = max(passive.debt.repayment_capacity, 4.0)
        ratio = growth / capacity / SWITCH_RATIO_SCALE
        t = max(0.0, min(1.0, ratio))
        return t * t * (3.0 - 2.0 * t)

    def switch_begin(self) -> None:
        """切换第一步（纯 CPU 变更，失败零副作用）：
        旧侧落回 CPU（腾出显存）→ 预检 → 债务交换 → 2% 对称插值 → 角色翻转。
        预检在旧侧腾出显存**之后**执行（此时空闲 ≈ 原空闲 + 1.14GB）。"""
        old = self.active
        new = self.passive_h().name
        old_h, new_h = self.hemispheres[old], self.hemispheres[new]
        old_h.to_cpu()   # 幂等；为新侧腾出显存
        free, _total = torch.cuda.mem_get_info()
        if free < int(1.3 * 1024**3):
            raise torch.cuda.OutOfMemoryError(
                f"切换预检：GPU 空闲 {free/1024**3:.2f}GB 不足新侧驻留（需 ~1.3GB），本轮不切换（零副作用）")
        for source_h, target_h in ((old_h, new_h), (new_h, old_h)):
            for item in source_h.replay.latest(2):
                target_h.admit_exchange(
                    Seq2SeqSample(
                        ids=item.ids.copy(), labels=item.labels.copy(),
                        weight=1.0, source=f"exchange:{item.source}",
                    )
                )
        mix = SYNC_MIX / 2
        with torch.no_grad():
            for left_p, right_p in zip(old_h.model.parameters(), new_h.model.parameters()):
                snapshot = left_p.detach().clone()
                left_p.lerp_(right_p, mix)
                right_p.lerp_(snapshot, mix)
        # 标记待完成 + 角色先行翻转：设备迁移由 complete 幂等补齐
        self._pending_switch = new
        self.active = new
        self.switch_count += 1

    def switch_complete(self) -> None:
        """切换第二步（幂等）：新侧上 GPU、旧侧驻留 CPU、基线重置。"""
        new = self._pending_switch
        old = "right" if new == "left" else "left"
        self.hemispheres[new].ensure_gpu()
        self.hemispheres[old].to_cpu()
        for name, hemisphere in self.hemispheres.items():
            self.pressure_baseline[name] = hemisphere.debt.total_clean_added
        self.switch_log.append({
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "from": old,
            "to": new,
            "index": self.switch_count,
        })
        self._pending_switch = None

    def maybe_switch(self) -> bool:
        # 未竟切换优先补完（只做幂等设备迁移，不重复交换/插值）
        if self._pending_switch is not None:
            self.switch_complete()
            return True
        urgency = self.switch_urgency()
        if urgency <= 0.0:
            return False
        self.switch_urgency_acc = min(1.0, self.switch_urgency_acc + urgency)
        if self.switch_urgency_acc < 1.0:
            return False
        self.switch_urgency_acc = 0.0
        self.switch_begin()
        self.switch_complete()
        return True

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
