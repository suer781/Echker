# 修复轮 1 v2：锚点已按实际文件文本校准
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "byt5_dual.py"
src = path.read_text(encoding="utf-8")

# ---------- P2-3: SYNC_MIX 语义 ----------
old = "SYNC_MIX = 0.01          # 每侧 1%，合计 2% 对称插值（legacy sync_mix=0.02）"
new = "SYNC_MIX = 0.02          # 合计 2% 对称插值；执行时每侧 lerp SYNC_MIX/2=1%"
assert old in src, "P2-3"
src = src.replace(old, new)

# ---------- P1-3: 驻留迁移失败时 device 再同步 ----------
old = '''    def ensure_gpu(self) -> None:
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
        torch.cuda.empty_cache()'''
new = '''    def _sync_device_field(self) -> None:
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
        torch.cuda.empty_cache()'''
assert old in src, "P1-3"
src = src.replace(old, new)

# ---------- P1-4: 巩固失败释放预留 ----------
old = '''        self.model.train()
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
        return {"steps": steps_done, "loss": last_loss}'''
new = '''        self.model.train()
        lr = BASE_LR * max(0.25, intensity)
        steps_done = 0
        last_loss = None
        try:
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
        except Exception:
            # P1-4：巩固失败必须释放 due() 的预留，否则这批债务本会话
            # 永远不会被再次选中（legacy B4 修复的对应移植）。
            self.debt.release_reservation(samples)
            raise
        if steps_done:
            self.debt.commit_repayment(samples, trained_steps=steps_done)
            self.debt.decay_after_training()
            stored_sample = samples[0]
            self.replay.add(Seq2SeqSample(
                ids=stored_sample.ids.copy(), labels=stored_sample.labels.copy(),
                weight=1.0, source=f"consolidate:{self.name}",
            ))
        else:
            self.debt.release_reservation(samples)
        return {"steps": steps_done, "loss": last_loss}'''
assert old in src, "P1-4"
src = src.replace(old, new)

# ---------- P2-1 + P0-1: save() 原子目录写；load() 优化器重建+隔离损坏 ----------
old = '''    def save(self) -> None:
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
        return True'''
new = '''    def save(self) -> None:
        import os
        out = self.state_dir / "hemispheres" / self.name
        tmp = out.with_suffix(".tmp")   # P2-1：先写临时目录，成功后整体替换（原子化）
        tmp.mkdir(parents=True, exist_ok=True)
        was_gpu = self.device == "cuda"
        self.to_cpu()   # 权重落盘前回到 CPU
        self.model.save_pretrained(tmp, safe_serialization=True)
        self.tokenizer.save_pretrained(tmp)
        torch.save(self.optimizer.state_dict(), tmp / "optimizer_state.pt")
        memory = {"replay": self.replay.to_dict(), "debt": self.debt.to_dict()}
        memory_tmp = tmp / "memory.json.tmp"
        memory_tmp.write_text(json.dumps(memory, ensure_ascii=False), encoding="utf-8")
        os.replace(memory_tmp, tmp / "memory.json")
        if out.exists():
            shutil.rmtree(out)
        os.replace(tmp, out)
        if was_gpu:
            self.ensure_gpu()   # 恢复原驻留

    def load(self) -> bool:
        import os
        out = self.state_dir / "hemispheres" / self.name
        if not (out / "config.json").exists():
            return False
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
                os.replace(out, corrupt)
            return False
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = "cpu"
        return True'''
assert old in src, "P2-1/P0-1"
src = src.replace(old, new)

# ---------- P2-4: 容量下限 0.5 → 4.0 ----------
old = '''        growth = max(0.0, active.debt.total_clean_added - self.pressure_baseline[active.name])
        capacity = max(passive.debt.repayment_capacity, 0.5)
        ratio = growth / capacity / SWITCH_RATIO_SCALE'''
new = '''        growth = max(0.0, active.debt.total_clean_added - self.pressure_baseline[active.name])
        # 容量下限提到满批巩固量级（4.0）：冷启动期被动侧尚未形成真实容量，
        # 0.5 会让单轮数据就饱和紧迫度，退化成每 2-3 轮一次的病态轮换。
        capacity = max(passive.debt.repayment_capacity, 4.0)
        ratio = growth / capacity / SWITCH_RATIO_SCALE'''
assert old in src, "P2-4"
src = src.replace(old, new)

# ---------- P1-5: brain.load 部分加载整体回退 + brain.json 容错 ----------
old = '''    def load(self) -> bool:
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
        return False'''
new = '''    def load(self) -> bool:
        loaded = [h.load() for h in self.hemispheres.values()]
        # P1-5：部分加载（一侧完整一侧损坏）时整体判为新档，绝不混血——
        # 混血 + 压力基线归零会让首轮出现假切换。
        if not all(loaded):
            for ok, hemisphere in zip(loaded, self.hemispheres.values()):
                if ok:
                    hemisphere.to_cpu()
            print("[双半球] 检查点不完整（可能上次保存中途断电），已放弃恢复，按全新档启动。", flush=True)
            return False
        meta_path = self.state_dir / "brain.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"[双半球] brain.json 损坏（{exc}），元数据按默认处理。", flush=True)
                return True
            self.active = meta.get("active", "left")
            self.switch_urgency_acc = float(meta.get("switch_urgency_acc", 0.0))
            self.pressure_baseline = meta.get("pressure_baseline", self.pressure_baseline)
            self.switch_count = int(meta.get("switch_count", 0))
            self.switch_log = meta.get("switch_log", [])
            return True
        return False'''
assert old in src, "P1-5 brain"
src = src.replace(old, new)

# ---------- P2-1: brain.json 原子写 ----------
old = '''        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "brain.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )'''
new = '''        import os
        self.state_dir.mkdir(parents=True, exist_ok=True)
        meta_tmp = self.state_dir / "brain.json.tmp"
        meta_tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(meta_tmp, self.state_dir / "brain.json")'''
assert old in src, "brain.json 原子写"
src = src.replace(old, new)

# ---------- shutil 导入 ----------
old = '''import json
from pathlib import Path
import random
import time'''
new = '''import json
from pathlib import Path
import random
import shutil
import time'''
assert old in src, "shutil"
src = src.replace(old, new)

path.write_text(src, encoding="utf-8")
print("byt5_dual.py 修复轮 1 完成")
