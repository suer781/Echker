# 一次性补丁：ByteHemisphere 重构为单对象原地迁移驻留
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "byt5_dual.py"
src = path.read_text(encoding="utf-8")

old_init = '''    def __init__(self, name: str, source: str, state_dir: Path) -> None:
        self.name = name
        self.source = source
        self.state_dir = state_dir
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.cpu_model = AutoModelForSeq2SeqLM.from_pretrained(source, dtype=torch.float32)
        for name_, param in self.cpu_model.named_parameters():
            param.requires_grad = "block" in name_ and (".q." in name_ or ".v." in name_)
        self.gpu_model = None
        self.optimizer = None
        self.optimizer_state = None      # 换出时保存的优化器状态（CPU）
        self.replay = ReplayBuffer(256)
        self.debt = DebtLedger(128, 0.75, 0.005)
        vocab = self.tokenizer.get_vocab()
        self.byte_to_id = {b: vocab[chr(b)] for b in range(256)}

    # ---------- 驻留管理 ----------

    def ensure_gpu(self) -> None:
        if self.gpu_model is not None:
            return
        config = self.cpu_model.config
        # 与微调时保持一致的非绑定拓扑（谷歌 checkpoint 的 shared/lm_head
        # 本就分离，见 HANDOVER 0.6.0）
        config.tie_word_embeddings = False
        gpu = type(self.cpu_model)(config)
        gpu.load_state_dict(self.cpu_model.state_dict())
        self.gpu_model = gpu.cuda()
        self.gpu_model.gradient_checkpointing_enable()
        self.gpu_model.config.use_cache = False
        trainable = [p for n, p in self.gpu_model.named_parameters() if p.requires_grad]
        self.optimizer = Adafactor(trainable, lr=BASE_LR, scale_parameter=False, relative_step=False)
        if self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)
            self.optimizer_state = None

    def to_cpu(self) -> None:
        if self.gpu_model is not None:
            return
        self.optimizer_state = self._optimizer_state_to_cpu()
        self.cpu_model.load_state_dict(self.gpu_model.state_dict())
        self.gpu_model = None
        self.optimizer = None
        torch.cuda.empty_cache()

    def _optimizer_state_to_cpu(self) -> dict:
        state = self.optimizer.state_dict()

        def to_cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {key: to_cpu(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(to_cpu(item) for item in value)
            return value

        return to_cpu(state)'''

new_init = '''    def __init__(self, name: str, source: str, state_dir: Path) -> None:
        self.name = name
        self.source = source
        self.state_dir = state_dir
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        # 单对象原地迁移：权重只有一份，驻留 = 整体 .to(device)。
        # 没有"主副本+工作副本"的双份驻留，也没有换入时的重建瞬时开销。
        self.model = AutoModelForSeq2SeqLM.from_pretrained(source, dtype=torch.float32)
        # 与微调时保持一致的非绑定拓扑（谷歌 checkpoint 的 shared/lm_head
        # 本就分离，见 HANDOVER 0.6.0）
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

    # ---------- 驻留管理（原地迁移） ----------

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
                    state[key] = value.to(device)'''

assert old_init in src, "init 锚点未命中"
src = src.replace(old_init, new_init)

pairs = [
    (
        "        self.gpu_model.train()\n        loss = self.gpu_model(input_ids=inputs, labels=label_tensor).loss",
        "        self.model.train()\n        loss = self.model(input_ids=inputs, labels=label_tensor).loss",
    ),
    (
        "        torch.nn.utils.clip_grad_norm_(\n            [p for p in self.gpu_model.parameters() if p.requires_grad], 1.0\n        )",
        "        torch.nn.utils.clip_grad_norm_(\n            [p for p in self.model.parameters() if p.requires_grad], 1.0\n        )",
    ),
    (
        "        self.gpu_model.train()\n        lr = BASE_LR * max(0.25, intensity)",
        "        self.model.train()\n        lr = BASE_LR * max(0.25, intensity)",
    ),
    (
        "            out = self.gpu_model(input_ids=inputs, labels=targets)\n            loss = out.loss",
        "            out = self.model(input_ids=inputs, labels=targets)\n            loss = out.loss",
    ),
    (
        "            torch.nn.utils.clip_grad_norm_(\n                [p for p in self.gpu_model.parameters() if p.requires_grad], 1.0\n            )",
        "            torch.nn.utils.clip_grad_norm_(\n                [p for p in self.model.parameters() if p.requires_grad], 1.0\n            )",
    ),
]
for old, new in pairs:
    assert old in src, f"替换锚点未命中: {old[:60]!r}"
    src = src.replace(old, new)

old_save = '''    def save(self) -> None:
        out = self.state_dir / "hemispheres" / self.name
        out.mkdir(parents=True, exist_ok=True)
        if self.gpu_model is not None:
            self.to_cpu()   # GPU 上更新的权重先同步回主副本
        self.cpu_model.save_pretrained(out, safe_serialization=True)
        self.tokenizer.save_pretrained(out)
        if self.optimizer_state is not None:
            torch.save(self.optimizer_state, out / "optimizer_state.pt")
        memory = {"replay": self.replay.to_dict(), "debt": self.debt.to_dict()}
        (out / "memory.json").write_text(json.dumps(memory, ensure_ascii=False), encoding="utf-8")

    def load(self) -> bool:
        out = self.state_dir / "hemispheres" / self.name
        if not (out / "config.json").exists():
            return False
        self.cpu_model = AutoModelForSeq2SeqLM.from_pretrained(out, dtype=torch.float32)
        for name_, param in self.cpu_model.named_parameters():
            param.requires_grad = "block" in name_ and (".q." in name_ or ".v." in name_)
        self.tokenizer = AutoTokenizer.from_pretrained(out)
        opt_path = out / "optimizer_state.pt"
        if opt_path.exists():
            self.optimizer_state = torch.load(opt_path, map_location="cpu", weights_only=True)
        memory_path = out / "memory.json"
        if memory_path.exists():
            memory = json.loads(memory_path.read_text(encoding="utf-8"))
            self.replay.load_dict(memory.get("replay", {}))
            self.debt.load_dict(memory.get("debt", {}))
        return True'''

new_save = '''    def save(self) -> None:
        out = self.state_dir / "hemispheres" / self.name
        out.mkdir(parents=True, exist_ok=True)
        was_gpu = self.device == "cuda"
        self.to_cpu()   # 权重落盘前回到 CPU
        self.model.save_pretrained(out, safe_serialization=True)
        self.tokenizer.save_pretrained(out)
        torch.save(self.optimizer.state_dict(), out / "optimizer_state.pt")
        memory = {"replay": self.replay.to_dict(), "debt": self.debt.to_dict()}
        (out / "memory.json").write_text(json.dumps(memory, ensure_ascii=False), encoding="utf-8")
        if was_gpu:
            self.ensure_gpu()   # 恢复原驻留

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
        self.tokenizer = AutoTokenizer.from_pretrained(out)
        trainable = [p for n, p in self.model.named_parameters() if p.requires_grad]
        self.optimizer = Adafactor(trainable, lr=BASE_LR, scale_parameter=False, relative_step=False)
        opt_path = out / "optimizer_state.pt"
        if opt_path.exists():
            self.optimizer.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=True))
        memory_path = out / "memory.json"
        if memory_path.exists():
            memory = json.loads(memory_path.read_text(encoding="utf-8"))
            self.replay.load_dict(memory.get("replay", {}))
            self.debt.load_dict(memory.get("debt", {}))
        self.device = "cpu"
        return True'''

assert old_save in src, "save/load 锚点未命中"
src = src.replace(old_save, new_save)

old_lerp = '''        # 2) CPU 主副本上 2% 对称插值（双方都已同步到 CPU 最新状态）
        old_h.to_cpu()
        mix = SYNC_MIX / 2
        with torch.no_grad():
            for left_p, right_p in zip(old_h.cpu_model.parameters(), new_h.cpu_model.parameters()):'''
new_lerp = '''        # 2) CPU 上 2% 对称插值（旧活动侧已迁回 CPU，权重即最新状态）
        old_h.to_cpu()
        mix = SYNC_MIX / 2
        with torch.no_grad():
            for left_p, right_p in zip(old_h.model.parameters(), new_h.model.parameters()):'''
assert old_lerp in src, "lerp 锚点未命中"
src = src.replace(old_lerp, new_lerp)

path.write_text(src, encoding="utf-8")
print("ByteHemisphere 原地迁移重构完成；cpu_model 残留引用数:", src.count("cpu_model"))
