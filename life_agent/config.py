"""Configuration for the local continual-learning agent."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 262
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    hidden_dim: int = 256
    min_hidden_dim: int = 128
    max_hidden_dim: int = 512
    max_seq_len: int = 512
    dropout: float = 0.0

    def validate(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not self.min_hidden_dim <= self.hidden_dim <= self.max_hidden_dim:
            raise ValueError("hidden_dim must be within configured structural bounds")
        if self.vocab_size != 262:
            raise ValueError("the byte codec requires vocab_size=262")
        if self.n_layers < 1 or self.max_seq_len < 32:
            raise ValueError("invalid model depth or context length")


@dataclass
class RuntimeConfig:
    inference_device: str = "cpu"
    training_device: str = "cuda"
    require_cuda: bool = True
    seed: int = 7
    max_new_tokens: int = 96
    ttt_chunk_tokens: int = 8
    ttt_steps: int = 1
    ttt_learning_rate: float = 0.0003
    consolidation_steps: int = 2
    batch_size: int = 4
    learning_rate: float = 0.0003
    min_learning_rate: float = 0.00001
    max_learning_rate: float = 0.003
    lr_up_factor: float = 1.05
    lr_down_factor: float = 0.6
    loss_spike_ratio: float = 1.35
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    replay_capacity: int = 1024
    debt_capacity: int = 512
    debt_retention: float = 0.35
    debt_decay: float = 0.75
    debt_min_weight: float = 0.005
    sync_mix: float = 0.02
    replay_probability: float = 0.35
    adaptive_switching: bool = True
    checkpoint_interval_s: float = 300.0
    context_turns: int = 6
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    structure_interval: int = 32
    structure_warmup: int = 64
    growth_size: int = 16
    prune_size: int = 8
    idle_sleep_s: float = 0.1
    max_frame_bytes: int = 1048576
    feed_steps: int = 1
    memory_budget_mb: int = 6144

    def validate(self) -> None:
        if self.inference_device != "cpu":
            raise ValueError("this implementation deliberately reserves inference for CPU")
        if not 0 < self.debt_retention <= 1 or not 0 < self.debt_decay <= 1:
            raise ValueError("debt retention and decay must be in (0, 1]")
        if not 0 <= self.sync_mix <= 0.25:
            raise ValueError("sync_mix must be in [0, 0.25]")
        if self.ttt_chunk_tokens < 1 or self.max_new_tokens < 1:
            raise ValueError("generation lengths must be positive")
        if self.min_learning_rate <= 0 or self.max_learning_rate < self.min_learning_rate:
            raise ValueError("invalid learning-rate bounds")
        if self.batch_size < 1 or self.replay_capacity < 1 or self.debt_capacity < 1:
            raise ValueError("batch and memory capacities must be positive")
        if self.max_frame_bytes < 32 or self.feed_steps < 1:
            raise ValueError("frame size and feed-step limits must be positive")
        if self.memory_budget_mb < 1024:
            raise ValueError("memory budget must leave room for the two hemisphere models")


@dataclass
class AppConfig:
    model: ModelConfig
    runtime: RuntimeConfig

    @classmethod
    def default(cls) -> "AppConfig":
        return cls(ModelConfig(), RuntimeConfig())

    def validate(self) -> None:
        self.model.validate()
        self.runtime.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        self.validate()
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        model_names = {f.name for f in fields(ModelConfig)}
        runtime_names = {f.name for f in fields(RuntimeConfig)}
        config = cls(
            ModelConfig(**{k: v for k, v in raw.get("model", {}).items() if k in model_names}),
            RuntimeConfig(**{k: v for k, v in raw.get("runtime", {}).items() if k in runtime_names}),
        )
        config.validate()
        return config
