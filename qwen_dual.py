"""Qwen dual-hemisphere training using the existing sequential-residency brain."""
from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
import random
import threading
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import Adafactor

from byt5_dual import ByteHemisphere, Byt5DualBrain
from life_agent.memory import DebtLedger, ReplayBuffer, Sample

MAX_SEQUENCE_LENGTH = 512
QWEN_LR = 2e-5
CONSOLIDATION_STEPS = 2


def _layers(model: nn.Module) -> list[nn.Module]:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise ValueError("dynamic Qwen growth requires a model.model.layers stack")
    return list(layers)


def _restore_tied_state_keys(
    model: nn.Module,
    config: Any,
    state: dict[str, torch.Tensor],
) -> None:
    if not getattr(config, "tie_word_embeddings", False):
        return
    model_state = model.state_dict()
    for name, tensor in model_state.items():
        if name in state:
            continue
        tied_source = next(
            (
                source_name
                for source_name, source_tensor in model_state.items()
                if source_name in state and source_tensor.data_ptr() == tensor.data_ptr()
            ),
            None,
        )
        if tied_source is not None:
            state[name] = state[tied_source]


def _activation(name: str, values: torch.Tensor) -> torch.Tensor:
    if name in ("silu", "swish"):
        return F.silu(values)
    if name == "gelu":
        return F.gelu(values)
    if name == "relu":
        return F.relu(values)
    raise ValueError(f"unsupported Qwen MLP activation for dynamic growth: {name}")


def _resize_mlp(layer: nn.Module, new_dim: int, activation: torch.Tensor) -> dict[str, Any]:
    mlp = layer.mlp
    gate = mlp.gate_proj
    up = mlp.up_proj
    down = mlp.down_proj
    old_dim = gate.out_features
    if new_dim == old_dim:
        return {"old_dim": old_dim, "new_dim": new_dim, "mode": "none"}
    if new_dim < 1:
        raise ValueError("Qwen MLP width must remain positive")

    incoming = gate.weight.detach().float().norm(dim=1) + up.weight.detach().float().norm(dim=1)
    outgoing = down.weight.detach().float().norm(dim=0)
    importance = outgoing * activation.to(outgoing.device).clamp_min(1e-6) + incoming * 1e-4
    growing = new_dim > old_dim
    new_gate = nn.Linear(
        gate.in_features, new_dim, bias=gate.bias is not None,
        device=gate.weight.device, dtype=gate.weight.dtype,
    )
    new_up = nn.Linear(
        up.in_features, new_dim, bias=up.bias is not None,
        device=up.weight.device, dtype=up.weight.dtype,
    )
    new_down = nn.Linear(
        new_dim, down.out_features, bias=down.bias is not None,
        device=down.weight.device, dtype=down.weight.dtype,
    )

    with torch.no_grad():
        if growing:
            count = new_dim - old_dim
            sources = importance.topk(min(count, old_dim)).indices
            if count > old_dim:
                sources = sources.repeat(math.ceil(count / old_dim))[:count]
            new_gate.weight[:old_dim].copy_(gate.weight)
            new_up.weight[:old_dim].copy_(up.weight)
            new_down.weight[:, :old_dim].copy_(down.weight)
            for offset, source in enumerate(sources.tolist()):
                target = old_dim + offset
                new_gate.weight[target].copy_(gate.weight[source])
                new_gate.weight[target].add_(torch.randn_like(new_gate.weight[target]) * 1e-5)
                new_up.weight[target].copy_(up.weight[source])
                new_down.weight[:, target].copy_(down.weight[:, source] * 0.5)
                new_down.weight[:, source].mul_(0.5)
            if gate.bias is not None:
                new_gate.bias[:old_dim].copy_(gate.bias)
                new_up.bias[:old_dim].copy_(up.bias)
                for offset, source in enumerate(sources.tolist()):
                    target = old_dim + offset
                    new_gate.bias[target].copy_(gate.bias[source])
                    new_up.bias[target].copy_(up.bias[source])
            if down.bias is not None:
                new_down.bias.copy_(down.bias)
            mapping = [[old_dim + offset, int(source)] for offset, source in enumerate(sources.tolist())]
            new_activation = torch.cat((activation, activation[sources.cpu()]))
        else:
            keep = importance.topk(new_dim, sorted=False).indices.sort().values
            new_gate.weight.copy_(gate.weight[keep])
            new_up.weight.copy_(up.weight[keep])
            new_down.weight.copy_(down.weight[:, keep])
            if gate.bias is not None:
                new_gate.bias.copy_(gate.bias[keep])
                new_up.bias.copy_(up.bias[keep])
            if down.bias is not None:
                new_down.bias.copy_(down.bias)
            mapping = [int(index) for index in keep.tolist()]
            new_activation = activation[keep.cpu()].clone()

    mlp.gate_proj = new_gate
    mlp.up_proj = new_up
    mlp.down_proj = new_down
    for parameter in mlp.parameters():
        parameter.requires_grad_(True)
    return {
        "old_dim": old_dim,
        "new_dim": new_dim,
        "mode": "grow" if growing else "prune",
        "row_map" if growing else "keep_map": mapping,
        "activation": new_activation.cpu(),
    }


class QwenStructureController:
    def __init__(self, model: nn.Module) -> None:
        self.losses: deque[float] = deque(maxlen=64)
        self.last_change_step = -10**9
        self.activation_ema = [
            torch.ones(layer.mlp.gate_proj.out_features, dtype=torch.float32)
            for layer in _layers(model)
        ]
        self.base_widths = [value.numel() for value in self.activation_ema]
        self.growth_size = 16
        self.prune_size = 8
        self.warmup = 64
        self.interval = 32

    def observe(self, layer_index: int, hidden: torch.Tensor) -> None:
        current = hidden.detach().abs().mean(dim=(0, 1)).float().cpu()
        ema = self.activation_ema[layer_index]
        rate = 0.05 if self.losses else 1.0
        if ema.numel() != current.numel():
            raise ValueError("activation statistics do not match the current MLP width")
        ema.lerp_(current, rate)

    def consider(self, model: nn.Module, step: int, loss: float) -> dict[str, Any] | None:
        self.losses.append(loss)
        layers = _layers(model)
        if step < self.warmup or step - self.last_change_step < self.interval:
            return None
        if step % self.interval or len(self.losses) < 24 or not math.isfinite(loss):
            return None

        values = list(self.losses)
        previous = sum(values[-24:-8]) / 16
        recent = sum(values[-8:]) / 8
        improvement = (previous - recent) / max(abs(previous), 1e-6)
        layer_index = (step // self.interval) % len(layers)
        layer = layers[layer_index]
        width = layer.mlp.gate_proj.out_features
        base_width = self.base_widths[layer_index]
        event = None
        if improvement < 0.002 and width < base_width * 2:
            event = {
                "layer": layer_index,
                "new_dim": min(width + self.growth_size, base_width * 2),
            }
        elif improvement > 0.01 and width > max(1, base_width // 2):
            importance = (
                layer.mlp.down_proj.weight.detach().float().norm(dim=0)
                * self.activation_ema[layer_index].to(layer.mlp.down_proj.weight.device).clamp_min(1e-6)
                + 1e-4
                * (
                    layer.mlp.gate_proj.weight.detach().float().norm(dim=1)
                    + layer.mlp.up_proj.weight.detach().float().norm(dim=1)
                )
            )
            if importance.min().item() < importance.median().item() * 0.02:
                event = {
                    "layer": layer_index,
                    "new_dim": max(width - self.prune_size, base_width // 2),
                }
        if event:
            self.last_change_step = step
        return event

    def to_dict(self, model: nn.Module) -> dict[str, Any]:
        return {
            "losses": list(self.losses),
            "last_change_step": self.last_change_step,
            "activation_ema": [value.tolist() for value in self.activation_ema],
            "base_widths": self.base_widths,
            "widths": [layer.mlp.gate_proj.out_features for layer in _layers(model)],
        }

    def load_dict(self, raw: dict[str, Any], model: nn.Module) -> None:
        self.losses = deque((float(value) for value in raw.get("losses", [])), maxlen=64)
        self.last_change_step = int(raw.get("last_change_step", -10**9))
        base_widths = raw.get("base_widths")
        if base_widths is not None:
            if len(base_widths) != len(self.base_widths):
                raise ValueError("checkpoint dynamic-growth base widths do not match model depth")
            self.base_widths = [int(value) for value in base_widths]
        values = raw.get("activation_ema", [])
        if values:
            if len(values) != len(self.activation_ema):
                raise ValueError("checkpoint dynamic-growth layer count does not match the model")
            restored = [torch.tensor(value, dtype=torch.float32) for value in values]
            widths = [layer.mlp.gate_proj.out_features for layer in _layers(model)]
            if [value.numel() for value in restored] != widths:
                raise ValueError("checkpoint activation statistics do not match MLP widths")
            self.activation_ema = restored


class QwenHemisphere(ByteHemisphere):
    """A full-parameter Qwen model with the existing hemisphere residency rules."""

    def __init__(
        self,
        name: str,
        source: str,
        state_dir: Path,
        training_device: str,
        load_base: bool = True,
    ) -> None:
        self.name = name
        self.source = source
        self.state_dir = state_dir
        self.training_device = training_device
        self.tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
        self.model: nn.Module | None = None
        self.optimizer: Any = None
        self.replay = ReplayBuffer(256, token_domain=True)
        self.debt = DebtLedger(128, 0.75, 0.005, token_domain=True)
        self._train_lock = threading.RLock()
        self._memory_lock = threading.RLock()
        self.structure: QwenStructureController | None = None
        self.global_step = 0
        self.ttt_steps = 0
        self.consolidate_steps = 0
        self.last_loss: float | None = None
        self.last_event = "initialized"
        self.device = "cpu"
        if load_base:
            self._load_base()

    def _load_base(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.source,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        self.optimizer = Adafactor(
            self.model.parameters(),
            lr=QWEN_LR,
            scale_parameter=False,
            relative_step=False,
        )
        self.structure = QwenStructureController(self.model)

    def ensure_gpu(self) -> None:
        if self.training_device == "cpu":
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable")
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Qwen hemisphere has not loaded its model checkpoint")
        super().ensure_gpu()

    def _named_optimizer_state(self) -> dict[str, dict[str, Any]]:
        return {
            name: self.optimizer.state[parameter]
            for name, parameter in self.model.named_parameters()
            if parameter in self.optimizer.state
        }

    @staticmethod
    def _remap_axis(
        value: torch.Tensor,
        axis: int,
        result: dict[str, Any],
    ) -> torch.Tensor:
        if value.ndim == 1:
            axis = 0
        if result["mode"] == "grow":
            old_dim = int(result["old_dim"])
            new_value = torch.zeros(
                tuple(result["new_dim"] if dim == axis else size for dim, size in enumerate(value.shape)),
                dtype=value.dtype,
                device=value.device,
            )
            src = [slice(None)] * value.ndim
            dst = [slice(None)] * value.ndim
            src[axis] = slice(0, old_dim)
            dst[axis] = slice(0, old_dim)
            new_value[tuple(dst)] = value[tuple(src)]
            for target, source in result["row_map"]:
                dst[axis] = target
                src[axis] = source
                new_value[tuple(dst)] = value[tuple(src)]
            return new_value
        indices = torch.tensor(result["keep_map"], dtype=torch.long, device=value.device)
        return value.index_select(axis, indices)

    def _rebuild_optimizer(
        self,
        previous: dict[str, dict[str, Any]],
        layer_index: int,
        result: dict[str, Any],
    ) -> None:
        self.optimizer = Adafactor(
            self.model.parameters(),
            lr=QWEN_LR,
            scale_parameter=False,
            relative_step=False,
        )
        prefix = f"model.layers.{layer_index}.mlp."
        row_names = (prefix + "gate_proj.", prefix + "up_proj.")
        col_names = (prefix + "down_proj.",)
        old_dim = int(result["old_dim"])
        for name, parameter in self.model.named_parameters():
            source = previous.get(name)
            if source is None:
                continue
            remapped: dict[str, Any] = {}
            for key, value in source.items():
                if not torch.is_tensor(value) or value.ndim == 0:
                    remapped[key] = value.clone() if torch.is_tensor(value) else value
                    continue
                if name.startswith(row_names):
                    if key == "exp_avg_sq_row" and value.shape == (old_dim,):
                        remapped[key] = self._remap_axis(value, 0, result)
                    elif key == "exp_avg_sq_col" and value.shape == (parameter.shape[1],):
                        remapped[key] = value.clone()
                    elif key == "exp_avg_sq" and value.ndim == 2 and value.shape[0] == old_dim:
                        remapped[key] = self._remap_axis(value, 0, result)
                    elif key == "exp_avg_sq" and value.shape == (old_dim,):
                        remapped[key] = self._remap_axis(value, 0, result)
                    elif key in ("exp_avg_sq_row", "exp_avg_sq_col", "exp_avg_sq"):
                        remapped = {}
                        break
                    else:
                        remapped[key] = value.clone()
                elif name.startswith(col_names):
                    if key == "exp_avg_sq_col" and value.shape == (old_dim,):
                        remapped[key] = self._remap_axis(value, 0, result)
                    elif key == "exp_avg_sq_row" and value.shape == (parameter.shape[0],):
                        remapped[key] = value.clone()
                    elif key == "exp_avg_sq" and value.ndim == 2 and value.shape[1] == old_dim:
                        remapped[key] = self._remap_axis(value, 1, result)
                    elif key == "exp_avg_sq" and value.shape == (old_dim,):
                        remapped[key] = self._remap_axis(value, 0, result)
                    elif key in ("exp_avg_sq_row", "exp_avg_sq_col", "exp_avg_sq"):
                        remapped = {}
                        break
                    else:
                        remapped[key] = value.clone()
                else:
                    if key == "exp_avg_sq_row" and value.shape == (parameter.shape[0],):
                        remapped[key] = value.clone()
                    elif key == "exp_avg_sq_col" and value.shape == (parameter.shape[1],):
                        remapped[key] = value.clone()
                    elif tuple(value.shape) == tuple(parameter.shape):
                        remapped[key] = value.clone()
                    else:
                        remapped = {}
                        break
            if remapped:
                self.optimizer.state[parameter] = remapped

    def _resize(self, layer_index: int, new_dim: int) -> dict[str, Any]:
        if self.model is None or self.structure is None:
            raise RuntimeError("Qwen hemisphere has not loaded its model checkpoint")
        old_optimizer_state = self._named_optimizer_state()
        layer = _layers(self.model)[layer_index]
        result = _resize_mlp(layer, new_dim, self.structure.activation_ema[layer_index])
        if result["mode"] == "none":
            return result
        self.structure.activation_ema[layer_index] = result.pop("activation")
        self._rebuild_optimizer(old_optimizer_state, layer_index, result)
        self.model.config.dynamic_intermediate_sizes = [
            item.mlp.gate_proj.out_features for item in _layers(self.model)
        ]
        self.last_event = (
            f"{result['mode']} L{layer_index}: "
            f"{result['old_dim']}->{result['new_dim']}"
        )
        return result

    def _activation_hooks(self) -> list[Any]:
        if self.model is None or self.structure is None:
            raise RuntimeError("Qwen hemisphere has not loaded its model checkpoint")
        hooks = []
        gates: dict[int, torch.Tensor] = {}
        activation_name = str(self.model.config.hidden_act)
        for index, layer in enumerate(_layers(self.model)):
            def gate_hook(_module, _inputs, output, layer_index=index):
                if torch.is_grad_enabled():
                    gates[layer_index] = output.detach()

            def up_hook(_module, _inputs, output, layer_index=index):
                if not torch.is_grad_enabled():
                    return
                gate = gates.pop(layer_index, None)
                if gate is not None:
                    hidden = _activation(activation_name, gate) * output.detach()
                    self.structure.observe(layer_index, hidden)

            hooks.append(layer.mlp.gate_proj.register_forward_hook(gate_hook))
            hooks.append(layer.mlp.up_proj.register_forward_hook(up_hook))
        return hooks

    def learn(
        self,
        ids: list[int],
        labels: list[int],
        intensity: float = 1.0,
        kind: str = "ttt",
    ) -> dict[str, Any]:
        with self._train_lock:
            return self._learn(ids, labels, intensity, kind)

    def _learn(
        self,
        ids: list[int],
        labels: list[int],
        intensity: float = 1.0,
        kind: str = "ttt",
    ) -> dict[str, Any]:
        if not ids or len(ids) != len(labels):
            raise ValueError("Qwen training ids and labels must have equal non-zero lengths")
        if kind not in ("ttt", "consolidate"):
            raise ValueError(f"unknown Qwen training kind: {kind}")
        if self.model is None or self.optimizer is None or self.structure is None:
            raise RuntimeError("Qwen hemisphere has not loaded its model checkpoint")
        if len(ids) > MAX_SEQUENCE_LENGTH:
            ids = ids[-MAX_SEQUENCE_LENGTH:]
            labels = labels[-MAX_SEQUENCE_LENGTH:]
        self.ensure_gpu()
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        target_ids = torch.tensor([labels], dtype=torch.long, device=device)
        self.model.train()
        self.model.config.use_cache = False
        self.optimizer.zero_grad(set_to_none=True)
        hooks = self._activation_hooks()
        try:
            output = self.model(input_ids=input_ids, labels=target_ids, use_cache=False)
            loss = output.loss
            if not torch.isfinite(loss):
                return {
                    "learned": False,
                    "reason": "non-finite-loss",
                    "loss": float(loss.detach().item()),
                }
            loss.backward()
            trainable = [
                parameter for parameter in self.model.parameters() if parameter.requires_grad
            ]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(grad_norm):
                return {
                    "learned": False,
                    "reason": "non-finite-grad",
                    "loss": float(loss.detach().item()),
                }
            learning_rate = QWEN_LR * max(0.05, intensity)
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            self.optimizer.step()
        finally:
            for hook in hooks:
                hook.remove()
            self.optimizer.zero_grad(set_to_none=True)
            self.model.eval()
        loss_value = float(loss.detach().item())
        self.global_step += 1
        if kind == "ttt":
            self.ttt_steps += 1
        elif kind == "consolidate":
            self.consolidate_steps += 1
        else:
            raise ValueError(f"unknown Qwen training kind: {kind}")
        self.last_loss = loss_value
        event = self.structure.consider(self.model, self.global_step, loss_value)
        if event is not None:
            self._resize(event["layer"], event["new_dim"])
        self.last_event = f"{kind}:loss={loss_value:.4f}"
        return {
            "learned": True,
            "loss": loss_value,
            "step": self.global_step,
            "event": self.last_event,
        }

    def admit_pressure(self, sample: Sample) -> None:
        with self._memory_lock:
            super().admit_pressure(sample)

    def consolidation_samples(self, limit: int = 4) -> list[Sample]:
        with self._memory_lock:
            samples = self.debt.due(limit)
            if self.replay.samples:
                samples.extend(self.replay.choose(max(1, limit // 4), random.Random()))
            return samples

    def consolidate(self, samples: list[Sample], intensity: float) -> dict[str, Any]:
        if not samples:
            return {"steps": 0, "loss": None}
        losses: list[float] = []
        trained: list[Sample] = []
        try:
            for index in range(min(CONSOLIDATION_STEPS, len(samples))):
                sample = samples[index]
                result = self.learn(sample.ids, sample.labels, intensity, kind="consolidate")
                if result["learned"]:
                    sample.loss = float(result["loss"])
                    losses.append(sample.loss)
                    trained.append(sample)
        except Exception:
            with self._memory_lock:
                self.debt.release_reservation(samples)
            raise
        with self._memory_lock:
            if trained:
                self.debt.commit_repayment(trained, trained_steps=len(trained))
                self.debt.decay_after_training()
                for sample in trained:
                    self.replay.add(sample.clone(source=f"consolidate:{self.name}"))
            self.debt.release_reservation(samples)
        return {
            "steps": len(trained),
            "loss": sum(losses) / len(losses) if losses else None,
        }

    def save(self) -> None:
        if self.model is None or self.structure is None:
            raise RuntimeError("cannot save an unloaded Qwen hemisphere")
        widths = [layer.mlp.gate_proj.out_features for layer in _layers(self.model)]
        self.model.config.dynamic_intermediate_sizes = widths
        structure_state = self.structure.to_dict(self.model)
        structure_state.update(
            {
                "global_step": self.global_step,
                "ttt_steps": self.ttt_steps,
                "consolidate_steps": self.consolidate_steps,
                "last_loss": self.last_loss,
                "last_event": self.last_event,
            }
        )
        self.model.config.dynamic_structure_state = structure_state
        super().save()

    def load(self) -> bool:
        out = self.state_dir / "hemispheres" / self.name
        if not (out / "config.json").exists():
            return False
        from safetensors.torch import load_file

        config = AutoConfig.from_pretrained(out, local_files_only=True)
        model = AutoModelForCausalLM.from_config(config)
        model.to(dtype=torch.float16)
        widths = getattr(config, "dynamic_intermediate_sizes", None)
        if widths:
            for index, width in enumerate(widths):
                layer = _layers(model)[index]
                if layer.mlp.gate_proj.out_features != int(width):
                    _resize_mlp(
                        layer,
                        int(width),
                        torch.ones(layer.mlp.gate_proj.out_features),
                    )
        weight_files = sorted(out.glob("model*.safetensors"))
        if not weight_files:
            return False
        saved_state: dict[str, torch.Tensor] = {}
        for weight_file in weight_files:
            saved_state.update(load_file(str(weight_file), device="cpu"))
        _restore_tied_state_keys(model, config, saved_state)
        model.load_state_dict(saved_state, strict=True)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        tokenizer = AutoTokenizer.from_pretrained(out, use_fast=True)
        optimizer = Adafactor(
            model.parameters(),
            lr=QWEN_LR,
            scale_parameter=False,
            relative_step=False,
        )
        optimizer_path = out / "optimizer_state.pt"
        if optimizer_path.exists():
            optimizer.load_state_dict(
                torch.load(optimizer_path, map_location="cpu", weights_only=True)
            )
        memory_path = out / "memory.json"
        if memory_path.exists():
            memory = json.loads(memory_path.read_text(encoding="utf-8"))
            self.replay.load_dict(memory.get("replay", {}))
            self.debt.load_dict(memory.get("debt", {}))
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.structure = QwenStructureController(model)
        structure_state = getattr(config, "dynamic_structure_state", None)
        if structure_state:
            self.structure.load_dict(structure_state, model)
            self.global_step = int(structure_state.get("global_step", 0))
            self.ttt_steps = int(structure_state.get("ttt_steps", 0))
            self.consolidate_steps = int(structure_state.get("consolidate_steps", 0))
            loss = structure_state.get("last_loss")
            self.last_loss = float(loss) if loss is not None else None
            self.last_event = str(structure_state.get("last_event", "restored"))
        self.device = "cpu"
        return True


class QwenDualBrain(Byt5DualBrain):
    def __init__(self, source: str, state_dir: Path, training_device: str) -> None:
        self.source = source
        self.state_dir = state_dir
        self.training_device = training_device
        self.hemispheres = {
            name: QwenHemisphere(
                name,
                source,
                state_dir,
                training_device,
                load_base=not (state_dir / "hemispheres" / name / "config.json").exists(),
            )
            for name in ("left", "right")
        }
        self.active = "left"
        self.switch_urgency_acc = 0.0
        self.pressure_baseline = {"left": 0.0, "right": 0.0}
        self.switch_count = 0
        self._pending_switch = None
        self.switch_log: list[dict[str, Any]] = []
        import threading

        self.training_lock = threading.RLock()

    def load(self) -> bool:
        loaded = [hemisphere.load() for hemisphere in self.hemispheres.values()]
        if not all(loaded):
            for (name, hemisphere), ok in zip(self.hemispheres.items(), loaded):
                if ok:
                    hemisphere.to_cpu()
                    self.pressure_baseline[name] = hemisphere.debt.total_clean_added
                elif hemisphere.model is None:
                    self.hemispheres[name] = QwenHemisphere(
                        name, self.source, self.state_dir, self.training_device
                    )
            return False
        meta_path = self.state_dir / "brain.json"
        if not meta_path.exists():
            for name, hemisphere in self.hemispheres.items():
                self.pressure_baseline[name] = hemisphere.debt.total_clean_added
            return False
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        active = meta.get("active")
        self.active = active if active in ("left", "right") else "left"
        self.switch_urgency_acc = max(
            0.0, min(1.0, float(meta.get("switch_urgency_acc", 0.0)))
        )
        baselines = meta.get("pressure_baseline", {})
        self.pressure_baseline = {
            name: max(0.0, float(baselines.get(name, 0.0)))
            for name in ("left", "right")
        }
        self.switch_count = int(meta.get("switch_count", 0))
        log = meta.get("switch_log", [])
        self.switch_log = log if isinstance(log, list) else []
        return True

    def switch_begin(self) -> None:
        old_h = self.active_h()
        new_h = self.passive_h()
        if old_h.model is None or new_h.model is None:
            raise RuntimeError("both Qwen hemispheres must be loaded before switching")
        old_h.to_cpu()
        new_h.to_cpu()
        left, right = old_h, new_h
        left_widths = [layer.mlp.gate_proj.out_features for layer in _layers(left.model)]
        right_widths = [layer.mlp.gate_proj.out_features for layer in _layers(right.model)]
        if len(left_widths) != len(right_widths):
            raise ValueError("Qwen hemisphere depths differ and cannot be synchronized")
        for index, (left_width, right_width) in enumerate(zip(left_widths, right_widths)):
            target = max(left_width, right_width)
            if left_width != target:
                left._resize(index, target)
            if right_width != target:
                right._resize(index, target)
        if self.training_device != "cpu":
            super().switch_begin()
            return

        old, new = old_h.name, new_h.name
        for source_h, target_h in ((old_h, new_h), (new_h, old_h)):
            for item in source_h.replay.latest(2):
                target_h.admit_exchange(item.clone(source=f"exchange:{item.source}"))
        mix = 0.01
        with torch.no_grad():
            for left_parameter, right_parameter in zip(old_h.model.parameters(), new_h.model.parameters()):
                snapshot = left_parameter.detach().clone()
                left_parameter.lerp_(right_parameter, mix)
                right_parameter.lerp_(snapshot, mix)
        self._pending_switch = new
        self.active = new
        self.switch_count += 1

    def admit_exchange_sample(self, sample: Sample) -> None:
        with self.training_lock:
            self.active_h().admit_pressure(sample)
            passive = self.passive_h()
            with passive._memory_lock:
                passive.debt.add(sample, 0.35)

    def consolidate_passive(self, intensity: float) -> dict[str, Any] | None:
        with self.training_lock:
            passive = self.passive_h()
            samples = passive.consolidation_samples()
            if not samples:
                return None
            passive.to_cpu()
            previous_device = passive.training_device
            passive.training_device = "cpu"
        try:
            return passive.consolidate(samples, intensity)
        finally:
            passive.training_device = previous_device
