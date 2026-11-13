"""Single-writer GPU training queue and automatic homeostasis controllers."""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
import itertools
import logging
import math
import queue
import threading
import time
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

from .byte_codec import IGNORE_INDEX, PAD, trim_training_pair
from .config import AppConfig
from .memory import DebtLedger, ReplayBuffer, Sample
from .model import DynamicByteLM

LOG = logging.getLogger(__name__)


class LearningRateController:
    def __init__(self, config: AppConfig) -> None:
        self.config = config.runtime
        self.value = config.runtime.learning_rate
        self.best_loss = math.inf
        self.ema_loss: float | None = None
        self.stale_updates = 0

    def update(self, loss: float) -> tuple[float, str]:
        if not math.isfinite(loss):
            self.value = max(self.config.min_learning_rate, self.value * self.config.lr_down_factor)
            return self.value, "non_finite_loss"
        self.ema_loss = loss if self.ema_loss is None else 0.95 * self.ema_loss + 0.05 * loss
        reason = "hold"
        if loss > self.ema_loss * self.config.loss_spike_ratio:
            self.value *= self.config.lr_down_factor
            self.stale_updates = 0
            reason = "loss_spike"
        elif loss < self.best_loss * 0.997:
            self.best_loss = min(self.best_loss, loss)
            self.stale_updates = 0
        else:
            self.best_loss = min(self.best_loss, loss)
            self.stale_updates += 1
            if self.stale_updates >= 10:
                self.value *= self.config.lr_up_factor
                self.stale_updates = 0
                reason = "plateau"
        self.value = max(self.config.min_learning_rate, min(self.config.max_learning_rate, self.value))
        return self.value, reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "best_loss": self.best_loss,
            "ema_loss": self.ema_loss,
            "stale_updates": self.stale_updates,
        }

    def load_dict(self, raw: dict[str, Any]) -> None:
        self.value = float(raw.get("value", self.config.learning_rate))
        best = float(raw.get("best_loss", math.inf))
        self.best_loss = best if math.isfinite(best) else math.inf
        ema = raw.get("ema_loss")
        self.ema_loss = float(ema) if ema is not None else None
        self.stale_updates = int(raw.get("stale_updates", 0))


class StructureController:
    """Loss/gradient-driven neurogenesis and importance-driven pruning."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.losses: deque[float] = deque(maxlen=64)
        self.last_change_step = -10**9

    def consider(self, model: DynamicByteLM, step: int, loss: float) -> dict[str, Any] | None:
        runtime = self.config.runtime
        self.losses.append(loss)
        if step < runtime.structure_warmup or step - self.last_change_step < runtime.structure_interval:
            return None
        if step % runtime.structure_interval or len(self.losses) < 24 or not math.isfinite(loss):
            return None
        values = list(self.losses)
        previous = sum(values[-24:-8]) / 16
        recent = sum(values[-8:]) / 8
        relative_improvement = (previous - recent) / max(abs(previous), 1e-6)
        layer_index = (step // runtime.structure_interval) % len(model.blocks)
        layer = model.blocks[layer_index]
        event = None
        if relative_improvement < 0.002 and layer.hidden_dim < self.config.model.max_hidden_dim:
            event = {"layer": layer_index, "new_dim": layer.hidden_dim + runtime.growth_size}
        elif relative_improvement > 0.01 and layer.hidden_dim > self.config.model.min_hidden_dim:
            importance = layer.importance()
            low = importance.min().item()
            median = importance.median().item()
            if low < median * 0.02:
                event = {"layer": layer_index, "new_dim": layer.hidden_dim - runtime.prune_size}
        if event:
            self.last_change_step = step
        return event

    def to_dict(self) -> dict[str, Any]:
        return {"losses": list(self.losses), "last_change_step": self.last_change_step}

    def load_dict(self, raw: dict[str, Any]) -> None:
        self.losses = deque((float(value) for value in raw.get("losses", [])), maxlen=64)
        self.last_change_step = int(raw.get("last_change_step", -10**9))


@dataclass
class HemisphereState:
    name: str
    cpu_model: DynamicByteLM
    model: DynamicByteLM
    optimizer: torch.optim.Optimizer
    replay: ReplayBuffer
    debt: DebtLedger
    lr: LearningRateController
    structure: StructureController
    lock: threading.RLock = field(default_factory=threading.RLock)
    global_step: int = 0
    train_steps: int = 0
    ttt_steps: int = 0
    consolidate_steps: int = 0
    total_trained_bytes: int = 0
    last_loss: float = 0.0
    last_event: str = "initialized"

    @property
    def online_learning_steps(self) -> int:
        """Steps that changed the online model through train or TTT."""
        return int(self.train_steps + self.ttt_steps)

    @property
    def passive_consolidation_steps(self) -> int:
        """Steps spent consolidating debts while this hemisphere was passive."""
        return int(self.consolidate_steps)

    @classmethod
    def create(cls, name: str, config: AppConfig, device: torch.device) -> "HemisphereState":
        cpu_model = DynamicByteLM(config.model).cpu().eval()
        model = DynamicByteLM(config.model).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.runtime.learning_rate, weight_decay=config.runtime.weight_decay
        )
        return cls(
            name=name,
            cpu_model=cpu_model,
            model=model,
            optimizer=optimizer,
            replay=ReplayBuffer(config.runtime.replay_capacity),
            debt=DebtLedger(
                config.runtime.debt_capacity, config.runtime.debt_decay, config.runtime.debt_min_weight
            ),
            lr=LearningRateController(config),
            structure=StructureController(config),
        )


@dataclass(order=True)
class TrainJob:
    priority: int
    sequence: int
    kind: str = field(compare=False)
    hemisphere: str | None = field(compare=False, default=None)
    samples: list[Sample] = field(compare=False, default_factory=list)
    steps: int = field(compare=False, default=1)
    future: Future = field(compare=False, default_factory=Future)
    event: dict[str, Any] = field(compare=False, default_factory=dict)


class TrainingWorker:
    """The only writer of GPU parameters; CPU snapshots update at chunk boundaries."""

    def __init__(self, config: AppConfig) -> None:
        config.validate()
        self.config = config
        if config.runtime.training_device.startswith("cuda"):
            if not torch.cuda.is_available():
                if config.runtime.require_cuda:
                    raise RuntimeError("CUDA training was required but torch.cuda.is_available() is false")
                device = torch.device("cpu")
            else:
                device = torch.device(config.runtime.training_device)
        else:
            device = torch.device(config.runtime.training_device)
        self.device = device
        self.states = {
            name: HemisphereState.create(name, config, device) for name in ("left", "right")
        }
        self.jobs: queue.PriorityQueue[TrainJob] = queue.PriorityQueue()
        self.sequence = itertools.count()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="gpu-training", daemon=True)
        self.started = False

    def start(self) -> None:
        if not self.started:
            self.started = True
            self.thread.start()

    def submit(
        self,
        kind: str,
        hemisphere: str | None = None,
        samples: list[Sample] | None = None,
        steps: int = 1,
        wait: bool = True,
        priority: int | None = None,
        event: dict[str, Any] | None = None,
    ) -> Any:
        if self.stop_event.is_set():
            raise RuntimeError("training worker has stopped")
        priorities = {"ttt": 0, "sync": 1, "resize": 1, "snapshot": 2, "train": 3, "consolidate": 4}
        job = TrainJob(
            priority=priorities[kind] if priority is None else priority,
            sequence=next(self.sequence),
            kind=kind,
            hemisphere=hemisphere,
            samples=samples or [],
            steps=steps,
            event=event or {},
        )
        self.jobs.put(job)
        return job.future.result() if wait else job.future

    def _optimizer_reset(self, state: HemisphereState) -> None:
        state.optimizer = torch.optim.AdamW(
            state.model.parameters(), lr=state.lr.value, weight_decay=self.config.runtime.weight_decay
        )
        state.last_event = "optimizer rebuilt after structural change"

    def _resize_both(self, state: HemisphereState, event: dict[str, Any]) -> dict[str, Any]:
        result = state.model.resize_hidden(event["layer"], event["new_dim"])
        target = state.model.blocks[event["layer"]].hidden_dim
        with state.lock:
            cpu_result = state.cpu_model.resize_hidden(event["layer"], target)
        if result["mode"] != "none" or cpu_result["mode"] != "none":
            self._optimizer_reset(state)
            state.last_event = f"{result['mode']} L{result['layer']}: {result['old_dim']}->{target}"
            LOG.info("%s %s", state.name, state.last_event)
        return result

    def resize(self, name: str, layer: int, new_dim: int) -> dict[str, Any]:
        """Queue structural changes behind the single GPU writer."""
        return self.submit(
            "resize", name, event={"layer": layer, "new_dim": new_dim}, wait=True
        )

    def _batch(self, samples: list[Sample], index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        runtime = self.config.runtime
        chosen = [samples[(index + offset) % len(samples)] for offset in range(min(runtime.batch_size, len(samples)))]
        pairs = [trim_training_pair(item.ids, item.labels, self.config.model.max_seq_len) for item in chosen]
        lengths = [len(ids) - 1 for ids, _ in pairs]
        width = max(lengths)
        inputs = torch.full((len(chosen), width), PAD, dtype=torch.long)
        targets = torch.full((len(chosen), width), IGNORE_INDEX, dtype=torch.long)
        weights = torch.tensor([max(0.001, item.weight) for item in chosen], dtype=torch.float32)
        for row, ((ids, labels), length) in enumerate(zip(pairs, lengths)):
            inputs[row, :length] = torch.tensor(ids[:-1], dtype=torch.long)
            targets[row, :length] = torch.tensor(labels[1:], dtype=torch.long)
        return inputs.to(self.device), targets.to(self.device), weights.to(self.device)

    def _train(self, state: HemisphereState, samples: list[Sample], steps: int, kind: str) -> dict[str, Any]:
        if not samples:
            return {"steps": 0, "loss": None, "event": "no_samples"}
        state.model.train()
        losses: list[float] = []
        structure_events: list[dict[str, Any]] = []
        for index in range(steps):
            batch_start = index * self.config.runtime.batch_size
            chosen_samples = [
                samples[(batch_start + offset) % len(samples)]
                for offset in range(min(self.config.runtime.batch_size, len(samples)))
            ]
            inputs, targets, sample_weights = self._batch(samples, batch_start)
            self.optimizer_zero(state)
            logits = state.model(inputs)
            token_losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                ignore_index=IGNORE_INDEX, reduction="none"
            ).reshape(targets.shape)
            valid = (targets != IGNORE_INDEX).to(token_losses.dtype)
            per_sample = (token_losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            loss = (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
            if not torch.isfinite(loss):
                state.lr.update(float("nan"))
                self._optimizer_reset(state)
                state.last_event = "non-finite batch skipped"
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), self.config.runtime.grad_clip)
            if not torch.isfinite(grad_norm):
                self._optimizer_reset(state)
                state.last_event = "non-finite gradient skipped"
                continue
            lr, lr_reason = state.lr.update(loss.item())
            for group in state.optimizer.param_groups:
                group["lr"] = lr * (self.config.runtime.ttt_learning_rate / self.config.runtime.learning_rate) if kind == "ttt" else lr
            state.optimizer.step()
            state.global_step += 1
            if kind == "ttt":
                state.ttt_steps += 1
            elif kind == "train":
                state.train_steps += 1
            elif kind == "consolidate":
                state.consolidate_steps += 1
            else:
                raise ValueError(f"unknown training kind for step accounting: {kind}")
            state.total_trained_bytes += int(valid.sum().item())
            state.last_loss = loss.item()
            losses.append(loss.item())
            with state.lock:
                for sample, sample_loss in zip(chosen_samples, per_sample.detach().cpu().tolist()):
                    sample.loss = sample_loss
                    robust_weight = state.replay.cleaner.loss_weight(sample_loss)
                    # Apply the robust score to stored copies so an outlier is
                    # genuinely downweighted on its next replay/debt visit.
                    for container in (state.replay.samples, state.debt.items):
                        stored = container.get(sample.digest)
                        if stored is not None:
                            stored.loss = 0.9 * stored.loss + 0.1 * sample_loss
                            stored.weight = max(0.001, stored.weight * max(0.05, robust_weight))
            event = state.structure.consider(state.model, state.global_step, loss.item())
            if event:
                structure_events.append(self._resize_both(state, event))
            state.last_event = f"{kind}:loss={loss.item():.4f};lr={lr:.2e};{lr_reason}"
        with state.lock:
            if state.cpu_model.hidden_dims() != state.model.hidden_dims():
                for layer, target in enumerate(state.model.hidden_dims()):
                    state.cpu_model.resize_hidden(layer, target)
            state.cpu_model.load_state_dict(state.model.state_dict())
            state.cpu_model.eval()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {
            "steps": len(losses),
            "loss": sum(losses) / len(losses) if losses else None,
            "structure_events": structure_events,
            "lr": state.lr.value,
        }

    @staticmethod
    def optimizer_zero(state: HemisphereState) -> None:
        state.optimizer.zero_grad(set_to_none=True)

    def _synchronize(self, first: HemisphereState, second: HemisphereState) -> dict[str, Any]:
        # Align MLP widths by growing the smaller side, then perform a small,
        # symmetric interpolation. Most semantic transfer remains in debt replay.
        for layer in range(self.config.model.n_layers):
            target = max(first.model.blocks[layer].hidden_dim, second.model.blocks[layer].hidden_dim)
            self._resize_both(first, {"layer": layer, "new_dim": target})
            self._resize_both(second, {"layer": layer, "new_dim": target})
        mix = self.config.runtime.sync_mix / 2
        if mix > 0:
            with torch.no_grad():
                for left_parameter, right_parameter in zip(first.model.parameters(), second.model.parameters()):
                    left_snapshot = left_parameter.detach().clone()
                    left_parameter.lerp_(right_parameter, mix)
                    right_parameter.lerp_(left_snapshot, mix)
        for state in (first, second):
            with state.lock:
                if state.cpu_model.hidden_dims() != state.model.hidden_dims():
                    for layer, target in enumerate(state.model.hidden_dims()):
                        state.cpu_model.resize_hidden(layer, target)
                state.cpu_model.load_state_dict(state.model.state_dict())
                state.cpu_model.eval()
            state.last_event = f"reciprocal sync mix={self.config.runtime.sync_mix}"
        return {"sync_mix": self.config.runtime.sync_mix}

    def _snapshot(self) -> dict[str, Any]:
        result = {}
        for name, state in self.states.items():
            with state.lock:
                if state.cpu_model.hidden_dims() != state.model.hidden_dims():
                    for layer, target in enumerate(state.model.hidden_dims()):
                        state.cpu_model.resize_hidden(layer, target)
                state.cpu_model.load_state_dict(state.model.state_dict())
                cpu_weights = {key: value.detach().cpu() for key, value in state.cpu_model.state_dict().items()}
                replay = state.replay.to_dict()
                debt = state.debt.to_dict()
                metrics = {
                    "global_step": state.global_step,
                    "train_steps": state.train_steps,
                    "ttt_steps": state.ttt_steps,
                    "consolidate_steps": state.consolidate_steps,
                    "total_trained_bytes": state.total_trained_bytes,
                    "last_loss": state.last_loss,
                    "last_event": state.last_event,
                }                # Optimizer state is part of the same checkpoint boundary as
                # weights, replay, debt and metrics. Reading it outside this
                # lock would let a concurrent update split the snapshot.
                optimizer = self._to_cpu(state.optimizer.state_dict())
            result[name] = {
                "model": cpu_weights,
                "optimizer": optimizer,
                "replay": replay,
                "debt": debt,
                "lr": state.lr.to_dict(),
                "structure": state.structure.to_dict(),
                "metrics": metrics,
                "hidden_dims": state.model.hidden_dims(),
            }
        return result

    @classmethod
    def _to_cpu(cls, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: cls._to_cpu(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cls._to_cpu(item) for item in value)
        return value

    def _run(self) -> None:
        # The stop sentinel has the lowest priority: every accepted training
        # job finishes and every Future resolves before the thread exits.
        while True:
            try:
                job = self.jobs.get(timeout=self.config.runtime.idle_sleep_s)
            except queue.Empty:
                continue
            try:
                if job.kind == "stop":
                    job.future.set_result({"stopped": True})
                    break
                if job.kind == "snapshot":
                    result = self._snapshot()
                elif job.kind == "sync":
                    result = self._synchronize(self.states["left"], self.states["right"])
                elif job.kind == "resize":
                    if job.hemisphere not in self.states:
                        raise ValueError(f"unknown hemisphere: {job.hemisphere}")
                    result = self._resize_both(self.states[job.hemisphere], job.event)
                else:
                    if job.hemisphere not in self.states:
                        raise ValueError(f"unknown hemisphere: {job.hemisphere}")
                    state = self.states[job.hemisphere]
                    result = self._train(state, job.samples, job.steps, job.kind)
                    if job.kind == "consolidate":
                        with state.lock:
                            if int(result.get("steps", 0)) > 0:
                                state.debt.commit_repayment(job.samples, int(result["steps"]))
                                state.debt.decay_after_training()
                            else:
                                state.debt.release_reservation(job.samples)
                job.future.set_result(result)
            except Exception as exc:  # propagate exact failure to the requester
                LOG.exception("training job failed: %s", job.kind)
                if job.kind == "consolidate" and job.hemisphere in self.states:
                    with self.states[job.hemisphere].lock:
                        self.states[job.hemisphere].debt.release_reservation(job.samples)
                job.future.set_exception(exc)
            finally:
                self.jobs.task_done()

    def stop(self) -> None:
        if not self.started:
            return
        job = TrainJob(99, next(self.sequence), "stop")
        self.jobs.put(job)
        self.stop_event.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise TimeoutError("GPU training worker did not stop within 30 seconds")
        self.started = False

    def metrics(self) -> dict[str, Any]:
        with torch.cuda.device(self.device) if self.device.type == "cuda" else _NullContext():
            memory = torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
        return {
            "training_device": str(self.device),
            "queued_jobs": self.jobs.qsize(),
            "gpu_memory_allocated": memory,
            "hemispheres": {
                name: {
                    "steps": state.global_step,
                    "train_steps": state.train_steps,
                    "ttt_steps": state.ttt_steps,
                    "consolidate_steps": state.consolidate_steps,
                    "online_learning_steps": state.online_learning_steps,
                    "passive_consolidation_steps": state.passive_consolidation_steps,
                    "loss": state.last_loss,
                    "learning_rate": state.lr.value,
                    "hidden_dims": state.model.hidden_dims(),
                    "replay": len(state.replay),
                    "debt": len(state.debt),
                    "repayment_capacity": state.debt.repayment_capacity,
                    "parameters": state.model.parameter_count(),
                    "last_event": state.last_event,
                }
                for name, state in self.states.items()
            },
        }


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None
