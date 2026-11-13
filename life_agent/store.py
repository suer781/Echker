"""Atomic local persistence for both hemispheres and their learning state."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
from .byte_codec import ByteFrame

if TYPE_CHECKING:
    from .agent import LifeAgent

LOG = logging.getLogger(__name__)
FORMAT_VERSION = 2


class StateStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.latest = directory / "latest.pt"
        self.backup = directory / "latest.backup.pt"

    @classmethod
    def _tensor_digest(cls, value: torch.Tensor) -> dict[str, Any]:
        tensor = value.detach().to("cpu")
        if tensor.is_floating_point():
            tensor = tensor.to(torch.float32)
        array = tensor.contiguous().numpy()
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }

    @classmethod
    def _value_digest(cls, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return cls._tensor_digest(value)
        if isinstance(value, dict):
            return {str(key): cls._value_digest(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        if isinstance(value, (list, tuple)):
            return [cls._value_digest(item) for item in value]
        if isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                return repr(value)
            return float(f"{value:.10g}")
        if isinstance(value, int) and not isinstance(value, bool):
            # Checkpoint counters can be ints while a restored controller
            # normalizes them to floats. Compare numeric values, not Python
            # container types.
            return float(f"{float(value):.10g}")
        return value

    @classmethod
    def _hemisphere_fingerprint(cls, saved: dict[str, Any]) -> dict[str, Any]:
        fields = ("hidden_dims", "model", "optimizer", "replay", "debt", "lr", "structure", "metrics")
        return {field: cls._value_digest(saved.get(field)) for field in fields}

    @classmethod
    def fingerprint_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        hemispheres = payload.get("hemispheres", {})
        fingerprint = {
            "active": payload.get("active"),
            "history": cls._value_digest(payload.get("history", [])),
            "interaction_count": int(payload.get("interaction_count", 0)),
            "observed_frames": int(payload.get("observed_frames", 0)),
            "switch_count": int(payload.get("switch_count", 0)),
            "adaptive_switch": cls._value_digest(payload.get("adaptive_switch", {})),
            "switch_decisions": cls._value_digest(payload.get("switch_decisions", [])),
            "hemispheres": {
                name: cls._hemisphere_fingerprint(hemispheres.get(name, {}))
                for name in ("left", "right")
            },
        }
        return fingerprint

    @classmethod
    def fingerprint_agent(cls, agent: "LifeAgent") -> dict[str, Any]:
        hemispheres: dict[str, Any] = {}
        for name, state in agent.worker.states.items():
            with state.lock:
                hemispheres[name] = cls._hemisphere_fingerprint(
                    {
                        "hidden_dims": state.model.hidden_dims(),
                        "model": state.model.state_dict(),
                        "optimizer": cls._to_cpu(state.optimizer.state_dict()),
                        "replay": state.replay.to_dict(),
                        "debt": state.debt.to_dict(),
                        "lr": state.lr.to_dict(),
                        "structure": state.structure.to_dict(),
                        "metrics": {
                            "global_step": state.global_step,
                            "train_steps": state.train_steps,
                            "ttt_steps": state.ttt_steps,
                            "consolidate_steps": state.consolidate_steps,
                            "total_trained_bytes": state.total_trained_bytes,
                            "last_loss": state.last_loss,
                            "last_event": state.last_event,
                        },
                    }
                )
        return {
            "active": agent.active,
            "history": cls._value_digest(
                [[stimulus.to_dict(), response.to_dict()] for stimulus, response in agent.history]
            ),
            "interaction_count": int(agent.interaction_count),
            "observed_frames": int(agent.observed_frames),
            "switch_count": int(agent.switch_count),
            "adaptive_switch": cls._value_digest(agent.switch_controller.to_dict()),
            "switch_decisions": cls._value_digest(agent.switch_decisions),
            "hemispheres": hemispheres,
        }

    @classmethod
    def _to_cpu(cls, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: cls._to_cpu(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cls._to_cpu(item) for item in value)
        return value

    @staticmethod
    def _fingerprint_differences(expected: Any, actual: Any, path: str = "") -> list[str]:
        differences: list[str] = []
        if type(expected) is not type(actual):
            return [f"{path or '<root>'}: type {type(expected).__name__} != {type(actual).__name__}"]
        if isinstance(expected, dict):
            for key in sorted(set(expected) | set(actual)):
                if key not in expected or key not in actual:
                    differences.append(f"{path}.{key}: missing")
                else:
                    differences.extend(
                        StateStore._fingerprint_differences(expected[key], actual[key], f"{path}.{key}")
                    )
        elif isinstance(expected, list):
            if len(expected) != len(actual):
                differences.append(f"{path}: length {len(expected)} != {len(actual)}")
            for index, (left, right) in enumerate(zip(expected, actual)):
                differences.extend(
                    StateStore._fingerprint_differences(left, right, f"{path}[{index}]")
                )
        elif expected != actual:
            differences.append(f"{path}: {expected!r} != {actual!r}")
        return differences[:12]

    def save(self, agent: "LifeAgent") -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        hemispheres = agent.worker.submit("snapshot", wait=True)
        payload = {
            "format_version": FORMAT_VERSION,
            "saved_at": time.time(),
            "config": agent.config.to_dict(),
            "active": agent.active,
            "history": [[stimulus.to_dict(), response.to_dict()] for stimulus, response in agent.history],
            "interaction_count": agent.interaction_count,
            "observed_frames": agent.observed_frames,
            "switch_count": agent.switch_count,
            "adaptive_switch": agent.switch_controller.to_dict(),
            "switch_decisions": list(agent.switch_decisions),
            "created_at": agent.created_at,
            "hemispheres": hemispheres,
        }
        payload["fingerprint"] = self.fingerprint_payload(payload)
        temporary = self.directory / "latest.writing.tmp"
        torch.save(payload, temporary)
        if self.latest.exists():
            shutil.copy2(self.latest, self.backup)
        os.replace(temporary, self.latest)
        return self.latest

    def load(self, agent: "LifeAgent") -> bool:
        if not self.latest.exists() and not self.backup.exists():
            return False
        errors = []
        for path in (self.latest, self.backup):
            if not path.exists():
                continue
            try:
                # Only load the agent's own trusted local state directory.
                payload = torch.load(path, map_location="cpu", weights_only=False)
                version_for_check = int(payload.get("format_version", 0))
                self._restore(agent, payload)
                saved_fingerprint = payload.get("fingerprint")
                # Version-1 checkpoints predate the fingerprint and may also
                # contain text history that is migrated during restore. They
                # are still loadable, but cannot be compared field-for-field.
                if saved_fingerprint and version_for_check != 1:
                    restored_fingerprint = self.fingerprint_agent(agent)
                    if restored_fingerprint != saved_fingerprint:
                        differences = self._fingerprint_differences(
                            saved_fingerprint, restored_fingerprint
                        )
                        raise ValueError(f"checkpoint fingerprint mismatch: {differences}")
                LOG.info("restored checkpoint: %s", path)
                return True
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
                LOG.exception("failed to restore checkpoint %s", path)
        raise RuntimeError("no readable checkpoint; " + " | ".join(errors))

    def _restore(self, agent: "LifeAgent", payload: dict[str, Any]) -> None:
        version = int(payload.get("format_version", 0))
        if version not in (1, FORMAT_VERSION):
            raise ValueError("unsupported checkpoint format")
        saved_model = payload.get("config", {}).get("model", {})
        fixed = ("vocab_size", "d_model", "n_heads", "n_layers", "max_seq_len")
        for key in fixed:
            if saved_model.get(key) != getattr(agent.config.model, key):
                raise ValueError(f"checkpoint model mismatch: {key}")
        for name, state in agent.worker.states.items():
            saved = payload["hemispheres"][name]
            with state.lock:
                for layer, width in enumerate(saved["hidden_dims"]):
                    state.model.resize_hidden(layer, width)
                    state.cpu_model.resize_hidden(layer, width)
                state.model.load_state_dict(saved["model"])
                state.cpu_model.load_state_dict(saved["model"])
                state.model.to(agent.worker.device)
                state.cpu_model.eval()
                agent.worker._optimizer_reset(state)
                state.optimizer.load_state_dict(saved["optimizer"])
                state.replay.load_dict(saved["replay"])
                state.debt.load_dict(saved["debt"])
                state.lr.load_dict(saved["lr"])
                state.structure.load_dict(saved["structure"])
                metrics = saved["metrics"]
                state.global_step = int(metrics["global_step"])
                # Legacy checkpoints lack the split counters. The total is
                # preserved, but it cannot be reconstructed into its kinds;
                # treating it as online work would contaminate the first
                # post-load switch decision, so unknown kinds stay at zero
                # until the first new job establishes them.
                state.train_steps = int(metrics.get("train_steps", 0))
                state.ttt_steps = int(metrics.get("ttt_steps", 0))
                state.consolidate_steps = int(metrics.get("consolidate_steps", 0))
                state.total_trained_bytes = int(metrics["total_trained_bytes"])
                state.last_loss = float(metrics["last_loss"])
                state.last_event = str(metrics["last_event"])
        agent.active = payload["active"]
        if agent.active not in agent.worker.states:
            raise ValueError("invalid active hemisphere in checkpoint")
        history = []
        for stimulus, response in payload.get("history", []):
            if version == 1:
                history.append((ByteFrame.text(str(stimulus)), ByteFrame.text(str(response))))
            else:
                history.append((ByteFrame.from_dict(stimulus), ByteFrame.from_dict(response)))
        agent.history = history
        agent.interaction_count = int(payload.get("interaction_count", 0))
        agent.observed_frames = int(payload.get("observed_frames", 0))
        agent.switch_count = int(payload.get("switch_count", 0))
        agent.switch_decisions = list(payload.get("switch_decisions", []))[-128:]
        saved_switch = payload.get("adaptive_switch")
        if saved_switch:
            agent.switch_controller.load_dict(saved_switch)
            if not agent.switch_controller._roles_match(agent.active, agent.passive):
                passive_state = agent.state(agent.passive)
                with passive_state.lock:
                    clean_pressure = passive_state.debt.clean_pressure()
                agent.switch_controller.rebase(
                    clean_pressure,
                    agent.state(agent.active).online_learning_steps,
                    agent.state(agent.passive).passive_consolidation_steps,
                    agent.active,
                    agent.passive,
                )
        else:
            passive_state = agent.state(agent.passive)
            with passive_state.lock:
                clean_pressure = passive_state.debt.clean_pressure()
            agent.switch_controller.rebase(
                clean_pressure,
                agent.state(agent.active).online_learning_steps,
                agent.state(agent.passive).passive_consolidation_steps,
                agent.active,
                agent.passive,
            )
        agent.created_at = float(payload.get("created_at", time.time()))
