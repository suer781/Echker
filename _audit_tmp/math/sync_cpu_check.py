# Minimal CPU-only check of _synchronize / resize shape safety (audit item 6/7).
# Builds two tiny models on CPU, widens one, runs the real worker._synchronize
# path, and reports shape alignment and optimizer-state reset behavior.
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from life_agent.config import AppConfig, ModelConfig, RuntimeConfig
from life_agent.worker import TrainingWorker

HERE = Path(__file__).resolve().parent


def tiny_config() -> AppConfig:
    return AppConfig(
        ModelConfig(
            vocab_size=262, d_model=64, n_heads=4, n_layers=1,
            hidden_dim=64, min_hidden_dim=16, max_hidden_dim=128, max_seq_len=64,
        ),
        RuntimeConfig(
            training_device="cpu", require_cuda=False,
            structure_interval=10_000, structure_warmup=10_000,
        ),
    )


def shapes(state):
    return {name: tuple(p.shape) for name, p in state.model.named_parameters()}


def main() -> None:
    worker = TrainingWorker(tiny_config())
    left, right = worker.states["left"], worker.states["right"]

    before = {"left": shapes(left), "right": shapes(right)}
    # Widen only the left hemisphere, mimicking an asymmetric structure event.
    resize_result = worker._resize_both(left, {"layer": 0, "new_dim": 96})
    after_resize = {"left": shapes(left), "right": shapes(right)}
    hidden_after_resize = {"left": left.model.hidden_dims(), "right": right.model.hidden_dims()}
    opt_state_left_after_resize = len(left.optimizer.state)

    # Run the real synchronize path (resize-to-max both sides, then zip lerp).
    sync_result = worker._synchronize(left, right)
    after_sync = {"left": shapes(left), "right": shapes(right)}

    mismatches = []
    for name in after_sync["left"]:
        if after_sync["left"][name] != after_sync["right"][name]:
            mismatches.append({"param": name, "left": after_sync["left"][name], "right": after_sync["right"][name]})

    # Parameter order / count check.
    left_names = [n for n, _ in left.model.named_parameters()]
    right_names = [n for n, _ in right.model.named_parameters()]

    pre_sync_mismatches = [
        {"param": name, "left": after_resize["left"][name], "right": after_resize["right"][name]}
        for name in after_resize["left"]
        if after_resize["left"][name] != after_resize["right"][name]
    ]
    report = {
        "resize_result": resize_result,
        "hidden_dims_after_resize": hidden_after_resize,
        "pre_sync_shape_mismatches": pre_sync_mismatches,
        "param_count_left": len(left_names),
        "param_count_right": len(right_names),
        "param_names_match": left_names == right_names,
        "shape_mismatches_after_sync": mismatches,
        "sync_result": sync_result,
        "optimizer_state_entries_left_after_resize": opt_state_left_after_resize,
        "optimizer_state_entries_left_after_sync": len(left.optimizer.state),
        "note": "_synchronize equalizes per-layer width before zipping (worker.py:336-339); zip pairs parameters by module order. Buffers (activation_ema) are excluded from parameters() and are not mixed.",
    }
    (HERE / "sync_cpu_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
