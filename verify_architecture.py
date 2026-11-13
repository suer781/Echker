"""Verify the actual dual-hemisphere runtime, not just class names.

Uses a fresh temporary state and the production LifeAgent/CUDA worker. It does
not read or overwrite the interactive, smoke, or future dataset checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

import torch

from life_agent import AppConfig, LifeAgent

ROOT = Path(__file__).resolve().parent


def wait_consolidation(agent: LifeAgent, name: str) -> None:
    future = agent.pending_consolidation.get(name)
    if future is not None:
        future.result(timeout=60)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "ARCHITECTURE_VALIDATION.json")
    args = parser.parse_args()
    config = AppConfig.load(ROOT / "config.json")
    config.runtime.max_new_tokens = 8
    config.runtime.ttt_chunk_tokens = 4
    config.runtime.consolidation_steps = 1
    config.runtime.adaptive_switching = False
    config.runtime.checkpoint_interval_s = 99999
    config.runtime.structure_interval = 10000
    config.runtime.structure_warmup = 10000
    config.runtime.idle_sleep_s = 0.01

    with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
        state_dir = Path(directory)
        agent = LifeAgent(config, state_dir)
        left = agent.state("left")
        right = agent.state("right")
        independent_objects = all(
            getattr(left, attribute) is not getattr(right, attribute)
            for attribute in ("model", "cpu_model", "optimizer", "replay", "debt", "lr", "structure")
        )
        initial_weights_distinct = not torch.equal(
            left.model.blocks[0].fc1.weight, right.model.blocks[0].fc1.weight
        )
        calls = {"left": 0, "right": 0}
        handles = [
            state.cpu_model.register_forward_hook(
                lambda _module, _inputs, _output, name=name: calls.__setitem__(name, calls[name] + 1)
            )
            for name, state in (("left", left), ("right", right))
        ]
        runtime: dict[str, Any] = {}
        agent.start()
        try:
            agent.interact("架构验证：左半球在线")
            wait_consolidation(agent, "right")
            runtime["first"] = {
                "active": agent.active,
                "cpu_calls": dict(calls),
                "left_steps": left.global_step,
                "right_offline_steps": right.global_step,
                "left_event": left.last_event,
                "right_event": right.last_event,
                "right_debt": len(right.debt),
            }
            first_checks = {
                "left_cpu_only": calls["left"] > 0 and calls["right"] == 0,
                "left_ttt": left.last_event.startswith("ttt:"),
                "right_offline_consolidation": right.global_step > 0,
            }

            agent.switch()
            left_calls = calls["left"]
            right_calls = calls["right"]
            agent.interact("架构验证：右半球在线")
            wait_consolidation(agent, "left")
            runtime["second"] = {
                "active": agent.active,
                "cpu_calls": dict(calls),
                "left_offline_steps": left.global_step,
                "right_steps": right.global_step,
                "left_event": left.last_event,
                "right_event": right.last_event,
            }
            second_checks = {
                "right_cpu_only": calls["right"] > right_calls and calls["left"] == left_calls,
                "right_ttt": right.last_event.startswith("ttt:"),
                "left_offline_consolidation": left.global_step > runtime["first"]["left_steps"],
            }

            agent.switch()
            runtime["cycle"] = {
                "active": agent.active,
                "switches": agent.switch_count,
                "left_debt": len(left.debt),
                "right_debt": len(right.debt),
                "sync_mix": config.runtime.sync_mix,
            }
            cycle_checks = {
                "full_switch_cycle": agent.active == "left" and agent.switch_count == 2,
                "reciprocal_debt": len(left.debt) > 0 and len(right.debt) > 0,
            }
            agent.save()
        finally:
            for handle in handles:
                handle.remove()
            agent.close()

        payload = torch.load(state_dir / "latest.pt", map_location="cpu", weights_only=False)
        restored = LifeAgent(config, state_dir)
        restored_ok = restored.load()
        persistence_checks = {
            "checkpoint_has_both": set(payload["hemispheres"]) == {"left", "right"},
            "checkpoint_weights_distinct": not torch.equal(
                payload["hemispheres"]["left"]["model"]["blocks.0.fc1.weight"],
                payload["hemispheres"]["right"]["model"]["blocks.0.fc1.weight"],
            ),
            "restored_active": restored_ok and restored.active == "left" and restored.switch_count == 2,
            "restored_both_learned": all(
                restored.state(name).global_step > 0 for name in ("left", "right")
            ),
        }
        checks = {
            "independent_cpu_gpu_models_optimizers_memory_controllers": independent_objects,
            "initial_weights_distinct": initial_weights_distinct,
            **first_checks,
            **second_checks,
            **cycle_checks,
            **persistence_checks,
            "byte_vocabulary_256_plus_6": config.model.vocab_size == 262,
            "cuda_training_cpu_inference": str(agent.worker.device) == "cuda"
            and next(left.cpu_model.parameters()).device.type == "cpu",
        }
        report = {
            "all_checks_passed": all(checks.values()),
            "checks": checks,
            "runtime": runtime,
            "checkpoint": {
                "format_version": payload["format_version"],
                "hemispheres": list(payload["hemispheres"]),
                "left_steps": payload["hemispheres"]["left"]["metrics"]["global_step"],
                "right_steps": payload["hemispheres"]["right"]["metrics"]["global_step"],
            },
            "boundaries": [
                "Only online generation is single-sided; the passive side intentionally performs offline consolidation.",
                "The sides are independent but not isolated: switch sync interpolates 2% total and debt replays samples.",
                "This is a short mechanism test, not evidence of long-term stability, generalization, or consciousness.",
            ],
        }
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
