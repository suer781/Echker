"""Continuous, machine-checked validation of the implemented learning mechanisms.

This is not a claim of consciousness or general intelligence. Criteria cover
observable requirements only. A non-zero exit means at least one measurable
invariant failed and the run must be repaired and repeated.
"""
from __future__ import annotations

import argparse
import ctypes
from collections import deque
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import traceback
import time
from typing import Any

import torch
from torch import nn

from ingest_dataset import DatasetStreamer, ProgressStore
from life_agent import AppConfig, ByteFrame, Codec, LifeAgent
from life_agent.memory import Sample
from life_agent.model import DynamicByteLM
from life_agent.store import StateStore
from life_agent.worker import StructureController
from life_agent.byte_codec import frame_dialogue_sample, observation_sample

ROOT = Path(__file__).resolve().parent
PVM = ROOT / "tests"


class ProcessMemory(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def rss_megabytes() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        pass
    try:
        counters = ProcessMemory()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemory),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        if counters.WorkingSetSize <= 0:
            raise RuntimeError("empty working set reading")
        return counters.WorkingSetSize / 1024 / 1024
    except Exception:
        return None


def parameter_vector(model: nn.Module) -> torch.Tensor:
    # Fixed-shape probe: token embeddings receive the next-byte gradient and
    # never resize, so neurogenesis cannot masquerade as a TTT update.
    return model.token_embedding.weight.detach().reshape(-1).cpu().clone()


def synthetic_frame(cycle: int) -> ByteFrame:
    kind = cycle % 5
    if kind == 0:
        payload = bytes((cycle + index * 37) & 0xFF for index in range(64))
        return ByteFrame.raw(payload, {"cycle": cycle, "kind": "raw", "media_type": "application/x-cycle"})
    if kind == 1:
        text = ["持续验证的字节文本", "推理与记忆同步", "左脑右脑交替", "自动结构调节"][cycle % 4]
        return ByteFrame.text(f"{text} 第{cycle}轮 中文与ASCII mixed {cycle * 17 % 97}")
    if kind == 2:
        payload = bytes((cycle + index * 11) & 0xFF for index in range(12))
        return ByteFrame.rgb(payload, width=2, height=2)
    if kind == 3:
        payload = bytes((cycle + index * 19) & 0xFF for index in range(8))
        return ByteFrame.pcm(payload, sample_rate=8000, sample_width=1, channels=1)
    return ByteFrame(7, {"kind": "unknown", "cycle": cycle, "shape": [4, 4]}, bytes(range(cycle, cycle + 16)))


def response_spec(stimulus: ByteFrame) -> ByteFrame:
    if stimulus.codec == Codec.RGB:
        return ByteFrame.rgb(bytes(2 * 2 * 3), width=2, height=2).empty()
    if stimulus.codec == Codec.PCM:
        return ByteFrame.pcm(
            bytes(8), sample_rate=8000, sample_width=1, channels=1
        ).empty()
    return stimulus.empty()


class ValidationRun:
    def __init__(self, config: AppConfig, directory: Path) -> None:
        self.config = config
        self.directory = directory
        self.state_dir = directory / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = directory / "metrics.jsonl"
        self.metrics_path.write_text("", encoding="utf-8")
        self.agent: LifeAgent | None = None
        self.cpu_calls = {"left": 0, "right": 0}
        self.hooks: list[Any] = []
        self.restart_records: list[dict[str, Any]] = []
        self.ttt_deltas: list[float] = []
        self.lr_values: dict[str, set[float]] = {"left": set(), "right": set()}
        self.hidden_history: dict[str, list[list[int]]] = {"left": [], "right": []}
        self.step_samples: list[tuple[int, int, int]] = []
        self.rss_samples: list[float] = []
        self.gpu_samples: list[int] = []
        self.passive_violations = 0
        self.nonfinite_losses = 0
        self.errors: list[str] = []
        self.background_errors: list[str] = []

    def create_agent(self, allow_load: bool = True) -> LifeAgent:
        agent = LifeAgent(self.config, self.state_dir)
        if allow_load and (self.state_dir / "latest.pt").exists():
            if not agent.load():
                raise RuntimeError("checkpoint existed but did not load")
        agent.start()
        self.cpu_calls = {"left": 0, "right": 0}
        self.hooks = [
            state.cpu_model.register_forward_hook(
                lambda _module, _inputs, _output, name=name: self._count_cpu(name)
            )
            for name, state in (("left", agent.state("left")), ("right", agent.state("right")))
        ]
        self.agent = agent
        return agent

    def _count_cpu(self, name: str) -> None:
        self.cpu_calls[name] += 1

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def record(self, value: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(value, ensure_ascii=False) + "\n")

    def snapshot_counts(self) -> dict[str, int]:
        return self.cpu_calls.copy()

    def sample_resources(self, device: torch.device) -> None:
        value = rss_megabytes()
        if value is not None:
            self.rss_samples.append(value)
        if device.type == "cuda":
            self.gpu_samples.append(int(torch.cuda.memory_allocated(device)))

    def step_window_fraction(self, window: int = 20, warmup: int = 40) -> float:
        """Smallest 20-cycle side ratio after warmup; 1.0 is balanced."""
        if self.step_samples.__len__() < window:
            return 0.0
        ratios: list[float] = []
        for end in range(window, len(self.step_samples) + 1):
            cycle = self.step_samples[end - 1][0]
            if cycle < warmup:
                continue
            start = self.step_samples[end - window]
            left_delta = self.step_samples[end - 1][1] - start[1]
            right_delta = self.step_samples[end - 1][2] - start[2]
            larger = max(left_delta, right_delta, 1)
            ratios.append(min(left_delta, right_delta) / larger)
        return min(ratios) if ratios else 0.0

    def save_restart_record(self, record: dict[str, Any]) -> None:
        self.restart_records.append(record)

    def verify_restore(self) -> dict[str, Any]:
        assert self.agent is not None
        self.remove_hooks()
        previous_active = self.agent.active
        previous_switches = self.agent.switch_count
        self.agent.close()
        payload = torch.load(self.state_dir / "latest.pt", map_location="cpu", weights_only=False)
        saved_fingerprint = payload.get("fingerprint") or StateStore.fingerprint_payload(payload)
        restored = LifeAgent(self.config, self.state_dir)
        loaded = restored.load()
        restored_fingerprint = StateStore.fingerprint_agent(restored)
        differences = StateStore._fingerprint_differences(saved_fingerprint, restored_fingerprint)
        result = {
            "loaded": loaded,
            "active_match": restored.active == previous_active,
            "switches_match": restored.switch_count == previous_switches,
            "steps_match": {
                name: int(payload["hemispheres"][name]["metrics"]["global_step"])
                for name in ("left", "right")
            }
            == {name: restored.state(name).global_step for name in ("left", "right")},
            "fingerprint_match": not differences,
            "fingerprint_differences": differences[:12],
            "step_kinds_match": {
                name: {
                    "train": int(payload["hemispheres"][name]["metrics"].get("train_steps", 0)),
                    "ttt": int(payload["hemispheres"][name]["metrics"].get("ttt_steps", 0)),
                    "consolidate": int(payload["hemispheres"][name]["metrics"].get("consolidate_steps", 0)),
                }
                for name in ("left", "right")
            }
            == {
                name: {
                    "train": restored.state(name).train_steps,
                    "ttt": restored.state(name).ttt_steps,
                    "consolidate": restored.state(name).consolidate_steps,
                }
                for name in ("left", "right")
            },
            "steps": {
                name: int(payload["hemispheres"][name]["metrics"]["global_step"])
                for name in ("left", "right")
            },
        }
        self.save_restart_record(result)
        self.agent = restored
        restored.start()
        self.cpu_calls = {"left": 0, "right": 0}
        self.hooks = [
            state.cpu_model.register_forward_hook(
                lambda _module, _inputs, _output, name=name: self._count_cpu(name)
            )
            for name, state in (("left", restored.state("left")), ("right", restored.state("right")))
        ]
        return result


def build_config() -> AppConfig:
    config = AppConfig.load(ROOT / "config.json")
    config.runtime.max_new_tokens = 16
    config.runtime.ttt_chunk_tokens = 4
    config.runtime.checkpoint_interval_s = 99999
    config.runtime.idle_sleep_s = 0.01
    return config


def validate_automatic_structure(config: AppConfig) -> dict[str, Any]:
    """Exercise the production controller's growth and prune decisions.

    This uses an isolated model so it cannot alter the continuously trained
    validation agent. Events are then applied through DynamicByteLM.resize_hidden.
    """
    growth_model = DynamicByteLM(config.model).cpu()
    growth_controller = StructureController(config)
    growth_controller.losses = deque([5.0] * 24, maxlen=64)
    growth_controller.last_change_step = -10**9
    structure_step = math.ceil(
        max(config.runtime.structure_warmup, config.runtime.structure_interval)
        / config.runtime.structure_interval
    ) * config.runtime.structure_interval
    growth_event = growth_controller.consider(growth_model, structure_step, 5.0)
    if growth_event:
        growth_model.resize_hidden(growth_event["layer"], growth_event["new_dim"])

    prune_model = DynamicByteLM(config.model).cpu()
    with torch.no_grad():
        prune_model.blocks[0].wo.weight[:, 0].zero_()
        prune_model.blocks[0].activation_ema[0] = 0.0
    prune_controller = StructureController(config)
    prune_controller.losses = deque([5.0] * 16 + [4.0] * 8, maxlen=64)
    prune_controller.last_change_step = -10**9
    prune_event = prune_controller.consider(
        prune_model, structure_step, 4.0
    )
    if prune_event:
        prune_model.resize_hidden(prune_event["layer"], prune_event["new_dim"])

    return {
        "growth_event": growth_event,
        "growth_dim": growth_model.hidden_dims(),
        "prune_event": prune_event,
        "prune_dim": prune_model.hidden_dims(),
        "passed": bool(
            growth_event
            and growth_event["new_dim"] > config.model.hidden_dim
            and prune_event
            and prune_event["new_dim"] < config.model.hidden_dim
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed-cycles", type=int, default=200)
    parser.add_argument("--real-bytes", type=int, default=20_000_000)
    parser.add_argument("--restart-every", type=int, default=50)
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=Path("O:/数据集"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.mixed_cycles < 64 or args.restart_every < 8:
        raise ValueError("validation duration is too short for meaningful checks")
    directory = args.output_dir or ROOT / "validation_runs" / f"continuous-{datetime.now():%Y%m%d-%H%M%S}"
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"refusing to reuse non-empty validation directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    config = build_config()
    run = ValidationRun(config, directory)
    started = time.monotonic()
    baseline_rss = rss_megabytes()
    baseline_gpu = None
    try:
        agent = run.create_agent()
        baseline_gpu = torch.cuda.memory_allocated(agent.worker.device)
        print(json.dumps({"stage": "mixed_start", "state_dir": str(run.state_dir)}), flush=True)

        for cycle in range(args.mixed_cycles):
            active = agent.active
            passive_name = agent.passive
            before_calls = run.snapshot_counts()
            before_weight = parameter_vector(agent.state(active).cpu_model) if cycle < 40 else None
            before_interactions = agent.interaction_count
            before_switches = agent.switch_count
            frame = synthetic_frame(cycle)
            response = agent.interact_frame(frame, response_spec(frame))
            after_calls = run.snapshot_counts()
            passive_before = before_calls[passive_name]
            if after_calls[passive_name] != passive_before:
                self_passive_violation = {
                    "cycle": cycle,
                    "active_before": active,
                    "passive_before": passive_name,
                    "active_after": agent.active,
                    "passive_after": agent.passive,
                    "interactions_before": before_interactions,
                    "interactions_after": agent.interaction_count,
                    "switches_before": before_switches,
                    "switches_after": agent.switch_count,
                    "calls_before": before_calls,
                    "calls_after": after_calls,
                }
                run.passive_violations += 1
                run.record({"stage": "passive_cpu_mismatch", **self_passive_violation})
                raise AssertionError(f"passive CPU inference occurred: {self_passive_violation}")
            run.record(
                {
                    "stage": "cycle",
                    "cycle": cycle,
                    "active_before": active,
                    "active_after": agent.active,
                    "passive_before": passive_name,
                    "calls_before": before_calls,
                    "calls_after": after_calls,
                    "interactions": agent.interaction_count,
                    "switches": agent.switch_count,
                }
            )
            if after_calls[active] <= before_calls[active]:
                raise AssertionError(f"active CPU inference missing at cycle {cycle}")
            if before_weight is not None:
                after_weight = parameter_vector(agent.state(active).cpu_model)
                delta = float((after_weight - before_weight).abs().sum().item())
                if not delta > 0:
                    state = agent.state(active)
                    raise AssertionError(
                        f"TTT did not publish changed CPU weights at cycle {cycle}; "
                        f"event={state.last_event!r}, steps={state.global_step}, "
                        f"interactions={agent.interaction_count}, switches={agent.switch_count}"
                    )
                run.ttt_deltas.append(delta)
            if not math.isfinite(agent.state(active).last_loss):
                run.nonfinite_losses += 1
                raise AssertionError(f"non-finite loss at cycle {cycle}")
            run.step_samples.append(
                (
                    cycle,
                    int(agent.state("left").global_step),
                    int(agent.state("right").global_step),
                )
            )
            run.sample_resources(agent.worker.device)
            for name in ("left", "right"):
                run.lr_values[name].add(float(agent.state(name).lr.value))
                dims = agent.state(name).model.hidden_dims()
                if not run.hidden_history[name] or run.hidden_history[name][-1] != dims:
                    run.hidden_history[name].append(dims)

            if cycle % 8 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "mixed",
                            "cycle": cycle + 1,
                            "active": agent.active,
                            "switches": agent.switch_count,
                            "left_steps": agent.state("left").global_step,
                            "right_steps": agent.state("right").global_step,
                            "hidden": agent.state("left").model.hidden_dims(),
                            "cpu_calls": run.cpu_calls,
                            "switch_decisions": len(agent.switch_decisions),
                            "memory_signals": agent.status().get("memory_signals", {}),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if cycle % 7 == 3:
                feed_frame = ByteFrame.raw(
                    bytes((cycle + index * 5) & 0xFF for index in range(256)),
                    {"cycle": cycle, "kind": "observation"},
                )
                agent.feed_frame(feed_frame)
            if (cycle + 1) % args.restart_every == 0:
                restored = run.verify_restore()
                agent = run.agent
                assert agent is not None
                run.record({"stage": "restart", "at_cycle": cycle + 1, **restored})
                print(json.dumps({"stage": "restart", "at_cycle": cycle + 1, **restored}), flush=True)

        # Statistical cleaning and duplicate consolidation on a deliberately
        # low-entropy frame. It should merge duplicate digests and low weight.
        before_unique = len(agent.state("left").replay) + len(agent.state("right").replay)
        before_digests = {
            digest
            for name in ("left", "right")
            for digest in agent.state(name).replay.samples
        }
        repeated = ByteFrame.raw(bytes(256), {"kind": "repeated-zero", "media_type": "application/x-zero"})
        for _ in range(5):
            agent.feed_frame(repeated)
        after_unique = len(agent.state("left").replay) + len(agent.state("right").replay)
        after_digests = {
            digest
            for name in ("left", "right")
            for digest in agent.state(name).replay.samples
        }
        new_digests = after_digests - before_digests
        stored_weights = [
            sample.weight
            for name in ("left", "right")
            for sample in agent.state(name).replay.samples.values()
            if sample.source.startswith("feed:")
        ] + [
            sample.weight
            for name in ("left", "right")
            for sample in agent.state(name).debt.items.values()
            if sample.source.startswith("debt:feed:")
        ]
        minimum_feed_weight = min(stored_weights) if stored_weights else 1.0
        cleaning = {
            "unique_before": before_unique,
            "unique_after": after_unique,
            "increase": after_unique - before_unique,
            "new_digests": len(new_digests),
            "new_digest_hashes": sorted(new_digests)[:5],
            "minimum_feed_weight": minimum_feed_weight,
        }
        run.record({"stage": "cleaning", **cleaning})
        print(json.dumps({"stage": "cleaning", **cleaning}), flush=True)

        # Real dataset bytes, read-only and with a validation-local progress file.
        real = {
            "requested_bytes": 0 if args.skip_real else args.real_bytes,
            "payload_bytes": 0,
            "frames": 0,
            "resume_verified": False,
            "source_unchanged": True,
        }
        if not args.skip_real:
            streamer = DatasetStreamer(args.dataset_root, 960_000, 0.01, [], [])
            progress = ProgressStore(run.directory / "dataset_progress.json", streamer.root, fresh=True)
            source_stats: dict[str, tuple[int, int]] = {}
            pending_progress: list[tuple[Any, int, str, int]] = []

            def commit_dataset_progress() -> None:
                # The checkpoint must land first. A crash between these two
                # operations can replay frames, but can never skip untrained
                # frames by advancing the dataset offset first.
                if not pending_progress:
                    return
                agent.save()
                for saved_record, saved_offset, chunk_hash, chunk_bytes in pending_progress:
                    progress.mark(saved_record, saved_offset, chunk_hash, chunk_bytes)
                pending_progress.clear()

            for frame, record, next_offset in streamer.frames(
                progress, args.real_bytes, max_files=0, deadline=None
            ):
                source_stats[record.relative] = (record.size, record.mtime_ns)
                agent.feed_frame(frame)
                pending_progress.append(
                    (
                        record,
                        next_offset,
                        str(frame.metadata.get("chunk_sha256", "")),
                        len(frame.payload),
                    )
                )
                real["payload_bytes"] += len(frame.payload)
                real["frames"] += 1
                if real["frames"] % 4 == 0:
                    commit_dataset_progress()
                    print(json.dumps({"stage": "real_dataset", **real, "last_file": record.relative}, ensure_ascii=False), flush=True)
            commit_dataset_progress()
            for relative, (size, mtime_ns) in source_stats.items():
                current = streamer.root / relative
                if current.stat().st_size != size or current.stat().st_mtime_ns != mtime_ns:
                    real["source_unchanged"] = False
                    break

            # Verify the next read starts at the saved offset, not at zero.
            resume_frames = streamer.frames(progress, 65536, max_files=1, deadline=None)
            try:
                frame, record, next_offset = next(resume_frames)
            except StopIteration:
                resume_frames.close()
                raise AssertionError("dataset ended before resume could be checked")
            else:
                expected_offset = progress.offset(record)
                if int(frame.metadata["offset"]) != expected_offset:
                    raise AssertionError(
                        f"resume offset mismatch: {frame.metadata['offset']} != {expected_offset}"
                    )
                agent.feed_frame(frame)
                agent.save()
                progress.mark(
                    record,
                    next_offset,
                    str(frame.metadata.get("chunk_sha256", "")),
                    len(frame.payload),
                )
                real["resume_verified"] = True
                real["resume_offset"] = expected_offset
            finally:
                resume_frames.close()
        else:
            real["resume_verified"] = "skipped"
            real["source_unchanged"] = "skipped"
            real["resume_offset"] = 0

        # Final checkpoint reload after all phases.
        run.verify_restore()
        final_agent = run.agent
        assert final_agent is not None
        structure = validate_automatic_structure(config)
        status = final_agent.status()
        run.background_errors = list(status.get("background_errors", []))
        left_steps = status["hemispheres"]["left"]["steps"]
        right_steps = status["hemispheres"]["right"]["steps"]
        final_gpu = torch.cuda.memory_allocated(final_agent.worker.device)
        total_gpu = torch.cuda.get_device_properties(final_agent.worker.device).total_memory
        final_rss = rss_megabytes()
        peak_rss = max(run.rss_samples + [final_rss]) if run.rss_samples else final_rss
        peak_gpu = max(run.gpu_samples + [final_gpu]) if run.gpu_samples else final_gpu
        auto_records = [
            record
            for record in final_agent.switch_decisions
            if record.get("switched")
            and not record.get("manual")
            and record.get("should_switch")
        ]
        recomputable_switches = [
            record
            for record in auto_records
            if float(record.get("pressure_growth", 0.0))
            >= float(record.get("learned_capacity", 0.0)) - 1e-9
            and abs(
                float(record.get("clean_pressure", 0.0))
                - float(record.get("baseline_pressure", 0.0))
                - float(record.get("pressure_growth", 0.0))
            )
            <= 1e-6
        ]
        statuses = {
            "cpu_inference_gpu_training": status["training_device"] == "cuda"
            and next(final_agent.state("left").cpu_model.parameters()).device.type == "cpu",
            "mixed_cycles_completed": args.mixed_cycles >= 200,
            "both_hemispheres_trained": left_steps >= 20
            and right_steps >= 20
            and run.step_window_fraction(window=20, warmup=40) >= 0.30,
            "automatic_switching": len(auto_records) >= 3
            and len({record.get("active_after") for record in auto_records}) >= 2,
            "adaptive_switch_without_fixed_countdown": (
                not hasattr(config.runtime, "switch_interactions")
                and not hasattr(config.runtime, "switch_interval_s")
                and len(recomputable_switches) >= 3
                and all(
                    record.get("active_work_kind") == "train+ttt"
                    and record.get("passive_work_kind") == "consolidate"
                    and record.get("active_name")
                    and record.get("passive_name")
                    for record in recomputable_switches
                )
            ),
            "ttt_weight_updates": len(run.ttt_deltas) > 0 and min(run.ttt_deltas) > 0,
            "passive_cpu_isolation": run.passive_violations == 0,
            "train_loss_finite": run.nonfinite_losses == 0
            and all(
                math.isfinite(final_agent.state(name).last_loss) for name in ("left", "right")
            ),
            "debt_bounded": all(
                len(final_agent.state(name).debt) <= config.runtime.debt_capacity
                for name in ("left", "right")
            ),
            "replay_bounded": all(
                len(final_agent.state(name).replay) <= config.runtime.replay_capacity
                for name in ("left", "right")
            ),
            "duplicate_cleaning": cleaning["new_digests"] <= 1 and minimum_feed_weight < 0.5,
            "automatic_structure_events": structure["passed"],
            "restarts_exact": len(run.restart_records) >= 4
            and all(
                record["loaded"]
                and record["active_match"]
                and record["switches_match"]
                and record["steps_match"]
                and record["step_kinds_match"]
                and record["fingerprint_match"]
                for record in run.restart_records
            ),
            "real_dataset_volume": (
                "skipped"
                if args.skip_real
                else real["payload_bytes"] >= min(args.real_bytes, 20_000_000)
            ),
            "real_dataset_resume": "skipped" if args.skip_real else real["resume_verified"],
            "source_files_unchanged": "skipped" if args.skip_real else real["source_unchanged"],
            "queue_drained": status["queued_jobs"] == 0,
            "learning_rate_in_bounds": all(
                config.runtime.min_learning_rate <= value <= config.runtime.max_learning_rate
                for values in run.lr_values.values()
                for value in values
            ),
            "host_memory_bounded": final_rss is not None
            and final_rss > 0
            and peak_rss is not None
            and peak_rss > 0
            and peak_rss <= config.runtime.memory_budget_mb,
            "gpu_memory_bounded": final_gpu <= total_gpu and peak_gpu <= total_gpu,
            "no_runtime_errors": len(run.errors) == 0 and len(run.background_errors) == 0,
        }
        failed = [name for name, value in statuses.items() if value is False]
        skipped = [name for name, value in statuses.items() if value == "skipped"]
        passed = [name for name, value in statuses.items() if value is True]
        report = {
            "created_at": datetime.now().isoformat(),
            "all_measurable_criteria_passed": len(failed) == 0 and len(skipped) == 0,
            "criteria": statuses,
            "criteria_counts": {"passed": len(passed), "failed": len(failed), "skipped": len(skipped)},
            "mixed_cycles": args.mixed_cycles,
            "elapsed_seconds": time.monotonic() - started,
            "left_steps": left_steps,
            "right_steps": right_steps,
            "step_accounting": status.get("step_accounting"),
            "starvation_window_fraction": run.step_window_fraction(window=20, warmup=40),
            "switches": status["switches"],
            "ttt_samples": len(run.ttt_deltas),
            "ttt_minimum_weight_l1_delta": min(run.ttt_deltas) if run.ttt_deltas else 0,
            "lr_values": {name: sorted(values) for name, values in run.lr_values.items()},
            "hidden_history": run.hidden_history,
            "automatic_structure": structure,
            "adaptive_switch": auto_records[-1] if auto_records else status.get("adaptive_switch"),
            "switch_decisions": auto_records[-12:],
            "switch_decisions_total": len(auto_records),
            "switch_decisions_recomputable": len(recomputable_switches),
            "cleaning": cleaning,
            "real_dataset": real,
            "restarts": run.restart_records,
            "memory_signals": status.get("memory_signals"),
            "memory": {
                "rss_baseline_mb": baseline_rss,
                "rss_peak_mb": peak_rss,
                "rss_final_mb": final_rss,
                "rss_samples_mb": run.rss_samples,
                "gpu_baseline_bytes": baseline_gpu,
                "gpu_peak_bytes": peak_gpu,
                "gpu_final_bytes": final_gpu,
                "gpu_total_bytes": total_gpu,
                "memory_budget_mb": config.runtime.memory_budget_mb,
            },
            "state_dir": str(run.state_dir),
            "scope": "Mechanism validation only; no claim of semantic understanding or consciousness.",
        }
        (directory / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0 if report["all_measurable_criteria_passed"] else 2
    except Exception as exc:
        run.errors.append(repr(exc))
        run.errors.append(traceback.format_exc())
        report = {
            "created_at": datetime.now().isoformat(),
            "all_measurable_criteria_passed": False,
            "errors": run.errors,
            "elapsed_seconds": time.monotonic() - started,
            "state_dir": str(run.state_dir),
        }
        (directory / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"stage": "failure", **report}, ensure_ascii=False, indent=2), flush=True)
        return 1
    finally:
        if run.agent is not None:
            run.remove_hooks()
            run.agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
