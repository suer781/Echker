"""Read-only behavioral probe: StatisticalCleaner / ReplayBuffer / DebtLedger / switch.

Audit-only. Does NOT import the life_agent package (whose __init__ pulls torch);
memory.py and switching.py are loaded directly by path. No model training, no GPU.
Writes only probe_results.json next to this file.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ROOT = Path(r"O:\Documents\智能自学习AI（新）")
AUDIT = ROOT / "_audit_tmp" / "novelty"


def _load(module_name: str, rel: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mem = _load("audit_memory", "life_agent/memory.py")
sw = _load("audit_switching", "life_agent/switching.py")

StatisticalCleaner = mem.StatisticalCleaner
ReplayBuffer = mem.ReplayBuffer
DebtLedger = mem.DebtLedger
Sample = mem.Sample
AdaptiveSwitchController = sw.AdaptiveSwitchController


def bit_entropy(ids: list[int]) -> float:
    if not ids:
        return 0.0
    counts: dict[int, int] = {}
    for v in ids:
        counts[v] = counts.get(v, 0) + 1
    n = len(ids)
    return -sum((c / n) * math.log(c / n) for c in counts.values()) / math.log(2)


def longest_run(ids: list[int]) -> int:
    if not ids:
        return 0
    run = best = 1
    for i in range(1, len(ids)):
        run = run + 1 if ids[i] == ids[i - 1] else 1
        best = max(best, run)
    return best


rng = random.Random(7)

# --- Construct probe corpora (ids, labels identical per observation_sample) ---
cases: dict[str, list[int]] = {
    "silence_pcm_256_zeros": [0] * 256,
    "solid_gray_256": [128] * 256,
    "random_high_entropy_256": [rng.randint(0, 255) for _ in range(256)],
    "alternating_two_values": [i % 2 for i in range(256)],
    "monotonic_0_to_255": list(range(256)),
    "periodic_16_cycle": [i % 16 for i in range(256)],
    "short_seq_len8": [5] * 8,
    "block_len32": [10, 20, 30, 40] * 8,
    "block_len64": [10, 20, 30, 40] * 16,
    "block_len128": [10, 20, 30, 40] * 32,
}
# Same inner block, different context wrapping -> different digest, same inner repetition.
inner = [7, 77, 13, 200] * 8  # 32 bytes
cases["wrapped_prefixA_suffixA"] = [1, 2, *inner, 254]
cases["wrapped_prefixB_suffixB"] = [9, 99, *inner, 250, 251]
# Structured table: 16 rows x 16 cols, last column jitters per row.
rows: list[int] = []
for r in range(16):
    row = [(c * 17 + r) % 256 for c in range(15)] + [(r * 31 + 5) % 256]
    rows.extend(row)
cases["table_16x16_with_jitter"] = rows
# Periodic stream frame k: base 16-cycle, one position jittered by frame index.
periodic_frames: list[list[int]] = []
for k in range(8):
    frame = [i % 16 for i in range(256)]
    frame[(k * 37) % 256] = (frame[(k * 37) % 256] + k + 1) % 256
    periodic_frames.append(frame)

cleaner = StatisticalCleaner()
quality_rows = []
for name, ids in cases.items():
    quality_rows.append(
        {
            "case": name,
            "length": len(ids),
            "distinct": len(set(ids)),
            "entropy_bits": round(bit_entropy(ids), 4),
            "longest_run": longest_run(ids),
            "structural_quality": round(cleaner.structural_quality(ids), 6),
        }
    )

# Digest equality: exact repeat vs wrapped vs micro-variant.
def make(ids: list[int], weight: float = 0.5, source: str = "probe") -> Sample:
    return Sample(ids=ids, labels=ids.copy(), weight=weight, source=source)


digest_samples = {
    "exact_zero": make([0] * 256),
    "exact_zero_dup": make([0] * 256),
    "wrapped_a": make(cases["wrapped_prefixA_suffixA"]),
    "wrapped_b": make(cases["wrapped_prefixB_suffixB"]),
    "table": make(cases["table_16x16_with_jitter"]),
}
digest_map = {k: v.digest for k, v in digest_samples.items()}

# --- ReplayBuffer behavior under repeated / micro-varied admission ---
replay = ReplayBuffer(capacity=1024)
replay_trace = []
zero = make([0] * 256, weight=0.5)
for admission in range(5):
    replay.add(zero.clone())
    stored = replay.samples[zero.digest]
    replay_trace.append(
        {"admission": admission + 1, "unique": len(replay), "weight": round(stored.weight, 6)}
    )
wrapped = make(cases["wrapped_prefixA_suffixA"])
replay.add(wrapped)
replay.add(make(cases["wrapped_prefixB_suffixB"]))
replay_after_wrap = {"unique": len(replay), "weights": {k: round(v.weight, 6) for k, v in replay.samples.items()}}

# --- DebtLedger: exact repeat accumulates weight; micro-variants each add pressure ---
ledger = DebtLedger(capacity=512, decay=0.75, min_weight=0.005)
debt_trace = []
for admission in range(5):
    ledger.add(zero.clone(), retention=0.35)
    item = ledger.items[zero.digest]
    debt_trace.append(
        {
            "admission": admission + 1,
            "unique": len(ledger),
            "weight": round(item.weight, 6),
            "clean_pressure": round(ledger.clean_pressure(), 6),
            "clean_backlog": round(ledger.clean_backlog(), 6),
        }
    )
micro_trace = []
for k, frame in enumerate(periodic_frames):
    ledger.add(make(frame, weight=0.5, source=f"probe:micro{k}"), retention=0.35)
    micro_trace.append(
        {
            "admission": k + 1,
            "unique": len(ledger),
            "clean_pressure": round(ledger.clean_pressure(), 6),
            "clean_backlog": round(ledger.clean_backlog(), 6),
        }
    )
due_batch = ledger.due(4)
capacity_after_due = ledger.repayment_capacity
ledger.decay_after_training()
decay1 = {d: round(item.weight, 6) for d, item in ledger.items.items()}
ledger.decay_after_training()
decay2_len = len(ledger)

# --- loss_weight trace: robust outlier score, not a repetition score ---
lw = StatisticalCleaner()
loss_weight_trace = []
# Warm-up 20 losses around mean 5.0.
for i in range(20):
    w = lw.loss_weight(5.0 + (i % 3) * 0.05)
loss_weight_trace.append({"phase": "warmup", "count": lw.count, "mean": round(lw.loss_mean, 4)})
for label, loss in [("typical", 5.0), ("high_outlier", 9.0), ("repeated_low", 3.0)]:
    w = lw.loss_weight(loss)
    loss_weight_trace.append({"phase": label, "loss": loss, "weight": round(w, 6), "count": lw.count})

# --- due() ordering: exact-repeat debt accumulates raw weight and is repaid first ---
order_ledger = DebtLedger(capacity=512, decay=0.75, min_weight=0.005)
order_ledger.add(zero.clone(), retention=0.35)
order_ledger.add(zero.clone(), retention=0.35)
order_ledger.add(zero.clone(), retention=0.35)
novel_item = make([rng.randint(0, 255) for _ in range(256)], weight=0.5)
order_ledger.add(novel_item, retention=0.35)
due_order = [
    {"source": item.source, "weight": round(item.weight, 6),
     "quality": round(StatisticalCleaner.structural_quality(item.ids), 6)}
    for item in order_ledger.due(4)
]

# --- Pure-python switch replay: repeated stream vs novel stream ---
def run_switch_replay(stream_name: str, frames: list[list[int]], rounds: int = 12) -> dict:
    ledger_local = DebtLedger(capacity=512, decay=0.75, min_weight=0.005)
    controller = AdaptiveSwitchController(enabled=True)
    # Baseline: one consolidation batch already repaid, so capacity is measurable.
    controller.note_switch(clean_pressure=0.0, active_work=0, passive_work=0)
    decisions = []
    intervals: list[int] = []
    last_switch_round = 0
    passive_work = 0
    active_work = 0
    for r in range(rounds):
        frame = frames[r % len(frames)]
        ledger_local.add(make(frame, weight=0.5), retention=0.35)
        # Synthetic consolidation: passive side repays a batch each round.
        ledger_local.due(4)
        passive_work += 2
        active_work += 2
        decision = controller.decide(
            clean_pressure=ledger_local.clean_pressure(),
            learned_capacity=ledger_local.repayment_capacity,
            active_work=active_work,
            passive_work=passive_work,
            has_debt=len(ledger_local) > 0,
        )
        if decision.should_switch:
            intervals.append(r + 1 - last_switch_round)
            last_switch_round = r + 1
            # Mirror _switch_locked: reset pressure/work baselines after a switch.
            controller.note_switch(
                clean_pressure=ledger_local.clean_pressure(),
                active_work=active_work,
                passive_work=passive_work,
            )
        decisions.append(
            {
                "round": r + 1,
                "pressure_growth": round(decision.pressure_growth, 6),
                "learned_capacity": round(decision.learned_capacity, 6),
                "passive_delta": round(decision.passive_work, 6),
                "active_delta": round(decision.active_work, 6),
                "should_switch": decision.should_switch,
                "reason": decision.reason,
            }
        )
    return {
        "stream": stream_name,
        "final_unique": len(ledger_local),
        "final_pressure": round(ledger_local.clean_pressure(), 6),
        "final_capacity": round(ledger_local.repayment_capacity, 6),
        "switches": sum(1 for d in decisions if d["should_switch"]),
        "switch_intervals": intervals,
        "decisions": decisions,
    }


novel_frames = [[rng.randint(0, 255) for _ in range(256)] for _ in range(12)]
repeated_frames = [[0] * 256 for _ in range(12)]
switch_repeated = run_switch_replay("exact_repeat_zeros", repeated_frames)
switch_novel = run_switch_replay("random_novel", novel_frames)

# --- Hypothetical hard-rejection accounting: what fraction of legal low-entropy cases ---
# would be excluded if a novelty gate rejected structural_quality below 0.05.
hard_reject = [
    row["case"]
    for row in quality_rows
    if row["structural_quality"] < 0.05 and row["length"] >= 16
]

report = {
    "probe": "novelty_signal_readonly",
    "note": "Pure Python replay; no torch, no training. memory.py/switching.py loaded by path.",
    "quality_rows": quality_rows,
    "digest_map": digest_map,
    "digest_equal_pairs": {
        "exact_zero==exact_zero_dup": digest_map["exact_zero"] == digest_map["exact_zero_dup"],
        "wrapped_a==wrapped_b": digest_map["wrapped_a"] == digest_map["wrapped_b"],
    },
    "replay_trace": replay_trace,
    "replay_after_wrap": replay_after_wrap,
    "debt_trace_exact_repeat": debt_trace,
    "debt_trace_micro_variants": micro_trace,
    "debt_after_due_capacity": round(capacity_after_due, 6),
    "debt_after_decay1_weights": decay1,
    "debt_len_after_decay2": decay2_len,
    "loss_weight_trace": loss_weight_trace,
    "due_ordering": due_order,
    "switch_repeated": switch_repeated,
    "switch_novel": switch_novel,
    "hard_reject_below_0.05": hard_reject,
}

out = AUDIT / "probe_results.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
