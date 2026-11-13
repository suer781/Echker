# Read-only analysis of metrics.jsonl for continuous-adaptive-full-7.
# No project files are modified; output written next to this script.
from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent.parent / "validation_runs" / "continuous-adaptive-full-7"
METRICS = RUN / "metrics.jsonl"
REPORT = RUN / "report.json"

cycles = []
restarts = []
with METRICS.open("r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec["stage"] == "cycle":
            cycles.append(rec)
        elif rec["stage"] == "restart":
            restarts.append(rec)

report = json.loads(REPORT.read_text(encoding="utf-8"))

switch_records = [c for c in cycles if c["active_before"] != c["active_after"]]
switch_cycles = [c["cycle"] for c in switch_records]

# Between-cycle flips: monitor thread / feed_frame switched after the previous
# cycle record but before this cycle's record.
between_flips = []
for prev, cur in zip(cycles, cycles[1:]):
    if prev["active_after"] != cur["active_before"]:
        between_flips.append({"cycle": cur["cycle"], "from": prev["active_after"], "to": cur["active_before"]})

# Inter-switch intervals in cycle index units.
intervals = [b - a for a, b in zip(switch_cycles, switch_cycles[1:])]

# Direction of each switch.
directions = [f"{c['active_before']}->{c['active_after']}" for c in switch_records]

# Consecutive immediate re-switch: interval == 1.
immediate = [
    (switch_cycles[i], switch_cycles[i + 1], directions[i], directions[i + 1])
    for i in range(len(intervals))
    if intervals[i] == 1
]

# Calls (CPU forward hook counts) deltas per cycle.
active_deltas = []
passive_deltas = []
for c in cycles:
    ab, aa = c["active_before"], c["active_after"]
    pb = c["passive_before"]
    active_deltas.append(c["calls_after"][ab] - c["calls_before"][ab])
    passive_deltas.append(c["calls_after"][pb] - c["calls_before"][pb])

# Switch cycles vs feed cycles (feed_frame at cycle % 7 == 3, after record).
feed_cycles = {c for c in range(len(cycles)) if c % 7 == 3}
switch_near_feed = [c for c in switch_cycles if c in feed_cycles or (c - 1) in feed_cycles]

# Per-interval active/passive CPU call accumulation between switches.
def segment_stats():
    bounds = [-1] + switch_cycles + [len(cycles)]
    segs = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i] + 1, bounds[i + 1]
        seg = cycles[lo:hi]
        if not seg:
            continue
        left_calls = seg[-1]["calls_after"]["left"] - seg[0]["calls_before"]["left"]
        right_calls = seg[-1]["calls_after"]["right"] - seg[0]["calls_before"]["right"]
        segs.append({"cycles": (lo, hi - 1), "len": len(seg), "left_calls": left_calls, "right_calls": right_calls})
    return segs

segs = segment_stats()

summary = {
    "total_cycles": len(cycles),
    "switches": len(switch_records),
    "switch_cycles": switch_cycles,
    "directions": directions,
    "interval_cycles": {
        "min": min(intervals) if intervals else None,
        "max": max(intervals) if intervals else None,
        "mean": statistics.fmean(intervals) if intervals else None,
        "median": statistics.median(intervals) if intervals else None,
        "histogram": {str(k): intervals.count(k) for k in sorted(set(intervals))},
    },
    "immediate_reswitch_count": len(immediate),
    "immediate_reswitch_pairs": immediate,
    "active_cpu_call_delta_per_cycle": {
        "min": min(active_deltas),
        "max": max(active_deltas),
        "mean": statistics.fmean(active_deltas),
    },
    "passive_cpu_call_delta_nonzero": [
        {"cycle": c["cycle"], "passive": c["passive_before"], "delta": c["calls_after"][c["passive_before"]] - c["calls_before"][c["passive_before"]]}
        for c in cycles
        if c["calls_after"][c["passive_before"]] != c["calls_before"][c["passive_before"]]
    ],
    "switch_cycles_adjacent_to_feed": switch_near_feed,
    "between_cycle_flips": between_flips,
    "within_cycle_switches": len(switch_records),
    "between_cycle_flip_count": len(between_flips),
    "restarts": restarts,
    "report_elapsed_seconds": report["elapsed_seconds"],
    "report_switches": report["switches"],
    "report_left_steps": report["left_steps"],
    "report_right_steps": report["right_steps"],
    "segments_between_switches": segs,
}

out = HERE / "metrics_analysis.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

# Console digest.
print(f"cycles={len(cycles)} switches={len(switch_records)}")
print(f"interval min/median/mean/max = {summary['interval_cycles']['min']}/"
      f"{summary['interval_cycles']['median']}/{summary['interval_cycles']['mean']:.2f}/"
      f"{summary['interval_cycles']['max']}")
print(f"interval histogram: {summary['interval_cycles']['histogram']}")
print(f"immediate (interval==1) re-switches: {len(immediate)}")
print(f"between-cycle flips: {len(between_flips)} -> {between_flips}")
print(f"passive CPU deltas nonzero: {len(summary['passive_cpu_call_delta_nonzero'])}")
print(f"elapsed={report['elapsed_seconds']:.1f}s, switches/sec="
      f"{len(switch_records) / report['elapsed_seconds']:.3f}")
print(f"avg seconds per switch interval = {report['elapsed_seconds'] / max(1, len(switch_records)):.2f}")
print(f"segments: {len(segs)}")
for s in segs[:12]:
    print("  ", s)
