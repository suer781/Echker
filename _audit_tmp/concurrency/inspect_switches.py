"""Inspect pending-future retention across switches in audit_metrics.json."""
from __future__ import annotations

import json
from pathlib import Path

AUDIT = Path(r"O:\Documents\智能自学习AI（新）\_audit_tmp\concurrency")
r = json.loads((AUDIT / "audit_metrics.json").read_text(encoding="utf-8"))

cycles = r["cycles"]
sched = r["schedule"]

print("=== cycles around switches (pending keys / active / steps) ===")
prev_sw = 0
for c in cycles:
    if c["switches_after"] != prev_sw or len(c["pending"]) > 1:
        print(json.dumps({
            "cycle": c["cycle"], "kind": c["kind"], "active": c["active"],
            "pending": c["pending"], "steps": c["steps_after"],
            "switches": c["switches_after"], "dur": c["dur"],
        }, ensure_ascii=False))
    prev_sw = c["switches_after"]

print("\n=== schedule deferred / by thread ===")
for s in sched:
    if s["thread"] == "homeostasis" or not s["new_future"]:
        print(json.dumps(s, ensure_ascii=False))

print("\n=== drain records (first 8) ===")
for d in r["drain"][:8]:
    print(json.dumps(d, ensure_ascii=False))

print("\n=== maybe_switch homeostasis invocations ===")
for m in r["maybe_switch"]:
    if m["thread"] == "homeostasis":
        print(json.dumps(m, ensure_ascii=False))

# pending retention: after a switch, count cycles carrying stale other-hemisphere future
dual = [c for c in cycles if len(c["pending"]) > 1]
print(f"\ncycles with dual pending: {len(dual)} / {len(cycles)}")
