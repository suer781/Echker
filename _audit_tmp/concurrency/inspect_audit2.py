"""Timeline around monitor save in audit2 raw metrics."""
from __future__ import annotations

import json
from pathlib import Path

AUDIT = Path(r"O:\Documents\智能自学习AI（新）\_audit_tmp\concurrency")
r = json.loads((AUDIT / "audit2_metrics.json").read_text(encoding="utf-8"))
saves = r["save"]
jobs = r["jobs"]

for sv in saves:
    print("=== save", sv["thread"], "t=", sv["t"], "pending_pending=", sv["pending_pending"])
    print("snapshot:", json.dumps(sv["snapshot"], ensure_ascii=False))
    # jobs enqueued within +-1.5s around save
    nearby = [j for j in jobs if abs(j["enqueue"] - sv["t"]) < 1.5]
    for j in sorted(nearby, key=lambda x: x["enqueue"]):
        print(f"  enq={j['enqueue']:.3f} done={j.get('done')} kind={j['kind']:12s} "
              f"hemi={j['hemisphere']} wait={j['wait']} exc={j.get('exception')}")
    print()

# phase B consolidate vs ttt timing
pb = [j for j in jobs if j.get("phase") == "B"]
from collections import Counter
print("phase B jobs:", Counter(j["kind"] for j in pb))
pending_left = [j for j in pb if j["hemisphere"] == "left" and j["kind"] == "consolidate"]
pending_right = [j for j in pb if j["hemisphere"] == "right" and j["kind"] == "consolidate"]
print(f"phase B consolidate left={len(pending_left)} right={len(pending_right)}")
