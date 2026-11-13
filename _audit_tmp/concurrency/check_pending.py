"""Check unresolved / failed futures in audit2 raw metrics."""
from __future__ import annotations

import json
from pathlib import Path

AUDIT = Path(r"O:\Documents\智能自学习AI（新）\_audit_tmp\concurrency")
r = json.loads((AUDIT / "audit2_metrics.json").read_text(encoding="utf-8"))
jobs = r["jobs"]

wf = [j for j in jobs if not j["wait"]]
none_done = [j for j in wf if j.get("done") is None]
exc = [j for j in jobs if j.get("exception")]
print(f"wait=False total: {len(wf)}, done=None: {len(none_done)}, exceptions: {len(exc)}")
for j in none_done:
    print("NONE:", json.dumps(j, ensure_ascii=False))
for j in exc:
    print("EXC:", json.dumps(j, ensure_ascii=False))

# also audit1
r1 = json.loads((AUDIT / "audit_metrics.json").read_text(encoding="utf-8"))
wf1 = [j for j in r1["jobs"] if not j["wait"]]
none1 = [j for j in wf1 if j.get("done") is None]
exc1 = [j for j in r1["jobs"] if j.get("exception")]
print(f"\naudit1 wait=False: {len(wf1)}, done=None: {len(none1)}, exceptions: {len(exc1)}")
for j in none1:
    print("NONE1:", json.dumps(j, ensure_ascii=False))
