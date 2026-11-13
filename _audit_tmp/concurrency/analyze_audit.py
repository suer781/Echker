"""Summarize audit_metrics.json for the concurrency review (read-only analysis)."""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict

AUDIT = Path(r"O:\Documents\智能自学习AI（新）\_audit_tmp\concurrency")
records = json.loads((AUDIT / "audit_metrics.json").read_text(encoding="utf-8"))

jobs = records["jobs"]
compute = records["compute"]
schedule = records["schedule"]
drain = records["drain"]
maybe = records["maybe_switch"]
save = records["save"]
samples = records["samples"]
status = records["status"]
cycles = records["cycles"]
close = records["close"]

summary: dict = {}

# ---- jobs by kind / wait / exception ------------------------------------
summary["jobs_by_kind"] = Counter(j["kind"] for j in jobs)
summary["jobs_wait_false"] = [
    {"kind": j["kind"], "hemisphere": j["hemisphere"], "exception": j.get("exception"),
     "enqueue": j["enqueue"], "done": j.get("done")}
    for j in jobs if not j["wait"]
]
summary["jobs_with_exception"] = [j for j in jobs if j.get("exception")]

# wait=True job span (queue wait + compute, since submit blocks)
wait_true = [j for j in jobs if j["wait"]]
spans = [round(j["returned"] - j["enqueue"], 6) for j in wait_true]
summary["wait_true_span"] = {
    "count": len(spans),
    "max": max(spans) if spans else None,
    "mean": round(sum(spans) / len(spans), 6) if spans else None,
    "p95": sorted(spans)[int(len(spans) * 0.95)] if spans else None,
}

# compute durations by kind
comp_by = defaultdict(list)
for c in compute:
    comp_by[c["kind"]].append(c["dur"])
summary["compute_dur"] = {
    k: {"count": len(v), "max": round(max(v), 6), "mean": round(sum(v) / len(v), 6)}
    for k, v in comp_by.items()
}

# ---- drain records -------------------------------------------------------
summary["drain"] = {
    "count": len(drain),
    "with_future": sum(1 for d in drain if d["had_future"]),
    "no_future": sum(1 for d in drain if not d["had_future"]),
    "step_delta": [d["step_delta"] for d in drain],
    "dur_max": round(max((d["dur"] for d in drain), default=0), 6),
    "dur_mean": round(sum(d["dur"] for d in drain) / len(drain), 6) if drain else None,
    "exc": [d["exc"] for d in drain if d["exc"]],
}

# ---- schedule records -----------------------------------------------------
summary["schedule"] = {
    "count": len(schedule),
    "new_future": sum(1 for s in schedule if s["new_future"]),
    "deferred_pending_running": sum(
        1 for s in schedule if not s["new_future"] and s["before_keys"] == s["after_keys"]
    ),
    "by_thread": Counter(s["thread"] for s in schedule),
    "exc": [s["exc"] for s in schedule if s["exc"]],
}

# ---- maybe_switch probes --------------------------------------------------
by_thread = defaultdict(list)
for m in maybe:
    by_thread[m["thread"]].append(m)
summary["maybe_switch"] = {
    "total": len(maybe),
    "by_thread": {
        t: {
            "calls": len(v),
            "early_return": sum(1 for x in v if x["early_return"]),
            "switched": sum(1 for x in v if x["switched"]),
            "should_switch": sum(1 for x in v if x["should_switch"]),
        }
        for t, v in by_thread.items()
    },
    "early_return_after_monitor": [],
}
# detect: main-thread early return where a homeostasis call wrote the probe last
last_writer = None
for m in maybe:
    if m["thread"] == "homeostasis":
        last_writer = "homeostasis"
    if m["thread"] != "homeostasis" and m["early_return"] and last_writer == "homeostasis":
        summary["maybe_switch"]["early_return_after_monitor"].append(
            {"cycle": m["cycle"], "count": m["count"], "probe_before": m["probe_before"]}
        )
    if m["thread"] != "homeostasis":
        last_writer = "main"

# ---- cycles ---------------------------------------------------------------
feeds = [c for c in cycles if c["kind"] == "feed"]
interacts = [c for c in cycles if c["kind"] == "interact"]
summary["cycles"] = {
    "total": len(cycles),
    "feed": len(feeds),
    "interact": len(interacts),
    "feed_dur_max": round(max((c["dur"] for c in feeds), default=0), 6),
    "feed_dur_mean": round(sum(c["dur"] for c in feeds) / len(feeds), 6) if feeds else None,
    "interact_dur_max": round(max((c["dur"] for c in interacts), default=0), 6),
    "interact_dur_mean": round(sum(c["dur"] for c in interacts) / len(interacts), 6) if interacts else None,
    "qsize_before_max": max((c["q_before"] for c in cycles), default=0),
    "qsize_after_max": max((c["q_after"] for c in cycles), default=0),
    "pending_at_cycle_end": {
        str(k): v for k, v in Counter(tuple(c["pending"]) for c in cycles).items()
    },
    "switches": cycles[-1]["switches_after"] if cycles else 0,
}

# ---- samples: queue / pending / thread states -----------------------------
summary["samples"] = {
    "count": len(samples),
    "qsize_max": max((s["qsize"] for s in samples), default=0),
    "pending_keys_seen": {
        str(k): v for k, v in Counter(tuple(s["pending"].keys()) for s in samples).items()
    },
    "threads_seen": sorted({t for s in samples for t in s["threads"]}),
}
# thread state histogram
state_hist = defaultdict(Counter)
for s in samples:
    for name, info in s["threads"].items():
        state_hist[name][info["state"]] += 1
summary["samples"]["thread_states"] = {k: dict(v) for k, v in state_hist.items()}

# ---- observer status (lock contention) -----------------------------------
sdur = [s["dur"] for s in status]
summary["status_observer"] = {
    "count": len(sdur),
    "max": round(max(sdur), 6) if sdur else None,
    "mean": round(sum(sdur) / len(sdur), 6) if sdur else None,
    "exc": [s.get("exc") for s in status if s.get("exc")],
}

# ---- close ----------------------------------------------------------------
summary["close"] = close

(AUDIT / "audit_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
