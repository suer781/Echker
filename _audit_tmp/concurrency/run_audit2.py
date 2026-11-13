"""Second concurrency probe: monitor-initiated save consistency + interact-only starvation.

Artifacts stay under _audit_tmp/concurrency. Source is not modified.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(r"O:\Documents\智能自学习AI（新）")
sys.path.insert(0, str(ROOT))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from life_agent.config import AppConfig  # noqa: E402
from life_agent.agent import LifeAgent  # noqa: E402
from life_agent.byte_codec import ByteFrame  # noqa: E402

AUDIT = ROOT / "_audit_tmp" / "concurrency"
STATE = AUDIT / "state2"
STATE.mkdir(parents=True, exist_ok=True)

records: dict[str, list[dict]] = {
    "jobs": [], "save": [], "cycles": [], "close": [], "phase_b": [],
}
lock = threading.Lock()
cycle = {"n": -1, "phase": "A"}


def log(bucket: str, rec: dict) -> None:
    rec["t"] = round(time.monotonic(), 6)
    with lock:
        records[bucket].append(rec)


def main() -> int:
    config = AppConfig.load(AUDIT / "config_audit2.json")
    agent = LifeAgent(config, STATE)
    worker = agent.worker

    orig_submit = worker.submit

    def instrumented_submit(kind, hemisphere=None, samples=None, steps=1,
                            wait=True, priority=None, event=None):
        t0 = time.monotonic()
        rec = {"kind": kind, "hemisphere": hemisphere, "steps": steps,
               "wait": wait, "enqueue": round(t0, 6),
               "thread": threading.current_thread().name, "phase": cycle["phase"]}
        fut = orig_submit(kind, hemisphere, samples, steps, wait,
                          priority=priority, event=event)
        if not wait:
            def done(f, rec=rec):
                rec["done"] = round(time.monotonic(), 6)
                rec["exception"] = repr(f.exception()) if f.exception() else None
                log("jobs", rec)
            fut.add_done_callback(done)
        else:
            rec["returned"] = round(time.monotonic(), 6)
            log("jobs", rec)
        return fut

    worker.submit = instrumented_submit

    orig_save = agent.save

    def wrap_save():
        t0 = time.monotonic()
        snap = {n: {"step": agent.state(n).global_step,
                     "debt": len(agent.state(n).debt),
                     "replay": len(agent.state(n).replay)}
                for n in ("left", "right")}
        pending = {k: v.done() for k, v in agent.pending_consolidation.items()}
        try:
            return orig_save()
        finally:
            log("save", {
                "thread": threading.current_thread().name,
                "dur": round(time.monotonic() - t0, 6),
                "snapshot": snap, "pending_done": pending,
                "pending_pending": [k for k, d in pending.items() if not d],
            })

    agent.save = wrap_save

    agent.start()
    close_exc = None
    try:
        # Phase A: mixed stream, lets the 2s monitor checkpoint fire.
        for c in range(24):
            cycle["n"] = c
            cycle["phase"] = "A"
            if c % 2 == 0:
                frame = ByteFrame.raw(bytes((c * 13 + i * 7) & 0xFF for i in range(48)), {"cycle": c})
                agent.feed_frame(frame)
            else:
                frame = ByteFrame.text(f"混合 {c}")
                agent.interact_frame(frame, frame.empty())
            records["cycles"].append({
                "cycle": c, "phase": "A", "active": agent.active,
                "pending": list(agent.pending_consolidation.keys()),
                "steps": {n: agent.state(n).global_step for n in ("left", "right")},
            })

        # Phase B: interact-only, no drain on this path.
        b_start_steps = {n: agent.state(n).global_step for n in ("left", "right")}
        b_start_cons = sum(1 for j in records["jobs"] if j["kind"] == "consolidate")
        for c in range(36):
            cycle["n"] = 24 + c
            cycle["phase"] = "B"
            frame = ByteFrame.text(f"纯交互 {c} 持续上下文")
            t0 = time.monotonic()
            agent.interact_frame(frame, frame.empty())
            records["phase_b"].append({
                "cycle": c, "dur": round(time.monotonic() - t0, 6),
                "active": agent.active,
                "steps": {n: agent.state(n).global_step for n in ("left", "right")},
                "pending": list(agent.pending_consolidation.keys()),
                "qsize": worker.jobs.qsize(),
            })
    except Exception as exc:  # noqa: BLE001
        records["close"].append({"stage": "stream_failed", "exc": repr(exc),
                                 "trace": traceback.format_exc()})
    finally:
        close_t0 = time.monotonic()
        try:
            agent.close()
        except Exception as exc:  # noqa: BLE001
            close_exc = repr(exc)
        records["close"].append({
            "close_dur": round(time.monotonic() - close_t0, 6),
            "exc": close_exc,
            "final": {
                "qsize": worker.jobs.qsize(),
                "pending": list(agent.pending_consolidation.keys()),
                "steps": {n: agent.state(n).global_step for n in ("left", "right")},
            },
        })

    jobs_b = [j for j in records["jobs"] if j["phase"] == "B"]
    summary = {
        "save_events": records["save"],
        "phase_b": {
            "interact_calls": len(records["phase_b"]),
            "consolidate_jobs": sum(1 for j in jobs_b if j["kind"] == "consolidate"),
            "ttt_jobs": sum(1 for j in jobs_b if j["kind"] == "ttt"),
            "train_jobs": sum(1 for j in jobs_b if j["kind"] == "train"),
            "steps_delta": {
                n: records["phase_b"][-1]["steps"][n] - b_start_steps[n]
                for n in ("left", "right")
            },
            "qsize_max": max((p["qsize"] for p in records["phase_b"]), default=0),
            "pending_at_end": records["phase_b"][-1]["pending"],
        },
        "close": records["close"],
    }
    (AUDIT / "audit2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (AUDIT / "audit2_metrics.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0 if close_exc is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
