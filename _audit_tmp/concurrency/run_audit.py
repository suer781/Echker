"""Concurrency audit harness (read-only w.r.t. life_agent source).

Drives a small isolated agent through a mixed feed/interact stream and samples
queue depth, pending futures, per-call wall segments, switch probes and thread
states. All artifacts are written under this audit directory only.
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
STATE = AUDIT / "state"
STATE.mkdir(parents=True, exist_ok=True)

records: dict[str, list[dict]] = {
    "jobs": [],
    "compute": [],
    "schedule": [],
    "drain": [],
    "maybe_switch": [],
    "save": [],
    "samples": [],
    "status": [],
    "cycles": [],
    "close": [],
}
lock = threading.Lock()
cycle = {"n": -1}


def log(bucket: str, rec: dict) -> None:
    rec["t"] = round(time.monotonic(), 6)
    rec["wall"] = time.time()
    with lock:
        records[bucket].append(rec)


def build_agent(device_override: str | None = None) -> LifeAgent:
    config = AppConfig.load(AUDIT / "config_audit.json")
    if device_override is not None:
        config.runtime.training_device = device_override
        config.runtime.require_cuda = False
    agent = LifeAgent(config, STATE)
    worker = agent.worker

    # ---- job queue instrumentation -------------------------------------
    orig_submit = worker.submit

    def instrumented_submit(kind, hemisphere=None, samples=None, steps=1,
                           wait=True, priority=None, event=None):
        t0 = time.monotonic()
        rec = {
            "kind": kind, "hemisphere": hemisphere, "steps": steps,
            "wait": wait, "priority": priority, "enqueue": round(t0, 6),
            "thread": threading.current_thread().name,
        }
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

    orig_train = worker._train

    def timed_train(state, samples, steps, kind):
        t0 = time.monotonic()
        try:
            return orig_train(state, samples, steps, kind)
        finally:
            log("compute", {
                "kind": kind, "state": state.name, "steps": steps,
                "dur": round(time.monotonic() - t0, 6),
                "thread": threading.current_thread().name,
            })

    worker._train = timed_train

    # ---- agent lifecycle instrumentation --------------------------------
    orig_schedule = agent._schedule_passive_consolidation

    def wrap_schedule():
        before = dict(agent.pending_consolidation)
        passive = agent.passive
        t0 = time.monotonic()
        exc = None
        try:
            return orig_schedule()
        except Exception as e:  # noqa: BLE001 - recorded, re-raised
            exc = repr(e)
            raise
        finally:
            after = dict(agent.pending_consolidation)
            log("schedule", {
                "thread": threading.current_thread().name,
                "cycle": cycle["n"], "passive": passive,
                "before_keys": list(before), "after_keys": list(after),
                "new_future": after.get(passive) is not None
                and before.get(passive) is not after.get(passive),
                "dur": round(time.monotonic() - t0, 6), "exc": exc,
            })

    agent._schedule_passive_consolidation = wrap_schedule

    orig_drain = agent._drain_passive_consolidation

    def wrap_drain():
        passive = agent.passive
        had_future = agent.pending_consolidation.get(passive) is not None
        steps_before = agent.state(passive).global_step
        t0 = time.monotonic()
        exc = None
        try:
            return orig_drain()
        except Exception as e:  # noqa: BLE001
            exc = repr(e)
            raise
        finally:
            steps_after = agent.state(passive).global_step
            log("drain", {
                "thread": threading.current_thread().name,
                "cycle": cycle["n"], "passive": passive,
                "had_future": had_future,
                "steps_before": steps_before, "steps_after": steps_after,
                "step_delta": steps_after - steps_before,
                "dur": round(time.monotonic() - t0, 6), "exc": exc,
            })

    agent._drain_passive_consolidation = wrap_drain

    orig_maybe = agent._maybe_switch

    def wrap_maybe():
        probe_before = agent.last_switch_probe
        count = agent.interaction_count
        early = count == probe_before
        switches_before = agent.switch_count
        t0 = time.monotonic()
        try:
            return orig_maybe()
        finally:
            dec = agent.switch_controller.last_decision
            log("maybe_switch", {
                "thread": threading.current_thread().name,
                "cycle": cycle["n"], "count": count,
                "probe_before": probe_before, "probe_after": agent.last_switch_probe,
                "early_return": early,
                "switched": agent.switch_count != switches_before,
                "reason": dec.reason if dec else None,
                "should_switch": dec.should_switch if dec else None,
                "dur": round(time.monotonic() - t0, 6),
            })

    agent._maybe_switch = wrap_maybe

    orig_save = agent.save

    def wrap_save():
        t0 = time.monotonic()
        snap = {
            n: {"step": agent.state(n).global_step,
                "debt": len(agent.state(n).debt),
                "replay": len(agent.state(n).replay)}
            for n in ("left", "right")
        }
        pending = {k: v.done() for k, v in agent.pending_consolidation.items()}
        try:
            return orig_save()
        finally:
            log("save", {
                "thread": threading.current_thread().name,
                "dur": round(time.monotonic() - t0, 6),
                "snapshot": snap, "pending_done": pending,
            })

    agent.save = wrap_save
    return agent


def state_snapshot(agent: LifeAgent) -> dict:
    return {
        "qsize": agent.worker.jobs.qsize(),
        "pending": {k: v.done() for k, v in agent.pending_consolidation.items()},
        "active": agent.active,
        "steps": {n: agent.state(n).global_step for n in ("left", "right")},
        "debt": {n: len(agent.state(n).debt) for n in ("left", "right")},
        "replay": {n: len(agent.state(n).replay) for n in ("left", "right")},
        "switches": agent.switch_count,
        "count": agent.interaction_count,
        "threads": {
            t.name: {"alive": t.is_alive(), "state": str(getattr(t, "state", "n/a"))}
            for t in threading.enumerate()
        },
    }


def main() -> int:
    agent = build_agent()
    stop_sampler = threading.Event()
    stop_observer = threading.Event()

    def sampler() -> None:
        while not stop_sampler.is_set():
            log("samples", state_snapshot(agent))
            time.sleep(0.1)

    def observer() -> None:
        while not stop_observer.is_set():
            t0 = time.monotonic()
            qsize = None
            try:
                st = agent.status()
                qsize = st.get("queued_jobs")
            except Exception as e:  # noqa: BLE001
                log("status", {"dur": round(time.monotonic() - t0, 6), "exc": repr(e)})
                time.sleep(0.15)
                continue
            log("status", {"dur": round(time.monotonic() - t0, 6), "qsize": qsize})
            time.sleep(0.15)

    agent.start()
    sampler_t = threading.Thread(target=sampler, name="audit-sampler", daemon=True)
    observer_t = threading.Thread(target=observer, name="audit-observer", daemon=True)
    sampler_t.start()
    observer_t.start()

    N = 64
    fallback_used = False
    try:
        for c in range(N):
            cycle["n"] = c
            q_before = agent.worker.jobs.qsize()
            steps_before = {n: agent.state(n).global_step for n in ("left", "right")}
            sw_before = agent.switch_count
            t0 = time.monotonic()
            if c % 2 == 0:
                frame = ByteFrame.raw(
                    bytes((c * 13 + i * 7) & 0xFF for i in range(48)),
                    {"cycle": c},
                )
                res = agent.feed_frame(frame)
                kind = "feed"
            else:
                frame = ByteFrame.text(f"探针 {c} 持续交互")
                res = agent.interact_frame(frame, frame.empty())
                kind = "interact"
            log("cycles", {
                "cycle": c, "kind": kind,
                "dur": round(time.monotonic() - t0, 6),
                "q_before": q_before, "q_after": agent.worker.jobs.qsize(),
                "steps_before": steps_before,
                "steps_after": {n: agent.state(n).global_step for n in ("left", "right")},
                "switches_before": sw_before, "switches_after": agent.switch_count,
                "active": agent.active,
                "pending": list(agent.pending_consolidation.keys()),
                "trained_steps": res.get("trained_steps") if isinstance(res, dict) else None,
            })
    except Exception as exc:  # noqa: BLE001
        records["close"].append({"stage": "stream_failed", "exc": repr(exc),
                                 "trace": traceback.format_exc()})
        if "OutOfMemoryError" in repr(exc) or "out of memory" in repr(exc).lower():
            fallback_used = True
    finally:
        stop_sampler.set()
        stop_observer.set()
        sampler_t.join(timeout=2)
        observer_t.join(timeout=2)

    close_t0 = time.monotonic()
    close_exc = None
    try:
        agent.close()
    except Exception as exc:  # noqa: BLE001
        close_exc = repr(exc)
    records["close"].append({
        "close_dur": round(time.monotonic() - close_t0, 6),
        "exc": close_exc,
        "final": state_snapshot(agent),
        "fallback_used": fallback_used,
    })

    out = AUDIT / "audit_metrics.json"
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "cycles": len(records["cycles"]),
        "switches": agent.switch_count,
        "steps": {n: agent.state(n).global_step for n in ("left", "right")},
        "jobs": len(records["jobs"]),
        "close_exc": close_exc,
        "out": str(out),
    }, ensure_ascii=False))
    return 0 if close_exc is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
