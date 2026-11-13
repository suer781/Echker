# Read-only audit experiment: persistence/restore consistency.
# Writes only under _audit_tmp/persistence/. Does not modify life_agent/tests/validation_runs.
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

ROOT = Path(r"O:\Documents\智能自学习AI（新）")
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from life_agent.agent import LifeAgent  # noqa: E402
from life_agent.byte_codec import ByteFrame  # noqa: E402
from life_agent.config import AppConfig, ModelConfig, RuntimeConfig  # noqa: E402

OUT = ROOT / "_audit_tmp" / "persistence"


def make_config() -> AppConfig:
    return AppConfig(
        ModelConfig(
            d_model=64,
            n_heads=4,
            n_layers=1,
            hidden_dim=64,
            min_hidden_dim=32,
            max_hidden_dim=128,
            max_seq_len=96,
        ),
        RuntimeConfig(
            training_device="cpu",
            require_cuda=False,
            seed=7,
            max_new_tokens=4,
            ttt_chunk_tokens=2,
            ttt_steps=1,
            consolidation_steps=1,
            batch_size=2,
            replay_capacity=32,
            debt_capacity=32,
            checkpoint_interval_s=99999,
            structure_interval=100000,
            structure_warmup=100000,
            idle_sleep_s=0.01,
            context_turns=2,
        ),
    )


def debt_items(state) -> dict:
    return {
        k: [round(v.weight, 6), round(v.loss, 6), v.source]
        for k, v in state.debt.items.items()
    }


def replay_items(state) -> dict:
    return {
        k: [round(v.weight, 6), round(v.loss, 6), v.source]
        for k, v in state.replay.samples.items()
    }


def opt_summary(state) -> dict:
    sd = state.optimizer.state_dict()
    state_entries = sd.get("state", {})
    exp_avg = 0
    for _, param_state in state_entries.items():
        if "exp_avg" in param_state:
            exp_avg += param_state["exp_avg"].numel()
    return {
        "param_groups": len(sd.get("param_groups", [])),
        "state_keys": len(state_entries),
        "exp_avg_numel": exp_avg,
    }


def snapshot(agent: LifeAgent) -> dict:
    hem = {}
    for name, state in agent.worker.states.items():
        hem[name] = {
            "metrics": {
                "global_step": state.global_step,
                "total_trained_bytes": state.total_trained_bytes,
                "last_loss": state.last_loss,
                "last_event": state.last_event,
            },
            "hidden_dims": state.model.hidden_dims(),
            "debt_items": debt_items(state),
            "debt_counters": {
                "total_created": state.debt.total_created,
                "total_repaid": state.debt.total_repaid,
                "repayment_batches": state.debt.repayment_batches,
                "repayment_capacity": round(state.debt.repayment_capacity, 6),
                "total_clean_added": round(state.debt.total_clean_added, 6),
            },
            "replay_items": replay_items(state),
            "replay_cleaner": state.replay.cleaner.to_dict(),
            "lr": state.lr.to_dict(),
            "structure": {
                "losses_len": len(state.structure.losses),
                "last_change_step": state.structure.last_change_step,
            },
            "optimizer": opt_summary(state),
        }
    return {
        "active": agent.active,
        "switch_count": agent.switch_count,
        "interaction_count": agent.interaction_count,
        "observed_frames": agent.observed_frames,
        "history_len": len(agent.history),
        "switch_controller": agent.switch_controller.to_dict(),
        "pending": sorted(agent.pending_consolidation.keys()),
        "hemispheres": hem,
    }


def diff_snapshots(a: dict, b: dict, prefix: str = "") -> list:
    issues = []

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for key in set(x) | set(y):
                if key not in x:
                    issues.append(f"{path}.{key}: missing in A (B={y[key]!r})")
                elif key not in y:
                    issues.append(f"{path}.{key}: missing in B (A={x[key]!r})")
                else:
                    walk(x[key], y[key], f"{path}.{key}")
        else:
            if x != y:
                issues.append(f"{path}: A={x!r} B={y!r}")

    walk(a, b, prefix)
    return issues


def disk_payload(state_dir: Path) -> dict:
    return torch.load(state_dir / "latest.pt", map_location="cpu", weights_only=False)


def _digest(item: dict) -> str:
    import hashlib

    payload = ",".join(map(str, item["ids"])).encode("ascii")
    payload += b"|" + ",".join(map(str, item["labels"])).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def payload_snapshot(payload: dict) -> dict:
    hem = {}
    for name, saved in payload["hemispheres"].items():
        hem[name] = {
            "metrics": saved["metrics"],
            "hidden_dims": saved["hidden_dims"],
            "debt_items": {
                _digest(i): [round(i["weight"], 6), round(i["loss"], 6), i["source"]]
                for i in saved["debt"]["items"]
            },
            "debt_counters": {
                "total_created": saved["debt"].get("total_created"),
                "total_repaid": saved["debt"].get("total_repaid"),
                "repayment_batches": saved["debt"].get("repayment_batches"),
                "repayment_capacity": round(saved["debt"].get("repayment_capacity", 0.0), 6),
                "total_clean_added": round(saved["debt"].get("total_clean_added", 0.0), 6),
            },
            "replay_items": {
                _digest(i): [round(i["weight"], 6), round(i["loss"], 6), i["source"]]
                for i in saved["replay"]["samples"]
            },
            "replay_cleaner": saved["replay"].get("cleaner"),
            "lr": saved["lr"],
            "structure": {
                "losses_len": len(saved["structure"].get("losses", [])),
                "last_change_step": saved["structure"].get("last_change_step"),
            },
            "optimizer": {"state_keys": len(saved["optimizer"].get("state", {}))},
        }
    return {
        "active": payload.get("active"),
        "switch_count": payload.get("switch_count"),
        "interaction_count": payload.get("interaction_count"),
        "observed_frames": payload.get("observed_frames"),
        "history_len": len(payload.get("history", [])),
        "switch_controller": payload.get("adaptive_switch"),
        "pending": "<not persisted>",
        "hemispheres": hem,
    }


results = {"roundtrip": {}, "race": {}, "migration": {}}

# ---------- Round 1: live -> save -> load, full-field comparison ----------
state1 = OUT / "run1" / "state"
state1.mkdir(parents=True, exist_ok=True)
cfg = make_config()
agent = LifeAgent(cfg, state1)
agent.start()
try:
    for text in ["第一轮观察字节流", "第二轮持续学习", "第三轮记忆回放"]:
        agent.interact(text)
    agent.feed_frame(ByteFrame.text("隔离实验数据样本 " * 8))
    live_before = snapshot(agent)
    agent.save()
    payload = disk_payload(state1)
    on_disk = payload_snapshot(payload)
finally:
    agent.close()

restored = LifeAgent(make_config(), state1)
restored.load()
restored_snap = snapshot(restored)
restored.start()
try:
    restored_after_start = snapshot(restored)
finally:
    restored.close()

live_vs_restore = diff_snapshots(live_before, restored_snap, "live→restored")
disk_vs_restore = diff_snapshots(on_disk, restored_snap, "disk→restored")
results["roundtrip"] = {
    "live_snapshot": live_before,
    "restored_snapshot": restored_snap,
    "live_vs_restored_issues": live_vs_restore,
    "disk_vs_restored_issues": disk_vs_restore,
    "note": "live_vs_restored is the real consistency check; disk_vs_restored replicates the validator's circular disk-to-disk comparison.",
}

# ---------- Round 2: save() races an in-flight consolidation ----------
state2 = OUT / "run2" / "state"
state2.mkdir(parents=True, exist_ok=True)
cfg2 = make_config()
racer = LifeAgent(cfg2, state2)
racer.start()
try:
    for text in ["竞态样本甲", "竞态样本乙"]:
        racer.interact(text)
    racer.feed_frame(ByteFrame.text("竞态观察数据 " * 6))
    passive = racer.passive
    samples = racer._debt_training_samples(passive)
    pre = snapshot(racer)
    future = None
    if samples:
        future = racer.worker.submit(
            "consolidate", passive, samples, cfg2.runtime.consolidation_steps, wait=False
        )
    # monitor-style save: no _wait_consolidations
    racer.save()
    race_payload = payload_snapshot(disk_payload(state2))
    if future is not None:
        future.result()
    post = snapshot(racer)
finally:
    racer.close()

race_issues = diff_snapshots(race_payload, post, "checkpoint→post-consolidation-live")
results["race"] = {
    "passive": passive,
    "consolidation_samples": len(samples),
    "checkpoint_snapshot": race_payload,
    "post_consolidation_live": post,
    "checkpoint_vs_post_issues": race_issues,
    "pending_persisted": False,
}

# ---------- Round 3: stripped v2 checkpoint migration ----------
state3 = OUT / "run3" / "state"
state3.mkdir(parents=True, exist_ok=True)
import shutil

shutil.copy2(state1 / "latest.pt", state3 / "latest.pt")
p3 = torch.load(state3 / "latest.pt", map_location="cpu", weights_only=False)
stripped_controller = p3.pop("adaptive_switch", None)
for name in p3["hemispheres"]:
    debt = p3["hemispheres"][name]["debt"]
    for k in ("total_repaid", "repayment_batches", "repayment_capacity", "total_clean_added"):
        debt.pop(k, None)
    p3["hemispheres"][name]["replay"].pop("cleaner", None)
torch.save(p3, state3 / "latest.pt")

migrant = LifeAgent(make_config(), state3)
migrant.load()
migrant_snap = snapshot(migrant)
migrant.start()
try:
    post_start = snapshot(migrant)
finally:
    migrant.close()

results["migration"] = {
    "original_controller_baseline": stripped_controller,
    "restored_controller_baseline": migrant_snap["switch_controller"],
    "restored_debt_counters": {
        n: migrant_snap["hemispheres"][n]["debt_counters"] for n in migrant_snap["hemispheres"]
    },
    "restored_replay_cleaner": {
        n: migrant_snap["hemispheres"][n]["replay_cleaner"] for n in migrant_snap["hemispheres"]
    },
    "post_start_snapshot": post_start,
}

(OUT / "audit_results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
)
print(json.dumps({
    "roundtrip_issues": len(live_vs_restore),
    "disk_vs_restore_issues": len(disk_vs_restore),
    "race_issues": len(race_issues),
}, ensure_ascii=False))
