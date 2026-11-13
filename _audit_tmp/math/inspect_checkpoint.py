# Read-only inspection of the final checkpoint from continuous-adaptive-full-7.
# Does NOT write into the state directory; only prints and dumps a summary JSON
# next to this script.
from __future__ import annotations

import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
STATE = HERE.parent.parent / "validation_runs" / "continuous-adaptive-full-7" / "state" / "latest.pt"

payload = torch.load(STATE, map_location="cpu", weights_only=False)

summary = {
    "format_version": payload.get("format_version"),
    "active": payload.get("active"),
    "interaction_count": payload.get("interaction_count"),
    "switch_count": payload.get("switch_count"),
    "adaptive_switch": payload.get("adaptive_switch"),
    "hemispheres": {},
}

for name, hemi in payload["hemispheres"].items():
    debt = hemi["debt"]
    replay = hemi["replay"]
    metrics = hemi["metrics"]
    opt = hemi.get("optimizer")
    opt_state = None
    if isinstance(opt, dict) and "state" in opt:
        opt_state = {k: (len(v) if isinstance(v, dict) else type(v).__name__) for k, v in opt["state"].items()}
    weights = [it["weight"] for it in debt["items"]]
    summary["hemispheres"][name] = {
        "global_step": metrics["global_step"],
        "last_event": metrics["last_event"],
        "last_loss": metrics["last_loss"],
        "hidden_dims": hemi["hidden_dims"],
        "replay_samples": len(replay["samples"]),
        "debt_items": len(debt["items"]),
        "debt_total_created": debt["total_created"],
        "debt_total_repaid": debt["total_repaid"],
        "debt_repayment_batches": debt["repayment_batches"],
        "debt_repayment_capacity": debt["repayment_capacity"],
        "debt_total_clean_added": debt["total_clean_added"],
        "debt_weight_min": min(weights) if weights else None,
        "debt_weight_max": max(weights) if weights else None,
        "debt_weight_mean": (sum(weights) / len(weights)) if weights else None,
        "debt_backlog_recomputed": None,
        "optimizer_state_param_count": len(opt["state"]) if isinstance(opt, dict) else None,
        "optimizer_param_groups_lr": [g.get("lr") for g in opt["param_groups"]] if isinstance(opt, dict) else None,
    }

# Recompute backlog from stored items (clean_backlog equivalent) without importing life_agent.
import math

def structural_quality(ids):
    raw = [v for v in ids if 0 <= v < 256]
    n = len(raw)
    if n < 16:
        return 1.0
    counts = {}
    for v in raw:
        counts[v] = counts.get(v, 0) + 1
    probs = [c / n for c in counts.values()]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs) / math.log(256)
    longest_run = run = 1
    for i in range(1, n):
        run = run + 1 if raw[i] == raw[i - 1] else 1
        longest_run = max(longest_run, run)
    unique_ratio = len(counts) / min(n, 256)
    ef = min(1.0, entropy / 0.12)
    rf = min(1.0, 12.0 / longest_run)
    uf = min(1.0, unique_ratio / 0.08)
    return max(0.001, min(1.0, ef * rf * uf))

for name, hemi in payload["hemispheres"].items():
    backlog = sum(max(0.0, it["weight"]) * structural_quality(it["ids"]) for it in hemi["debt"]["items"])
    summary["hemispheres"][name]["debt_backlog_recomputed"] = backlog
    # weight cap violations
    over = [it["weight"] for it in hemi["debt"]["items"] if it["weight"] > 4.0 + 1e-9]
    summary["hemispheres"][name]["debt_weights_over_cap4"] = len(over)

out = HERE / "checkpoint_inspection.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
