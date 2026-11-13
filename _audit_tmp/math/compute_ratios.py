# Derive checkpoint/run ratios for the audit findings (read-only).
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
cp = json.loads((HERE / "checkpoint_inspection.json").read_text(encoding="utf-8"))
ma = json.loads((HERE / "metrics_analysis.json").read_text(encoding="utf-8"))

LIMIT = 8  # batch_size 4 * consolidation_steps 2
out: dict[str, object] = {}
for name, h in cp["hemispheres"].items():
    backlog = h["debt_backlog_recomputed"]
    items = h["debt_items"]
    oracle_batch = backlog / items * min(LIMIT, items)
    out[name] = {
        "global_step": h["global_step"],
        "debt_items": items,
        "backlog": round(backlog, 4),
        "total_clean_added": round(h["debt_total_clean_added"], 4),
        "backlog_share_of_cumulative": round(backlog / h["debt_total_clean_added"], 4),
        "repayment_batches": h["debt_repayment_batches"],
        "repayment_capacity": round(h["debt_repayment_capacity"], 4),
        "oracle_batch_mass": round(oracle_batch, 4),
        "capacity_over_oracle_ratio": round(h["debt_repayment_capacity"] / oracle_batch, 3),
        "total_created": h["debt_total_created"],
        "total_repaid": h["debt_total_repaid"],
        "avg_repaid_per_batch": round(h["debt_total_repaid"] / max(1, h["debt_repayment_batches"]), 3),
    }

total_created = cp["hemispheres"]["left"]["debt_total_created"] + cp["hemispheres"]["right"]["debt_total_created"]
switches = cp["switch_count"]
out["run_level"] = {
    "switches": switches,
    "total_debt_created": total_created,
    "exchange_adds_upper_bound": 4 * switches,
    "exchange_add_share_upper_bound": round(4 * switches / total_created, 5),
    "mixed_phase_observable_flips": ma["within_cycle_switches"] + ma["between_cycle_flip_count"],
    "post_mixed_switches_unobserved": switches - (ma["within_cycle_switches"] + ma["between_cycle_flip_count"]),
    "elapsed_seconds": ma["report_elapsed_seconds"],
    "seconds_per_switch_avg": round(ma["report_elapsed_seconds"] / switches, 3),
    "mixed_interval_cycles": ma["interval_cycles"],
    "immediate_reswitch_count": ma["immediate_reswitch_count"],
    "passive_cpu_violations": len(ma["passive_cpu_call_delta_nonzero"]),
}

(HERE / "derived_ratios.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
