# CPU-only audit simulation for the adaptive-switch math.
# Replicates DebtLedger / ReplayBuffer / AdaptiveSwitchController semantics at
# the ledger+controller level. No GPU, no full training. Writes sim_results.json
# next to this script; does not touch life_agent/, tests/ or validation_runs/.
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from life_agent.memory import DebtLedger, ReplayBuffer, Sample, StatisticalCleaner
from life_agent.switching import AdaptiveSwitchController

HERE = Path(__file__).resolve().parent

# Match config.json runtime defaults used by continuous-adaptive-full-7.
RETENTION = 0.35
DECAY = 0.75
MIN_WEIGHT = 0.005
DEBT_CAPACITY = 512
BATCH_SIZE = 4
CONSOLIDATION_STEPS = 2
LIMIT = BATCH_SIZE * CONSOLIDATION_STEPS  # 8


def make_sample(rng: random.Random, weight: float, tag: str, length: int = 32) -> Sample:
    ids = [rng.randrange(256) for _ in range(length)]
    labels = ids.copy()
    return Sample(ids=ids, labels=labels, weight=weight, source=tag)


class TaggedDebtLedger(DebtLedger):
    """Adds per-tag clean-mass accounting without changing DebtLedger math."""

    def __init__(self, capacity, decay, min_weight, mass_by_tag: dict[str, float]):
        super().__init__(capacity, decay, min_weight)
        self.mass_by_tag = mass_by_tag

    def add(self, sample: Sample, retention: float, tag: str = "regular") -> None:
        before = self.total_clean_added
        super().add(sample, retention)
        self.mass_by_tag[tag] += self.total_clean_added - before


results: dict[str, object] = {}

# ---------------------------------------------------------------------------
# Exp A1: healthy population -- running repayment_capacity vs per-batch oracle.
# ---------------------------------------------------------------------------
def exp_a1_healthy() -> dict:
    rng = random.Random(7)
    ledger = DebtLedger(DEBT_CAPACITY, DECAY, MIN_WEIGHT)
    for _ in range(DEBT_CAPACITY):
        ledger.add(make_sample(rng, rng.uniform(0.2, 0.9), "seed"), RETENTION)
    per_batch_capacity: list[float] = []
    running_capacity: list[float] = []
    oracle_per_batch: list[float] = []
    rounds = 0
    while len(ledger) > 0 and rounds < 30:
        due = ledger.due(LIMIT)
        if due:
            batch_cap = sum(max(0.0, s.weight) * StatisticalCleaner.structural_quality(s.ids) for s in due)
            per_batch_capacity.append(batch_cap)
            # Oracle: current population mean clean mass * batch size.
            backlog = ledger.clean_backlog()
            oracle = backlog / max(1, len(ledger)) * min(LIMIT, len(ledger))
            oracle_per_batch.append(oracle)
        ledger.decay_after_training()
        running_capacity.append(ledger.repayment_capacity)
        rounds += 1
    return {
        "nonempty_batches": len(per_batch_capacity),
        "batch_capacity_first5": [round(x, 4) for x in per_batch_capacity[:5]],
        "batch_capacity_last5": [round(x, 4) for x in per_batch_capacity[-5:]],
        "oracle_per_batch_last5": [round(x, 4) for x in oracle_per_batch[-5:]],
        "final_running_capacity": round(ledger.repayment_capacity, 4),
        "final_population": len(ledger),
        "mean_batch_capacity": round(sum(per_batch_capacity) / len(per_batch_capacity), 4),
        "mean_oracle": round(sum(oracle_per_batch) / len(oracle_per_batch), 4),
        "running_vs_last_batch_ratio": round(
            ledger.repayment_capacity / max(1e-9, per_batch_capacity[-1]), 3
        ),
    }


# ---------------------------------------------------------------------------
# Exp A2: starvation -- ledger drains to a handful of items; does the running
# average roll back down with the smaller batches?
# ---------------------------------------------------------------------------
def exp_a2_starvation() -> dict:
    rng = random.Random(11)
    ledger = DebtLedger(DEBT_CAPACITY, DECAY, MIN_WEIGHT)
    for _ in range(DEBT_CAPACITY):
        ledger.add(make_sample(rng, rng.uniform(0.2, 0.9), "seed"), RETENTION)
    # Fill the running average with healthy batches first.
    healthy_batches = 0
    for _ in range(20):
        due = ledger.due(LIMIT)
        if due:
            healthy_batches += 1
        ledger.decay_after_training()
    capacity_healthy = ledger.repayment_capacity
    # Sparse-replenishment phase: incoming stream slows, population hovers at
    # a handful of items, so each batch is small. Capture those late batches.
    late_batches: list[float] = []
    population_trace: list[int] = []
    for _ in range(40):
        if len(ledger) < 6:
            ledger.add(make_sample(rng, 0.5, "replenish"), RETENTION)
        due = ledger.due(LIMIT)
        if due:
            late_batches.append(
                sum(max(0.0, s.weight) * StatisticalCleaner.structural_quality(s.ids) for s in due)
            )
        ledger.decay_after_training()
        population_trace.append(len(ledger))
    late = late_batches[-10:] if len(late_batches) >= 10 else late_batches
    mean_late = sum(late) / max(1, len(late))
    return {
        "healthy_batches": healthy_batches,
        "population_late_phase_min": min(population_trace),
        "population_late_phase_max": max(population_trace),
        "capacity_healthy": round(capacity_healthy, 4),
        "running_capacity_after_sparse_phase": round(ledger.repayment_capacity, 4),
        "mean_late_small_batch_capacity": round(mean_late, 4),
        "capacity_lag_ratio": round(ledger.repayment_capacity / max(1e-9, mean_late), 3),
        "note": "repayment_capacity is a cumulative arithmetic mean (memory.py:216-218); it never decays per-se when later batches shrink, so the threshold lags above the current per-batch mass (old batches retain weight 1/n).",
    }


# ---------------------------------------------------------------------------
# Exp A3: repeated digest stacking toward the 4.0 cap; top-weight selection.
# ---------------------------------------------------------------------------
def exp_a3_duplicate_stacking() -> dict:
    rng = random.Random(13)
    ledger = DebtLedger(DEBT_CAPACITY, DECAY, MIN_WEIGHT)
    proto = make_sample(rng, 0.5, "dup")
    for _ in range(6):
        ledger.add(proto.clone(), RETENTION)
    weights = [s.weight for s in ledger.items.values()]
    due = ledger.due(LIMIT)
    top = due[0]
    return {
        "items": len(ledger),
        "weights": [round(w, 4) for w in weights],
        "top_weight": round(top.weight, 4),
        "top_is_duplicate": top.digest == proto.digest,
        "cap4_violation": any(w > 4.0 + 1e-9 for w in weights),
    }


# ---------------------------------------------------------------------------
# Exp A4: due() called twice before decay (as schedule + _switch_locked do).
# Items are NOT removed by due(); the second view counts another batch.
# ---------------------------------------------------------------------------
def exp_a4_double_due() -> dict:
    rng = random.Random(17)
    ledger = DebtLedger(DEBT_CAPACITY, DECAY, MIN_WEIGHT)
    for _ in range(40):
        ledger.add(make_sample(rng, rng.uniform(0.2, 0.9), "seed"), RETENTION)
    first = [s.digest for s in ledger.due(LIMIT)]
    batches_before = ledger.repayment_batches
    second = [s.digest for s in ledger.due(LIMIT)]
    batches_after = ledger.repayment_batches
    overlap = len(set(first) & set(second))
    return {
        "batches_counted_before_second_due": batches_before,
        "batches_counted_after_second_due": batches_after,
        "overlap_digests": overlap,
        "batch_size": len(first),
        "note": "due() clones but never removes items; second due() re-counts the same top-weight view as a new repayment batch.",
    }


# ---------------------------------------------------------------------------
# Exp B: exchange pressure injection over a 61-switch synthetic trajectory.
# Mirrors LifeAgent: active replay holds dialogue/observe; passive debt receives
# regular remembers; each switch exchanges replay.latest(2) both ways.
# ---------------------------------------------------------------------------
def exp_b_exchange_pressure() -> dict:
    rng = random.Random(23)
    per_ledger_mass = {
        "left": {"regular": 0.0, "exchange": 0.0},
        "right": {"regular": 0.0, "exchange": 0.0},
    }
    ledgers = {
        "left": TaggedDebtLedger(DEBT_CAPACITY, DECAY, MIN_WEIGHT, per_ledger_mass["left"]),
        "right": TaggedDebtLedger(DEBT_CAPACITY, DECAY, MIN_WEIGHT, per_ledger_mass["right"]),
    }
    replays = {"left": ReplayBuffer(1024), "right": ReplayBuffer(1024)}
    controller = AdaptiveSwitchController()
    active = "left"
    switches = 0
    cycles_per_switch: list[int] = []
    cycles_since = 0
    baseline_exchange_share: list[float] = []
    # Switch at cycles 4,8,...,244 -> 61 switches over 245 cycles.
    switch_cycles = {4 * k for k in range(1, 62)}

    def passive() -> str:
        return "right" if active == "left" else "left"

    for cycle in range(245):
        dialogue = make_sample(rng, 0.8, f"dialogue:{cycle}")
        observe = make_sample(rng, 0.25, f"observe:{cycle}")
        replays[active].add(dialogue)
        replays[active].add(observe)
        ledgers[passive()].add(dialogue, RETENTION, "regular")
        ledgers[passive()].add(observe, RETENTION, "regular")
        ledger = ledgers[passive()]
        ledger.due(LIMIT)
        ledger.decay_after_training()

        if cycle in switch_cycles:
            old, new = active, passive()
            for src, tgt in ((old, new), (new, old)):
                for sample in replays[src].latest(2):
                    ledgers[tgt].add(sample, RETENTION, "exchange")
            ledgers[new].due(LIMIT)
            ledgers[new].decay_after_training()
            active = new
            switches += 1
            cycles_per_switch.append(cycles_since + 1)
            cycles_since = 0
            passive_ledger = ledgers[passive()]
            mass = per_ledger_mass[passive()]
            controller.note_switch(passive_ledger.clean_pressure(), rng.randrange(100), rng.randrange(100))
            baseline_exchange_share.append(mass["exchange"] / max(1e-9, mass["regular"] + mass["exchange"]))
        else:
            cycles_since += 1

    regular = sum(m["regular"] for m in per_ledger_mass.values())
    exchange = sum(m["exchange"] for m in per_ledger_mass.values())
    return {
        "switches": switches,
        "regular_clean_mass": round(regular, 4),
        "exchange_clean_mass": round(exchange, 4),
        "exchange_share_of_total_pressure": round(exchange / max(1e-9, regular + exchange), 4),
        "exchange_mass_per_switch": round(exchange / max(1, switches), 4),
        "cycles_per_switch_histogram": {str(k): cycles_per_switch.count(k) for k in sorted(set(cycles_per_switch))},
        "final_total_clean_added": {
            "left": round(ledgers["left"].total_clean_added, 4),
            "right": round(ledgers["right"].total_clean_added, 4),
        },
        "per_ledger_exchange_share": {
            name: round(m["exchange"] / max(1e-9, m["regular"] + m["exchange"]), 4)
            for name, m in per_ledger_mass.items()
        },
        "baseline_exchange_share_final": round(baseline_exchange_share[-1], 4),
        "caveat": "Synthetic replay samples keep raw dialogue/observe weights; real replay weights are shrunk by robust loss weighting (worker.py:309), so real exchange mass share is expected lower. Real-run add-count upper bound: <=4*61=244 exchange adds of 40124 total_created (<=0.61%).",
    }


# ---------------------------------------------------------------------------
# Exp C1: consolidate steps count toward passive work delta and can block switch.
# ---------------------------------------------------------------------------
def exp_c1_consolidate_steps_block() -> dict:
    controller = AdaptiveSwitchController()
    controller.note_switch(0.0, 0.0, 0.0)
    active_step = passive_step = 0
    trace = []
    blocked_from = None
    for cycle in range(12):
        active_step += 3   # ttt/train updates on active
        passive_step += 5  # several consolidation jobs draining debt on passive
        decision = controller.decide(
            clean_pressure=100.0,
            learned_capacity=1.0,
            active_work=active_step,
            passive_work=passive_step,
            has_debt=True,
        )
        entry = {
            "cycle": cycle,
            "active_step": active_step,
            "passive_step": passive_step,
            "active_delta": round(decision.active_work, 2),
            "passive_delta": round(decision.passive_work, 2),
            "should_switch": decision.should_switch,
            "reason": decision.reason,
        }
        trace.append(entry)
        if not decision.should_switch and blocked_from is None:
            blocked_from = cycle
    return {
        "trace": trace,
        "first_cycle_blocked_by_work": blocked_from,
        "note": "global_step increments per optimizer step for ttt/train/consolidate alike (worker.py:295, kind-agnostic); passive consolidate steps inflate passive_delta and can satisfy passive_delta >= active_delta, blocking the switch on the work criterion.",
    }


# ---------------------------------------------------------------------------
# Exp C2: note_switch records left/right steps positionally, not active/passive.
# Demonstrate delta misalignment after a switch to the right hemisphere.
# ---------------------------------------------------------------------------
def exp_c2_baseline_misalignment() -> dict:
    # At switch moment: active=right, passive=left. True steps: R0=200, L0=180.
    controller = AdaptiveSwitchController()
    # Actual agent call (agent.py:316-317): active_work=left.step, passive_work=right.step.
    controller.note_switch(0.0, 180, 200)
    # Later: active=right now at R1=230, passive=left at L1=190.
    decision = controller.decide(
        clean_pressure=10.0, learned_capacity=1.0,
        active_work=230, passive_work=190, has_debt=True,
    )
    # Corrected mapping would baseline active=R0=200, passive=L0=180.
    active_delta_wrong = 230 - 180
    passive_delta_wrong = 190 - 200
    active_delta_correct = 230 - 200
    passive_delta_correct = 190 - 180
    return {
        "switch_moment_steps": {"left": 180, "right": 200, "active": "right", "passive": "left"},
        "recorded_baseline": {"baseline_active_work": 180, "baseline_passive_work": 200},
        "later_steps": {"active_right": 230, "passive_left": 190},
        "delta_as_recorded": {"active_delta": active_delta_wrong, "passive_delta": passive_delta_wrong},
        "delta_correct_mapping": {"active_delta": active_delta_correct, "passive_delta": passive_delta_correct},
        "decision_reason": decision.reason,
        "should_switch": decision.should_switch,
        "note": "agent.py:316-317 passes left.global_step as active_work and right.global_step as passive_work regardless of actual active/passive role.",
    }


# ---------------------------------------------------------------------------
# Exp D: boundaries.
# ---------------------------------------------------------------------------
def exp_d_boundaries() -> dict:
    out: dict[str, object] = {}
    # Empty debt.
    c = AdaptiveSwitchController()
    d = c.decide(clean_pressure=0.0, learned_capacity=0.0, active_work=1, passive_work=1, has_debt=False)
    out["empty_debt"] = {"should_switch": d.should_switch, "reason": d.reason}

    # Ledger holds debt but no repayment batch has ever been measured:
    # repayment_capacity == 0 -> required == 0 -> sufficient_data False.
    ledger = DebtLedger(DEBT_CAPACITY, DECAY, MIN_WEIGHT)
    rng0 = random.Random(3)
    ledger.add(make_sample(rng0, 0.6, "seed"), RETENTION)
    c2 = AdaptiveSwitchController()
    d2 = c2.decide(clean_pressure=ledger.clean_pressure(), learned_capacity=ledger.repayment_capacity,
                   active_work=1, passive_work=1, has_debt=len(ledger) > 0)
    out["zero_capacity"] = {"should_switch": d2.should_switch, "reason": d2.reason,
                            "repayment_capacity": ledger.repayment_capacity, "has_debt": len(ledger) > 0}

    # Reload fallback: debt restored with total_clean_added=X, controller has no
    # saved baseline; store._restore reconstructs baseline from current pressure.
    ledger2 = DebtLedger(DEBT_CAPACITY, DECAY, MIN_WEIGHT)
    rng = random.Random(5)
    for _ in range(20):
        ledger2.add(make_sample(rng, 0.6, "restored"), RETENTION)
    restored_pressure = ledger2.clean_pressure()
    c3 = AdaptiveSwitchController()
    c3.note_switch(restored_pressure, 5558, 5618)  # store.py:118-122 fallback
    d3 = c3.decide(clean_pressure=restored_pressure, learned_capacity=1.0,
                   active_work=5558, passive_work=5618, has_debt=True)
    out["reload_fallback"] = {
        "restored_pressure": round(restored_pressure, 4),
        "pressure_growth": round(d3.pressure_growth, 4),
        "should_switch": d3.should_switch,
        "note": "debt.load_dict restores total_clean_added (memory.py:256) BEFORE controller baseline rebuild (store.py:111-122); growth starts at zero, no immediate switch.",
    }

    # Baseline above current pressure after eviction/decay -> growth clamped 0.
    c4 = AdaptiveSwitchController()
    c4.note_switch(100.0, 0, 0)
    d4 = c4.decide(clean_pressure=50.0, learned_capacity=1.0, active_work=10, passive_work=10, has_debt=True)
    out["baseline_above_current"] = {"pressure_growth": d4.pressure_growth, "should_switch": d4.should_switch}
    return out


results["A1_healthy_capacity"] = exp_a1_healthy()
results["A2_starvation_lag"] = exp_a2_starvation()
results["A3_duplicate_stacking"] = exp_a3_duplicate_stacking()
results["A4_double_due"] = exp_a4_double_due()
results["B_exchange_pressure"] = exp_b_exchange_pressure()
results["C1_consolidate_steps_block"] = exp_c1_consolidate_steps_block()
results["C2_baseline_misalignment"] = exp_c2_baseline_misalignment()
results["D_boundaries"] = exp_d_boundaries()

(HERE / "sim_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(results, ensure_ascii=False, indent=2))
