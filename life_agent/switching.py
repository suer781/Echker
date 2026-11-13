"""Adaptive hemisphere switching from learnable data pressure, not clocks."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any


@dataclass
class SwitchDecision:
    should_switch: bool
    reason: str
    clean_pressure: float = 0.0
    baseline_pressure: float = 0.0
    pressure_growth: float = 0.0
    learned_capacity: float = 0.0
    work_share: float = 0.0
    active_work: float = 0.0
    passive_work: float = 0.0
    active_name: str = "left"
    passive_name: str = "right"
    active_work_kind: str = "train+ttt"
    passive_work_kind: str = "consolidate"
    decided_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_switch": self.should_switch,
            "reason": self.reason,
            "clean_pressure": self.clean_pressure,
            "baseline_pressure": self.baseline_pressure,
            "pressure_growth": self.pressure_growth,
            "learned_capacity": self.learned_capacity,
            "work_share": self.work_share,
            "active_work": self.active_work,
            "passive_work": self.passive_work,
            "active_name": self.active_name,
            "passive_name": self.passive_name,
            "active_work_kind": self.active_work_kind,
            "passive_work_kind": self.passive_work_kind,
            "decided_at": self.decided_at,
        }


@dataclass
class AdaptiveSwitchController:
    """Switch when new clean data exceeds learned consolidation capacity.

    The decision threshold is not an interaction count or a wall-clock interval.
    It is the passive hemisphere's own measured clean mass for one consolidation
    batch. Pressure is the cumulative non-noise data admitted to debt since the
    last switch, not the buffer's capped residue; a full debt buffer therefore
    cannot make incoming data look static.

    Work is measured in gradient-update steps, but the two sides are different
    operations: the online side's learning steps (train + TTT) are compared
    against the passive side's consolidation steps. ``global_step`` remains a
    total for logging and is never used as the work criterion.

    The baselines are role-indexed. When ``active`` becomes ``right`` after a
    switch, ``baseline_active_work`` follows the new active side rather than
    remaining attached to the left hemisphere name.
    """

    enabled: bool = True
    baseline_clean_pressure: float = 0.0
    baseline_active_work: float = 0.0
    baseline_passive_work: float = 0.0
    baseline_active_name: str | None = "left"
    baseline_passive_name: str | None = "right"
    last_decision: SwitchDecision | None = field(default=None, repr=False)

    def note_switch(
        self,
        clean_pressure: float,
        active_work: float,
        passive_work: float,
        active_name: str | None = "left",
        passive_name: str | None = "right",
    ) -> None:
        self.baseline_clean_pressure = float(max(0.0, clean_pressure))
        self.baseline_active_work = float(max(0.0, active_work))
        self.baseline_passive_work = float(max(0.0, passive_work))
        self.baseline_active_name = active_name or "left"
        self.baseline_passive_name = passive_name or "right"

    def _roles_match(self, active_name: str, passive_name: str) -> bool:
        if self.baseline_active_name is None or self.baseline_passive_name is None:
            # A legacy checkpoint has numeric baselines but no role labels.
            # Refuse to guess which side they belonged to.
            return False
        return self.baseline_active_name == active_name and self.baseline_passive_name == passive_name

    def rebase(
        self,
        clean_pressure: float,
        active_work: float,
        passive_work: float,
        active_name: str,
        passive_name: str,
    ) -> None:
        """Re-establish a role-correct baseline without causing a switch."""
        self.note_switch(clean_pressure, active_work, passive_work, active_name, passive_name)

    def decide(
        self,
        *,
        clean_pressure: float,
        learned_capacity: float,
        active_work: float,
        passive_work: float,
        has_debt: bool,
        active_name: str = "left",
        passive_name: str = "right",
    ) -> SwitchDecision:
        if not self.enabled or not has_debt:
            decision = SwitchDecision(
                False,
                "disabled or no unlearned data",
                clean_pressure=clean_pressure,
                learned_capacity=learned_capacity,
                active_name=active_name,
                passive_name=passive_name,
            )
            self.last_decision = decision
            return decision
        if not self._roles_match(active_name, passive_name):
            decision = SwitchDecision(
                False,
                "switch baseline role mismatch; rebase required",
                clean_pressure=clean_pressure,
                learned_capacity=learned_capacity,
                active_name=active_name,
                passive_name=passive_name,
            )
            self.last_decision = decision
            return decision

        active_delta = max(0.0, active_work - self.baseline_active_work)
        passive_delta = max(0.0, passive_work - self.baseline_passive_work)
        pressure_growth = max(0.0, clean_pressure - self.baseline_clean_pressure)
        required = max(0.0, learned_capacity)
        sufficient_data = required > 0 and pressure_growth >= required
        # This compares online learning work (train + TTT) with passive
        # consolidation work. It is deliberately not a comparison of two
        # undifferentiated global_step counters.
        sufficient_work = passive_delta > 0 and active_delta >= passive_delta
        should_switch = sufficient_data and sufficient_work
        work_share = (
            active_delta / (active_delta + passive_delta)
            if active_delta + passive_delta > 0
            else 0.0
        )
        if should_switch:
            reason = "new clean data exceeded learned repayment capacity"
        elif not sufficient_data:
            reason = "insufficient new clean data pressure"
        elif passive_delta <= 0:
            reason = "passive side has not yet carried measurable consolidation work"
        else:
            reason = "online side has not carried at least as much recent learning work"
        decision = SwitchDecision(
            should_switch=should_switch,
            reason=reason,
            clean_pressure=clean_pressure,
            baseline_pressure=self.baseline_clean_pressure,
            pressure_growth=pressure_growth,
            learned_capacity=learned_capacity,
            work_share=work_share,
            active_work=active_delta,
            passive_work=passive_delta,
            active_name=active_name,
            passive_name=passive_name,
        )
        self.last_decision = decision
        return decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_clean_pressure": self.baseline_clean_pressure,
            "baseline_active_work": self.baseline_active_work,
            "baseline_passive_work": self.baseline_passive_work,
            "baseline_active_name": self.baseline_active_name,
            "baseline_passive_name": self.baseline_passive_name,
        }

    def load_dict(self, raw: dict[str, Any]) -> None:
        self.baseline_clean_pressure = float(raw.get("baseline_clean_pressure", 0.0))
        self.baseline_active_work = float(raw.get("baseline_active_work", 0.0))
        self.baseline_passive_work = float(raw.get("baseline_passive_work", 0.0))
        active_name = raw.get("baseline_active_name")
        passive_name = raw.get("baseline_passive_name")
        self.baseline_active_name = str(active_name) if active_name else None
        self.baseline_passive_name = str(passive_name) if passive_name else None
