"""Statistical sample cleaning, bounded replay, and lossy training debt."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import random
import statistics
import time
from collections import deque
from typing import Any, Iterable


@dataclass
class Sample:
    ids: list[int]
    labels: list[int]
    weight: float = 1.0
    source: str = "interaction"
    created_at: float = 0.0
    loss: float = 0.0

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = time.time()
        if len(self.ids) != len(self.labels) or len(self.ids) < 2:
            raise ValueError("sample ids and labels must have equal length >= 2")

    @property
    def digest(self) -> str:
        payload = ",".join(map(str, self.ids)).encode("ascii")
        payload += b"|" + ",".join(map(str, self.labels)).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    def clone(self, weight: float | None = None, source: str | None = None) -> "Sample":
        return Sample(
            ids=self.ids.copy(),
            labels=self.labels.copy(),
            weight=self.weight if weight is None else weight,
            source=self.source if source is None else source,
            created_at=self.created_at,
            loss=self.loss,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Sample":
        return cls(**raw)


class Seq2SeqSample(Sample):
    """seq2seq 语义样本：ids=编码器（用户侧），labels=解码器（回应侧）。

    长度天然不等——放宽 legacy 的等长校验；digest/clone 语义保持一致，
    债务/回放账本可无差别持有两类样本。
    """

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = time.time()
        if len(self.ids) < 2:
            raise ValueError("sample ids must have length >= 2")

    def clone(self, weight: float | None = None, source: str | None = None) -> "Seq2SeqSample":
        return Seq2SeqSample(
            ids=self.ids.copy(),
            labels=self.labels.copy(),
            weight=self.weight if weight is None else weight,
            source=self.source if source is None else source,
            created_at=self.created_at,
            loss=self.loss,
        )


def sample_from_dict(raw: dict) -> Sample:
    """容错重建：legacy 等长样本走正常构造；ByT5 双半球的 seq2seq 样本
    （encoder/decoder 长度不等）重建为 Seq2SeqSample。"""
    try:
        return Sample(**raw)
    except ValueError:
        return Seq2SeqSample(**raw)


class StatisticalCleaner:
    """Robust EWMA statistics; downweights outliers instead of judging truth.

    A bounded n-gram familiarity trace lives here because repeated stimulation
    changes the *value* of a memory, not because the agent needs a separate
    emotional subsystem. Familiarity lowers the stored sample's salience
    toward a non-zero floor; it never rejects a sample. This is the
    psychological habituation principle applied inside the existing cleaner.
    """

    def __init__(self, familiarity_window: int = 4096, token_domain: bool = False) -> None:
        self.count = 0
        self.loss_mean = 0.0
        self.loss_var = 1.0
        self.token_domain = token_domain
        self.familiarity_window = max(64, familiarity_window)
        self.familiarity_order: deque[int] = deque()
        self.familiarity_counts: dict[int, int] = {}
        self.frames_seen = 0
        self.last_novelty = 1.0
        self.last_familiarity = 0.0
        self.familiarity_signal = 1.0

    @staticmethod
    def _ngram_hashes(
        ids: list[int],
        ngram_size: int = 4,
        limit: int = 2048,
        token_domain: bool = False,
    ) -> list[int]:
        if token_domain:
            raw = [int(value) for value in ids if 0 <= int(value) < 2**32]
        else:
            raw = [int(value) for value in ids if 0 <= int(value) < 256]
        if len(raw) < ngram_size:
            return []
        hashes: list[int] = []
        count = min(len(raw) - ngram_size + 1, limit)
        for index in range(count):
            values = raw[index : index + ngram_size]
            window = (
                b"".join(value.to_bytes(4, "big") for value in values)
                if token_domain
                else bytes(values)
            )
            digest = hashlib.blake2b(window, digest_size=8).digest()
            hashes.append(int.from_bytes(digest, "big"))
        return hashes

    def _remember_ngrams(self, hashes: list[int]) -> None:
        for value in hashes:
            self.familiarity_counts[value] = self.familiarity_counts.get(value, 0) + 1
            self.familiarity_order.append(value)
        while len(self.familiarity_order) > self.familiarity_window:
            old = self.familiarity_order.popleft()
            count = self.familiarity_counts.get(old, 0) - 1
            if count <= 0:
                self.familiarity_counts.pop(old, None)
            else:
                self.familiarity_counts[old] = count

    def observe(self, ids: list[int]) -> dict[str, float]:
        """Update the bounded familiarity trace and return its current value."""
        hashes = self._ngram_hashes(ids, token_domain=self.token_domain)
        self.frames_seen += 1
        if not hashes:
            novelty = 1.0
        else:
            repeated = sum(1 for value in hashes if value in self.familiarity_counts)
            novelty = 1.0 - repeated / len(hashes)
            self._remember_ngrams(hashes)
        familiarity = 1.0 - novelty
        self.last_novelty = max(0.0, min(1.0, novelty))
        self.last_familiarity = max(0.0, min(1.0, familiarity))
        # Habituation reduces salience but never below 20% of the nominal
        # weight. Silent audio, solid colour and sparse tables stay learnable.
        self.familiarity_signal = max(0.2, 0.2 + 0.8 * self.last_novelty)
        return {
            "novelty": self.last_novelty,
            "familiarity": self.last_familiarity,
            "sampling_weight": self.familiarity_signal,
            "frames": float(self.frames_seen),
        }

    def signal(self) -> dict[str, float]:
        return {
            "novelty": self.last_novelty,
            "familiarity": self.last_familiarity,
            "sampling_weight": self.familiarity_signal,
            "frames": float(self.frames_seen),
            "known_ngrams": float(len(self.familiarity_counts)),
        }

    @staticmethod
    def structural_quality(ids: list[int], token_domain: bool = False) -> float:
        raw = (
            [value for value in ids if 0 <= value < 2**32]
            if token_domain
            else [value for value in ids if 0 <= value < 256]
        )
        n = len(raw)
        if n < 16:
            return 1.0
        # Assess a fixed leading window so the same bytes do not change
        # quality merely because a longer wrapper was appended. This removes
        # the 32/64/128 length drift found by the audit without rejecting
        # short legal frames.
        sample = raw[:32]
        counts: dict[int, int] = {}
        for value in sample:
            counts[value] = counts.get(value, 0) + 1
        probabilities = [count / len(sample) for count in counts.values()]
        entropy = -sum(p * math.log(p + 1e-12) for p in probabilities) / math.log(256)
        longest_run = 1
        run = 1
        for index in range(1, len(sample)):
            run = run + 1 if sample[index] == sample[index - 1] else 1
            longest_run = max(longest_run, run)
        unique_ratio = len(counts) / len(sample)
        entropy_factor = min(1.0, entropy / 0.12)
        run_factor = min(1.0, 12.0 / longest_run)
        unique_factor = min(1.0, unique_ratio / 0.08)
        return max(0.001, min(1.0, entropy_factor * run_factor * unique_factor))

    def loss_weight(self, loss: float) -> float:
        if not math.isfinite(loss):
            return 0.0
        if self.count < 20:
            weight = 1.0
        else:
            deviation = max(0.0, loss - self.loss_mean) / math.sqrt(self.loss_var + 1e-6)
            weight = 1.0 if deviation <= 2.5 else math.exp(-(deviation - 2.5))
        rate = 0.02
        delta = loss - self.loss_mean
        self.loss_mean = (1 - rate) * self.loss_mean + rate * loss
        self.loss_var = (1 - rate) * self.loss_var + rate * delta * delta
        self.count += 1
        return weight

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "loss_mean": self.loss_mean,
            "loss_var": self.loss_var,
            "token_domain": self.token_domain,
            "frames_seen": self.frames_seen,
            "last_novelty": self.last_novelty,
            "last_familiarity": self.last_familiarity,
            "familiarity_signal": self.familiarity_signal,
            "familiarity_order": [format(value, "016x") for value in self.familiarity_order],
        }

    def load_dict(self, raw: dict[str, Any]) -> None:
        self.count = int(raw.get("count", 0))
        self.loss_mean = float(raw.get("loss_mean", 0.0))
        self.loss_var = float(raw.get("loss_var", 1.0))
        self.token_domain = bool(raw.get("token_domain", self.token_domain))
        self.frames_seen = int(raw.get("frames_seen", 0))
        self.last_novelty = float(raw.get("last_novelty", 1.0))
        self.last_familiarity = float(raw.get("last_familiarity", 0.0))
        self.familiarity_signal = float(raw.get("familiarity_signal", 1.0))
        order = [int(value, 16) for value in raw.get("familiarity_order", [])]
        self.familiarity_order = deque(order[-self.familiarity_window :], maxlen=self.familiarity_window)
        self.familiarity_counts = {}
        for value in self.familiarity_order:
            self.familiarity_counts[value] = self.familiarity_counts.get(value, 0) + 1


class ReplayBuffer:
    def __init__(self, capacity: int, token_domain: bool = False) -> None:
        self.capacity = capacity
        self.samples: dict[str, Sample] = {}
        self.cleaner = StatisticalCleaner(
            familiarity_window=max(1024, capacity * 4),
            token_domain=token_domain,
        )

    def add(self, sample: Sample) -> bool:
        quality = self.cleaner.structural_quality(
            sample.ids, token_domain=self.cleaner.token_domain
        )
        familiarity = self.cleaner.observe(sample.ids)
        # Habituation is a multiplier on value, not a gate. It lowers the
        # salience of repeated n-grams toward a non-zero floor so silence,
        # solid images, tables and protocol headers remain learnable.
        sample.weight = max(
            0.001,
            min(4.0, sample.weight * quality * float(familiarity["sampling_weight"])),
        )
        digest = sample.digest
        if digest in self.samples:
            previous = self.samples[digest]
            # Repetition is not a reason to make a sample louder. The stored
            # weight follows the incoming (habituation-adjusted) weight instead
            # of being boosted every time the same bytes reappear.
            previous.weight = max(0.001, min(4.0, 0.7 * previous.weight + 0.3 * sample.weight))
            previous.created_at = max(previous.created_at, sample.created_at)
            return True
        self.samples[digest] = sample
        self._evict()
        return True

    def _evict(self) -> None:
        while len(self.samples) > self.capacity:
            oldest = min(self.samples, key=lambda key: self.samples[key].weight / (1.0 + self.samples[key].loss))
            del self.samples[oldest]

    def choose(self, limit: int, rng: random.Random) -> list[Sample]:
        if not self.samples:
            return []
        population = list(self.samples.values())
        chosen: list[Sample] = []
        available = population.copy()
        while available and len(chosen) < min(limit, len(population)):
            weights = [max(1e-6, item.weight) for item in available]
            item = rng.choices(available, weights=weights, k=1)[0]
            chosen.append(item.clone())
            available.remove(item)
        return chosen

    def latest(self, limit: int = 1) -> list[Sample]:
        return sorted(self.samples.values(), key=lambda item: item.created_at, reverse=True)[:limit]

    def __len__(self) -> int:
        return len(self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "samples": [sample.to_dict() for sample in self.samples.values()],
            "cleaner": self.cleaner.to_dict(),
        }

    def load_dict(self, raw: dict[str, Any]) -> None:
        self.samples = {}
        for item in raw.get("samples", []):
            sample = sample_from_dict(item)
            self.samples[sample.digest] = sample
        self.cleaner.load_dict(raw.get("cleaner", {}))
        self._evict()


class DebtLedger:
    """Directional, compressed debt: quantized weights and exponential forgetting."""

    def __init__(
        self,
        capacity: int,
        decay: float,
        min_weight: float,
        token_domain: bool = False,
    ) -> None:
        self.capacity = capacity
        self.decay = decay
        self.min_weight = min_weight
        self.token_domain = token_domain
        self.items: dict[str, Sample] = {}
        self.total_created = 0
        self.total_repaid = 0
        self.repayment_batches = 0
        self.repayment_capacity = 0.0
        self.total_clean_added = 0.0
        # A due() selection is a reservation, not a repayment. The same sample
        # cannot be selected twice before its training job has committed.
        self._reserved: set[str] = set()
        # Median of recent actually-trained batch capacities is less sensitive
        # to the top-weight selection bias than a cumulative arithmetic mean.
        self.capacity_history: deque[float] = deque(maxlen=32)

    def _quality(self, item: Sample) -> float:
        return StatisticalCleaner.structural_quality(
            item.ids, token_domain=self.token_domain
        )

    def clean_backlog(self) -> float:
        """Current unlearned mass after repetition and structural-noise weights."""
        return sum(max(0.0, item.weight) * self._quality(item) for item in self.items.values())

    def clean_pressure(self) -> float:
        """Cumulative clean mass admitted since the ledger was created."""
        return self.total_clean_added

    def add(self, sample: Sample, retention: float, count_pressure: bool = True) -> bool:
        weight = max(1 / 256, round(sample.weight * retention * 256) / 256)
        debt = sample.clone(weight=weight, source=f"debt:{sample.source}")
        digest = debt.digest
        quality = self._quality(debt)
        if digest in self.items:
            # Repeated identical debt is one debt with at most its strongest
            # observed weight; adding it again must not manufacture pressure.
            previous = self.items[digest]
            previous.weight = min(4.0, max(previous.weight, weight))
            previous.loss = max(previous.loss, debt.loss)
            added_clean = 0.0
        else:
            self.items[digest] = debt
            added_clean = weight * quality
        self.total_created += 1
        if count_pressure:
            self.total_clean_added += added_clean
        while len(self.items) > self.capacity:
            weakest = min(
                self.items,
                key=lambda key: max(0.0, self.items[key].weight) * self._quality(self.items[key]),
            )
            del self.items[weakest]
            self._reserved.discard(weakest)
        return added_clean > 0

    def due(self, limit: int) -> list[Sample]:
        """Reserve the highest expected-value debt; do not account repayment.

        Accounting is committed by the worker only after the corresponding
        gradient steps actually completed. This closes the window where a
        checkpoint could record a repayment that never trained.
        """
        if limit <= 0:
            return []
        selected = sorted(
            (item for digest, item in self.items.items() if digest not in self._reserved),
            key=lambda item: max(0.0, item.weight) * self._quality(item),
            reverse=True,
        )[:limit]
        for item in selected:
            self._reserved.add(item.digest)
        return [item.clone() for item in selected]

    def commit_repayment(self, samples: Iterable[Sample], trained_steps: int = 1) -> int:
        """Commit a successful consolidation batch.

        Only samples still present in the ledger count. Replay extras are
        ignored, and a failed/non-finite training job never reaches this call.
        """
        if trained_steps <= 0:
            return 0
        committed: list[Sample] = []
        for sample in samples:
            digest = sample.digest
            self._reserved.discard(digest)
            item = self.items.get(digest)
            if item is not None:
                committed.append(item)
        if not committed:
            return 0
        # MATH-05: switch-exchanged material is already-learned replay re-entered
        # for cross-hemisphere exposure. Training it is real consolidation work,
        # but its mass must not inflate the measured new-data batch capacity that
        # the switch controller compares against incoming clean pressure.
        batch_capacity = sum(
            max(0.0, item.weight) * self._quality(item)
            for item in committed
            if not item.source.startswith("debt:exchange")
        )
        self.total_repaid += len(committed)
        self.repayment_batches += 1
        if batch_capacity > 0:
            self.capacity_history.append(batch_capacity)
            self.repayment_capacity = float(statistics.median(self.capacity_history))
        return len(committed)

    def release_reservation(self, samples: Iterable[Sample]) -> None:
        """Release a due() reservation when the training job failed or was skipped."""
        for sample in samples:
            self._reserved.discard(sample.digest)

    def decay_after_training(self) -> None:
        expired = []
        for digest, item in self.items.items():
            item.weight *= self.decay
            if item.weight < self.min_weight:
                expired.append(digest)
        for digest in expired:
            del self.items[digest]
            self._reserved.discard(digest)

    def clear(self) -> None:
        self.items.clear()
        self._reserved.clear()

    def __len__(self) -> int:
        return len(self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "token_domain": self.token_domain,
            "items": [item.to_dict() for item in self.items.values()],
            "total_created": self.total_created,
            "total_repaid": self.total_repaid,
            "repayment_batches": self.repayment_batches,
            "repayment_capacity": self.repayment_capacity,
            "capacity_history": list(self.capacity_history),
            "total_clean_added": self.total_clean_added,
        }

    def load_dict(self, raw: dict[str, Any]) -> None:
        self.token_domain = bool(raw.get("token_domain", self.token_domain))
        self.items = {}
        for value in raw.get("items", []):
            item = sample_from_dict(value)
            self.items[item.digest] = item
        self.total_created = int(raw.get("total_created", 0))
        self.total_repaid = int(raw.get("total_repaid", 0))
        self.repayment_batches = int(raw.get("repayment_batches", 0))
        self.repayment_capacity = float(raw.get("repayment_capacity", 0.0))
        self.capacity_history = deque(
            (float(value) for value in raw.get("capacity_history", [])), maxlen=32
        )
        if not self.capacity_history and self.repayment_capacity > 0:
            self.capacity_history.append(self.repayment_capacity)
        self.total_clean_added = float(raw.get("total_clean_added", 0.0))
        self._reserved.clear()
