"""Regression tests for the online-learning reliability fixes.

CONC-01: submitted jobs must fail loudly on a wedged worker instead of
freezing the agent inside future.result() forever.
SYNC-02: structural growth/prune must carry AdamW momentum across the rebuild.
MATH-05: switch-exchanged material must not inflate clean pressure or the
measured new-data repayment capacity.
"""
from collections import deque
import unittest

from life_agent.config import AppConfig, ModelConfig, RuntimeConfig
from life_agent.memory import DebtLedger, Sample, StatisticalCleaner
from life_agent.worker import TrainingWorker


def worker_config():
    return AppConfig(
        ModelConfig(
            d_model=32,
            n_heads=4,
            n_layers=1,
            hidden_dim=32,
            min_hidden_dim=16,
            max_hidden_dim=64,
            max_seq_len=64,
        ),
        RuntimeConfig(
            training_device="cpu",
            require_cuda=False,
            learning_rate=0.001,
            structure_interval=10_000,
            structure_warmup=10_000,
            batch_size=2,
            consolidation_steps=2,
        ),
    )


def make_samples(count: int, length: int = 8) -> list[Sample]:
    return [
        Sample(
            ids=list(range(1, length + 1)),
            labels=list(range(2, length + 2)),
            weight=1.0,
            source=f"stream:test{i}",
        )
        for i in range(count)
    ]


def param_index(state, parameter) -> int:
    return [i for i, p in enumerate(state.model.parameters()) if p is parameter][0]


class SubmitTimeoutTests(unittest.TestCase):
    def test_hung_job_raises_instead_of_freezing(self):
        worker = TrainingWorker(worker_config())
        worker.start()
        original_train = worker._train

        def slow_train(state, samples, steps, kind):
            import time

            time.sleep(0.5)
            return original_train(state, samples, steps, kind)

        worker._train = slow_train
        with self.assertRaises(RuntimeError) as context:
            worker.submit("train", "left", make_samples(2), 1, wait=True, timeout=0.05)
        self.assertIn("'train'", str(context.exception))
        worker.stop()

    def test_healthy_job_still_returns_result(self):
        worker = TrainingWorker(worker_config())
        worker.start()
        result = worker.submit("train", "left", make_samples(2), 1, wait=True, timeout=60.0)
        self.assertEqual(result["steps"], 1)
        worker.stop()


class OptimizerMomentumTests(unittest.TestCase):
    def setUp(self):
        self.worker = TrainingWorker(worker_config())
        self.worker.start()
        self.state = self.worker.states["left"]
        # A few real gradient steps so the optimizer state exists and is
        # non-zero for every trained parameter.
        self.worker.submit("train", "left", make_samples(4), 3, wait=True)

    def tearDown(self):
        self.worker.stop()

    def test_growth_carries_momentum_for_existing_neurons(self):
        state = self.state
        old_params = list(state.model.parameters())
        old_dim = state.model.blocks[0].hidden_dim
        fc1_index = param_index(state, state.model.blocks[0].fc1.weight)
        old_entry = state.optimizer.state_dict()["state"][fc1_index]
        old_avg = old_entry["exp_avg"].detach().clone()
        self.assertGreater(float(old_avg.abs().sum()), 0.0)

        result = self.worker.resize("left", 0, old_dim + 8)
        self.assertEqual(result["mode"], "grow")
        self.assertIn("row_map", result)

        fc1_new = state.model.blocks[0].fc1.weight
        new_index = param_index(state, fc1_new)
        new_entry = state.optimizer.state_dict()["state"][new_index]
        self.assertTrue(new_entry, "grown fc1.weight lost its optimizer state entirely")
        self.assertEqual(float(new_entry["step"]), float(old_entry["step"]))
        self.assertTrue(
            torch_equal(new_entry["exp_avg"][:old_dim], old_avg[:old_dim]),
            "surviving neurons must keep their exact momentum",
        )
        self.assertGreater(float(new_entry["exp_avg"][old_dim:].abs().sum()), 0.0)

    def test_prune_keeps_only_kept_neurons_momentum(self):
        state = self.state
        old_dim = state.model.blocks[0].hidden_dim
        fc1_index = param_index(state, state.model.blocks[0].fc1.weight)
        old_entry = state.optimizer.state_dict()["state"][fc1_index]
        old_avg = old_entry["exp_avg"].detach().clone()

        result = self.worker.resize("left", 0, old_dim - 8)
        self.assertEqual(result["mode"], "prune")
        keep_map = result["keep_map"]
        self.assertEqual(len(keep_map), old_dim - 8)

        new_index = param_index(state, state.model.blocks[0].fc1.weight)
        new_entry = state.optimizer.state_dict()["state"][new_index]
        self.assertTrue(new_entry)
        for new_row, old_row in enumerate(keep_map):
            self.assertTrue(
                torch_equal(new_entry["exp_avg"][new_row], old_avg[old_row]),
                f"pruned remap lost momentum for row {new_row}",
            )


def torch_equal(left, right) -> bool:
    import torch

    return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))


class ExchangeDebtTests(unittest.TestCase):
    def test_exchange_admission_does_not_count_pressure(self):
        ledger = DebtLedger(64, 0.75, 0.005)
        stream_sample = Sample(ids=[1, 2, 3, 4], labels=[2, 3, 4, 5], weight=1.0, source="stream:a")
        self.assertTrue(ledger.add(stream_sample, 1.0))
        pressure_before = ledger.clean_pressure()
        self.assertGreater(pressure_before, 0.0)
        exchange = Sample(ids=[9, 9, 9, 9], labels=[9, 9, 9, 9], weight=1.0, source="exchange:stream:b")
        self.assertTrue(ledger.add(exchange, 0.35, count_pressure=False))
        self.assertEqual(ledger.clean_pressure(), pressure_before)

    def test_exchange_only_batch_does_not_pollute_capacity(self):
        ledger = DebtLedger(64, 0.75, 0.005)
        exchange = Sample(ids=[5, 6, 7, 8], labels=[6, 7, 8, 9], weight=4.0, source="exchange:stream:b")
        ledger.add(exchange, 0.35, count_pressure=False)
        selected = ledger.due(8)
        self.assertEqual(len(selected), 1)
        committed = ledger.commit_repayment(selected, trained_steps=2)
        self.assertEqual(committed, 1)
        self.assertEqual(ledger.total_repaid, 1)
        self.assertEqual(len(ledger.capacity_history), 0)
        self.assertEqual(ledger.repayment_capacity, 0.0)

    def test_mixed_batch_capacity_counts_only_new_mass(self):
        ledger = DebtLedger(64, 0.75, 0.005)
        stream_sample = Sample(ids=[1, 2, 3, 4], labels=[2, 3, 4, 5], weight=1.0, source="stream:a")
        exchange = Sample(ids=[5, 6, 7, 8], labels=[6, 7, 8, 9], weight=4.0, source="exchange:stream:b")
        ledger.add(stream_sample, 1.0)
        ledger.add(exchange, 0.35, count_pressure=False)
        selected = ledger.due(8)
        self.assertEqual(len(selected), 2)
        expected = sum(
            max(0.0, item.weight) * StatisticalCleaner.structural_quality(item.ids)
            for item in selected
            if item.source.startswith("debt:stream")
        )
        self.assertGreater(expected, 0.0)
        ledger.commit_repayment(selected, trained_steps=2)
        self.assertEqual(len(ledger.capacity_history), 1)
        self.assertAlmostEqual(ledger.capacity_history[0], expected, places=6)
        self.assertEqual(ledger.repayment_capacity, expected)

    def test_exchange_tag_survives_roundtrip(self):
        ledger = DebtLedger(64, 0.75, 0.005)
        exchange = Sample(ids=[5, 6, 7, 8], labels=[6, 7, 8, 9], weight=4.0, source="exchange:stream:b")
        ledger.add(exchange, 0.35, count_pressure=False)
        restored = DebtLedger(64, 0.75, 0.005)
        restored.load_dict(ledger.to_dict())
        digest = next(iter(restored.items))
        self.assertTrue(restored.items[digest].source.startswith("debt:exchange"))


if __name__ == "__main__":
    unittest.main()
