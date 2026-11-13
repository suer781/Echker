import unittest

from life_agent.memory import DebtLedger, ReplayBuffer, Sample, StatisticalCleaner


def sample(value=65, length=32, weight=1.0):
    ids = [value + (index % 3) for index in range(length)]
    return Sample(ids=ids, labels=ids.copy(), weight=weight)


class MemoryTests(unittest.TestCase):
    def test_extreme_repetition_is_cleaned(self):
        cleaner = StatisticalCleaner()
        repeated = [97] * 256
        varied = [(index * 37) % 256 for index in range(256)]
        self.assertLess(cleaner.structural_quality(repeated), 0.005)
        self.assertGreater(cleaner.structural_quality(varied), 0.5)

    def test_replay_deduplicates(self):
        replay = ReplayBuffer(capacity=4)
        item = sample()
        replay.add(item)
        replay.add(item.clone())
        self.assertEqual(len(replay), 1)

    def test_debt_is_lossy_and_decays_after_repayment(self):
        ledger = DebtLedger(capacity=2, decay=0.5, min_weight=0.05)
        ledger.add(sample(weight=0.8), retention=0.35)
        first = ledger.due(1)
        self.assertAlmostEqual(first[0].weight, 0.28125)
        self.assertEqual(ledger.commit_repayment(first), 1)
        ledger.decay_after_training()
        second = ledger.due(1)
        self.assertAlmostEqual(second[0].weight, 0.140625)
        self.assertEqual(ledger.commit_repayment(second), 1)
        ledger.decay_after_training()
        ledger.decay_after_training()
        self.assertEqual(len(ledger), 0)
        self.assertEqual(ledger.total_repaid, 2)

    def test_clean_pressure_and_capacity_are_measured(self):
        ledger = DebtLedger(capacity=4, decay=0.5, min_weight=0.01)
        ledger.add(sample(weight=1.0), retention=0.5)
        self.assertGreater(ledger.clean_pressure(), 0.0)
        self.assertGreater(ledger.clean_backlog(), 0.0)
        self.assertEqual(ledger.repayment_batches, 0)
        selected = ledger.due(1)
        self.assertEqual(ledger.repayment_batches, 0)
        ledger.commit_repayment(selected)
        self.assertEqual(ledger.repayment_batches, 1)
        self.assertGreater(ledger.repayment_capacity, 0.0)

    def test_capacity_drops_weakest_debt(self):
        ledger = DebtLedger(capacity=1, decay=0.5, min_weight=0.01)
        ledger.add(sample(value=65, weight=0.1), retention=1.0)
        ledger.add(sample(value=80, weight=1.0), retention=1.0)
        self.assertEqual(len(ledger), 1)
        self.assertGreater(ledger.due(1)[0].weight, 0.5)


if __name__ == "__main__":
    unittest.main()
