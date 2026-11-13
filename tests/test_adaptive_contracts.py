from pathlib import Path
import tempfile
import unittest

import torch

from life_agent.agent import LifeAgent
from life_agent.byte_codec import dialogue_sample
from life_agent.config import AppConfig, ModelConfig, RuntimeConfig
from life_agent.memory import DebtLedger, ReplayBuffer, Sample, StatisticalCleaner
from life_agent.store import StateStore
from life_agent.switching import AdaptiveSwitchController
from life_agent.worker import TrainingWorker
from ingest_dataset import ProgressStore

PROJECT = Path(__file__).resolve().parents[1]


def small_config(training_device="cpu"):
    return AppConfig(
        ModelConfig(
            d_model=32,
            n_heads=4,
            n_layers=1,
            hidden_dim=32,
            min_hidden_dim=16,
            max_hidden_dim=64,
            max_seq_len=96,
        ),
        RuntimeConfig(
            training_device=training_device,
            require_cuda=training_device.startswith("cuda"),
            max_new_tokens=4,
            ttt_chunk_tokens=2,
            ttt_steps=1,
            consolidation_steps=1,
            batch_size=2,
            replay_capacity=16,
            debt_capacity=16,
            checkpoint_interval_s=99999,
            structure_interval=10000,
            structure_warmup=10000,
            idle_sleep_s=0.01,
            context_turns=2,
        ),
    )


class SwitchRoleTests(unittest.TestCase):
    def test_baseline_follows_role_not_hemisphere_name(self):
        controller = AdaptiveSwitchController()
        controller.note_switch(10.0, 130, 100, "right", "left")
        decision = controller.decide(
            clean_pressure=11.0,
            learned_capacity=0.5,
            active_work=133,
            passive_work=105,
            has_debt=True,
            active_name="right",
            passive_name="left",
        )
        self.assertFalse(decision.should_switch)
        self.assertEqual(decision.active_work, 3.0)
        self.assertEqual(decision.passive_work, 5.0)
        self.assertNotIn("passive side has not", decision.reason)

    def test_role_roundtrip_keeps_right_active_baseline(self):
        controller = AdaptiveSwitchController()
        controller.note_switch(10.0, 130, 100, "right", "left")
        restored = AdaptiveSwitchController()
        restored.load_dict(controller.to_dict())
        decision = restored.decide(
            clean_pressure=12.0,
            learned_capacity=0.5,
            active_work=140,
            passive_work=105,
            has_debt=True,
            active_name="right",
            passive_name="left",
        )
        self.assertTrue(decision.should_switch)
        self.assertEqual(decision.active_work, 10.0)
        self.assertEqual(decision.passive_work, 5.0)


class WorkAccountingTests(unittest.TestCase):
    def test_kind_counters_sum_to_global_step(self):
        worker = TrainingWorker(small_config())
        ids, labels = dialogue_sample("计数", "测试")
        worker.start()
        try:
            worker.submit("ttt", "left", [Sample(ids, labels)], 1)
            worker.submit("train", "left", [Sample(ids, labels)], 2)
            worker.submit("consolidate", "left", [Sample(ids, labels)], 3)
            state = worker.states["left"]
            self.assertEqual(state.ttt_steps, 1)
            self.assertEqual(state.train_steps, 2)
            self.assertEqual(state.consolidate_steps, 3)
            self.assertEqual(state.global_step, 6)
            self.assertEqual(state.online_learning_steps, 3)
            self.assertEqual(state.passive_consolidation_steps, 3)
        finally:
            worker.stop()


class CheckpointFingerprintTests(unittest.TestCase):
    def test_full_fingerprint_roundtrip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tests") as directory:
            state_dir = Path(directory)
            config = small_config()
            config.runtime.adaptive_switching = False
            agent = LifeAgent(config, state_dir)
            agent.start()
            try:
                agent.interact("指纹测试")
                agent.switch()
                agent.save()
            finally:
                agent.close()
            payload = torch.load(state_dir / "latest.pt", map_location="cpu", weights_only=False)
            self.assertIn("fingerprint", payload)
            restored = LifeAgent(config, state_dir)
            self.assertTrue(restored.load())
            self.assertEqual(
                StateStore.fingerprint_agent(restored),
                payload["fingerprint"],
            )
            restored.start()
            restored.close()

            payload["adaptive_switch"]["baseline_active_work"] = 999999.0
            torch.save(payload, state_dir / "latest.pt")
            backup = state_dir / "latest.backup.pt"
            if backup.exists():
                backup.unlink()
            tampered = LifeAgent(config, state_dir)
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                tampered.load()


class DebtReservationTests(unittest.TestCase):
    def test_due_does_not_count_until_commit(self):
        ledger = DebtLedger(capacity=4, decay=0.5, min_weight=0.01)
        ledger.add(Sample([65 + index % 3 for index in range(32)], [65 + index % 3 for index in range(32)]), 0.5)
        selected = ledger.due(1)
        self.assertEqual(ledger.repayment_batches, 0)
        self.assertEqual(ledger.total_repaid, 0)
        # A second due() must not return the same reserved digest.
        self.assertEqual(ledger.due(1), [])
        ledger.commit_repayment(selected)
        self.assertEqual(ledger.repayment_batches, 1)
        self.assertEqual(ledger.total_repaid, 1)

    def test_repeated_digest_does_not_manufacture_pressure(self):
        ledger = DebtLedger(capacity=4, decay=0.5, min_weight=0.01)
        item = Sample([65 + index % 3 for index in range(32)], [65 + index % 3 for index in range(32)])
        ledger.add(item, 0.5)
        pressure = ledger.clean_pressure()
        ledger.add(item.clone(), 0.5)
        self.assertEqual(ledger.clean_pressure(), pressure)
        self.assertEqual(len(ledger), 1)


class FamiliarityTests(unittest.TestCase):
    def test_repetition_reduces_salience_but_never_rejects(self):
        cleaner = StatisticalCleaner()
        ids = [65 + (index % 17) for index in range(64)]
        first = cleaner.observe(ids)
        second = cleaner.observe(ids)
        self.assertEqual(first["sampling_weight"], 1.0)
        self.assertLess(second["sampling_weight"], first["sampling_weight"])
        self.assertGreaterEqual(second["sampling_weight"], 0.2)
        replay = ReplayBuffer(capacity=8)
        replay.add(Sample(ids, ids.copy()))
        self.assertEqual(len(replay), 1)
        self.assertGreaterEqual(next(iter(replay.samples.values())).weight, 0.001)

    def test_structural_quality_is_length_invariant_for_same_prefix(self):
        base = [65 + (index % 17) for index in range(32)]
        self.assertEqual(
            StatisticalCleaner.structural_quality(base),
            StatisticalCleaner.structural_quality(base + [97] * 32),
        )
        self.assertEqual(
            StatisticalCleaner.structural_quality(base),
            StatisticalCleaner.structural_quality(base + [98] * 96),
        )


class ProgressIntegrityTests(unittest.TestCase):
    def test_chunk_digest_mismatch_forces_file_restart(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tests") as directory:
            root = Path(directory) / "data"
            root.mkdir()
            file = root / "a.bin"
            file.write_bytes(bytes(range(256)))
            progress = ProgressStore(Path(directory) / "progress.json", root, fresh=True)
            record = type("Record", (), {})()
            record.relative = "a.bin"
            record.path = file
            record.size = 256
            record.mtime_ns = file.stat().st_mtime_ns
            progress.mark(record, 100, "0" * 64, 100)
            self.assertEqual(progress.offset(record), 0)


if __name__ == "__main__":
    unittest.main()
