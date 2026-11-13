from concurrent.futures import Future
from pathlib import Path
import tempfile
import time
import unittest

import torch

from life_agent.agent import LifeAgent
from life_agent.byte_codec import ByteFrame, Codec, dialogue_sample
from life_agent.config import AppConfig, ModelConfig, RuntimeConfig
from life_agent.memory import Sample
from life_agent.switching import AdaptiveSwitchController
from life_agent.worker import TrainingWorker

PROJECT = Path(__file__).resolve().parents[1]


def test_config(training_device="cpu"):
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


class WorkerTests(unittest.TestCase):
    def test_cuda_training_publishes_cpu_weights(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        worker = TrainingWorker(test_config("cuda"))
        state = worker.states["left"]
        before = state.cpu_model.blocks[0].wi_0.weight.detach().clone()
        ids, labels = dialogue_sample("你好", "测试")
        worker.start()
        try:
            result = worker.submit("ttt", "left", [Sample(ids, labels)], 1)
            self.assertEqual(result["steps"], 1)
            self.assertEqual(str(worker.device), "cuda")
            self.assertFalse(torch.equal(before, state.cpu_model.blocks[0].wi_0.weight))
        finally:
            worker.stop()
    def test_resize_repairs_stale_cpu_width(self):
        worker = TrainingWorker(test_config())
        state = worker.states["left"]
        initial_width = state.cpu_model.blocks[0].hidden_dim
        target = initial_width + 16
        state.model.resize_hidden(0, target)
        self.assertEqual(state.cpu_model.blocks[0].hidden_dim, initial_width)
        worker._resize_both(state, {"layer": 0, "new_dim": target})
        self.assertEqual(state.cpu_model.hidden_dims(), state.model.hidden_dims())

    def test_cuda_agent_completes_interaction_and_switch(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        with tempfile.TemporaryDirectory(dir=PROJECT / "tests") as directory:
            config = test_config("cuda")
            config.runtime.adaptive_switching = False
            agent = LifeAgent(config, Path(directory))
            agent.start()
            try:
                response = agent.interact("GPU 在线学习")
                self.assertIsInstance(response, str)
                agent.switch()
                self.assertEqual(agent.switch_count, 1)
                self.assertEqual(agent.status()["training_device"], "cuda")
            finally:
                agent.close()


class AgentIntegrationTests(unittest.TestCase):
    def test_manual_switch_and_checkpoint_restore(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tests") as directory:
            state_dir = Path(directory)
            config = test_config()
            config.runtime.adaptive_switching = False
            agent = LifeAgent(config, state_dir)
            agent.start()
            try:
                response = agent.interact("你好")
                self.assertIsInstance(response, str)
                self.assertGreater(agent.state("left").global_step + agent.state("right").global_step, 0)
                agent.switch()
                self.assertEqual(agent.switch_count, 1)
                agent.worker.resize("left", 0, 40)
                ids, labels = dialogue_sample("结构变化后", "继续训练")
                agent.worker.submit("train", "left", [Sample(ids, labels)], 1, wait=True)
                self.assertEqual(agent.state("left").model.blocks[0].activation_ema.numel(), 40)
                checkpoint = agent.save()
                self.assertTrue(checkpoint.exists())
                expected_active = agent.active
            finally:
                agent.close()

            restored = LifeAgent(test_config(), state_dir)
            self.assertTrue(restored.load())
            self.assertEqual(restored.active, expected_active)
            self.assertEqual(restored.interaction_count, 1)
            self.assertEqual(restored.state("left").model.hidden_dims(), [40])
            self.assertGreater(len(restored.history), 0)
            restored.start()
            restored.close()

    def test_adaptive_switch_requires_clean_pressure_beyond_learned_capacity(self):
        controller = AdaptiveSwitchController()
        low = controller.decide(
            clean_pressure=1.0,
            learned_capacity=4.0,
            active_work=10_000,
            passive_work=10_000,
            has_debt=True,
        )
        self.assertFalse(low.should_switch)
        high = controller.decide(
            clean_pressure=9.0,
            learned_capacity=4.0,
            active_work=10_000,
            passive_work=10_000,
            has_debt=True,
        )
        self.assertTrue(high.should_switch)
        self.assertGreater(high.pressure_growth, high.learned_capacity)
        no_debt = controller.decide(
            clean_pressure=9.0,
            learned_capacity=4.0,
            active_work=10_000,
            passive_work=10_000,
            has_debt=False,
        )
        self.assertFalse(no_debt.should_switch)

    def test_adaptive_switch_baseline_moves_after_switch(self):
        controller = AdaptiveSwitchController()
        controller.note_switch(9.0, 100, 100)
        self.assertFalse(
            controller.decide(
                clean_pressure=9.0,
                learned_capacity=1.0,
                active_work=200,
                passive_work=200,
                has_debt=True,
            ).should_switch
        )
        self.assertTrue(
            controller.decide(
                clean_pressure=12.0,
                learned_capacity=1.0,
                active_work=200,
                passive_work=200,
                has_debt=True,
            ).should_switch
        )

    def test_running_load_rejected_and_background_failure_is_not_silent(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tests") as directory:
            agent = LifeAgent(test_config(), Path(directory))
            agent.start()
            try:
                with self.assertRaisesRegex(RuntimeError, "before the worker starts"):
                    agent.load()
                failed = Future()
                failed.set_exception(RuntimeError("deliberate consolidation failure"))
                agent.pending_consolidation["right"] = failed
                with self.assertRaisesRegex(RuntimeError, "deliberate consolidation failure"):
                    agent.close()
                self.assertFalse(agent.worker.started)
            finally:
                if agent.worker.started:
                    agent.worker.stop()

    def test_dual_hemisphere_roles_debt_and_persistence(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tests") as directory:
            config = test_config()
            config.runtime.adaptive_switching = False
            agent = LifeAgent(config, Path(directory))
            left = agent.state("left")
            right = agent.state("right")
            self.assertIsNot(left.model, right.model)
            self.assertIsNot(left.cpu_model, right.cpu_model)
            self.assertIsNot(left.optimizer, right.optimizer)
            self.assertIsNot(left.replay, right.replay)
            self.assertIsNot(left.debt, right.debt)
            self.assertFalse(torch.equal(left.model.blocks[0].wi_0.weight, right.model.blocks[0].wi_0.weight))
            calls = {"left": 0, "right": 0}
            handles = [
                state.cpu_model.register_forward_hook(
                    lambda _module, _inputs, _output, name=name: calls.__setitem__(name, calls[name] + 1)
                )
                for name, state in (("left", left), ("right", right))
            ]
            agent.start()
            try:
                agent.interact("左侧在线推理")
                self.assertEqual(agent.active, "left")
                self.assertGreater(calls["left"], 0)
                self.assertEqual(calls["right"], 0)
                self.assertTrue(left.last_event.startswith("ttt:"))
                self.assertGreater(len(right.debt), 0)

                agent.switch()
                self.assertEqual(agent.active, "right")
                self.assertEqual(agent.switch_count, 1)
                self.assertGreater(len(right.debt), 0)
                left_calls = calls["left"]
                right_calls = calls["right"]
                agent.interact("右侧在线推理")
                self.assertEqual(calls["left"], left_calls)
                self.assertGreater(calls["right"], right_calls)
                self.assertTrue(right.last_event.startswith("ttt:"))

                # The reverse debt exists after the right side has produced
                # its own online replay; an unused passive side has none to send.
                agent.switch()
                self.assertEqual(agent.active, "left")
                self.assertEqual(agent.switch_count, 2)
                self.assertGreater(len(left.debt), 0)
                self.assertGreater(len(right.debt), 0)
                agent.save()
            finally:
                for handle in handles:
                    handle.remove()
                agent.close()

            restored = LifeAgent(config, Path(directory))
            self.assertTrue(restored.load())
            self.assertEqual(restored.active, "left")
            self.assertEqual(restored.switch_count, 2)
            self.assertEqual(set(restored.worker.states), {"left", "right"})
            self.assertGreater(restored.state("left").global_step, 0)
            self.assertGreater(restored.state("right").global_step, 0)

    def test_solid_rgb_frame_is_fully_observed_and_restored(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tests") as directory:
            state_dir = Path(directory)
            agent = LifeAgent(test_config(), state_dir)
            agent.start()
            try:
                frame = ByteFrame.rgb(bytes(8 * 8 * 3), width=8, height=8)
                result = agent.feed_frame(frame)
                self.assertEqual(result["payload_bytes"], 192)
                self.assertEqual(result["trained_steps"], (result["chunks"] + 1) // 2)
                self.assertEqual(agent.observed_frames, 1)
                self.assertGreater(len(agent.state("left").replay) + len(agent.state("right").replay), 0)
            finally:
                agent.close()
            restored = LifeAgent(test_config(), state_dir)
            self.assertTrue(restored.load())
            self.assertEqual(restored.observed_frames, 1)
            restored.start()
            restored.close()

    def test_fixed_shape_rgb_generation_has_exact_payload_length(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tests") as directory:
            config = test_config()
            config.runtime.max_new_tokens = 12
            config.runtime.ttt_chunk_tokens = 4
            agent = LifeAgent(config, Path(directory))
            agent.start()
            try:
                specification = ByteFrame.rgb(bytes(12), 2, 2).empty()
                response = agent.interact_frame(ByteFrame.text("生成 2x2 RGB"), specification)
                self.assertEqual(len(response.payload), 12)
                response.validate_payload()
            finally:
                agent.close()

    def test_feed_frame_is_not_starved_by_passive_queue(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tests") as directory:
            config = test_config()
            config.runtime.adaptive_switching = False
            agent = LifeAgent(config, Path(directory))
            agent.start()
            try:
                before = agent.state(agent.passive).global_step
                agent.feed_frame(ByteFrame.text("连续数据流" * 400))
                self.assertGreater(agent.state(agent.passive).global_step, before)
                self.assertFalse(agent.pending_consolidation)
            finally:
                agent.close()

    def test_custom_codec_interaction_and_v1_history_migration(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tests") as directory:
            state_dir = Path(directory)
            agent = LifeAgent(test_config(), state_dir)
            agent.start()
            try:
                stimulus = ByteFrame(88, {"format": "unknown"}, bytes(range(64)))
                specification = ByteFrame(99, {"purpose": "arbitrary-output"}, b"")
                response = agent.interact_frame(stimulus, specification)
                self.assertEqual(response.codec, 99)
                self.assertEqual(response.metadata["purpose"], "arbitrary-output")
                self.assertEqual(len(response.payload), 4)
            finally:
                agent.close()

            checkpoint = state_dir / "latest.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["format_version"] = 1
            payload["history"] = [["旧文本输入", "旧文本输出"]]
            torch.save(payload, checkpoint)
            restored = LifeAgent(test_config(), state_dir)
            self.assertTrue(restored.load())
            self.assertEqual(restored.history[0][0].as_text(), "旧文本输入")
            self.assertEqual(restored.history[0][1].as_text(), "旧文本输出")
            restored.start()
            restored.close()


if __name__ == "__main__":
    unittest.main()
