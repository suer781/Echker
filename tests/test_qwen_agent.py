import gc
import unittest
import tempfile
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM
from transformers.optimization import Adafactor

from qwen_agent import QwenAgent, completion_labels, transcript_samples
from qwen_dual import (
    QwenDualBrain,
    QwenHemisphere,
    QwenStructureController,
    _resize_mlp,
    _restore_tied_state_keys,
)


class CompletionLabelTests(unittest.TestCase):
    def test_only_assistant_completion_is_trainable(self):
        labels = completion_labels([10, 11], [10, 11, 12, 13])
        self.assertEqual(labels, [-100, -100, 12, 13])

    def test_rejects_non_prefix_chat_template(self):
        with self.assertRaisesRegex(ValueError, "not a prefix"):
            completion_labels([10, 11], [10, 12, 13])

    def test_rejects_empty_completion(self):
        with self.assertRaisesRegex(ValueError, "non-empty answer"):
            completion_labels([10, 11], [10, 11])

    def test_transcript_replay_targets_every_role_and_keeps_final_tokens(self):
        ids = list(range(1000, 1000 + 513))
        samples = transcript_samples(ids)
        self.assertEqual([token for sample in samples for token in sample.ids], ids)
        self.assertTrue(all(sample.labels == sample.ids for sample in samples))
        self.assertTrue(all(sample.source == "conversation:all-roles" for sample in samples))


class QwenStructureTests(unittest.TestCase):
    @staticmethod
    def small_model(tie_word_embeddings=False):
        config = Qwen2Config(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=tie_word_embeddings,
        )
        return Qwen2ForCausalLM(config)

    def test_growth_splits_mlp_contribution(self):
        model = self.small_model()
        layer = model.model.layers[0]
        inputs = torch.randn(1, 4, 16)
        with torch.no_grad():
            before = layer.mlp(inputs)
        result = _resize_mlp(layer, 40, torch.ones(32))
        with torch.no_grad():
            after = layer.mlp(inputs)
        self.assertEqual(layer.mlp.gate_proj.out_features, 40)
        self.assertEqual(result["mode"], "grow")
        self.assertTrue(torch.allclose(before, after, atol=1e-4, rtol=1e-4))

    def test_optimizer_state_survives_mlp_growth(self):
        model = self.small_model()
        hemisphere = QwenHemisphere.__new__(QwenHemisphere)
        hemisphere.model = model
        hemisphere.optimizer = Adafactor(
            model.parameters(),
            lr=2e-5,
            scale_parameter=False,
            relative_step=False,
        )
        hemisphere.structure = QwenStructureController(model)
        ids = torch.randint(0, 128, (1, 8))
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        hemisphere.optimizer.step()
        hemisphere.optimizer.zero_grad(set_to_none=True)

        old_row_state = hemisphere.optimizer.state[
            model.model.layers[0].mlp.gate_proj.weight
        ]["exp_avg_sq_row"].clone()
        result = hemisphere._resize(0, 40)
        new_parameter = model.model.layers[0].mlp.gate_proj.weight
        new_row_state = hemisphere.optimizer.state[new_parameter]["exp_avg_sq_row"]
        self.assertEqual(result["mode"], "grow")
        self.assertEqual(new_row_state.shape[0], 40)
        self.assertTrue(torch.equal(new_row_state[:32], old_row_state))
        self.assertEqual(len(hemisphere.optimizer.param_groups[0]["params"]), len(list(model.parameters())))

    def test_ttt_updates_the_full_qwen_model(self):
        model = self.small_model()
        model.to(dtype=torch.float16)
        hemisphere = QwenHemisphere.__new__(QwenHemisphere)
        hemisphere.model = model
        hemisphere.optimizer = Adafactor(
            model.parameters(),
            lr=2e-5,
            scale_parameter=False,
            relative_step=False,
        )
        hemisphere.structure = QwenStructureController(model)
        hemisphere.training_device = "cpu"
        hemisphere.device = "cpu"
        hemisphere._train_lock = threading.RLock()
        hemisphere.global_step = 0
        hemisphere.ttt_steps = 0
        hemisphere.consolidate_steps = 0
        hemisphere.last_loss = None
        hemisphere.last_event = "initialized"
        before = model.model.embed_tokens.weight.detach().clone()
        ids = torch.randint(0, 128, (12,)).tolist()

        result = hemisphere.learn(ids, ids, kind="ttt")

        self.assertTrue(result["learned"])
        self.assertEqual(hemisphere.ttt_steps, 1)
        self.assertFalse(torch.equal(before, model.model.embed_tokens.weight))

    def test_dynamic_width_checkpoint_restores_exact_weights(self):
        model = self.small_model()
        model.to(dtype=torch.float16)
        structure = QwenStructureController(model)
        _resize_mlp(model.model.layers[0], 40, structure.activation_ema[0])
        structure.activation_ema[0] = torch.ones(40)
        structure_state = structure.to_dict(model)
        model.config.dynamic_intermediate_sizes = structure_state["widths"]
        model.config.dynamic_structure_state = structure_state
        expected = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            config = AutoConfig.from_pretrained(directory, local_files_only=True)
            restored = AutoModelForCausalLM.from_config(config)
            restored.to(dtype=torch.float16)
            for index, width in enumerate(config.dynamic_intermediate_sizes):
                layer = restored.model.layers[index]
                _resize_mlp(layer, int(width), torch.ones(layer.mlp.gate_proj.out_features))
            restored.load_state_dict(load_file(f"{directory}\\model.safetensors"), strict=True)

        self.assertEqual(restored.model.layers[0].mlp.gate_proj.out_features, 40)
        self.assertTrue(
            all(torch.equal(expected[name], tensor) for name, tensor in restored.state_dict().items())
        )

    def test_tied_embedding_checkpoint_restores_omitted_output_alias(self):
        model = self.small_model(tie_word_embeddings=True)
        expected = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            config = AutoConfig.from_pretrained(directory, local_files_only=True)
            restored = AutoModelForCausalLM.from_config(config)
            state = load_file(f"{directory}\\model.safetensors")
            _restore_tied_state_keys(restored, config, state)
            restored.load_state_dict(state, strict=True)
            del state
            gc.collect()

        self.assertTrue(
            all(torch.equal(expected[name], tensor) for name, tensor in restored.state_dict().items())
        )

    def test_passive_consolidation_keeps_active_hemisphere_on_gpu(self):
        class FakeHemisphere:
            def __init__(self, device):
                self.device = device
                self.training_device = "cuda"
                self.sample = object()
                self.training_device_during_consolidation = None

            def consolidation_samples(self):
                return [self.sample]

            def to_cpu(self):
                self.device = "cpu"

            def consolidate(self, samples, intensity):
                self.training_device_during_consolidation = self.training_device
                return {"steps": len(samples), "intensity": intensity}

        class FakeBrain:
            training_lock = threading.RLock()

            def __init__(self):
                self.active = FakeHemisphere("cuda")
                self.passive = FakeHemisphere("cpu")

            def active_h(self):
                return self.active

            def passive_h(self):
                return self.passive

        brain = FakeBrain()
        result = QwenDualBrain.consolidate_passive(brain, 0.75)

        self.assertEqual(result, {"steps": 1, "intensity": 0.75})
        self.assertEqual(brain.active.device, "cuda")
        self.assertEqual(brain.passive.training_device_during_consolidation, "cpu")
        self.assertEqual(brain.passive.training_device, "cuda")

    def test_generation_does_not_wait_for_background_consolidation(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(use_cache=False)

            def generate(self, input_ids, **kwargs):
                token = torch.tensor([[3]], device=input_ids.device)
                return torch.cat((input_ids, token), dim=1)

        class TinyTokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return [1, 2]

            def decode(self, ids, skip_special_tokens):
                return "ok"

        model = TinyModel()
        hemisphere = SimpleNamespace(
            model=model,
            learn=Mock(return_value={"learned": True}),
        )
        brain = SimpleNamespace(
            training_lock=threading.RLock(),
            ensure_active_gpu=Mock(),
            active_h=Mock(return_value=hemisphere),
        )
        agent = QwenAgent.__new__(QwenAgent)
        agent.brain = brain
        agent.tokenizer = TinyTokenizer()
        agent._lock = brain.training_lock
        agent._consolidation_future = Future()
        agent.last_learning_result = None
        agent.learn_from_transcript = Mock(return_value=1)
        agent._await_consolidation = Mock(side_effect=AssertionError("generation waited"))

        response = agent.generate("hello", max_new_tokens=1, temperature=0)

        self.assertEqual(response, "ok")
        hemisphere.learn.assert_called_once()
        agent.learn_from_transcript.assert_called_once()


if __name__ == "__main__":
    unittest.main()
