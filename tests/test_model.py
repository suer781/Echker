import unittest

import torch

from life_agent.config import ModelConfig
from life_agent.model import DynamicByteLM


def tiny_config():
    return ModelConfig(
        d_model=32,
        n_heads=4,
        n_layers=1,
        hidden_dim=32,
        min_hidden_dim=16,
        max_hidden_dim=64,
        max_seq_len=64,
    )


class DynamicModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.model = DynamicByteLM(tiny_config()).eval()
        self.ids = torch.randint(0, 262, (2, 12))

    def test_growth_preserves_function_within_noise_bound(self):
        with torch.no_grad():
            before = self.model(self.ids)
            event = self.model.resize_hidden(0, 40)
            after = self.model(self.ids)
        self.assertEqual(event["mode"], "grow")
        self.assertEqual(self.model.hidden_dims(), [40])
        self.assertTrue(torch.allclose(before, after, atol=2e-3, rtol=2e-3))

    def test_pruning_reduces_parameters_and_keeps_forward_valid(self):
        before = self.model.parameter_count()
        event = self.model.resize_hidden(0, 24)
        with torch.no_grad():
            output = self.model(self.ids)
        self.assertEqual(event["mode"], "prune")
        self.assertEqual(self.model.hidden_dims(), [24])
        self.assertLess(self.model.parameter_count(), before)
        self.assertEqual(output.shape, (2, 12, 262))
        self.assertTrue(torch.isfinite(output).all())

    def test_resized_model_remains_trainable(self):
        self.model.resize_hidden(0, 48)
        self.model.train()
        logits = self.model(self.ids)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 262), self.ids[:, 1:].reshape(-1))
        loss.backward()
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))


if __name__ == "__main__":
    unittest.main()
