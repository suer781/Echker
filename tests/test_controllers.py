import unittest

from life_agent.config import AppConfig, ModelConfig, RuntimeConfig
from life_agent.model import DynamicByteLM
from life_agent.worker import LearningRateController, StructureController


def controller_config():
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
            structure_interval=2,
            structure_warmup=2,
            growth_size=8,
            prune_size=8,
        ),
    )


class ControllerTests(unittest.TestCase):
    def test_learning_rate_increases_on_plateau_and_drops_on_spike(self):
        controller = LearningRateController(controller_config())
        initial = controller.value
        for _ in range(11):
            controller.update(1.0)
        increased = controller.value
        self.assertGreater(increased, initial)
        controller.update(100.0)
        self.assertLess(controller.value, increased)

    def test_plateau_triggers_neurogenesis(self):
        config = controller_config()
        model = DynamicByteLM(config.model)
        controller = StructureController(config)
        for step in range(1, 24):
            self.assertIsNone(controller.consider(model, step, 1.0))
        event = controller.consider(model, 24, 1.0)
        self.assertIsNotNone(event)
        self.assertEqual(event["new_dim"], 40)

    def test_low_importance_neurons_trigger_pruning_when_learning(self):
        config = controller_config()
        model = DynamicByteLM(config.model)
        with torch_no_grad(model):
            model.blocks[0].activation_ema.fill_(1.0)
            model.blocks[0].activation_ema[0] = 0.0
        controller = StructureController(config)
        for step in range(1, 24):
            controller.consider(model, step, 100.0 / step)
        event = controller.consider(model, 24, 100.0 / 24)
        self.assertIsNotNone(event)
        self.assertEqual(event["new_dim"], 24)


class torch_no_grad:
    def __init__(self, model):
        import torch
        self.context = torch.no_grad()

    def __enter__(self):
        return self.context.__enter__()

    def __exit__(self, *args):
        return self.context.__exit__(*args)


if __name__ == "__main__":
    unittest.main()
