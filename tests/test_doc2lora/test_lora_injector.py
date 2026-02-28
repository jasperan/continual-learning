import torch
import torch.nn as nn
from continual_learning.doc2lora.lora_injector import LoRAInjector


def _make_simple_model():
    """A tiny model with named linear layers to inject LoRA into."""
    model = nn.Sequential()
    model.add_module("layer_0_mlp", nn.Linear(16, 32, bias=False))
    model.add_module("layer_1_mlp", nn.Linear(16, 32, bias=False))
    return model


class TestLoRAInjector:
    def setup_method(self):
        self.model = _make_simple_model()
        self.injector = LoRAInjector()

    def test_inject_changes_output(self):
        x = torch.randn(1, 16)
        with torch.no_grad():
            out_before = self.model.layer_0_mlp(x).clone()

        lora_weights = {
            "layer_0_A": torch.randn(16, 4) * 0.1,
            "layer_0_B": torch.randn(4, 32) * 0.1,
        }
        self.injector.inject(self.model, lora_weights, layer_prefix="layer_{i}_mlp")

        with torch.no_grad():
            out_after = self.model.layer_0_mlp(x)

        assert not torch.allclose(out_before, out_after, atol=1e-6)

    def test_remove_restores_original(self):
        x = torch.randn(1, 16)
        with torch.no_grad():
            out_before = self.model.layer_0_mlp(x).clone()

        lora_weights = {
            "layer_0_A": torch.randn(16, 4) * 0.1,
            "layer_0_B": torch.randn(4, 32) * 0.1,
        }
        self.injector.inject(self.model, lora_weights, layer_prefix="layer_{i}_mlp")
        self.injector.remove(self.model)

        with torch.no_grad():
            out_after = self.model.layer_0_mlp(x)

        assert torch.allclose(out_before, out_after, atol=1e-6)

    def test_is_injected_flag(self):
        assert not self.injector.is_injected
        lora_weights = {
            "layer_0_A": torch.randn(16, 4) * 0.1,
            "layer_0_B": torch.randn(4, 32) * 0.1,
        }
        self.injector.inject(self.model, lora_weights, layer_prefix="layer_{i}_mlp")
        assert self.injector.is_injected
        self.injector.remove(self.model)
        assert not self.injector.is_injected

    def test_inject_multiple_layers(self):
        lora_weights = {
            "layer_0_A": torch.randn(16, 4) * 0.1,
            "layer_0_B": torch.randn(4, 32) * 0.1,
            "layer_1_A": torch.randn(16, 4) * 0.1,
            "layer_1_B": torch.randn(4, 32) * 0.1,
        }
        self.injector.inject(self.model, lora_weights, layer_prefix="layer_{i}_mlp")
        assert self.injector.is_injected
        assert len(self.injector._original_weights) == 2
