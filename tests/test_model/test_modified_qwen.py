import torch
import pytest
from continual_learning.model.modified_qwen import (
    inject_dual_mlps,
    get_modified_layer_indices,
)
from continual_learning.model.dual_mlp import DualMLP


class FakeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(64, 128, bias=False)
        self.up_proj = torch.nn.Linear(64, 128, bias=False)
        self.down_proj = torch.nn.Linear(128, 64, bias=False)
    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = FakeMLP()


class FakeModel(torch.nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([FakeLayer() for _ in range(num_layers)])


class TestGetModifiedLayerIndices:
    def test_default_range(self):
        indices = get_modified_layer_indices(num_layers=28, start=21, end=28)
        assert indices == list(range(21, 28))

    def test_custom_range(self):
        indices = get_modified_layer_indices(num_layers=28, start=24, end=28)
        assert indices == [24, 25, 26, 27]


class TestInjectDualMLPs:
    def test_injection_replaces_mlp_with_dual_mlp(self):
        model = FakeModel(4)
        inject_dual_mlps(model, layer_indices=[2, 3], alpha_initial=1.0, tfidf_threshold=0.3)
        assert isinstance(model.model.layers[0].mlp, FakeMLP)
        assert isinstance(model.model.layers[1].mlp, FakeMLP)
        assert isinstance(model.model.layers[2].mlp, DualMLP)
        assert isinstance(model.model.layers[3].mlp, DualMLP)

    def test_frozen_weights_match_original(self):
        model = FakeModel(1)
        original_weight = model.model.layers[0].mlp.gate_proj.weight.data.clone()
        inject_dual_mlps(model, layer_indices=[0], alpha_initial=1.0, tfidf_threshold=0.3)
        frozen_weight = model.model.layers[0].mlp.frozen_mlp.gate_proj.weight.data
        assert torch.equal(original_weight, frozen_weight)

    def test_early_layers_remain_frozen(self):
        model = FakeModel(4)
        inject_dual_mlps(model, layer_indices=[2, 3], alpha_initial=1.0, tfidf_threshold=0.3)
        for param in model.model.layers[0].parameters():
            assert not param.requires_grad
        for param in model.model.layers[1].parameters():
            assert not param.requires_grad
