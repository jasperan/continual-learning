import torch
import pytest
from continual_learning.model.dual_mlp import DualMLP
from continual_learning.model.tfidf_gate import TFIDFGate


class TestDualMLP:
    def setup_method(self):
        self.hidden_size = 64
        self.intermediate_size = 128
        self.dual_mlp = DualMLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            hidden_act="silu",
            alpha_initial=1.0,
            tfidf_threshold=0.3,
        )

    def test_frozen_mlp_requires_no_grad(self):
        for param in self.dual_mlp.frozen_mlp.parameters():
            assert not param.requires_grad

    def test_trainable_mlp_requires_grad(self):
        for param in self.dual_mlp.trainable_mlp.parameters():
            assert param.requires_grad

    def test_trainable_mlp_starts_near_zero(self):
        for param in self.dual_mlp.trainable_mlp.parameters():
            assert param.abs().max() < 0.01  # Small random init, not zeros

    def test_output_shape(self):
        x = torch.randn(2, 10, self.hidden_size)
        out = self.dual_mlp(x)
        assert out.shape == x.shape

    def test_initial_output_approximately_equals_frozen(self):
        x = torch.randn(2, 10, self.hidden_size)
        frozen_out = self.dual_mlp.frozen_mlp(x)
        dual_out = self.dual_mlp(x)
        # With small random init and alpha close to 1, output is very close to frozen
        relative_diff = (dual_out - frozen_out).abs().max() / (frozen_out.abs().max() + 1e-8)
        assert relative_diff < 0.1  # Within 10% of frozen output

    def test_alpha_blending(self):
        for param in self.dual_mlp.trainable_mlp.parameters():
            param.data.fill_(0.01)
        x = torch.randn(2, 10, self.hidden_size)
        self.dual_mlp.alpha = 1.0
        out_frozen_heavy = self.dual_mlp(x)
        self.dual_mlp.alpha = 0.0
        out_trainable_heavy = self.dual_mlp(x)
        assert not torch.allclose(out_frozen_heavy, out_trainable_heavy)

    def test_gradient_flows_to_trainable_only(self):
        x = torch.randn(2, 10, self.hidden_size)
        for param in self.dual_mlp.trainable_mlp.parameters():
            param.data.normal_(0, 0.01)
        self.dual_mlp.alpha = 0.5
        out = self.dual_mlp(x)
        loss = out.sum()
        loss.backward()
        for param in self.dual_mlp.trainable_mlp.parameters():
            assert param.grad is not None
        for param in self.dual_mlp.frozen_mlp.parameters():
            assert param.grad is None

    def test_from_existing_mlp(self):
        class FakeMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = torch.nn.Linear(64, 128, bias=False)
                self.up_proj = torch.nn.Linear(64, 128, bias=False)
                self.down_proj = torch.nn.Linear(128, 64, bias=False)
            def forward(self, x):
                return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

        original = FakeMLP()
        dual = DualMLP.from_existing_mlp(original, alpha_initial=1.0, tfidf_threshold=0.3)
        assert torch.equal(dual.frozen_mlp.gate_proj.weight, original.gate_proj.weight)
        assert dual.trainable_mlp.gate_proj.weight.abs().max() < 0.01

    def test_get_trainable_params(self):
        params = list(self.dual_mlp.get_trainable_params())
        assert len(params) == 3  # gate_proj, up_proj, down_proj weights
        for p in params:
            assert p.requires_grad
