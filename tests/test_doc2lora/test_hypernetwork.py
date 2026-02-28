import torch
from continual_learning.doc2lora.hypernetwork import (
    SimulatedHypernetwork,
    PerceiverHypernetwork,
    PerceiverCrossAttentionBlock,
)


class TestSimulatedHypernetwork:
    def setup_method(self):
        self.hypernet = SimulatedHypernetwork(
            hidden_dim=256,
            num_layers=4,
            lora_rank=8,
            intermediate_dim=512,
        )

    def test_init(self):
        assert self.hypernet.lora_rank == 8
        assert self.hypernet.num_layers == 4

    def test_generate_lora_returns_dict(self):
        lora_weights = self.hypernet.generate_lora(["Hello world"])
        assert isinstance(lora_weights, dict)

    def test_generate_lora_has_correct_keys(self):
        lora_weights = self.hypernet.generate_lora(["Hello world"])
        for layer_idx in range(4):
            assert f"layer_{layer_idx}_A" in lora_weights
            assert f"layer_{layer_idx}_B" in lora_weights

    def test_generate_lora_correct_shapes(self):
        lora_weights = self.hypernet.generate_lora(["Hello world"])
        assert lora_weights["layer_0_A"].shape == (256, 8)
        assert lora_weights["layer_0_B"].shape == (8, 512)

    def test_generate_lora_deterministic_same_input(self):
        w1 = self.hypernet.generate_lora(["Hello"])
        w2 = self.hypernet.generate_lora(["Hello"])
        assert torch.allclose(w1["layer_0_A"], w2["layer_0_A"])

    def test_generate_lora_different_for_different_input(self):
        w1 = self.hypernet.generate_lora(["Hello"])
        w2 = self.hypernet.generate_lora(["Goodbye"])
        assert not torch.allclose(w1["layer_0_A"], w2["layer_0_A"])

    def test_multiple_chunks_compose_rank(self):
        w = self.hypernet.generate_lora(["chunk one", "chunk two"])
        assert w["layer_0_A"].shape == (256, 16)
        assert w["layer_0_B"].shape == (16, 512)

    def test_small_weights_magnitude(self):
        w = self.hypernet.generate_lora(["test"])
        assert w["layer_0_A"].abs().mean() < 0.1


class TestPerceiverHypernetwork:
    def setup_method(self):
        self.hypernet = PerceiverHypernetwork(
            hidden_dim=64,
            num_layers=4,
            intermediate_dim=128,
            lora_rank=4,
            latent_dim=32,
            num_cross_attn_blocks=2,
            num_heads=2,
        )

    def test_init_parameter_count(self):
        total = sum(p.numel() for p in self.hypernet.parameters())
        assert total > 0
        # Should be reasonably sized (not millions for small test config)
        assert total < 500_000

    def test_forward_returns_correct_keys(self):
        activations = torch.randn(1, 10, 64)  # (batch, seq_len, hidden_dim)
        weights = self.hypernet(activations)
        for i in range(4):
            assert f"layer_{i}_A" in weights
            assert f"layer_{i}_B" in weights

    def test_forward_correct_shapes(self):
        activations = torch.randn(1, 10, 64)
        weights = self.hypernet(activations)
        assert weights["layer_0_A"].shape == (64, 4)
        assert weights["layer_0_B"].shape == (4, 128)

    def test_forward_unbatched(self):
        activations = torch.randn(10, 64)  # no batch dim
        weights = self.hypernet(activations)
        assert weights["layer_0_A"].shape == (64, 4)

    def test_forward_is_differentiable(self):
        activations = torch.randn(1, 10, 64, requires_grad=True)
        weights = self.hypernet(activations)
        loss = sum(w.sum() for w in weights.values())
        loss.backward()
        assert activations.grad is not None

    def test_output_weights_small_magnitude(self):
        activations = torch.randn(1, 10, 64)
        weights = self.hypernet(activations)
        for key, w in weights.items():
            assert w.abs().mean() < 1.0, f"{key} has large magnitude"

    def test_cross_attention_block_standalone(self):
        block = PerceiverCrossAttentionBlock(latent_dim=32, num_heads=2, dropout=0.0)
        latents = torch.randn(1, 8, 32)
        context = torch.randn(1, 10, 32)
        output = block(latents, context)
        assert output.shape == latents.shape
