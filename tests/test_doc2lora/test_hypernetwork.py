import torch
from continual_learning.doc2lora.hypernetwork import SimulatedHypernetwork


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
