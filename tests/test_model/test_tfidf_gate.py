import torch
import pytest
from continual_learning.model.tfidf_gate import TFIDFGate


class TestTFIDFGate:
    def setup_method(self):
        self.hidden_size = 64
        self.gate = TFIDFGate(hidden_size=self.hidden_size, threshold=0.3)

    def test_initialization(self):
        assert self.gate.hidden_size == self.hidden_size
        assert self.gate.threshold == 0.3
        assert self.gate.idf_scores is None

    def test_calibrate_computes_idf(self):
        activations = [torch.randn(1, 10, self.hidden_size).abs() for _ in range(100)]
        self.gate.calibrate(activations)
        assert self.gate.idf_scores is not None
        assert self.gate.idf_scores.shape == (self.hidden_size,)
        assert (self.gate.idf_scores > 0).all()

    def test_compute_mask_returns_binary_mask(self):
        activations = [torch.randn(1, 10, self.hidden_size).abs() for _ in range(100)]
        self.gate.calibrate(activations)
        new_activation = torch.randn(1, 10, self.hidden_size).abs()
        mask = self.gate.compute_mask(new_activation)
        assert mask.shape == (self.hidden_size,)
        assert ((mask == 0) | (mask == 1)).all()

    def test_high_tfidf_neurons_pass(self):
        activations = [torch.ones(1, 10, self.hidden_size) for _ in range(100)]
        self.gate.calibrate(activations)
        new_activation = torch.zeros(1, 10, self.hidden_size)
        new_activation[:, :, :10] = 100.0
        mask = self.gate.compute_mask(new_activation)
        assert mask[:10].sum() >= mask[10:].sum()

    def test_uncalibrated_gate_passes_all(self):
        new_activation = torch.randn(1, 10, self.hidden_size).abs()
        mask = self.gate.compute_mask(new_activation)
        assert (mask == 1).all()

    def test_state_dict_roundtrip(self):
        activations = [torch.randn(1, 10, self.hidden_size).abs() for _ in range(100)]
        self.gate.calibrate(activations)
        state = self.gate.state_dict_custom()
        new_gate = TFIDFGate(hidden_size=self.hidden_size, threshold=0.3)
        new_gate.load_state_dict_custom(state)
        assert torch.equal(self.gate.idf_scores, new_gate.idf_scores)
