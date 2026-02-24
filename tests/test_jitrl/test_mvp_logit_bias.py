import torch
import pytest
from continual_learning.jitrl.mvp.logit_bias import LogitBiaser


class FakeTokenizer:
    def __init__(self):
        self.vocab_size = 100

    def encode(self, text, add_special_tokens=False):
        return [hash(w) % self.vocab_size for w in text.split()]


class TestLogitBiaser:
    def setup_method(self):
        self.tokenizer = FakeTokenizer()
        self.biaser = LogitBiaser(tokenizer=self.tokenizer, bias_strength=2.0)

    def test_compute_bias_returns_correct_shape(self):
        bias = self.biaser.compute_bias(["Oracle Database supports SQL"], vocab_size=100)
        assert bias.shape == (100,)

    def test_compute_bias_boosts_document_tokens(self):
        bias = self.biaser.compute_bias(["Oracle Database"], vocab_size=100)
        doc_token_ids = self.tokenizer.encode("Oracle Database", add_special_tokens=False)
        for tid in doc_token_ids:
            assert bias[tid] > 0

    def test_compute_bias_zero_for_non_document_tokens(self):
        bias = self.biaser.compute_bias(["Oracle"], vocab_size=100)
        zero_count = (bias == 0).sum().item()
        assert zero_count > 90

    def test_empty_chunks_returns_zero_bias(self):
        bias = self.biaser.compute_bias([], vocab_size=100)
        assert torch.all(bias == 0)

    def test_bias_strength_scales_output(self):
        weak = LogitBiaser(tokenizer=self.tokenizer, bias_strength=1.0)
        strong = LogitBiaser(tokenizer=self.tokenizer, bias_strength=3.0)
        bias_weak = weak.compute_bias(["Oracle"], vocab_size=100)
        bias_strong = strong.compute_bias(["Oracle"], vocab_size=100)
        nonzero = bias_weak != 0
        if nonzero.any():
            ratio = bias_strong[nonzero] / bias_weak[nonzero]
            assert torch.allclose(ratio, torch.tensor(3.0))
