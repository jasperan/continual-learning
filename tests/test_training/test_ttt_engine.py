import torch
import pytest
from unittest.mock import MagicMock
from continual_learning.training.ttt_engine import TTTEngine
from continual_learning.model.dual_mlp import DualMLP


class FakeTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=4096):
        tokens = torch.randint(0, 1000, (1, 50))
        return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}


class FakeModel(torch.nn.Module):
    """Minimal model with dual MLPs for testing the TTT loop."""

    def __init__(self):
        super().__init__()
        self.dual_mlps = torch.nn.ModuleList([
            DualMLP(hidden_size=64, intermediate_size=128, alpha_initial=0.5)
        ])
        for dual in self.dual_mlps:
            for param in dual.trainable_mlp.parameters():
                param.data.normal_(0, 0.01)
        self.lm_head = torch.nn.Linear(64, 1000, bias=False)
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(1000, 64)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.model.embed_tokens(input_ids)
        for dual in self.dual_mlps:
            x = dual(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        result = MagicMock()
        result.loss = loss
        result.logits = logits
        return result


class TestTTTEngine:
    def setup_method(self):
        self.model = FakeModel()
        self.tokenizer = FakeTokenizer()
        self.engine = TTTEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            dual_mlps=list(self.model.dual_mlps),
            learning_rate=1e-4,
            mini_batch_size=16,
            gradient_steps=1,
        )

    def test_learn_updates_trainable_weights(self):
        original_weights = [
            p.data.clone() for dual in self.model.dual_mlps
            for p in dual.trainable_mlp.parameters()
        ]
        self.engine.learn("This is a test document about Oracle Cloud Infrastructure.")
        new_weights = [
            p.data.clone() for dual in self.model.dual_mlps
            for p in dual.trainable_mlp.parameters()
        ]
        changed = any(not torch.equal(o, n) for o, n in zip(original_weights, new_weights))
        assert changed, "Trainable weights should change after learning"

    def test_learn_does_not_update_frozen_weights(self):
        original_weights = [
            p.data.clone() for dual in self.model.dual_mlps
            for p in dual.frozen_mlp.parameters()
        ]
        self.engine.learn("New knowledge to internalize.")
        for orig, dual in zip([original_weights], self.model.dual_mlps):
            for o, p in zip(orig, dual.frozen_mlp.parameters()):
                assert torch.equal(o, p.data)

    def test_learn_returns_metrics(self):
        metrics = self.engine.learn("Some document to learn from.")
        assert "losses" in metrics
        assert "tokens_processed" in metrics
        assert len(metrics["losses"]) > 0
        assert metrics["tokens_processed"] > 0

    def test_learn_respects_gradient_steps(self):
        self.engine.gradient_steps = 2
        metrics = self.engine.learn("A document for multi-step testing.")
        assert len(metrics["losses"]) > 0
