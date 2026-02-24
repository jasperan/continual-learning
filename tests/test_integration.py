import torch
import pytest
from unittest.mock import MagicMock
from continual_learning.model.dual_mlp import DualMLP
from continual_learning.model.tfidf_gate import TFIDFGate
from continual_learning.training.ttt_engine import TTTEngine
from continual_learning.checkpointing.manager import CheckpointManager
from continual_learning.evaluation.forgetting_metrics import compute_forgetting_ratio


class FakeTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=4096):
        tokens = torch.randint(0, 1000, (1, 60))
        return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}


class FakeModel(torch.nn.Module):
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
            loss = loss_fn(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        result = MagicMock()
        result.loss = loss
        result.logits = logits
        return result


class TestEndToEndPipeline:
    def test_full_learn_checkpoint_restore_cycle(self, tmp_path):
        """Full pipeline: calibrate gate -> learn -> save checkpoint -> restore -> verify."""
        model = FakeModel()
        tokenizer = FakeTokenizer()
        dual_mlps = list(model.dual_mlps)

        # Step 1: Calibrate TF-IDF gates
        calibration_data = [torch.randn(1, 10, 64).abs() for _ in range(50)]
        for dual in dual_mlps:
            dual.gate.calibrate(calibration_data)
            assert dual.gate.idf_scores is not None

        # Step 2: Learn from a document
        engine = TTTEngine(
            model=model,
            tokenizer=tokenizer,
            dual_mlps=dual_mlps,
            learning_rate=1e-3,
            mini_batch_size=16,
            gradient_steps=1,
        )
        metrics = engine.learn("Test document about Oracle Cloud Infrastructure services.")
        assert metrics["tokens_processed"] > 0
        assert len(metrics["losses"]) > 0

        # Save weights after learning
        learned_weights = [
            p.data.clone() for dual in dual_mlps
            for p in dual.trainable_mlp.parameters()
        ]

        # Step 3: Save checkpoint
        manager = CheckpointManager(checkpoint_dir=str(tmp_path / "checkpoints"))
        manager.save("test_cp", dual_mlps=dual_mlps, metadata={"test": True})

        # Step 4: Zero out weights (simulate restart)
        for dual in dual_mlps:
            for p in dual.trainable_mlp.parameters():
                p.data.zero_()

        # Verify weights are zeroed
        for dual in dual_mlps:
            for p in dual.trainable_mlp.parameters():
                assert torch.all(p == 0)

        # Step 5: Restore checkpoint
        manager.load("test_cp", dual_mlps=dual_mlps)

        # Step 6: Verify weights match post-learning state
        restored_weights = [
            p.data.clone() for dual in dual_mlps
            for p in dual.trainable_mlp.parameters()
        ]
        for learned, restored in zip(learned_weights, restored_weights):
            assert torch.allclose(learned, restored)

    def test_forgetting_measurement_workflow(self):
        """Simulate before/after benchmark to measure forgetting."""
        before_accuracy = 0.75
        after_accuracy = 0.68
        ratio = compute_forgetting_ratio(before_accuracy, after_accuracy)
        assert 0 < ratio < 0.15  # Within our target
