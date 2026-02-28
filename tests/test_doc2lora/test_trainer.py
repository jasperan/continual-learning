import torch
import torch.nn as nn
from continual_learning.doc2lora.trainer import HypernetworkTrainer
from continual_learning.doc2lora.hypernetwork import PerceiverHypernetwork
from continual_learning.doc2lora.lora_injector import LoRAInjector
from unittest.mock import MagicMock


def _make_tiny_model():
    """A tiny model that mimics a causal LM for testing.

    The MLP layers (model.layers.{i}.mlp) are Linear(16, 32) and are
    actually used in the forward pass so that LoRA forward hooks fire
    and produce a differentiable gradient path through the hypernetwork.
    Each MLP maps 16 -> 32; a projection maps 32 -> 16 so dimensions
    stay consistent across layers.
    """
    class MLPLayer(nn.Module):
        """Wraps a layer with a .mlp attribute for LoRA injection."""
        def __init__(self, in_features, intermediate):
            super().__init__()
            self.mlp = nn.Linear(in_features, intermediate, bias=False)
            self.proj = nn.Linear(intermediate, in_features, bias=False)

        def forward(self, x):
            return x + self.proj(self.mlp(x))

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 16)
            # Build nested structure: model.layers.{i}.mlp as nn.Linear
            model_inner = nn.Module()
            layers = nn.ModuleList([MLPLayer(16, 32) for _ in range(2)])
            model_inner.add_module("layers", layers)
            self.add_module("model", model_inner)
            self.lm_head = nn.Linear(16, 100, bias=False)

        def forward(self, input_ids, output_hidden_states=False, **kwargs):
            x = self.embed(input_ids)
            for layer in self.model.layers:
                x = layer(x)
            logits = self.lm_head(x)

            class Output:
                pass
            out = Output()
            out.logits = logits
            if output_hidden_states:
                out.hidden_states = [x, x]  # just repeat for simplicity
            return out

    return TinyLM()


def _make_mock_tokenizer():
    tok = MagicMock()
    tok.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
    return tok


class TestHypernetworkTrainer:
    def setup_method(self):
        self.model = _make_tiny_model()
        self.tokenizer = _make_mock_tokenizer()
        self.hypernet = PerceiverHypernetwork(
            hidden_dim=16,
            num_layers=2,
            intermediate_dim=32,
            lora_rank=2,
            latent_dim=8,
            num_cross_attn_blocks=1,
            num_heads=2,
        )
        self.injector = LoRAInjector()

    def test_init(self):
        trainer = HypernetworkTrainer(
            hypernetwork=self.hypernet,
            base_model=self.model,
            tokenizer=self.tokenizer,
            lora_injector=self.injector,
            device="cpu",
        )
        assert trainer.step == 0
        assert len(trainer.log_history) == 0

    def test_train_step_returns_loss(self):
        trainer = HypernetworkTrainer(
            hypernetwork=self.hypernet,
            base_model=self.model,
            tokenizer=self.tokenizer,
            lora_injector=self.injector,
            device="cpu",
            log_interval=1,
        )
        loss = trainer.train_step("Oracle was founded in 1977.", "When was Oracle founded?")
        assert isinstance(loss, float)
        assert loss >= 0
        assert trainer.step == 1

    def test_train_step_updates_hypernetwork(self):
        trainer = HypernetworkTrainer(
            hypernetwork=self.hypernet,
            base_model=self.model,
            tokenizer=self.tokenizer,
            lora_injector=self.injector,
            device="cpu",
        )
        params_before = {k: v.clone() for k, v in self.hypernet.named_parameters()}
        trainer.train_step("Test context.", "Test question?")

        changed = False
        for k, v in self.hypernet.named_parameters():
            if not torch.allclose(params_before[k], v):
                changed = True
                break
        assert changed, "Hypernetwork parameters should change after training step"

    def test_lora_removed_after_step(self):
        trainer = HypernetworkTrainer(
            hypernetwork=self.hypernet,
            base_model=self.model,
            tokenizer=self.tokenizer,
            lora_injector=self.injector,
            device="cpu",
        )
        trainer.train_step("Context.", "Question?")
        assert not self.injector.is_injected

    def test_save_and_load_checkpoint(self, tmp_path):
        trainer = HypernetworkTrainer(
            hypernetwork=self.hypernet,
            base_model=self.model,
            tokenizer=self.tokenizer,
            lora_injector=self.injector,
            device="cpu",
        )
        trainer.train_step("Context.", "Question?")

        path = str(tmp_path / "test_ckpt.pt")
        trainer.save_checkpoint(path)

        # Create fresh trainer and load
        hypernet2 = PerceiverHypernetwork(
            hidden_dim=16, num_layers=2, intermediate_dim=32,
            lora_rank=2, latent_dim=8, num_cross_attn_blocks=1, num_heads=2,
        )
        trainer2 = HypernetworkTrainer(
            hypernetwork=hypernet2,
            base_model=self.model,
            tokenizer=self.tokenizer,
            lora_injector=LoRAInjector(),
            device="cpu",
        )
        trainer2.load_checkpoint(path)
        assert trainer2.step == 1

    def test_multiple_steps(self):
        trainer = HypernetworkTrainer(
            hypernetwork=self.hypernet,
            base_model=self.model,
            tokenizer=self.tokenizer,
            lora_injector=self.injector,
            device="cpu",
            log_interval=2,
        )
        losses = []
        for _ in range(3):
            loss = trainer.train_step("Test doc.", "Test q?")
            losses.append(loss)
        assert trainer.step == 3
        assert len(losses) == 3

    def test_base_model_frozen(self):
        trainer = HypernetworkTrainer(
            hypernetwork=self.hypernet,
            base_model=self.model,
            tokenizer=self.tokenizer,
            lora_injector=self.injector,
            device="cpu",
        )
        for p in trainer.base_model.parameters():
            assert not p.requires_grad
