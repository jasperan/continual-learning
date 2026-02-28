import torch
import torch.nn as nn
from unittest.mock import MagicMock
from continual_learning.doc2lora.engine import Doc2LoRAEngine


def _make_mock_model():
    """Minimal model with linear layers for LoRA injection."""
    model = nn.Sequential()
    for i in range(4):
        model.add_module(f"layer_{i}_mlp", nn.Linear(16, 32, bias=False))
    return model


def _make_mock_tokenizer():
    tok = MagicMock()
    tok.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
    tok.decode.return_value = "Generated answer about the document."
    return tok


class TestDoc2LoRAEngine:
    def setup_method(self):
        self.model = _make_mock_model()
        self.tokenizer = _make_mock_tokenizer()
        self.engine = Doc2LoRAEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            hidden_dim=16,
            num_target_layers=4,
            intermediate_dim=32,
            lora_rank=4,
            chunk_size=10,
            simulated=True,
            layer_prefix="layer_{i}_mlp",
        )

    def test_init(self):
        assert self.engine.mode == "doc"
        assert self.engine.num_documents == 0

    def test_learn_stores_document(self):
        metrics = self.engine.learn("This is a test document about Oracle.")
        assert metrics["tokens_processed"] > 0
        assert metrics["method"] == "doc2lora"
        assert self.engine.num_documents == 1

    def test_learn_injects_lora(self):
        self.engine.learn("Test document.")
        assert self.engine._injector.is_injected

    def test_learn_returns_correct_metrics(self):
        metrics = self.engine.learn("Test doc with several words in it.")
        assert "tokens_processed" in metrics
        assert "method" in metrics
        assert "num_chunks" in metrics
        assert "effective_rank" in metrics
        assert "mode" in metrics
        assert metrics["method"] == "doc2lora"

    def test_generate_returns_string(self):
        self.engine.learn("Test document.")
        self.model.generate = MagicMock(
            return_value=torch.zeros(1, 20, dtype=torch.long)
        )
        self.model.config = MagicMock(vocab_size=32000)
        result = self.engine.generate("What is Oracle?")
        assert isinstance(result, str)

    def test_clear_resets_state(self):
        self.engine.learn("Test document.")
        self.engine.clear()
        assert self.engine.num_documents == 0
        assert not self.engine._injector.is_injected

    def test_multiple_documents(self):
        self.engine.learn("First document.")
        self.engine.learn("Second document.")
        assert self.engine.num_documents == 2

    def test_set_mode(self):
        self.engine.set_mode("text")
        assert self.engine.mode == "text"
        self.engine.set_mode("doc")
        assert self.engine.mode == "doc"

    def test_set_mode_invalid_raises(self):
        import pytest
        with pytest.raises(ValueError):
            self.engine.set_mode("invalid")

    def test_text_mode_no_chunking(self):
        self.engine.set_mode("text")
        metrics = self.engine.learn("Summarize scientific papers about AI.")
        assert metrics["num_chunks"] == 1
        assert metrics["mode"] == "text"

    def test_callback_invoked(self):
        calls = []
        self.engine.learn(
            "Test document.",
            callback=lambda **kwargs: calls.append(kwargs),
        )
        assert len(calls) >= 1
