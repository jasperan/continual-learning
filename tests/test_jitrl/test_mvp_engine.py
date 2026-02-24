import torch
import pytest
from unittest.mock import MagicMock
from continual_learning.jitrl.mvp.engine import JitRLMVPEngine


class FakeTokenizer:
    vocab_size = 1000

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=4096):
        tokens = torch.randint(0, 1000, (1, min(len(text.split()) + 5, 30)))
        return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}

    def encode(self, text, add_special_tokens=False):
        return [hash(w) % 1000 for w in text.split()]

    def decode(self, token_ids, skip_special_tokens=True):
        return "Generated response about Oracle databases"


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.config = MagicMock()
        self.config.vocab_size = 1000

    def generate(self, input_ids, attention_mask=None, max_new_tokens=256,
                 do_sample=False, logits_processor=None):
        new_tokens = torch.randint(0, 1000, (1, max_new_tokens))
        return torch.cat([input_ids, new_tokens], dim=1)


class TestJitRLMVPEngine:
    def setup_method(self):
        self.model = FakeModel()
        self.tokenizer = FakeTokenizer()
        self.engine = JitRLMVPEngine(model=self.model, tokenizer=self.tokenizer, top_k=2, bias_strength=2.0)

    def test_learn_returns_metrics(self):
        metrics = self.engine.learn("Oracle Cloud Infrastructure provides compute.")
        assert "tokens_processed" in metrics
        assert metrics["method"] == "jitrl_mvp"
        assert metrics["tokens_processed"] > 0

    def test_learn_adds_to_retriever(self):
        self.engine.learn("Document one about databases.")
        self.engine.learn("Document two about networking.")
        assert self.engine.num_documents == 2

    def test_generate_returns_string(self):
        self.engine.learn("Oracle AI Vector Search supports embeddings.")
        result = self.engine.generate("What is Oracle AI Vector Search?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_without_documents_works(self):
        result = self.engine.generate("Hello world")
        assert isinstance(result, str)

    def test_clear_resets_state(self):
        self.engine.learn("Some document text.")
        self.engine.clear()
        assert self.engine.num_documents == 0
        assert self.engine._retriever.num_chunks == 0
