import torch
import pytest
from unittest.mock import MagicMock
from continual_learning.jitrl.full.engine import JitRLFullEngine


class FakeTokenizer:
    vocab_size = 200

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=4096):
        tokens = torch.randint(0, 200, (1, min(len(text.split()) + 5, 30)))
        return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}

    def decode(self, token_ids, skip_special_tokens=True):
        return "Generated answer about Oracle Cloud"


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(200, 64)
        self.lm_head = torch.nn.Linear(64, 200, bias=False)
        self.config = MagicMock()
        self.config.vocab_size = 200
        self.config.hidden_size = 64

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        result = MagicMock()
        result.logits = logits
        if output_hidden_states:
            result.hidden_states = (hidden, hidden)
        return result

    def generate(self, input_ids, attention_mask=None, max_new_tokens=256,
                 do_sample=False, logits_processor=None):
        new_tokens = torch.randint(0, 200, (1, max_new_tokens))
        return torch.cat([input_ids, new_tokens], dim=1)


class TestJitRLFullEngine:
    def setup_method(self):
        self.model = FakeModel()
        self.tokenizer = FakeTokenizer()
        self.engine = JitRLFullEngine(model=self.model, tokenizer=self.tokenizer, modulation_temperature=1.0)

    def test_learn_extracts_knowledge(self):
        metrics = self.engine.learn("Oracle provides cloud database services.")
        assert metrics["method"] == "jitrl_full"
        assert metrics["tokens_processed"] > 0
        assert self.engine._knowledge_store.num_entries == 1

    def test_learn_multiple_documents(self):
        self.engine.learn("First doc about Oracle.")
        self.engine.learn("Second doc about AI.")
        assert self.engine._knowledge_store.num_entries == 2
        assert self.engine.num_documents == 2

    def test_generate_returns_string(self):
        self.engine.learn("Oracle AI Vector Search enables embeddings.")
        result = self.engine.generate("What is Oracle AI Vector Search?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_without_knowledge_works(self):
        result = self.engine.generate("Hello")
        assert isinstance(result, str)

    def test_clear_resets_knowledge(self):
        self.engine.learn("Some text")
        self.engine.clear()
        assert self.engine.num_documents == 0
        assert self.engine._knowledge_store.num_entries == 0
