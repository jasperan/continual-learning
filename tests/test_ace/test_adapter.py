import json
from unittest.mock import MagicMock
from continual_learning.ace.adapter import ACEAdapter
from continual_learning.ace.engine import ACEEngine
from continual_learning.jitrl.base import BaseJitRLEngine


class TestACEAdapter:
    def setup_method(self):
        self.client = MagicMock()
        self.client.generate.return_value = "Test answer"
        self.ace_engine = ACEEngine(ollama_model="qwen2.5:7b", num_loops=1)
        self.ace_engine._client = self.client
        self.ace_engine._generator.client = self.client
        self.ace_engine._reflector.client = self.client
        self.ace_engine._curator.client = self.client

    def test_adapter_is_base_engine(self):
        adapter = ACEAdapter(self.ace_engine, model=MagicMock(), tokenizer=MagicMock())
        assert isinstance(adapter, BaseJitRLEngine)

    def test_learn_delegates_to_ace(self):
        adapter = ACEAdapter(self.ace_engine, model=MagicMock(), tokenizer=MagicMock())
        metrics = adapter.learn("Test document")
        assert metrics["method"] == "ace"
        assert metrics["tokens_processed"] > 0

    def test_generate_delegates_to_ace(self):
        adapter = ACEAdapter(self.ace_engine, model=MagicMock(), tokenizer=MagicMock())
        adapter.learn("Some context")
        result = adapter.generate("Question?")
        assert isinstance(result, str)

    def test_clear_delegates_to_ace(self):
        adapter = ACEAdapter(self.ace_engine, model=MagicMock(), tokenizer=MagicMock())
        adapter.learn("Text")
        adapter.clear()
        assert adapter.num_documents == 0

    def test_num_documents_delegates(self):
        adapter = ACEAdapter(self.ace_engine, model=MagicMock(), tokenizer=MagicMock())
        adapter.learn("Doc one")
        adapter.learn("Doc two")
        assert adapter.num_documents == 2
