import pytest
from continual_learning.jitrl.mvp.retriever import TFIDFRetriever


class TestTFIDFRetriever:
    def setup_method(self):
        self.retriever = TFIDFRetriever()

    def test_add_document_increases_count(self):
        self.retriever.add_document("Oracle Cloud provides compute services.")
        assert self.retriever.num_chunks == 1

    def test_add_document_with_chunking(self):
        long_text = "First sentence about databases. " * 50 + "Second topic about AI. " * 50
        self.retriever.add_document(long_text, max_chunk_words=100)
        assert self.retriever.num_chunks > 1

    def test_retrieve_returns_relevant_chunks(self):
        self.retriever.add_document("Oracle Database supports SQL queries and PL/SQL programming.")
        self.retriever.add_document("Python is a popular programming language for data science.")
        self.retriever.add_document("The weather today is sunny with clear skies.")
        results = self.retriever.retrieve("What databases support SQL?", top_k=1)
        assert len(results) == 1
        assert "Oracle" in results[0] or "SQL" in results[0]

    def test_retrieve_respects_top_k(self):
        for i in range(5):
            self.retriever.add_document(f"Document number {i} about topic {i}.")
        results = self.retriever.retrieve("topic", top_k=3)
        assert len(results) == 3

    def test_retrieve_empty_store_returns_empty(self):
        results = self.retriever.retrieve("anything", top_k=3)
        assert results == []

    def test_clear_removes_all(self):
        self.retriever.add_document("Some document.")
        self.retriever.clear()
        assert self.retriever.num_chunks == 0
        assert self.retriever.retrieve("document") == []

    def test_add_returns_chunk_count(self):
        count = self.retriever.add_document("Short doc.")
        assert count == 1
