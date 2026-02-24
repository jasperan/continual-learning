import torch
import pytest
from continual_learning.jitrl.full.knowledge_store import KnowledgeStore


class TestKnowledgeStore:
    def setup_method(self):
        self.hidden_size = 64
        self.store = KnowledgeStore(hidden_size=self.hidden_size)

    def test_add_embeddings(self):
        embeddings = torch.randn(10, self.hidden_size)
        self.store.add("doc1", embeddings)
        assert self.store.num_entries == 1

    def test_add_multiple_documents(self):
        self.store.add("doc1", torch.randn(5, self.hidden_size))
        self.store.add("doc2", torch.randn(8, self.hidden_size))
        assert self.store.num_entries == 2

    def test_query_returns_scores_and_embeddings(self):
        self.store.add("doc1", torch.randn(10, self.hidden_size))
        query = torch.randn(1, self.hidden_size)
        scores, embeddings = self.store.query(query, top_k=1)
        assert scores.shape == (1,)
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == self.hidden_size

    def test_query_top_k_limits_results(self):
        for i in range(5):
            self.store.add(f"doc{i}", torch.randn(3, self.hidden_size))
        query = torch.randn(1, self.hidden_size)
        scores, embeddings = self.store.query(query, top_k=2)
        assert scores.shape == (2,)

    def test_query_empty_store(self):
        query = torch.randn(1, self.hidden_size)
        scores, embeddings = self.store.query(query, top_k=3)
        assert scores.shape == (0,)
        assert embeddings.shape == (0, self.hidden_size)

    def test_clear_removes_all(self):
        self.store.add("doc1", torch.randn(5, self.hidden_size))
        self.store.clear()
        assert self.store.num_entries == 0
