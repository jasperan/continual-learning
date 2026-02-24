import pytest
from continual_learning.data.streaming_qa import load_squad_splits


class TestLoadSquadSplits:
    def test_returns_learn_and_holdout(self):
        learn, holdout = load_squad_splits(learn_size=10, holdout_size=10, seed=42)
        assert len(learn) == 10
        assert len(holdout) == 10

    def test_each_item_has_context_and_question(self):
        learn, _ = load_squad_splits(learn_size=5, holdout_size=5, seed=42)
        for item in learn:
            assert "context" in item
            assert "question" in item
            assert "answer" in item

    def test_no_overlap(self):
        learn, holdout = load_squad_splits(learn_size=20, holdout_size=20, seed=42)
        learn_contexts = {item["context"] for item in learn}
        holdout_contexts = {item["context"] for item in holdout}
        assert learn_contexts.isdisjoint(holdout_contexts)

    def test_deterministic_with_seed(self):
        learn1, holdout1 = load_squad_splits(learn_size=10, holdout_size=10, seed=42)
        learn2, holdout2 = load_squad_splits(learn_size=10, holdout_size=10, seed=42)
        assert learn1[0]["context"] == learn2[0]["context"]
