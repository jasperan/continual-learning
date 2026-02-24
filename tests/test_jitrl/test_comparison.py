import pytest
from unittest.mock import MagicMock
from continual_learning.jitrl.comparison import ComparisonHarness
from continual_learning.jitrl.base import BaseJitRLEngine


class FakeEngine(BaseJitRLEngine):
    def learn(self, text, callback=None):
        self._documents.append(text)
        return {"tokens_processed": len(text.split()), "method": "fake"}

    def generate(self, prompt, max_new_tokens=256):
        return "Paris"


class TestComparisonHarness:
    def setup_method(self):
        self.model = MagicMock()
        self.tokenizer = MagicMock()
        self.engines = {
            "engine_a": FakeEngine(self.model, self.tokenizer),
            "engine_b": FakeEngine(self.model, self.tokenizer),
        }
        self.harness = ComparisonHarness(engines=self.engines)

    def test_run_comparison_returns_results_per_engine(self):
        learn_texts = ["Oracle Cloud provides compute services."]
        qa_items = [{"question": "What is the capital of France?", "answer": "Paris"}]
        results = self.harness.run_comparison(learn_texts, qa_items)
        assert "engine_a" in results
        assert "engine_b" in results

    def test_results_contain_accuracy(self):
        learn_texts = ["Some text"]
        qa_items = [{"question": "What is the capital of France?", "answer": "Paris"}]
        results = self.harness.run_comparison(learn_texts, qa_items)
        assert "accuracy" in results["engine_a"]
        assert "accuracy" in results["engine_b"]

    def test_results_contain_learn_time(self):
        learn_texts = ["Some text"]
        qa_items = [{"question": "Q?", "answer": "A"}]
        results = self.harness.run_comparison(learn_texts, qa_items)
        assert "learn_time_s" in results["engine_a"]

    def test_accuracy_computed_correctly(self):
        qa_items = [
            {"question": "Capital of France?", "answer": "Paris"},
            {"question": "Capital of Germany?", "answer": "Berlin"},
        ]
        results = self.harness.run_comparison([], qa_items)
        assert results["engine_a"]["accuracy"] == 0.5
