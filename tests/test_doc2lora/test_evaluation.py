from unittest.mock import MagicMock
from continual_learning.doc2lora.evaluation import Doc2LoRAEvaluator


def _make_mock_engine(answers: list[str] = None):
    """Create a mock engine that returns predefined answers."""
    engine = MagicMock()
    if answers is None:
        answers = ["Oracle was founded in 1977"]

    call_count = [0]
    def mock_generate(prompt, **kwargs):
        idx = call_count[0] % len(answers)
        call_count[0] += 1
        return answers[idx]

    engine.generate.side_effect = mock_generate
    engine.learn.return_value = {"tokens_processed": 100, "method": "doc2lora"}
    engine.clear.return_value = None
    return engine


class TestDoc2LoRAEvaluator:
    def test_evaluate_qa_all_correct(self):
        engine = _make_mock_engine(["1977"])
        evaluator = Doc2LoRAEvaluator(engine)

        results = evaluator.evaluate_qa([
            {"context": "Oracle was founded in 1977.", "question": "When?", "answer": "1977"},
        ])
        assert results["accuracy"] == 1.0
        assert results["correct"] == 1
        assert results["total"] == 1

    def test_evaluate_qa_partial_correct(self):
        engine = _make_mock_engine(["1977", "I don't know"])
        evaluator = Doc2LoRAEvaluator(engine)

        results = evaluator.evaluate_qa([
            {"context": "Oracle founded 1977.", "question": "When?", "answer": "1977"},
            {"context": "Google founded 1998.", "question": "When?", "answer": "1998"},
        ])
        assert results["accuracy"] == 0.5
        assert results["correct"] == 1

    def test_evaluate_qa_empty_list(self):
        engine = _make_mock_engine()
        evaluator = Doc2LoRAEvaluator(engine)
        results = evaluator.evaluate_qa([])
        assert results["accuracy"] == 0.0
        assert results["total"] == 0

    def test_evaluate_qa_case_insensitive(self):
        engine = _make_mock_engine(["ORACLE was founded in 1977"])
        evaluator = Doc2LoRAEvaluator(engine)
        results = evaluator.evaluate_qa([
            {"context": "c", "question": "q", "answer": "oracle"},
        ])
        assert results["accuracy"] == 1.0

    def test_evaluate_needle_found(self):
        engine = _make_mock_engine(["The secret code is 42."])
        evaluator = Doc2LoRAEvaluator(engine)

        result = evaluator.evaluate_needle_in_haystack(
            needle_fact="The secret code is 42.",
            haystack_texts=["Paragraph one about nothing.", "Paragraph two about nothing."],
            question="What is the secret code?",
            expected_answer="42",
        )
        assert result["found"]
        assert result["total_paragraphs"] == 3

    def test_evaluate_needle_not_found(self):
        engine = _make_mock_engine(["I have no idea"])
        evaluator = Doc2LoRAEvaluator(engine)

        result = evaluator.evaluate_needle_in_haystack(
            needle_fact="The secret code is 42.",
            haystack_texts=["Filler text."],
            question="What is the secret code?",
            expected_answer="42",
        )
        assert not result["found"]

    def test_evaluate_forgetting_no_forgetting(self):
        engine = _make_mock_engine(["Paris is the capital"])
        evaluator = Doc2LoRAEvaluator(engine)

        result = evaluator.evaluate_forgetting(
            baseline_prompts=[
                {"prompt": "Capital of France?", "expected": "Paris"},
            ],
            document="Oracle was founded in 1977.",
        )
        assert result["passed"]
        assert result["forgetting_ratio"] == 0.0

    def test_evaluate_forgetting_with_forgetting(self):
        # First call returns correct answer (baseline), second returns wrong (after learning)
        engine = _make_mock_engine(["Paris", "I forgot"])
        evaluator = Doc2LoRAEvaluator(engine)

        result = evaluator.evaluate_forgetting(
            baseline_prompts=[
                {"prompt": "Capital of France?", "expected": "Paris"},
            ],
            document="Some unrelated document.",
        )
        assert result["forgetting_ratio"] == 1.0
        assert not result["passed"]

    def test_evaluate_qa_clears_between_items(self):
        engine = _make_mock_engine(["answer"])
        evaluator = Doc2LoRAEvaluator(engine)

        evaluator.evaluate_qa([
            {"context": "c1", "question": "q1", "answer": "answer"},
            {"context": "c2", "question": "q2", "answer": "answer"},
        ])
        # clear should be called once per item
        assert engine.clear.call_count == 2
