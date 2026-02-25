import json
from unittest.mock import MagicMock
from continual_learning.ace.reflector import Reflector


class TestReflector:
    def setup_method(self):
        self.client = MagicMock()
        self.reflector = Reflector(client=self.client)

    def test_reflect_returns_parsed_feedback(self):
        feedback = {
            "assessment": "partially_correct",
            "strengths": ["Found the right document section"],
            "weaknesses": ["Missed the specific date"],
            "suggested_strategies": ["Look for temporal markers when questions ask about dates"],
        }
        self.client.generate.return_value = json.dumps(feedback)
        result = self.reflector.reflect(
            question="When was Oracle founded?",
            generated_answer="Oracle is a database company",
            expected_answer="1977",
        )
        assert result["assessment"] == "partially_correct"
        assert len(result["suggested_strategies"]) == 1

    def test_reflect_handles_malformed_json(self):
        self.client.generate.return_value = "This is not JSON at all"
        result = self.reflector.reflect(
            question="Test?",
            generated_answer="Answer",
            expected_answer="Expected",
        )
        assert result["assessment"] == "unknown"
        assert "raw_response" in result

    def test_reflect_without_expected_answer(self):
        feedback = {
            "assessment": "unknown",
            "strengths": [],
            "weaknesses": [],
            "suggested_strategies": ["Provide more specific answers"],
        }
        self.client.generate.return_value = json.dumps(feedback)
        result = self.reflector.reflect(
            question="What is AI?",
            generated_answer="AI is artificial intelligence",
            expected_answer=None,
        )
        assert "suggested_strategies" in result
