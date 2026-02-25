from unittest.mock import MagicMock
from continual_learning.ace.generator import Generator
from continual_learning.ace.playbook import Playbook


class TestGenerator:
    def setup_method(self):
        self.client = MagicMock()
        self.generator = Generator(client=self.client)

    def test_generate_answer_calls_client(self):
        self.client.generate.return_value = "Paris is the capital"
        pb = Playbook()
        result = self.generator.generate_answer(
            question="What is the capital of France?",
            document_context="France is a country. Paris is the capital of France.",
            playbook=pb,
        )
        assert result == "Paris is the capital"
        self.client.generate.assert_called_once()

    def test_generate_answer_includes_playbook_in_prompt(self):
        self.client.generate.return_value = "Answer"
        pb = Playbook()
        pb.add_strategy("Always quote from the document", source="test")
        self.generator.generate_answer(
            question="Test?",
            document_context="Some context",
            playbook=pb,
        )
        call_args = self.client.generate.call_args
        prompt = call_args[1].get("prompt", call_args[0][0] if call_args[0] else "")
        assert "Always quote from the document" in prompt

    def test_generate_answer_includes_document_context(self):
        self.client.generate.return_value = "Answer"
        pb = Playbook()
        self.generator.generate_answer(
            question="Test?",
            document_context="Oracle Cloud provides compute services.",
            playbook=pb,
        )
        call_args = self.client.generate.call_args
        prompt = call_args[1].get("prompt", call_args[0][0] if call_args[0] else "")
        assert "Oracle Cloud provides compute services" in prompt
