from continual_learning.ace.ollama_client import OllamaClient
from continual_learning.text_utils import parse_json_block


class Reflector:
    """Reflector role: critiques generated answers and suggests strategy improvements."""

    SYSTEM_PROMPT = (
        "You are a reflection agent. Given a question, a generated answer, and optionally the expected answer, "
        "produce a JSON object with these fields:\n"
        '- "assessment": one of "correct", "partially_correct", "incorrect", "unknown"\n'
        '- "strengths": list of strings describing what went well\n'
        '- "weaknesses": list of strings describing what went wrong\n'
        '- "suggested_strategies": list of strings — new rules or strategies to improve future answers\n'
        "Respond with ONLY valid JSON, no other text."
    )

    def __init__(self, client: OllamaClient):
        self.client = client

    def reflect(self, question: str, generated_answer: str, expected_answer: str = None) -> dict:
        prompt = f"Question: {question}\nGenerated Answer: {generated_answer}\n"
        if expected_answer:
            prompt += f"Expected Answer: {expected_answer}\n"
        else:
            prompt += "Expected Answer: (not provided — assess based on quality and relevance)\n"
        prompt += "\nProvide your reflection as JSON:"

        raw = self.client.generate(prompt=prompt, system=self.SYSTEM_PROMPT, temperature=0.3)
        return self._parse_feedback(raw)

    def _parse_feedback(self, raw: str) -> dict:
        parsed = parse_json_block(raw)
        if parsed is None:
            return {
                "assessment": "unknown",
                "strengths": [],
                "weaknesses": [],
                "suggested_strategies": [],
                "raw_response": raw,
            }
        return parsed
