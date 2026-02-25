from continual_learning.ace.ollama_client import OllamaClient
from continual_learning.ace.playbook import Playbook


class Generator:
    """Generator role: answers questions using document context and playbook strategies."""

    def __init__(self, client: OllamaClient):
        self.client = client

    def generate_answer(self, question: str, document_context: str, playbook: Playbook) -> str:
        playbook_text = playbook.render()
        prompt = (
            f"Document:\n{document_context}\n\n"
            f"Playbook:\n{playbook_text}\n\n"
            f"Question: {question}\n"
            f"Answer concisely based on the document above:"
        )
        return self.client.generate(prompt=prompt)
