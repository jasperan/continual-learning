import os
from typing import Optional
from continual_learning.ace.ollama_client import OllamaClient
from continual_learning.ace.generator import Generator
from continual_learning.ace.reflector import Reflector
from continual_learning.ace.curator import Curator
from continual_learning.ace.playbook import Playbook
from continual_learning.text_utils import approx_tokens


class ACEEngine:
    """Agentic Context Engineering engine.

    Evolves a playbook through Generate-Reflect-Curate loops
    to improve document QA without any weight updates.
    """

    def __init__(
        self,
        ollama_model: str = "qwen2.5:7b",
        ollama_base_url: str = "http://localhost:11434",
        num_loops: int = 3,
        playbook_dir: str = "playbooks",
        max_strategies: int = 50,
    ):
        self.num_loops = num_loops
        self.playbook_dir = playbook_dir

        self._client = OllamaClient(base_url=ollama_base_url, model=ollama_model)
        self._generator = Generator(client=self._client)
        self._reflector = Reflector(client=self._client)
        self._curator = Curator(client=self._client)
        self._playbook = Playbook(max_strategies=max_strategies)
        self._documents: list[str] = []

    def learn(
        self,
        text: str,
        qa_pairs: list[dict] = None,
        callback: Optional[callable] = None,
    ) -> dict:
        """Ingest a document and optionally run ACE loops with QA pairs.

        Args:
            text: Document text to learn from.
            qa_pairs: Optional list of {"question": str, "answer": str} for ACE loops.
            callback: Optional progress callback(loop_num, assessment).

        Returns:
            Dict with tokens_processed, method, loops_completed, num_strategies.
        """
        self._documents.append(text)
        tokens_approx = approx_tokens(text)
        loops_completed = self.num_loops if qa_pairs else 0

        if qa_pairs:
            for loop_num in range(1, self.num_loops + 1):
                for qa in qa_pairs:
                    # Generate
                    answer = self._generator.generate_answer(
                        question=qa["question"],
                        document_context=text,
                        playbook=self._playbook,
                    )
                    # Reflect
                    feedback = self._reflector.reflect(
                        question=qa["question"],
                        generated_answer=answer,
                        expected_answer=qa.get("answer"),
                    )
                    # Curate
                    self._curator.curate(
                        playbook=self._playbook,
                        feedback=feedback,
                        loop_num=loop_num,
                    )

                    if callback:
                        callback(loop_num, feedback.get("assessment", "unknown"))

        self._playbook.stats["documents_processed"] = len(self._documents)

        return {
            "tokens_processed": tokens_approx,
            "method": "ace",
            "loops_completed": loops_completed,
            "num_strategies": len(self._playbook.strategies),
        }

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate a response using the evolved playbook."""
        context = "\n\n---\n\n".join(self._documents) if self._documents else ""
        return self._generator.generate_answer(
            question=prompt,
            document_context=context,
            playbook=self._playbook,
        )

    def clear(self) -> None:
        """Reset all documents and the playbook."""
        self._documents.clear()
        self._playbook = Playbook(max_strategies=self._playbook.max_strategies)

    @property
    def num_documents(self) -> int:
        """Number of documents currently stored."""
        return len(self._documents)

    def save_playbook(self, name: str) -> str:
        """Save the current playbook to disk.

        Args:
            name: Name for the playbook file (without extension).

        Returns:
            Path to the saved file.
        """
        path = os.path.join(self.playbook_dir, f"{name}.json")
        self._playbook.save(path)
        return path

    def load_playbook(self, name: str) -> None:
        """Load a playbook from disk.

        Args:
            name: Name of the playbook file (without extension).
        """
        path = os.path.join(self.playbook_dir, f"{name}.json")
        self._playbook = Playbook.load(path)
