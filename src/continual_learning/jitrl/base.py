from abc import ABC, abstractmethod
from typing import Optional
import torch.nn as nn


class BaseJitRLEngine(ABC):
    """Common interface for all JitRL engines."""

    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._documents: list[str] = []

    @abstractmethod
    def learn(self, text: str, callback: Optional[callable] = None) -> dict:
        """Ingest a document (no weight updates).
        Returns dict with at least 'tokens_processed' and 'method' keys.
        """

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate a response, incorporating learned knowledge."""

    def clear(self) -> None:
        """Reset all learned knowledge."""
        self._documents.clear()

    @property
    def num_documents(self) -> int:
        return len(self._documents)
