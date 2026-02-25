from typing import Optional
import torch.nn as nn
from continual_learning.jitrl.base import BaseJitRLEngine
from continual_learning.ace.engine import ACEEngine


class ACEAdapter(BaseJitRLEngine):
    """Wraps ACEEngine to conform to BaseJitRLEngine interface for ComparisonHarness."""

    def __init__(self, ace_engine: ACEEngine, model: nn.Module, tokenizer):
        super().__init__(model, tokenizer)
        self._ace = ace_engine

    def learn(self, text: str, callback: Optional[callable] = None) -> dict:
        return self._ace.learn(text, callback=callback)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        return self._ace.generate(prompt, max_new_tokens=max_new_tokens)

    def clear(self) -> None:
        super().clear()
        self._ace.clear()

    @property
    def num_documents(self) -> int:
        return self._ace.num_documents
