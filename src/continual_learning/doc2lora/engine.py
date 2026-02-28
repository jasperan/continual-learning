import torch
from typing import Optional

from continual_learning.doc2lora.chunker import DocumentChunker
from continual_learning.doc2lora.hypernetwork import SimulatedHypernetwork
from continual_learning.doc2lora.lora_injector import LoRAInjector


class Doc2LoRAEngine:
    """Doc-to-LoRA / Text-to-LoRA unified engine.

    Uses a hypernetwork to generate LoRA adapters from documents (doc mode)
    or task descriptions (text mode) in a single forward pass.

    In 'doc' mode: documents are chunked, each chunk produces a rank-r LoRA,
    and chunks are composed by rank concatenation.

    In 'text' mode: the full task description is processed as a single chunk.
    """

    def __init__(
        self,
        model,
        tokenizer,
        hidden_dim: int = 2048,
        num_target_layers: int = 40,
        intermediate_dim: int = 512,
        lora_rank: int = 8,
        chunk_size: int = 1024,
        mode: str = "doc",
        simulated: bool = False,
        layer_prefix: str = "model.layers.{i}.mlp",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.mode = mode
        self._layer_prefix = layer_prefix

        self._chunker = DocumentChunker(chunk_size=chunk_size)
        self._injector = LoRAInjector()
        self._documents: list[str] = []

        if simulated:
            self._hypernetwork = SimulatedHypernetwork(
                hidden_dim=hidden_dim,
                num_layers=num_target_layers,
                lora_rank=lora_rank,
                intermediate_dim=intermediate_dim,
            )
        else:
            # TODO: Load real Perceiver hypernetwork from checkpoint
            self._hypernetwork = SimulatedHypernetwork(
                hidden_dim=hidden_dim,
                num_layers=num_target_layers,
                lora_rank=lora_rank,
                intermediate_dim=intermediate_dim,
            )

    def learn(self, text: str, callback: Optional[callable] = None) -> dict:
        """Ingest a document or task description via hypernetwork-generated LoRA.

        Args:
            text: Document text (doc mode) or task description (text mode).
            callback: Optional callback(**kwargs) called after LoRA injection.

        Returns:
            Dict with tokens_processed, method, num_chunks, effective_rank, mode.
        """
        self._documents.append(text)

        chunks = self._chunker.chunk(text, mode=self.mode)
        if not chunks:
            return {
                "tokens_processed": 0,
                "method": "doc2lora",
                "num_chunks": 0,
                "effective_rank": 0,
                "mode": self.mode,
            }

        lora_weights = self._hypernetwork.generate_lora(chunks)

        self._injector.inject(
            self.model, lora_weights, layer_prefix=self._layer_prefix
        )

        word_count = len(text.split())
        tokens_approx = int(word_count / 0.75)
        effective_rank = self._hypernetwork.lora_rank * len(chunks)

        if callback:
            callback(
                tokens_processed=tokens_approx,
                num_chunks=len(chunks),
                effective_rank=effective_rank,
            )

        return {
            "tokens_processed": tokens_approx,
            "method": "doc2lora",
            "num_chunks": len(chunks),
            "effective_rank": effective_rank,
            "mode": self.mode,
        }

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate a response using the LoRA-adapted model."""
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        return response

    def clear(self) -> None:
        """Reset all learned knowledge and remove injected LoRA."""
        self._documents.clear()
        if self._injector.is_injected:
            self._injector.remove(self.model)

    def set_mode(self, mode: str) -> None:
        """Switch between 'doc' and 'text' modes."""
        if mode not in ("doc", "text"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'doc' or 'text'.")
        self.mode = mode

    @property
    def num_documents(self) -> int:
        return len(self._documents)
