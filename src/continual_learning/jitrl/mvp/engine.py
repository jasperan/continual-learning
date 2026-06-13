import torch
from typing import Optional
from transformers import LogitsProcessor, LogitsProcessorList

from continual_learning.jitrl.base import BaseJitRLEngine
from continual_learning.jitrl.mvp.retriever import TFIDFRetriever
from continual_learning.jitrl.mvp.logit_bias import LogitBiaser
from continual_learning.text_utils import approx_tokens


class _BiasLogitsProcessor(LogitsProcessor):
    def __init__(self, bias: torch.Tensor):
        self.bias = bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores + self.bias.to(scores.device)


class JitRLMVPEngine(BaseJitRLEngine):
    """JitRL MVP: Retrieval-Augmented Generation with logit biasing."""

    def __init__(self, model, tokenizer, top_k: int = 3, bias_strength: float = 2.0,
                 max_chunk_words: int = 300, max_context_tokens: int = 1024):
        super().__init__(model, tokenizer)
        self.top_k = top_k
        self.max_chunk_words = max_chunk_words
        self.max_context_tokens = max_context_tokens
        self._retriever = TFIDFRetriever()
        self._biaser = LogitBiaser(tokenizer=tokenizer, bias_strength=bias_strength)

    def learn(self, text: str, callback: Optional[callable] = None) -> dict:
        self._documents.append(text)
        num_chunks = self._retriever.add_document(text, max_chunk_words=self.max_chunk_words)
        tokens_approx = approx_tokens(text)
        if callback:
            callback(batch_idx=0, loss=0.0, tokens_so_far=tokens_approx)
        return {"tokens_processed": tokens_approx, "chunks_added": num_chunks, "method": "jitrl_mvp"}

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        chunks = self._retriever.retrieve(prompt, top_k=self.top_k)
        if chunks:
            context = "\n\n".join(chunks)
            augmented = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
        else:
            augmented = prompt

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(augmented, return_tensors="pt", truncation=True, max_length=self.max_context_tokens)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        vocab_size = self.model.config.vocab_size
        bias = self._biaser.compute_bias(chunks, vocab_size)

        processors = LogitsProcessorList()
        if chunks and bias.any():
            processors.append(_BiasLogitsProcessor(bias))

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                logits_processor=processors if processors else None,
            )

        response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response

    def clear(self) -> None:
        super().clear()
        self._retriever.clear()
