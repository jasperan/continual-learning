import torch
from collections import Counter


class LogitBiaser:
    """Computes logit bias vectors from retrieved document chunks."""

    def __init__(self, tokenizer, bias_strength: float = 2.0):
        self.tokenizer = tokenizer
        self.bias_strength = bias_strength

    def compute_bias(self, chunks: list[str], vocab_size: int) -> torch.Tensor:
        if not chunks:
            return torch.zeros(vocab_size)

        token_counts: Counter = Counter()
        for chunk in chunks:
            token_ids = self.tokenizer.encode(chunk, add_special_tokens=False)
            token_counts.update(token_ids)

        bias = torch.zeros(vocab_size)
        if not token_counts:
            return bias

        max_count = max(token_counts.values())
        for tid, count in token_counts.items():
            if 0 <= tid < vocab_size:
                bias[tid] = self.bias_strength * (count / max_count)

        return bias
