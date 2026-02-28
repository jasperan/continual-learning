import hashlib
import torch


class SimulatedHypernetwork:
    """Simulated hypernetwork that generates LoRA weights from text chunks.

    Uses a deterministic hash-seeded RNG to produce small random LoRA matrices.
    This enables full pipeline testing without a real pretrained checkpoint.

    When a real Perceiver checkpoint is available, replace this class with
    PerceiverHypernetwork that loads and runs the actual model.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_layers: int = 40,
        lora_rank: int = 8,
        intermediate_dim: int = 512,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lora_rank = lora_rank
        self.intermediate_dim = intermediate_dim

    def _text_to_seed(self, text: str) -> int:
        return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)

    def generate_lora(self, chunks: list[str]) -> dict[str, torch.Tensor]:
        """Generate LoRA weight matrices from text chunks.

        For multiple chunks, LoRA matrices are composed by concatenating
        along the rank dimension (effective rank = lora_rank * num_chunks).

        Args:
            chunks: List of text chunks to generate LoRA weights for.

        Returns:
            Dict mapping "layer_{i}_A" and "layer_{i}_B" to weight tensors.
            A shape: (hidden_dim, effective_rank)
            B shape: (effective_rank, intermediate_dim)
        """
        all_A = {i: [] for i in range(self.num_layers)}
        all_B = {i: [] for i in range(self.num_layers)}

        for chunk in chunks:
            seed = self._text_to_seed(chunk)
            gen = torch.Generator().manual_seed(seed)

            for layer_idx in range(self.num_layers):
                A = torch.randn(
                    self.hidden_dim, self.lora_rank, generator=gen
                ) * 0.01
                B = torch.randn(
                    self.lora_rank, self.intermediate_dim, generator=gen
                ) * 0.01
                all_A[layer_idx].append(A)
                all_B[layer_idx].append(B)

        weights = {}
        for layer_idx in range(self.num_layers):
            weights[f"layer_{layer_idx}_A"] = torch.cat(all_A[layer_idx], dim=1)
            weights[f"layer_{layer_idx}_B"] = torch.cat(all_B[layer_idx], dim=0)

        return weights
