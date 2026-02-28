import hashlib
import torch
import torch.nn as nn


class PerceiverCrossAttentionBlock(nn.Module):
    """Single cross-attention block: latents (queries) attend to activations (keys/values)."""

    def __init__(self, latent_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Cross-attention: latents attend to context (activations)
        attended, _ = self.cross_attn(self.norm1(latents), context, context)
        latents = latents + attended
        # FFN
        latents = latents + self.ffn(self.norm2(latents))
        return latents


class PerceiverHypernetwork(nn.Module):
    """Perceiver-based hypernetwork that generates LoRA adapters from frozen LLM activations.

    Architecture:
    1. Input projection: maps per-layer activations to a common latent space
    2. Learned latent queries (one per target layer x 2 for A and B matrices)
    3. N cross-attention blocks: latents attend to projected activations
    4. Output heads: project latents to LoRA A and B matrices
    """

    def __init__(
        self,
        hidden_dim: int = 640,
        num_layers: int = 18,
        intermediate_dim: int = 2048,
        lora_rank: int = 8,
        latent_dim: int = 256,
        num_cross_attn_blocks: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.intermediate_dim = intermediate_dim
        self.lora_rank = lora_rank
        self.latent_dim = latent_dim

        # Input projection: activations -> latent space
        self.input_proj = nn.Linear(hidden_dim, latent_dim)

        # Learned latent queries: num_layers for A matrices + num_layers for B matrices
        self.latent_queries = nn.Parameter(
            torch.randn(num_layers * 2, latent_dim)
        )

        # Cross-attention blocks
        self.cross_attn_blocks = nn.ModuleList([
            PerceiverCrossAttentionBlock(latent_dim, num_heads, dropout)
            for _ in range(num_cross_attn_blocks)
        ])

        # Output heads
        self.output_head_A = nn.Linear(latent_dim, hidden_dim * lora_rank)
        self.output_head_B = nn.Linear(latent_dim, lora_rank * intermediate_dim)

    def forward(self, activations: torch.Tensor) -> dict[str, torch.Tensor]:
        """Generate LoRA weights from frozen model activations.

        Args:
            activations: Tensor of shape (batch, seq_len, hidden_dim) --
                         mean-pooled or last-token activations from frozen model.
                         Can also be (seq_len, hidden_dim) for single-sample.

        Returns:
            Dict mapping "layer_{i}_A" and "layer_{i}_B" to weight tensors.
        """
        # Handle single-sample case
        if activations.dim() == 2:
            activations = activations.unsqueeze(0)

        # Project activations to latent space
        context = self.input_proj(activations)  # (batch, seq_len, latent_dim)

        # Expand latent queries for batch
        batch_size = context.shape[0]
        latents = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Run through cross-attention blocks
        for block in self.cross_attn_blocks:
            latents = block(latents, context)

        # Split latents into A and B groups
        # latents shape: (batch, num_layers * 2, latent_dim)
        # First num_layers are for A matrices, next num_layers for B
        latents_A = latents[:, :self.num_layers, :]  # (batch, num_layers, latent_dim)
        latents_B = latents[:, self.num_layers:, :]

        # Generate LoRA matrices
        weights = {}
        for i in range(self.num_layers):
            A_flat = self.output_head_A(latents_A[:, i, :])  # (batch, hidden_dim * lora_rank)
            B_flat = self.output_head_B(latents_B[:, i, :])  # (batch, lora_rank * intermediate_dim)

            # Reshape and scale down
            A = A_flat.view(-1, self.hidden_dim, self.lora_rank) * 0.01
            B = B_flat.view(-1, self.lora_rank, self.intermediate_dim) * 0.01

            # Squeeze batch dim if single sample
            if batch_size == 1:
                A = A.squeeze(0)
                B = B.squeeze(0)

            weights[f"layer_{i}_A"] = A
            weights[f"layer_{i}_B"] = B

        return weights

    def generate_lora(
        self,
        chunks: list[str],
        tokenizer=None,
        base_model=None,
    ) -> dict[str, torch.Tensor]:
        """Generate LoRA weights from text chunks using the frozen base model.

        This is the high-level API matching SimulatedHypernetwork.
        Requires tokenizer and base_model to extract activations.
        """
        if tokenizer is None or base_model is None:
            raise ValueError(
                "PerceiverHypernetwork.generate_lora requires tokenizer and base_model"
            )

        device = next(self.parameters()).device
        all_activations = []

        for chunk in chunks:
            inputs = tokenizer(
                chunk, return_tensors="pt", truncation=True, max_length=512
            )
            input_ids = inputs["input_ids"].to(device)

            with torch.no_grad():
                outputs = base_model(input_ids, output_hidden_states=True)
                # Mean-pool the last hidden state
                hidden = outputs.hidden_states[-1].mean(dim=1)  # (1, hidden_dim)
                all_activations.append(hidden)

        # Stack activations as sequence
        activations = torch.cat(all_activations, dim=0).unsqueeze(0)  # (1, num_chunks, hidden_dim)

        return self.forward(activations)


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
