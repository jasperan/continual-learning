import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterator

from continual_learning.model.tfidf_gate import TFIDFGate


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP matching Qwen2's architecture."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        if hidden_act == "silu":
            self.act_fn = F.silu
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DualMLP(nn.Module):
    """Dual-MLP module: frozen (original) + trainable (zero-init) with TF-IDF gating.

    Output: frozen_output + (1 - alpha) * gated_trainable_output

    Uses a residual connection so the frozen MLP output is always fully preserved.
    The trainable MLP adds new knowledge on top, scaled by (1 - alpha).
    When trainable weights are zero-initialized, the output equals frozen_output exactly.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        alpha_initial: float = 0.95,
        tfidf_threshold: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.alpha = alpha_initial

        self.frozen_mlp = SwiGLUMLP(hidden_size, intermediate_size, hidden_act)
        for param in self.frozen_mlp.parameters():
            param.requires_grad = False

        self.trainable_mlp = SwiGLUMLP(hidden_size, intermediate_size, hidden_act)
        for param in self.trainable_mlp.parameters():
            nn.init.normal_(param, mean=0.0, std=0.001)

        self.gate = TFIDFGate(hidden_size=hidden_size, threshold=tfidf_threshold)
        self._cached_input = None

    @classmethod
    def from_existing_mlp(
        cls,
        original_mlp: nn.Module,
        alpha_initial: float = 0.95,
        tfidf_threshold: float = 0.3,
    ) -> "DualMLP":
        hidden_size = original_mlp.gate_proj.in_features
        intermediate_size = original_mlp.gate_proj.out_features
        dual = cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            alpha_initial=alpha_initial,
            tfidf_threshold=tfidf_threshold,
        )
        dual.frozen_mlp.gate_proj.weight.data.copy_(original_mlp.gate_proj.weight.data)
        dual.frozen_mlp.up_proj.weight.data.copy_(original_mlp.up_proj.weight.data)
        dual.frozen_mlp.down_proj.weight.data.copy_(original_mlp.down_proj.weight.data)
        return dual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._cached_input = x.detach()

        frozen_out = self.frozen_mlp(x)
        trainable_out = self.trainable_mlp(x)

        mask = self.gate.compute_mask(trainable_out)
        mask = mask.to(trainable_out.device)
        gated_trainable_out = trainable_out * mask

        return frozen_out + (1.0 - self.alpha) * gated_trainable_out

    def get_trainable_params(self) -> Iterator[nn.Parameter]:
        return self.trainable_mlp.parameters()
