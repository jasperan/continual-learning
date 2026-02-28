import re
import torch
import torch.nn as nn


class LoRAInjector:
    """Injects and removes LoRA weight deltas from model linear layers.

    Applies LoRA as a direct weight modification: W' = W + A @ B.
    Stores original weights for clean removal.
    """

    def __init__(self):
        self._original_weights: dict[str, torch.Tensor] = {}
        self.is_injected: bool = False

    def inject(
        self,
        model: nn.Module,
        lora_weights: dict[str, torch.Tensor],
        layer_prefix: str = "layer_{i}_mlp",
    ) -> None:
        """Inject LoRA weights into model layers.

        Args:
            model: The model to modify.
            lora_weights: Dict with "layer_{i}_A" and "layer_{i}_B" keys.
            layer_prefix: Pattern for finding target modules. {i} is replaced
                with layer index.
        """
        if self.is_injected:
            self.remove(model)

        layer_indices = set()
        for key in lora_weights:
            match = re.match(r"layer_(\d+)_[AB]", key)
            if match:
                layer_indices.add(int(match.group(1)))

        for idx in sorted(layer_indices):
            module_name = layer_prefix.replace("{i}", str(idx))
            A = lora_weights[f"layer_{idx}_A"]
            B = lora_weights[f"layer_{idx}_B"]

            module = dict(model.named_modules()).get(module_name)
            if module is None:
                continue
            if not isinstance(module, nn.Linear):
                continue

            self._original_weights[module_name] = module.weight.data.clone()

            delta = (A @ B).T  # (out_features, in_features)
            device = module.weight.device
            module.weight.data += delta.to(device)

        self.is_injected = True

    def remove(self, model: nn.Module) -> None:
        """Remove injected LoRA weights, restoring originals."""
        for module_name, original in self._original_weights.items():
            module = dict(model.named_modules()).get(module_name)
            if module is not None:
                module.weight.data = original

        self._original_weights.clear()
        self.is_injected = False
