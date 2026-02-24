import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from continual_learning.model.dual_mlp import DualMLP


def get_modified_layer_indices(num_layers: int, start: int, end: int) -> list[int]:
    """Return the list of layer indices that should receive dual-MLP injection."""
    return list(range(start, end))


def inject_dual_mlps(
    model: nn.Module,
    layer_indices: list[int],
    alpha_initial: float = 1.0,
    tfidf_threshold: float = 0.3,
) -> None:
    """Replace MLP modules in specified layers with DualMLP modules.
    Freezes all layers first, then injects DualMLP into target layers.
    """
    layers = model.model.layers

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Inject DualMLP into target layers
    for idx in layer_indices:
        original_mlp = layers[idx].mlp
        dual_mlp = DualMLP.from_existing_mlp(
            original_mlp,
            alpha_initial=alpha_initial,
            tfidf_threshold=tfidf_threshold,
        )
        layers[idx].mlp = dual_mlp


def load_modified_model(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    device: str = "auto",
    start_layer: int = 21,
    end_layer: int = 28,
    alpha_initial: float = 1.0,
    tfidf_threshold: float = 0.3,
) -> tuple:
    """Load the base Qwen model and inject dual-MLP layers.
    Returns (model, tokenizer) tuple.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    num_layers = len(model.model.layers)
    layer_indices = get_modified_layer_indices(num_layers, start_layer, end_layer)
    inject_dual_mlps(model, layer_indices, alpha_initial, tfidf_threshold)

    return model, tokenizer
