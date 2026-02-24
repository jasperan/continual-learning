import torch
from continual_learning.model.dual_mlp import DualMLP


def collect_calibration_activations(
    model,
    tokenizer,
    texts: list[str],
    dual_mlps: list[DualMLP],
    max_tokens: int = 512,
) -> dict[int, list[torch.Tensor]]:
    """Run calibration texts through the model and collect DualMLP input activations.

    Returns a dict mapping DualMLP index to list of cached input activations.
    Each activation has shape (batch, seq_len, hidden_size).
    """
    activations = {i: [] for i in range(len(dual_mlps))}

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_tokens,
            )
            input_ids = encoded["input_ids"].to(device)
            model(input_ids=input_ids)

            for i, dual in enumerate(dual_mlps):
                if dual._cached_input is not None:
                    activations[i].append(dual._cached_input.cpu())

    return activations


def calibrate_gates(
    model,
    tokenizer,
    texts: list[str],
    dual_mlps: list[DualMLP],
    max_tokens: int = 512,
) -> None:
    """Calibrate all TF-IDF gates using real model activations on calibration text."""
    activations = collect_calibration_activations(
        model, tokenizer, texts, dual_mlps, max_tokens
    )

    for i, dual in enumerate(dual_mlps):
        if activations[i]:
            dual.gate.calibrate(activations[i])
