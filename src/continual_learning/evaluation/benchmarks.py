import torch
from typing import Optional

from continual_learning.evaluation.forgetting_metrics import compute_forgetting_ratio


def evaluate_qa_accuracy(
    model,
    tokenizer,
    qa_items: list[dict],
    max_new_tokens: int = 50,
    device: Optional[str] = None,
) -> float:
    """Evaluate the model's ability to answer questions without context.

    For each item, prompts the model with just the question (no context passage).
    Checks if the expected answer appears in the model's generated response.

    Returns accuracy as float between 0 and 1.
    """
    if not qa_items:
        return 0.0

    if device is None:
        device = next(model.parameters()).device

    correct = 0
    total = 0

    for item in qa_items:
        if not item["answer"]:
            continue

        prompt = f"Answer the following question concisely:\nQuestion: {item['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        if item["answer"].lower() in response.lower():
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def run_benchmark_suite(
    model,
    tokenizer,
    holdout_items: list[dict],
    before_score: Optional[float] = None,
) -> dict:
    """Run the full benchmark suite on holdout data.

    Returns dict with 'accuracy', 'forgetting_ratio' (if before_score provided),
    and 'num_evaluated'.
    """
    accuracy = evaluate_qa_accuracy(model, tokenizer, holdout_items)
    result = {
        "accuracy": accuracy,
        "num_evaluated": len([i for i in holdout_items if i["answer"]]),
    }
    if before_score is not None:
        result["forgetting_ratio"] = compute_forgetting_ratio(before_score, accuracy)
    return result
