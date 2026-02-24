import random
from datasets import load_dataset


def load_squad_splits(
    learn_size: int = 250,
    holdout_size: int = 250,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Load SQuAD 2.0 and split into learn and holdout sets.

    Each item is a dict with 'context', 'question', and 'answer'.
    Learn set: passages the model will learn via TTT-E2E.
    Holdout set: passages for measuring catastrophic forgetting.
    """
    dataset = load_dataset("rajpurkar/squad_v2", split="validation")

    context_groups = {}
    for item in dataset:
        ctx = item["context"]
        if ctx not in context_groups:
            context_groups[ctx] = []
        answer_text = ""
        if item["answers"]["text"]:
            answer_text = item["answers"]["text"][0]
        context_groups[ctx].append({
            "context": ctx,
            "question": item["question"],
            "answer": answer_text,
        })

    rng = random.Random(seed)
    contexts = list(context_groups.keys())
    rng.shuffle(contexts)

    learn_items = []
    holdout_items = []

    for ctx in contexts:
        items = context_groups[ctx]
        if len(learn_items) < learn_size:
            learn_items.extend(items)
        elif len(holdout_items) < holdout_size:
            holdout_items.extend(items)
        else:
            break

    return learn_items[:learn_size], holdout_items[:holdout_size]
