def compute_forgetting_ratio(before: float, after: float) -> float:
    """Compute the catastrophic forgetting ratio.

    Returns (before - after) / before.
    0 = no forgetting. 1 = total forgetting. Negative = improvement.
    """
    if before == 0.0:
        return 0.0
    return (before - after) / before
