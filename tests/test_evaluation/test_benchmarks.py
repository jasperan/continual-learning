import pytest
from continual_learning.evaluation.forgetting_metrics import compute_forgetting_ratio


class TestForgettingMetrics:
    def test_no_forgetting(self):
        ratio = compute_forgetting_ratio(before=0.8, after=0.8)
        assert ratio == 0.0

    def test_full_forgetting(self):
        ratio = compute_forgetting_ratio(before=0.8, after=0.0)
        assert ratio == 1.0

    def test_partial_forgetting(self):
        ratio = compute_forgetting_ratio(before=1.0, after=0.89)
        assert abs(ratio - 0.11) < 0.001

    def test_improvement_returns_negative(self):
        ratio = compute_forgetting_ratio(before=0.5, after=0.7)
        assert ratio < 0

    def test_zero_before_returns_zero(self):
        ratio = compute_forgetting_ratio(before=0.0, after=0.5)
        assert ratio == 0.0
