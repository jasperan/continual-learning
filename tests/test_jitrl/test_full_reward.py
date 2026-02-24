import torch
import pytest
from continual_learning.jitrl.full.reward import RewardComputer, LogitModulator


class TestRewardComputer:
    def setup_method(self):
        self.hidden_size = 64
        self.computer = RewardComputer(hidden_size=self.hidden_size)

    def test_compute_reward_shape(self):
        query_hidden = torch.randn(1, 10, self.hidden_size)
        knowledge = torch.randn(3, self.hidden_size)
        reward = self.computer.compute(query_hidden, knowledge)
        assert reward.shape == (self.hidden_size,)

    def test_reward_higher_for_similar_input(self):
        knowledge = torch.randn(1, self.hidden_size)
        similar = knowledge.unsqueeze(0).expand(1, 5, -1) + torch.randn(1, 5, self.hidden_size) * 0.01
        different = torch.randn(1, 5, self.hidden_size)
        reward_similar = self.computer.compute(similar, knowledge)
        reward_different = self.computer.compute(different, knowledge)
        assert reward_similar.sum() > reward_different.sum()

    def test_reward_no_knowledge_returns_zeros(self):
        query = torch.randn(1, 5, self.hidden_size)
        knowledge = torch.zeros(0, self.hidden_size)
        reward = self.computer.compute(query, knowledge)
        assert torch.all(reward == 0)


class TestLogitModulator:
    def setup_method(self):
        self.modulator = LogitModulator(temperature=1.0)

    def test_modulate_shape(self):
        logits = torch.randn(1, 100)
        reward = torch.randn(64)
        projection = torch.randn(64, 100)
        result = self.modulator.modulate(logits, reward, projection)
        assert result.shape == logits.shape

    def test_modulate_changes_logits(self):
        logits = torch.randn(1, 100)
        reward = torch.ones(64) * 0.5
        projection = torch.randn(64, 100)
        result = self.modulator.modulate(logits, reward, projection)
        assert not torch.allclose(result, logits)

    def test_zero_reward_preserves_logits(self):
        logits = torch.randn(1, 100)
        reward = torch.zeros(64)
        projection = torch.randn(64, 100)
        result = self.modulator.modulate(logits, reward, projection)
        assert torch.allclose(result, logits)

    def test_temperature_scales_effect(self):
        logits = torch.randn(1, 100)
        reward = torch.ones(64) * 0.5
        projection = torch.randn(64, 100)
        weak = LogitModulator(temperature=0.1).modulate(logits, reward, projection)
        strong = LogitModulator(temperature=2.0).modulate(logits, reward, projection)
        weak_diff = (weak - logits).abs().sum()
        strong_diff = (strong - logits).abs().sum()
        assert strong_diff > weak_diff
