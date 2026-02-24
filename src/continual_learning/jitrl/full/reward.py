import torch
import torch.nn.functional as F


class RewardComputer:
    """Computes reward signals by comparing hidden states to stored knowledge."""

    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size

    def compute(self, query_hidden: torch.Tensor, knowledge_embeddings: torch.Tensor) -> torch.Tensor:
        if knowledge_embeddings.numel() == 0:
            return torch.zeros(self.hidden_size)

        query_mean = query_hidden.detach().mean(dim=(0, 1))
        knowledge_mean = knowledge_embeddings.detach().mean(dim=0).to(query_mean.device)

        query_norm = F.normalize(query_mean, dim=0)
        knowledge_norm = F.normalize(knowledge_mean, dim=0)
        reward = query_norm * knowledge_norm

        return reward


class LogitModulator:
    """Applies reward-based modulation to logits during generation."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def modulate(self, logits: torch.Tensor, reward: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        if torch.all(reward == 0):
            return logits

        reward_logits = reward.to(logits.device) @ projection.to(logits.device)
        return logits + self.temperature * reward_logits.unsqueeze(0)
