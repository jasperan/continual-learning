import torch
import torch.nn as nn
from typing import Optional


class TFIDFGate(nn.Module):
    """TF-IDF based gradient gating for Sparse Memory Finetuning.

    Computes per-neuron importance scores using TF-IDF logic:
    - Term Frequency: how strongly a neuron activates on the current (new) input
    - Inverse Document Frequency: inverse of how often a neuron activates across
      a general calibration corpus

    Neurons with high TF-IDF are specialized to the new content (gradients pass).
    Neurons with low TF-IDF are general-purpose (gradients masked to zero).
    """

    def __init__(self, hidden_size: int, threshold: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.idf_scores: Optional[torch.Tensor] = None
        self._doc_freq: Optional[torch.Tensor] = None
        self._num_docs: int = 0

    def calibrate(self, activations: list[torch.Tensor]) -> None:
        """Compute IDF scores from a list of activation tensors.
        Each tensor shape: (batch, seq_len, hidden_size).
        """
        self._num_docs = len(activations)
        doc_freq = torch.zeros(self.hidden_size)
        for act in activations:
            neuron_mean = act.abs().mean(dim=(0, 1))
            global_mean = neuron_mean.mean()
            doc_freq += (neuron_mean > global_mean).float()
        self._doc_freq = doc_freq
        self.idf_scores = torch.log(
            torch.tensor(self._num_docs, dtype=torch.float32)
            / (1.0 + self._doc_freq)
        )

    def compute_mask(self, activation: torch.Tensor) -> torch.Tensor:
        """Compute binary gradient mask. 1 = gradient passes, 0 = masked."""
        if self.idf_scores is None:
            return torch.ones(self.hidden_size)

        tf = activation.abs().mean(dim=(0, 1))
        tf_max = tf.max()
        if tf_max > 0:
            tf = tf / tf_max

        idf_device = self.idf_scores.to(tf.device)
        tfidf = tf * idf_device

        tfidf_max = tfidf.max()
        if tfidf_max > 0:
            tfidf = tfidf / tfidf_max

        mask = (tfidf >= self.threshold).float()
        return mask

    def state_dict_custom(self) -> dict:
        return {
            "idf_scores": self.idf_scores,
            "doc_freq": self._doc_freq,
            "num_docs": self._num_docs,
            "threshold": self.threshold,
        }

    def load_state_dict_custom(self, state: dict) -> None:
        self.idf_scores = state["idf_scores"]
        self._doc_freq = state["doc_freq"]
        self._num_docs = state["num_docs"]
        self.threshold = state["threshold"]
