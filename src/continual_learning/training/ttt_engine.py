import torch
import torch.nn as nn
from typing import Optional

from continual_learning.model.dual_mlp import DualMLP


class TTTEngine:
    """Test-Time Training End-to-End (TTT-E2E) engine.

    Performs mini-batch gradient descent on incoming documents at inference time,
    updating only the trainable MLP weights in the dual-MLP layers.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        dual_mlps: list[DualMLP],
        learning_rate: float = 1e-5,
        mini_batch_size: int = 32,
        gradient_steps: int = 1,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dual_mlps = dual_mlps
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.gradient_steps = gradient_steps
        self.max_tokens = max_tokens

        # Optimizer only over trainable params
        trainable_params = []
        for dual in self.dual_mlps:
            trainable_params.extend(dual.get_trainable_params())
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)

    def learn(
        self,
        text: str,
        callback: Optional[callable] = None,
    ) -> dict:
        """Run TTT-E2E inner loop on the given text.

        Tokenizes the text, splits into mini-batches, and performs gradient
        descent to update the trainable MLP weights.

        Args:
            text: The document text to learn from.
            callback: Optional function called after each mini-batch with
                      (batch_idx, loss, tokens_so_far) for progress reporting.

        Returns:
            Dict with 'losses' (list of floats) and 'tokens_processed' (int).
        """
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
        )
        input_ids = encoded["input_ids"]

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        total_tokens = input_ids.shape[1]
        losses = []
        tokens_processed = 0

        for start in range(0, total_tokens - 1, self.mini_batch_size):
            end = min(start + self.mini_batch_size + 1, total_tokens)
            batch_ids = input_ids[:, start:end]

            if batch_ids.shape[1] < 2:
                break

            labels = batch_ids.clone()

            for step in range(self.gradient_steps):
                self.optimizer.zero_grad()

                outputs = self.model(input_ids=batch_ids, labels=labels)
                loss = outputs.loss

                if loss is not None:
                    loss.backward()

                    # Apply TF-IDF masking to gradients
                    for dual in self.dual_mlps:
                        if dual.gate.idf_scores is not None:
                            mask = dual.gate.compute_mask(
                                torch.zeros(1, 1, dual.hidden_size).to(device)
                            ).to(device)
                            for param in dual.trainable_mlp.parameters():
                                if param.grad is not None:
                                    if param.grad.dim() == 2:
                                        if param.grad.shape[0] == dual.hidden_size:
                                            param.grad *= mask.unsqueeze(1)
                                        elif param.grad.shape[1] == dual.hidden_size:
                                            param.grad *= mask.unsqueeze(0)

                    self.optimizer.step()
                    losses.append(loss.item())

            tokens_processed += batch_ids.shape[1] - 1

            if callback:
                callback(
                    batch_idx=start // self.mini_batch_size,
                    loss=losses[-1] if losses else 0.0,
                    tokens_so_far=tokens_processed,
                )

        return {
            "losses": losses,
            "tokens_processed": tokens_processed,
        }
