import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional
import time


class HypernetworkTrainer:
    """Teacher-student trainer for hypernetwork meta-training.

    The hypernetwork learns to generate LoRA weights that internalize
    document knowledge, so the adapted model can answer questions
    without seeing the original document in context.
    """

    def __init__(
        self,
        hypernetwork: nn.Module,
        base_model: nn.Module,
        tokenizer,
        lora_injector,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_steps: int = 50000,
        warmup_steps: int = 500,
        max_context_tokens: int = 256,
        max_question_tokens: int = 128,
        log_interval: int = 100,
        checkpoint_interval: int = 5000,
        checkpoint_dir: str = "checkpoints/hypernetwork",
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.hypernetwork = hypernetwork.to(device)
        self.base_model = base_model.to(device)
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = False

        self.tokenizer = tokenizer
        self.lora_injector = lora_injector
        self.max_context_tokens = max_context_tokens
        self.max_question_tokens = max_question_tokens
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir

        self.optimizer = AdamW(
            hypernetwork.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=1e-6
        )

        self.step = 0
        self.total_loss = 0.0
        self.log_history: list[dict] = []

    def _get_activations(self, text: str) -> torch.Tensor:
        """Extract mean-pooled hidden states from frozen base model."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.max_context_tokens
        )
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.base_model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1].mean(dim=1)  # (1, hidden_dim)

        return hidden

    def _get_teacher_logits(self, context: str, question: str) -> torch.Tensor:
        """Get logits from teacher (base model with full context)."""
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.max_context_tokens + self.max_question_tokens
        )
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.base_model(input_ids)
            # Take logits at the last position
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)

        return logits

    def _install_lora_hooks(
        self,
        lora_weights: dict[str, torch.Tensor],
        layer_prefix: str = "model.layers.{i}.mlp",
    ) -> list[torch.utils.hooks.RemovableHandle]:
        """Install forward hooks that add LoRA deltas differentiably.

        Unlike LoRAInjector.inject (which uses .data += and breaks the
        computation graph), these hooks add the delta inside the forward
        pass so gradients flow back through the hypernetwork.

        Returns a list of hook handles that must be removed after the
        forward pass.
        """
        handles = []

        # Discover layer indices from lora_weights keys
        layer_indices = set()
        for key in lora_weights:
            match = re.match(r"layer_(\d+)_[AB]", key)
            if match:
                layer_indices.add(int(match.group(1)))

        named_modules = dict(self.base_model.named_modules())

        for idx in sorted(layer_indices):
            module_name = layer_prefix.replace("{i}", str(idx))
            module = named_modules.get(module_name)
            if module is None or not isinstance(module, nn.Linear):
                continue

            A = lora_weights[f"layer_{idx}_A"]
            B = lora_weights[f"layer_{idx}_B"]
            # delta shape: (out_features, in_features) after transpose
            delta = (A @ B).T.to(module.weight.device)

            def make_hook(d):
                def hook(mod, input, output):
                    # input[0] has shape (batch, seq, in_features)
                    # Add the LoRA contribution: input @ delta^T
                    lora_out = F.linear(input[0], d)
                    return output + lora_out
                return hook

            h = module.register_forward_hook(make_hook(delta))
            handles.append(h)

        return handles

    def _get_student_logits(self, question: str) -> torch.Tensor:
        """Get logits from student (base model with LoRA hooks, no context)."""
        prompt = f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.max_question_tokens
        )
        input_ids = inputs["input_ids"].to(self.device)

        # Forward pass with LoRA hooks active (hooks installed by caller)
        outputs = self.base_model(input_ids)
        logits = outputs.logits[:, -1, :]  # (1, vocab_size)

        return logits

    def train_step(self, context: str, question: str) -> float:
        """Execute a single training step.

        Args:
            context: Document passage
            question: Question about the passage

        Returns:
            Loss value for this step.
        """
        self.hypernetwork.train()
        self.optimizer.zero_grad()

        # 1. Extract activations from context
        activations = self._get_activations(context)  # (1, hidden_dim)
        # Add sequence dimension
        activations = activations.unsqueeze(1)  # (1, 1, hidden_dim)

        # 2. Generate LoRA weights (differentiable w.r.t. hypernetwork)
        lora_weights = self.hypernetwork(activations)

        # 3. Get teacher logits (base model with full context, no LoRA)
        teacher_logits = self._get_teacher_logits(context, question)

        # 4. Install differentiable LoRA hooks for student forward pass
        hooks = self._install_lora_hooks(
            lora_weights, layer_prefix="model.layers.{i}.mlp"
        )

        try:
            # 5. Get student logits (question only, with LoRA via hooks)
            student_logits = self._get_student_logits(question)

            # 6. Compute KL divergence loss
            teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

            # 7. Backprop through hypernetwork
            loss.backward()
        finally:
            # 8. Always remove hooks to keep the base model clean
            for h in hooks:
                h.remove()

        torch.nn.utils.clip_grad_norm_(self.hypernetwork.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Also ensure injector state is clean
        if self.lora_injector.is_injected:
            self.lora_injector.remove(self.base_model)

        self.step += 1
        loss_val = loss.item()
        self.total_loss += loss_val

        # Log
        if self.step % self.log_interval == 0:
            avg_loss = self.total_loss / self.log_interval
            lr = self.scheduler.get_last_lr()[0]
            entry = {
                "step": self.step,
                "loss": avg_loss,
                "lr": lr,
                "time": time.time(),
            }
            self.log_history.append(entry)
            self.total_loss = 0.0

        return loss_val

    def save_checkpoint(self, path: Optional[str] = None):
        """Save hypernetwork checkpoint."""
        import os
        if path is None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            path = os.path.join(self.checkpoint_dir, f"checkpoint-{self.step}.pt")

        torch.save({
            "step": self.step,
            "hypernetwork_state_dict": self.hypernetwork.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "log_history": self.log_history,
        }, path)
        return path

    def load_checkpoint(self, path: str):
        """Load hypernetwork checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.hypernetwork.load_state_dict(ckpt["hypernetwork_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.step = ckpt["step"]
        self.log_history = ckpt.get("log_history", [])
