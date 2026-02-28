#!/usr/bin/env python3
"""End-to-end training script for the Perceiver hypernetwork.

Trains a PerceiverHypernetwork via teacher-student distillation on SQuAD v2
or a synthetic meta-dataset. The hypernetwork learns to generate LoRA weights
from document activations so the adapted model can answer questions without
seeing the original document in context.

Usage:
    # Train on synthetic data (no download required)
    python scripts/train_hypernetwork.py --dataset synthetic --max-steps 1000

    # Train on SQuAD v2 (downloads ~30MB on first run)
    python scripts/train_hypernetwork.py --dataset squad --max-steps 50000

    # Resume from checkpoint
    python scripts/train_hypernetwork.py --resume checkpoints/hypernetwork/checkpoint-5000.pt
"""

import argparse
import os
import sys
import time

import torch

# Ensure project is importable when run from repo root
sys.path.insert(0, "src")

from continual_learning.doc2lora.hypernetwork import PerceiverHypernetwork
from continual_learning.doc2lora.lora_injector import LoRAInjector
from continual_learning.doc2lora.trainer import HypernetworkTrainer
from continual_learning.doc2lora.meta_dataset import MetaTrainingDataset, SyntheticMetaDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Perceiver hypernetwork for Doc-to-LoRA generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-model", type=str, default="google/gemma-3-270m-it",
        help="HuggingFace model ID for the base causal LM.",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=8,
        help="Rank of the generated LoRA adapters.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="Peak learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=50000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size (1 = one document per step).",
    )
    parser.add_argument(
        "--max-context-tokens", type=int, default=256,
        help="Maximum number of tokens for document context.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/hypernetwork",
        help="Directory for saving checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=5000,
        help="Save a checkpoint every N steps.",
    )
    parser.add_argument(
        "--log-interval", type=int, default=100,
        help="Print training metrics every N steps.",
    )
    parser.add_argument(
        "--eval-interval", type=int, default=1000,
        help="Run evaluation every N steps.",
    )
    parser.add_argument(
        "--dataset", type=str, choices=["squad", "synthetic"], default="squad",
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to train on (auto, cuda, cpu).",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def print_header(text: str):
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}\n")


def print_gpu_status():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {allocated:.1f}GB allocated / {reserved:.1f}GB reserved / {total:.1f}GB total")
    else:
        print("  Device: CPU (no GPU available)")


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def load_base_model(model_name: str, device: str):
    """Load base model and tokenizer from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model = model.to(device)
    model.eval()

    return model, tokenizer


def extract_model_dims(model) -> dict:
    """Extract hidden_dim, num_layers, and intermediate_dim from model config."""
    config = model.config
    hidden_dim = config.hidden_size
    num_layers = config.num_hidden_layers
    # intermediate_dim may be stored under different attribute names
    intermediate_dim = getattr(config, "intermediate_size", None)
    if intermediate_dim is None:
        intermediate_dim = getattr(config, "intermediate_dim", hidden_dim * 4)

    return {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "intermediate_dim": intermediate_dim,
    }


def load_dataset(dataset_name: str):
    """Load the training dataset."""
    if dataset_name == "squad":
        print("  Loading SQuAD v2 dataset (this may download ~30MB on first run)...")
        dataset = MetaTrainingDataset(split="train", max_context_words=256)
        print(f"  Loaded {len(dataset)} training items from SQuAD v2.")
    else:
        print("  Generating synthetic meta-dataset...")
        dataset = SyntheticMetaDataset(num_items=1000, seed=42)
        print(f"  Generated {len(dataset)} synthetic items.")
    return dataset


def run_eval(trainer, dataset, num_eval_steps: int = 5) -> float:
    """Run a quick evaluation by averaging loss over a few steps.

    Returns the average loss.
    """
    import random

    total_loss = 0.0
    dataset_len = len(dataset)

    for _ in range(num_eval_steps):
        idx = random.randint(0, dataset_len - 1)
        item = dataset[idx]
        loss = trainer.train_step(item["context"], item["question"])
        total_loss += loss

    return total_loss / num_eval_steps


def main():
    args = parse_args()
    device = resolve_device(args.device)

    # Optional Rich progress bar
    try:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
        use_rich = True
    except ImportError:
        use_rich = False

    print_header("Perceiver Hypernetwork Training")
    print(f"  Device:             {device}")
    print(f"  Base model:         {args.base_model}")
    print(f"  LoRA rank:          {args.lora_rank}")
    print(f"  Learning rate:      {args.learning_rate}")
    print(f"  Max steps:          {args.max_steps}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Max context tokens: {args.max_context_tokens}")
    print(f"  Dataset:            {args.dataset}")
    print(f"  Checkpoint dir:     {args.checkpoint_dir}")
    print(f"  Resume from:        {args.resume or 'None'}")
    print()
    print_gpu_status()

    # --- Load base model ---
    print_header("Loading Base Model")
    base_model, tokenizer = load_base_model(args.base_model, device)
    dims = extract_model_dims(base_model)
    print(f"  Architecture: hidden={dims['hidden_dim']}, layers={dims['num_layers']}, "
          f"intermediate={dims['intermediate_dim']}")
    print_gpu_status()

    # --- Create hypernetwork ---
    print_header("Creating Hypernetwork")
    hypernetwork = PerceiverHypernetwork(
        hidden_dim=dims["hidden_dim"],
        num_layers=dims["num_layers"],
        intermediate_dim=dims["intermediate_dim"],
        lora_rank=args.lora_rank,
    )
    param_count = sum(p.numel() for p in hypernetwork.parameters())
    trainable_count = sum(p.numel() for p in hypernetwork.parameters() if p.requires_grad)
    print(f"  Total params:     {param_count:,}")
    print(f"  Trainable params: {trainable_count:,}")

    # --- Create trainer ---
    print_header("Initializing Trainer")
    lora_injector = LoRAInjector()
    trainer = HypernetworkTrainer(
        hypernetwork=hypernetwork,
        base_model=base_model,
        tokenizer=tokenizer,
        lora_injector=lora_injector,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        max_context_tokens=args.max_context_tokens,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
    )

    # --- Resume from checkpoint ---
    if args.resume:
        print(f"  Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print(f"  Resumed at step {trainer.step}")

    # --- Load dataset ---
    print_header("Loading Dataset")
    dataset = load_dataset(args.dataset)

    # --- Training loop ---
    print_header("Training")
    start_time = time.time()
    start_step = trainer.step
    dataset_len = len(dataset)
    sample_idx = start_step % dataset_len

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if use_rich:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
        task = progress.add_task("Training", total=args.max_steps - start_step)
        progress.start()
    else:
        progress = None

    try:
        for step_num in range(start_step, args.max_steps):
            # Sample from dataset (cycle through)
            item = dataset[sample_idx % dataset_len]
            sample_idx += 1

            # Train step
            loss = trainer.train_step(item["context"], item["question"])
            current_step = trainer.step  # trainer increments internally

            # Update Rich progress bar
            if progress is not None:
                progress.advance(task)

            # Log metrics
            if current_step % args.log_interval == 0:
                elapsed = time.time() - start_time
                steps_done = current_step - start_step
                steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
                lr = trainer.scheduler.get_last_lr()[0]
                eta = (args.max_steps - current_step) / steps_per_sec if steps_per_sec > 0 else 0

                print(
                    f"  [Step {current_step:>6d}/{args.max_steps}]  "
                    f"loss={loss:.6f}  lr={lr:.2e}  "
                    f"speed={steps_per_sec:.2f} step/s  "
                    f"elapsed={format_time(elapsed)}  "
                    f"ETA={format_time(eta)}"
                )

            # Checkpoint
            if current_step % args.checkpoint_interval == 0:
                ckpt_path = trainer.save_checkpoint()
                print(f"  >> Checkpoint saved: {ckpt_path}")
                print_gpu_status()

            # Evaluation
            if current_step % args.eval_interval == 0:
                print(f"\n  --- Eval at step {current_step} ---")
                avg_eval_loss = run_eval(trainer, dataset, num_eval_steps=5)
                lr = trainer.scheduler.get_last_lr()[0]
                print(f"  Avg eval loss: {avg_eval_loss:.6f}  LR: {lr:.2e}")
                print()

    except KeyboardInterrupt:
        print("\n  Training interrupted by user.")
    finally:
        if progress is not None:
            progress.stop()

    # --- Save final checkpoint ---
    print_header("Saving Final Checkpoint")
    final_path = trainer.save_checkpoint()
    print(f"  Saved: {final_path}")

    # --- Summary ---
    total_time = time.time() - start_time
    total_steps = trainer.step - start_step
    print_header("Training Complete")
    print(f"  Total steps:  {total_steps}")
    print(f"  Total time:   {format_time(total_time)}")
    if total_time > 0:
        print(f"  Avg speed:    {total_steps / total_time:.2f} step/s")
    if trainer.log_history:
        final_loss = trainer.log_history[-1]["loss"]
        print(f"  Final avg loss: {final_loss:.6f}")
    print_gpu_status()
    print()


if __name__ == "__main__":
    main()
