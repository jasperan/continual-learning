#!/usr/bin/env python3
"""GPU validation for Doc-to-LoRA pipeline.

Requires A10 GPU (24GB VRAM).
Downloads Gemma-2-2b-it + Sakana hypernetwork checkpoint.

Usage:
    python scripts/validate_doc2lora.py
"""

import sys
import time
import torch

sys.path.insert(0, "src")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cpu":
        print("WARNING: Running on CPU. Results will be slow and for testing only.")

    # Load base model
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "google/gemma-2-2b-it"
    print(f"\n=== Loading {model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )

    if device == "cuda":
        vram_gb = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM after model load: {vram_gb:.1f} GB")

    # Initialize engine (simulated hypernetwork for now)
    from continual_learning.doc2lora.engine import Doc2LoRAEngine

    print("\n=== Initializing Doc-to-LoRA Engine ===")
    engine = Doc2LoRAEngine(
        model=model,
        tokenizer=tokenizer,
        hidden_dim=2048,
        num_target_layers=18,  # Gemma-2-2b has 18 layers
        intermediate_dim=16384,  # Gemma-2-2b MLP intermediate
        lora_rank=8,
        chunk_size=1024,
        simulated=True,  # Use simulated hypernetwork for demo
    )

    # Learn a document
    sample_doc = """
    Oracle Database is a multi-model database management system produced by Oracle Corporation.
    It supports SQL and is available on-premises and in Oracle Cloud.
    Oracle Database 23ai introduced AI Vector Search for storing and querying vector embeddings.
    The database supports JSON, spatial, graph, and key-value data models.
    Oracle Autonomous Database provides self-driving, self-securing capabilities.
    """

    print("\n=== Learning Document ===")
    t0 = time.time()
    metrics = engine.learn(sample_doc)
    learn_time = time.time() - t0
    print(f"Learn time: {learn_time:.3f}s")
    print(f"Metrics: {metrics}")

    # Generate response
    print("\n=== Generating Response ===")
    t0 = time.time()
    response = engine.generate("What is Oracle Database 23ai?")
    gen_time = time.time() - t0
    print(f"Generate time: {gen_time:.3f}s")
    print(f"Response: {response[:500]}")

    # Clear and verify
    engine.clear()
    print(f"\nAfter clear: {engine.num_documents} documents")

    if device == "cuda":
        vram_gb = torch.cuda.memory_allocated() / 1e9
        print(f"Final VRAM: {vram_gb:.1f} GB")

    print("\n=== VALIDATION COMPLETE ===")


if __name__ == "__main__":
    main()
