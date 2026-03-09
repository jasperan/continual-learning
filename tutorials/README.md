# Continual Learning Tutorials

A hands-on course teaching 5 strategies for teaching language models new knowledge without catastrophic forgetting.

## Learning Path

| # | Tutorial | Strategy | Key Concept | Time |
|---|----------|----------|-------------|------|
| 0 | Introduction | — | What is continual learning? Why does forgetting happen? | 15 min |
| 1 | TTT-E2E | Test-Time Training | DualMLP, TF-IDF gating, alpha decay | 30 min |
| 2 | JitRL MVP | Retrieval-Augmented | TF-IDF retrieval, logit biasing, zero weight updates | 20 min |
| 3 | JitRL Full | Reward-Guided | Hidden-state embeddings, reward signals, knowledge store | 25 min |
| 4 | ACE | Agentic Context | Generate-Reflect-Curate loops, playbook evolution | 20 min |
| 5 | Doc-to-LoRA | Hypernetwork Adapters | LoRA, hypernetworks, document chunking | 30 min |
| 6 | Benchmark | Comparison | Side-by-side strategy evaluation | 15 min |

## Prerequisites

- Python 3.11+
- NVIDIA GPU with 24GB+ VRAM (A10, RTX 4090)
- Conda environment: `continual-learning`
- Ollama with qwen3.5:9b (for ACE strategy)

## Quick Start

```bash
conda activate continual-learning
cd tutorials
jupyter notebook 00_introduction.ipynb
```

---

## The Forgetting Problem

Neural networks are powerful function approximators, but they have an Achilles' heel: **catastrophic forgetting**. When you fine-tune a model on new data, the gradient updates that teach it new knowledge simultaneously erode the weights that encode old knowledge.

This is not a minor nuisance. In production LLM deployments, you need models that can:

1. **Learn new facts** (product updates, policy changes, breaking news)
2. **Retain existing knowledge** (general reasoning, language understanding, domain expertise)
3. **Do both without full retraining** (which costs thousands of GPU-hours)

The five strategies in this course each attack the problem differently, ranging from surgical weight modifications to approaches that never touch the weights at all.

## How Strategies Differ: Weight vs. No-Weight

The most fundamental distinction between continual learning strategies is whether they modify the model's weights:

### Weight-Modifying Strategies

- **TTT-E2E** (Tutorial 1): Injects a trainable copy of specific MLP layers (layers 21-28) and blends their output with the frozen original via an alpha parameter. TF-IDF gating protects general-purpose neurons from being overwritten.
- **Doc-to-LoRA** (Tutorial 5): A hypernetwork generates small Low-Rank Adaptation (LoRA) matrices from document text and injects them into the base model. The base weights stay frozen; only the LoRA deltas are added.

### Non-Weight-Modifying Strategies

- **JitRL MVP** (Tutorial 2): Retrieves relevant passages via TF-IDF and biases the model's output logits toward retrieved content. Zero parameters are updated.
- **JitRL Full** (Tutorial 3): Encodes documents as hidden-state embeddings in a knowledge store. Uses reward signals to guide retrieval quality. Still zero weight updates.
- **ACE** (Tutorial 4): Uses an LLM in a Generate-Reflect-Curate loop to evolve context strategies. The model itself is never modified; instead, the *prompting strategy* improves over time.

## Strategy Comparison at a Glance

| Strategy | Weight Updates? | Learn Speed | Gen Speed | Forgetting Risk | Best For |
|----------|----------------|-------------|-----------|-----------------|----------|
| TTT-E2E | Yes (trainable MLP only) | ~12s | ~2s | Low (TF-IDF gated) | Deep knowledge internalization |
| JitRL MVP | No | ~0.002s | ~2.4s | Zero | Fast prototyping, simple QA |
| JitRL Full | No | ~0.04s | ~33s | Zero | Complex retrieval tasks |
| ACE | No | ~30s (LLM loops) | ~5s | Zero | Strategy evolution, meta-learning |
| Doc-to-LoRA | Yes (LoRA injection) | ~0.8s | ~2.2s | Very low | Single-pass document learning |

## Hardware Requirements

All tutorials assume access to an NVIDIA GPU with at least 24GB of VRAM:

| Hardware | Status | Notes |
|----------|--------|-------|
| A10 (24GB) | Recommended | Primary development/testing target |
| RTX 4090 (24GB) | Supported | Equivalent VRAM, faster consumer card |
| A100 (40/80GB) | Supported | More headroom for larger batch sizes |
| RTX 3090 (24GB) | Should work | Not tested, but sufficient VRAM |
| < 24GB VRAM | Not supported | Model + DualMLP + training state won't fit |

Tutorials 0 (Introduction) and 2 (JitRL MVP) can run on CPU for the conceptual demonstrations, though GPU is still recommended.

## Which Tutorial Should I Start With?

**"I want to understand the theory first"** -- Start with Tutorial 0 (Introduction), then proceed sequentially.

**"I want the fastest possible demo"** -- Jump to Tutorial 2 (JitRL MVP). It requires no weight updates, learns in 0.002 seconds, and demonstrates the core idea immediately.

**"I want the deepest technical understanding"** -- Start with Tutorial 1 (TTT-E2E). It covers DualMLP injection, TF-IDF gradient masking, and alpha decay -- the most architecturally involved strategy.

**"I care about production deployment"** -- Start with Tutorial 5 (Doc-to-LoRA). Single-pass hypernetwork inference is the most deployment-friendly approach.

**"I want to see all strategies compared"** -- Work through Tutorials 1-5 first, then run Tutorial 6 (Benchmark) for a side-by-side evaluation.
