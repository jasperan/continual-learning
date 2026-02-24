# Continual Learning SLM

A continual learning system that enables Small Language Models to learn from new documents in real-time by updating neural weights at inference time.

## How It Works

This project modifies Qwen2.5-1.5B with two key mechanisms:

- **TTT-E2E (Test-Time Training)**: Performs mini-batch gradient descent on new documents during inference, writing knowledge directly into the model's weights
- **Sparse Memory Finetuning**: Uses TF-IDF scoring to protect general-purpose neurons from updates, preventing catastrophic forgetting

The final 25% of transformer layers are modified with a dual-MLP architecture: one frozen (original intelligence) and one trainable (new knowledge).

## Quick Start

```bash
pip install -e ".[dev]"
continual-learning
```

The interactive CLI will guide you through:
1. Loading the model
2. Feeding it new documents to learn
3. Chatting with it to verify learning
4. Running benchmarks to measure forgetting

## Project Structure

```
src/continual_learning/
├── model/          # Dual-MLP architecture and TF-IDF gating
├── training/       # TTT-E2E engine
├── evaluation/     # Benchmarks and forgetting metrics
├── data/           # SQuAD and Oracle docs pipelines
├── checkpointing/  # Save/load learned state
└── cli/            # Interactive menu-driven interface
```

## Requirements

- Python 3.11+
- NVIDIA GPU with 24GB+ VRAM (tested on A10)
- CUDA toolkit

## License

MIT
