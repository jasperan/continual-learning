# Continual Learning SLM

A continual learning system that enables Small Language Models to learn from new documents in real-time by updating neural weights at inference time.

## How It Works

This project modifies Qwen2.5-1.5B with two key mechanisms:

- **TTT-E2E (Test-Time Training)**: Performs mini-batch gradient descent on new documents during inference, writing knowledge directly into the model's weights
- **Sparse Memory Finetuning**: Uses TF-IDF scoring to protect general-purpose neurons from updates, preventing catastrophic forgetting

The final 25% of transformer layers are modified with a dual-MLP architecture: one frozen (original intelligence) and one trainable (new knowledge).

## Requirements

- Python 3.11+
- NVIDIA GPU with 24GB+ VRAM (tested on A10)
- CUDA toolkit

## Installation

```bash
git clone <repo-url>
cd continual-learning
pip install -e ".[dev]"
```

This installs the package in editable mode with all dependencies (PyTorch, Transformers, JAX, Rich, Questionary, scikit-learn, etc.) plus dev tools (pytest).

## Quick Start

Launch the interactive CLI:

```bash
continual-learning
```

The CLI presents a menu-driven interface. On first use that requires the model, it will automatically download Qwen2.5-1.5B (~3GB) and inject the DualMLP architecture.

### Typical Workflow

1. **Chat with Model** or **Ask a Question** — the model loads automatically on first interaction
2. **Learn from Document** — feed a `.txt`, `.md`, or `.jsonl` file; the TTT engine writes new knowledge into the trainable weights
3. **Learn from Directory** — batch-learn all documents in a folder
4. **Chat with Model** — ask questions about what it just learned to verify retention
5. **Run Benchmarks** — evaluate accuracy and forgetting ratio against SQuAD holdout data
6. **Save Checkpoint** — persist the learned state for later

### CLI Menu Options

| Option | Description |
|--------|-------------|
| **Chat with Model** | Free-form conversation (type `quit` to return) |
| **Ask a Question** | Single question/answer |
| **Learn from Document** | TTT-E2E on a single file |
| **Learn from Directory** | TTT-E2E on all `.txt`/`.md`/`.jsonl` files in a directory |
| **Run Benchmarks** | Evaluate against SQuAD holdout set |
| **View Forgetting Metrics** | Check catastrophic forgetting indicators |
| **Model Info** | Architecture details, parameter counts, alpha values |
| **Learning History** | Documents learned, token counts, loss values |
| **Save/Load/List Checkpoints** | Persist and restore learned state |
| **JitRL MVP** | Learn + query via TF-IDF retrieval + logit biasing (fast) |
| **JitRL Full** | Learn + query via knowledge store + reward modulation |
| **Compare All Engines** | A/B benchmark across TTT, JitRL MVP, and JitRL Full |
| **Configure** | View and edit runtime settings (learning rate, alpha, etc.) |

## Configuration

Default settings are in `configs/default.yaml`. The CLI's **Configure** option lets you edit settings at runtime. Key parameters:

```yaml
model:
  name: "Qwen/Qwen2.5-1.5B"
  modified_layers_start: 21    # Inject DualMLP into layers 21-28
  modified_layers_end: 28
  device: "auto"

ttt:
  learning_rate: 1.0e-5
  mini_batch_size: 32
  gradient_steps: 1
  max_tokens_per_document: 4096

alpha:
  initial: 1.0          # Blend weight for frozen vs trainable MLP
  decay_rate: 0.01       # Decay per learning step
  min_value: 0.3

tfidf_gate:
  threshold: 0.3         # Neuron masking threshold
  calibration_samples: 2000
```

## Running Tests

```bash
python -m pytest tests/              # All 101 tests
python -m pytest tests/test_model/   # Model tests only
python -m pytest tests/test_jitrl/   # JitRL tests only
```

## GPU Validation

Scripts that verify end-to-end functionality on GPU:

```bash
python scripts/validate_gpu.py     # Model loading, TTT updates, forgetting ratio, Oracle docs
python scripts/validate_jitrl.py   # JitRL MVP vs Full accuracy comparison
```

## Project Structure

```
src/continual_learning/
├── model/           # DualMLP architecture, modified Qwen, TF-IDF gating
├── training/        # TTT-E2E engine, calibration
├── evaluation/      # Benchmarks, forgetting metrics
├── data/            # SQuAD pipeline, Oracle docs fetcher
├── jitrl/           # JitRL MVP + Full engines, comparison harness
├── checkpointing/   # Save/restore learned state
├── cli/             # Interactive menu interface
└── config.py        # YAML config loader
```

## License

MIT
