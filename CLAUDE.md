# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Option 1: venv (matches install.sh)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Option 2: conda (per workspace convention)
conda create -n continual-learning python=3.12
conda activate continual-learning
pip install -e ".[dev]"

continual-learning          # Launch interactive CLI (entry point)
```

## Testing

```bash
python -m pytest tests/                        # All tests
python -m pytest tests/test_model/             # Just model tests
python -m pytest tests/test_jitrl/             # Just JitRL tests
python -m pytest tests/test_ace/               # Just ACE tests
python -m pytest tests/test_config.py -k "test_load"  # Single test by name
```

Pytest is configured in `pyproject.toml`: testpaths=`tests/`, pythonpath=`src/`.

## Linting

```bash
ruff check src/ tests/    # Lint (E, F, I, N, W rules; line-length 100)
ruff format src/ tests/   # Auto-format (double quotes)
ty check src/             # Type checking (ty, targets Python 3.11)
```

## GPU Validation Scripts

These live in `tests/` (not `scripts/`):

```bash
python tests/validate_gpu.py        # Milestones 1-3, 5 (requires A10/24GB)
python tests/validate_jitrl.py      # JitRL MVP vs Full comparison
python tests/validate_doc2lora.py   # Doc-to-LoRA pipeline (Gemma-2-2b-it)
python tests/train_hypernetwork.py  # Hypernetwork training run
```

## Architecture

**Core idea**: Modify Qwen2.5-1.5B so it can learn from new documents at inference time without forgetting existing knowledge.

### DualMLP Injection (`model/dual_mlp.py`, `model/modified_qwen.py`)
Layers 21-28 of Qwen have their MLP replaced with DualMLP: a frozen original MLP + a trainable copy blended via `frozen_out + (1-alpha) * trainable_out`. Alpha decays from 0.95 toward 0.5 as documents are learned. Trainable MLP uses small random init (std=0.001) — not zeros (SwiGLU dead gradient issue). DualMLP caches input activations for TF-IDF gradient masking.

### TF-IDF Gating (`model/tfidf_gate.py`)
Protects general-purpose neurons by masking gradients during backprop. Calibrated against sample activations to identify which neurons are document-specific vs general.

### TTT-E2E Engine (`training/ttt_engine.py`)
Test-Time Training via mini-batch gradient descent on tokenized documents. Only updates trainable MLP parameters. Manages alpha decay across learning steps.

### JitRL Engines (`jitrl/`)
Two alternative retrieval-augmented approaches:
- **MVP** (`jitrl/mvp/`): TF-IDF retriever + logit biasing. Fast (~0.002s learn, ~2.4s eval).
- **Full** (`jitrl/full/`): Knowledge store + reward-guided logit modulation. Slower, more sophisticated.
- **Comparison harness** (`jitrl/comparison.py`): A/B benchmarking across engines.

### Doc-to-LoRA / Text-to-LoRA (`doc2lora/`)
Hypernetwork-generated LoRA pipeline. A Perceiver-based hypernetwork takes document text
(or task descriptions), generates rank-8 LoRA matrices in a single forward pass, and
injects them into the base model's MLP layers. Two modes: `doc` (document internalization)
and `text` (task specialization). Uses simulated hypernetwork for tests; real Sakana AI
checkpoint for GPU demos with Gemma-2-2b-it.

### ACE — Agentic Context Engineering (`ace/`)
Ollama-backed agentic pipeline that runs multi-loop context generation. Components: `engine.py` (main loop, `num_loops` iterations), `generator.py` (prompt → text), `curator.py` (filters/deduplicates outputs), `adapter.py` (format conversion), `reflector.py` (self-critique), `playbook.py` (strategy store, persisted to `playbooks/`), `ollama_client.py` (HTTP client for `localhost:11434`). Configured via the `ace:` block in `configs/default.yaml` (model: `qwen2.5:7b`, 3 loops, up to 50 strategies).

### CLI (`cli/main.py`)
Interactive menu (Questionary + Rich) that lazy-loads the model on first use. All state is global module-level variables. Menu handlers map to `MENU_HANDLERS` dict.

## Configuration

`configs/default.yaml` — model name, layer range, TTT hyperparams, alpha schedule, TF-IDF threshold. Loaded via `config.py` which merges defaults with any user overrides.

## Package Layout

```
src/continual_learning/
├── ace/             # Agentic Context Engineering (Ollama-backed pipeline)
├── model/           # DualMLP, modified Qwen, TF-IDF gate
├── training/        # TTT engine, calibration
├── evaluation/      # Benchmarks, forgetting metrics
├── data/            # SQuAD pipeline, Oracle docs fetcher
├── jitrl/           # MVP + Full engines, comparison harness
├── doc2lora/        # Hypernetwork → LoRA pipeline (Doc/Text-to-LoRA)
├── checkpointing/   # Save/restore learned state
├── cli/             # Interactive menu interface
└── config.py        # YAML config loader
```

## Git Conventions

- No AI attribution in commit messages
- Commit and push after completing changes (standard workspace policy)
- Design docs live in `docs/plans/` (gitignored)
