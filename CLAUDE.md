# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
pip install -e ".[dev]"    # Install package + dev deps (editable)
continual-learning          # Launch interactive CLI (entry point)
```

## Testing

```bash
python -m pytest tests/                        # All 101 tests (~10s)
python -m pytest tests/test_model/             # Just model tests
python -m pytest tests/test_jitrl/             # Just JitRL tests
python -m pytest tests/test_config.py -k "test_load"  # Single test by name
```

Pytest is configured in `pyproject.toml`: testpaths=`tests/`, pythonpath=`src/`.

## GPU Validation Scripts

```bash
python scripts/validate_gpu.py    # Milestones 1-3, 5 (requires A10/24GB)
python scripts/validate_jitrl.py  # JitRL MVP vs Full comparison
python scripts/validate_doc2lora.py  # Doc-to-LoRA pipeline (Gemma-2-2b-it)
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

### CLI (`cli/main.py`)
Interactive menu (Questionary + Rich) that lazy-loads the model on first use. All state is global module-level variables. Menu handlers map to `MENU_HANDLERS` dict.

## Configuration

`configs/default.yaml` — model name, layer range, TTT hyperparams, alpha schedule, TF-IDF threshold. Loaded via `config.py` which merges defaults with any user overrides.

## Package Layout

```
src/continual_learning/
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
- Never push to GitHub — local commits only
- Design docs live in `docs/plans/` (gitignored)
