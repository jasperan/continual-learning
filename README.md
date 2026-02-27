# Continual Learning SLM

[![PyPI](https://img.shields.io/pypi/v/continual-learning-slm?style=for-the-badge)](https://pypi.org/project/continual-learning-slm/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/continual-learning-slm?style=for-the-badge)](https://pypi.org/project/continual-learning-slm/)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=for-the-badge)](https://www.python.org/)

A continual learning system that enables Small Language Models to learn from new documents in real-time by updating neural weights at inference time, without catastrophic forgetting.

## How It Works

This project implements four distinct continual learning strategies on top of Qwen2.5-1.5B:

### Strategy 1: TTT-E2E (Test-Time Training End-to-End)

The core approach. The final 25% of transformer layers (layers 21-28) are modified with a **Dual-MLP architecture**: each layer gets a frozen MLP (preserving original intelligence) and a trainable MLP (absorbing new knowledge). When you feed the model a document, it runs mini-batch gradient descent to write knowledge directly into the trainable weights.

**Sparse Memory Finetuning** (TF-IDF gating) protects general-purpose neurons from being overwritten. Neurons that activate broadly across a calibration corpus are masked during gradient updates, while neurons specialized to the new content receive full gradients.

The **alpha blending** parameter controls the mix: `frozen_out + (1-alpha) * trainable_out`. Alpha starts at 0.95 and decays toward 0.5 as documents are learned, gradually increasing the influence of newly learned knowledge.

### Strategy 2: JitRL MVP (Just-in-Time Retrieval Learning - Minimum Viable)

A lightweight retrieval-augmented approach. Instead of modifying weights, it indexes document chunks using TF-IDF and retrieves relevant passages at query time. Retrieved chunks are prepended as context, and a logit biaser nudges the model toward tokens that appear in the retrieved content.

**Tradeoff**: No weight modification means no forgetting risk, but knowledge is limited to what fits in the context window. Very fast (~0.002s to learn, ~2.4s to generate).

### Strategy 3: JitRL Full (Reward-Guided Logit Modulation)

A more sophisticated retrieval approach. Documents are encoded into hidden-state embeddings and stored in a knowledge store. At query time, the system retrieves relevant knowledge embeddings, computes a reward signal via cosine similarity, and modulates the model's logit distribution to favor knowledge-aligned tokens.

**Tradeoff**: More expressive than MVP but slower (~0.04s to learn, ~33s to generate). Currently less accurate than MVP; needs hyperparameter tuning.

### Strategy 4: ACE (Agentic Context Engineering)

A zero-weight-update approach inspired by the [Stanford + SambaNova ACE paper](https://arxiv.org). Instead of modifying model weights or retrieving passages, ACE evolves a **playbook** — a structured set of strategies — through iterative self-improvement loops powered by a local LLM via Ollama.

Each learning cycle runs three roles:
- **Generator**: Answers questions using the document and current playbook strategies
- **Reflector**: Critiques the answer (what went right, what went wrong, suggested improvements)
- **Curator**: Patch-updates the playbook with minimal, targeted changes (never a full rewrite — preventing "context collapse")

The playbook grows smarter with each loop: failures become strategies, successes become rules. Playbooks persist as JSON files and can be saved/loaded across sessions.

**Tradeoff**: Requires Ollama running locally with a 7B+ model. No weight modification and no forgetting risk. Quality depends on the Ollama model's reasoning ability. Configurable loop count (default: 3 Generate-Reflect-Curate cycles).

## Requirements

- Python 3.11+
- NVIDIA GPU with 24GB+ VRAM (tested on A10) — for TTT-E2E and JitRL strategies
- CUDA toolkit
- [Ollama](https://ollama.com/) — required only for ACE strategy (install and `ollama pull qwen2.5:7b`)

## Installation

From PyPI:

```bash
pip install continual-learning-slm
```

Or from source (for development):

```bash
git clone https://github.com/jasperan/continual-learning.git
cd continual-learning
pip install -e ".[dev]"
```

## Quick Start

```bash
continual-learning
```

This launches the interactive CLI. The model downloads automatically (~3GB) on first use.

### Typical Workflow

1. Select **Chat with Model** or **Ask a Question** - the model loads and injects DualMLP automatically
2. Select **Learn from Document** - point it at a `.txt`, `.md`, or `.jsonl` file
3. Select **Chat with Model** again - ask questions about what it just learned
4. Select **Run Benchmarks** - measure accuracy and forgetting ratio against SQuAD holdout data
5. Select **Save Checkpoint** - persist the learned state for later

## Using Each Learning Strategy

### TTT-E2E: Weight-Based Learning

Feed documents directly into the model's weights via the CLI:

| CLI Option | What It Does |
|---|---|
| **Learn from Document** | Runs TTT-E2E on a single file. Tokenizes the text, splits into mini-batches of 32 tokens, and performs gradient descent. Shows per-batch loss and token count as it learns. |
| **Learn from Directory** | Batch-learns all `.txt`, `.md`, and `.jsonl` files in a directory sequentially. |

After learning, the trainable MLP weights are updated and alpha is decayed. The model's responses immediately reflect the new knowledge.

### JitRL MVP: Fast Retrieval + Logit Biasing

| CLI Option | What It Does |
|---|---|
| **JitRL MVP (Learn Doc)** | Indexes a document by chunking it and building a TF-IDF index. Then prompts you with a question - retrieves the top-3 most relevant chunks, prepends them as context, and applies logit biasing toward tokens found in the retrieved chunks. |

The MVP engine does not modify model weights. You can learn multiple documents and they accumulate in the TF-IDF index.

### JitRL Full: Knowledge Store + Reward Modulation

| CLI Option | What It Does |
|---|---|
| **JitRL Full (Learn Doc)** | Encodes a document through the full model, captures the last hidden-state embeddings, and stores them in a knowledge store. At query time, it retrieves the closest knowledge embeddings via cosine similarity, computes a reward vector, and modulates the output logits through the model's language model head. |

### ACE: Agentic Context Engineering

Requires Ollama running locally (`ollama serve`).

| CLI Option | What It Does |
|---|---|
| **ACE Learn Document** | Loads a document and optionally collects QA pairs. If QA pairs are provided, runs N Generate-Reflect-Curate loops (default: 3) to evolve a playbook of answering strategies. Without QA pairs, simply stores the document for context. |
| **ACE Ask Question** | Generates an answer using all stored documents and the evolved playbook strategies via Ollama. |
| **ACE Save Playbook** | Saves the current playbook (strategies + stats) to a named JSON file in the `playbooks/` directory. |
| **ACE Load Playbook** | Loads a previously saved playbook by name, restoring its strategies for future generation. |

### Comparing All Strategies

| CLI Option | What It Does |
|---|---|
| **Compare All Engines** | A/B benchmarks across JitRL MVP, JitRL Full, and ACE on the same document and QA pairs. Provide a document path, then enter question/answer pairs. The harness feeds the same data to each engine and reports accuracy, learn time, eval time, and tokens learned in a comparison table. |

## Evaluation and Benchmarks

| CLI Option | What It Does |
|---|---|
| **Run Benchmarks** | Loads 50 items from SQuAD 2.0 validation set and evaluates the model's QA accuracy. Checks if the expected answer substring appears in the model's generated response. Reports accuracy and forgetting ratio if a baseline exists. |
| **View Forgetting Metrics** | Shows catastrophic forgetting indicators. Forgetting ratio = `(before - after) / before`. A value of 0 means no forgetting, negative means the model improved. Target: < 0.15. |
| **Model Info** | Displays architecture details: total/modified/frozen layers, total/trainable parameter counts, current alpha value, and whether TF-IDF gates are calibrated. |
| **Learning History** | Shows a table of all documents learned in the current session: file name, token count, final loss, and timestamp. |

## Checkpointing

Checkpoints save only the trainable MLP weights and TF-IDF gate statistics (~50-100MB), not the full 3GB model.

| CLI Option | What It Does |
|---|---|
| **Save Checkpoint** | Saves trainable MLP state dicts, TF-IDF gate stats (IDF scores, document frequencies), alpha values, learning history, and config to a named subdirectory under `checkpoints/`. |
| **Load Checkpoint** | Presents a selection menu of saved checkpoints. Restores trainable weights, TF-IDF calibration, alpha values, and learning history. |
| **List Checkpoints** | Shows all saved checkpoint names. |

## Configuration

Default settings are in `configs/default.yaml`. The CLI's **Configure** option lets you view and edit settings at runtime (changes persist to the YAML file).

```yaml
model:
  name: "Qwen/Qwen2.5-1.5B"
  modified_layers_start: 21    # First layer to inject DualMLP
  modified_layers_end: 28      # Last layer (exclusive)
  device: "auto"               # "auto", "cuda", or "cpu"

ttt:
  learning_rate: 1.0e-5        # Adam learning rate for TTT-E2E
  mini_batch_size: 32          # Tokens per mini-batch
  gradient_steps: 1            # Gradient steps per mini-batch
  max_tokens_per_document: 4096  # Truncation limit

alpha:
  initial: 1.0                 # Starting blend weight (1.0 = fully frozen)
  decay_rate: 0.01             # Alpha decrease per learning step
  min_value: 0.3               # Floor for alpha decay

tfidf_gate:
  threshold: 0.3               # TF-IDF score below which gradients are masked
  calibration_samples: 2000    # Number of samples for IDF calibration

ace:
  ollama_model: "qwen2.5:7b"            # Ollama model for ACE roles
  ollama_base_url: "http://localhost:11434"  # Ollama API endpoint
  num_loops: 3                           # Generate-Reflect-Curate cycles per learn
  playbook_dir: "playbooks"             # Directory for saved playbooks
  max_strategies: 50                     # Max strategies before FIFO eviction
```

## Running Tests

```bash
# All 141 tests (~10 seconds, no GPU needed)
python -m pytest tests/

# By component
python -m pytest tests/test_model/          # DualMLP, modified Qwen, TF-IDF gate
python -m pytest tests/test_training/       # TTT-E2E engine
python -m pytest tests/test_jitrl/          # JitRL MVP, Full, comparison harness
python -m pytest tests/test_ace/            # ACE engine, roles, playbook, adapter
python -m pytest tests/test_evaluation/     # Benchmarks, forgetting metrics
python -m pytest tests/test_data/           # SQuAD pipeline, Oracle docs
python -m pytest tests/test_checkpointing/  # Checkpoint save/load
python -m pytest tests/test_cli/            # CLI menu and handlers
python -m pytest tests/test_config.py       # YAML config loading

# Single test by name
python -m pytest tests/test_model/test_dual_mlp.py -k "test_forward"
```

## GPU Validation Scripts

End-to-end validation on real GPU hardware (requires A10 or equivalent with 24GB VRAM):

```bash
# Validates 4 milestones sequentially:
#   1. Architecture: Loads Qwen2.5-1.5B + DualMLP injection, verifies 7 modified layers
#   2. TTT-E2E: Learns a test document, verifies weights change and loss is recorded
#   3. Sparse Memory: Calibrates TF-IDF gates, learns domain docs, measures forgetting ratio (<0.15)
#   4. Oracle Docs: Fetches live Oracle documentation, learns from it, tests Oracle-specific Q&A
python scripts/validate_gpu.py

# Compares JitRL MVP vs Full on identical Oracle AI Vector Search content:
#   - Tests each engine individually (learn time, generate time, response quality)
#   - Runs comparison harness with 3 QA items, reports accuracy/timing side by side
python scripts/validate_jitrl.py
```

## Project Structure

```
src/continual_learning/
├── model/
│   ├── dual_mlp.py          # DualMLP: frozen + trainable MLPs with alpha blending
│   ├── modified_qwen.py     # Loads Qwen2.5-1.5B and injects DualMLP into layers 21-28
│   └── tfidf_gate.py        # TF-IDF gate: calibrates IDF scores, computes gradient masks
├── training/
│   ├── ttt_engine.py        # TTT-E2E: mini-batch gradient descent with TF-IDF masking
│   └── calibration.py       # Collects activations and calibrates TF-IDF gates
├── evaluation/
│   ├── benchmarks.py        # QA accuracy evaluation on holdout sets
│   └── forgetting_metrics.py # Catastrophic forgetting ratio computation
├── data/
│   ├── streaming_qa.py      # SQuAD 2.0 loader with learn/holdout splits
│   └── oracle_docs.py       # Fetches, parses, and chunks Oracle documentation
├── jitrl/
│   ├── base.py              # Abstract BaseJitRLEngine interface (learn/generate/clear)
│   ├── mvp/
│   │   ├── engine.py        # JitRL MVP: TF-IDF retrieval + context prepending + logit bias
│   │   ├── retriever.py     # TF-IDF document retriever with chunking
│   │   └── logit_bias.py    # Computes per-token bias from retrieved chunks
│   ├── full/
│   │   ├── engine.py        # JitRL Full: hidden-state knowledge store + reward modulation
│   │   ├── knowledge_store.py # Stores and retrieves document embeddings by cosine similarity
│   │   └── reward.py        # Computes reward vectors and modulates logits
│   └── comparison.py        # A/B harness: runs identical benchmarks across engines
├── ace/
│   ├── engine.py            # ACE orchestrator: Generate-Reflect-Curate loop
│   ├── generator.py         # Generator role: answers questions using playbook + context
│   ├── reflector.py         # Reflector role: critiques answers, suggests improvements
│   ├── curator.py           # Curator role: patch-updates playbook (never full rewrite)
│   ├── playbook.py          # Evolving strategy playbook with JSON persistence
│   ├── ollama_client.py     # Thin sync HTTP client for Ollama /api/generate
│   └── adapter.py           # Wraps ACEEngine as BaseJitRLEngine for comparison harness
├── checkpointing/
│   └── manager.py           # Saves/loads trainable weights, TF-IDF stats, alpha, metadata
├── cli/
│   └── main.py              # Interactive menu (Questionary + Rich) with all handlers
└── config.py                # YAML config loader with defaults merge
```

## License

MIT

---

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-jasperan-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/jasperan)&nbsp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-jasperan-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jasperan/)

</div>
