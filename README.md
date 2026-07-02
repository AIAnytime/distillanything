# Distill Anything

**The open-source lifecycle framework for building specialized models.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange)](#)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](#development)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)

Big models know things. Small models ship. Distill Anything covers the full loop —
**generate → clean → distill → evaluate → benchmark** — with one config schema that
scales from a MacBook to a GPU cluster.

<p align="center">
  <img src="media/architecture.png" alt="Distill Anything architecture: seed prompts through teachers, synthetic dataset pipeline, distillation engine, evaluation, benchmark, to Python SDK / CLI / REST API" width="100%">
</p>

The diagram is the north star for the project, not a feature list of what's in your
`pip install` today. The table below maps every box to its real status so there's no
surprise between the picture and the code.

**Legend:** ✅ shipped &nbsp;·&nbsp; 🚧 partial &nbsp;·&nbsp; 🗺️ planned, not yet built

| Diagram section | Box | Status | Notes |
|---|---|:---:|---|
| Seed Prompts | Text / Docs, Custom | ✅ | `.txt` (one prompt/line) and `.jsonl` (`prompt`/`messages`/`text`) |
| Seed Prompts | Code | 🚧 | No special handling — treated as plain text, works but untuned |
| Teacher | Claude API, GPT API, Hugging Face (Local), Ollama (Local) | ✅ | `hf:` / `claude` / `openai:` / `ollama:` teacher specs |
| Synthetic Dataset Pipeline | Generate, Deduplication, Curated Dataset | ✅ | `distill generate`, normalized-content dedup |
| Synthetic Dataset Pipeline | Filtering | 🚧 | Empty/too-short response filter only — no content or toxicity filters yet |
| Synthetic Dataset Pipeline | Quality Scoring | ✅ | LLM-judge 1-10 scoring: `distill generate --judge claude --min-score 7` |
| Distillation Engine | Forward KL, Reverse KL, JSD, Top-k/Temp | ✅ | White-box logit KD, all three divergences + top-k truncation |
| Distillation Engine | Response Supervision, Instruction Tuning | ✅ | Black-box seqKD (fine-tune on teacher-generated text) |
| Distillation Engine | Preference (Optional) | 🗺️ | No DPO/preference-based distillation yet |
| Evaluation | Perplexity | ✅ | |
| Evaluation | Teacher Agreement | ✅ | Win/tie/lose via blind, position-swapped LLM judge (`distill report --judge`) + token-level agreement |
| Evaluation | Accuracy/EM, BLEU/ROUGE/BERTScore, Safety/Bias Checks | 🗺️ | No task-benchmark or safety-eval harness yet |
| Benchmark | Tokens/s, Memory, Model Size | ✅ | `distill benchmark` |
| Benchmark | Latency | ✅ | p50/p95 over repeated runs (`--n-runs`) |
| Benchmark | Cost / 1K Tokens | ✅ | From measured throughput × your `--cost-per-hour` |
| Outputs | Model Weights, Tokenizer & Config, Training Logs, Benchmarks | ✅ | Saved to `output_dir` on every run |
| Outputs | Evaluation Report | ✅ | `distill report` writes a shareable REPORT.md + report.json per run |
| Core Capabilities | Reproducible Pipelines | ✅ | Seeded runs + full config snapshot saved alongside the checkpoint |
| Core Capabilities | Multi-Teacher Support, Multi-Modal, Distributed Training, Quantization/Export, Experiment Tracking | 🗺️ | One teacher/one device per run today; text-only; no W&B/MLflow hooks |
| Integrations | Hugging Face | ✅ | Models, tokenizers, chat templates |
| Integrations | Weights & Biases, MLflow, S3/GCS/Azure Blob, Docker/Kubernetes | 🗺️ | Not integrated yet |
| How you interact | Python SDK, CLI | ✅ | `Student().learn(...)` and `distill ...` |
| How you interact | REST API, Web UI | 🗺️ | CLI/SDK only for now |

## Why

Everyone wants GPT-class quality at 1/100th the size and cost. Almost nobody has
tooling for the *whole lifecycle* — existing libraries stop at the training loop.
Distill Anything treats distillation as a pipeline, not a loss function.

## Install

```bash
pip install -e .                    # core (torch, transformers)
pip install -e ".[anthropic]"       # + Claude as a teacher
pip install -e ".[openai]"          # + OpenAI / vLLM / Ollama teachers
```

## 60-second sanity check (no downloads)

```bash
distill smoke
```

Trains a tiny random student against a tiny random teacher with logit KD, evaluates,
and benchmarks — entirely offline. If this passes, your machine is ready.

## Quickstart 1 — white-box KD on a laptop

Distill SmolLM2-360M into SmolLM2-135M (same tokenizer family — required for logit KD):

```bash
# 1. Build a dataset with the teacher itself
distill generate examples/data/seed_prompts.txt \
  --teacher hf:HuggingFaceTB/SmolLM2-360M-Instruct \
  --out data/train.jsonl --expand 5

# 2. Distill
distill train recipes/mac-small.yaml

# 3. Inspect
distill benchmark runs/mac-small
distill chat runs/mac-small -p "Explain gradient descent using a hiking analogy."
```

## Quickstart 2 — Claude as the teacher (black-box seqKD)

```bash
export ANTHROPIC_API_KEY=...
distill generate examples/data/seed_prompts.txt --teacher claude \
  --out data/claude_train.jsonl --expand 10
distill train recipes/claude-blackbox.yaml
```

## The report card — "was it worth it?"

After training, build a shareable REPORT.md that answers the only question that
matters: *did the student keep the teacher's quality at a fraction of the cost?*

```bash
distill report runs/mac-small \
  --dataset data/train.jsonl \
  --teacher hf:HuggingFaceTB/SmolLM2-360M-Instruct \
  --judge claude \
  --n 32 --cost-per-hour 1.20
```

What it does:

- **Quality**: a blind, position-swapped LLM judge (any teacher spec works as judge)
  compares student vs reference answers → win/tie/lose and a single headline number,
  *quality retention* (how often the student is at least as good).
- **Efficiency**: side-by-side student-vs-teacher table — params, tokens/s, p50/p95
  latency, memory, and $ per 1K tokens at your hardware price.
- **Receipts**: sample outputs + the run's training metrics, written to
  `REPORT.md` (human) and `report.json` (machines).

The same judge can gate your synthetic data before training:

```bash
distill generate seeds.txt --teacher claude --judge claude --min-score 7
```

## The SDK

```python
from distillanything import Student

student = Student("HuggingFaceTB/SmolLM2-135M-Instruct")

# White-box: local teacher, KL on logits
student.learn(
    teacher="hf:HuggingFaceTB/SmolLM2-360M-Instruct",
    dataset="data/train.jsonl",
    epochs=2, lr=1e-4,
)

# Black-box: API teacher generates missing responses, student fine-tunes on them
student.learn(teacher="claude", dataset="data/prompts_only.jsonl")

print(student.generate("Explain what a database index is."))
print(student.benchmark())
student.save("runs/my-model")
```

## Teachers

One string selects any knowledge source:

| Spec | Backend | KD mode |
|---|---|---|
| `hf:HuggingFaceTB/SmolLM2-360M-Instruct` | local Hugging Face model | white-box (logits) |
| `claude` / `claude:claude-opus-4-8` | Anthropic API | black-box (seqKD) |
| `openai:gpt-4o-mini` | OpenAI API | black-box |
| `ollama:llama3.2` | local Ollama server | black-box |

## Recipes

Everything is a YAML recipe (`distill init` writes a starter):

```yaml
mode: logit                # or seqkd
teacher: { spec: hf:HuggingFaceTB/SmolLM2-360M-Instruct }
student: { model: HuggingFaceTB/SmolLM2-135M-Instruct }
data:    { path: data/train.jsonl, max_seq_len: 512 }
loss:    { kind: forward_kl, temperature: 2.0, alpha: 0.5, top_k: 256 }
train:   { output_dir: runs/out, lr: 1.0e-4, epochs: 2, batch_size: 2, grad_accum: 8 }
```

Loss options: `forward_kl` (classic Hinton), `reverse_kl` (mode-seeking, MiniLLM-style),
`jsd` (interpolated, numerically gentle). `top_k` truncates KD to the teacher's top-k
logits — a large memory win on 50k+ vocabularies.

**Logit KD requires a shared tokenizer** (use models from the same family). Across
tokenizers, use `mode: seqkd`. Cross-tokenizer logit alignment (ULD) is on the roadmap.

## Hardware

Device selection is automatic: CUDA (bf16 autocast) → Apple Silicon MPS (fp32) → CPU.
Recipes are sized so `mac-small` runs on a 16GB MacBook.

## Roadmap

Beyond closing the 🗺️ gaps in the status table above, also planned (not pictured in
the diagram):

- [ ] Hidden-state / feature KD with learned projectors
- [ ] Cross-tokenizer logit distillation (ULD)
- [ ] LoRA/QLoRA students for larger models on small hardware
- [ ] VLM, embedding, and reranker distillation
- [ ] Eval harness integration (lm-eval-harness) and regression tracking

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest            # fully offline test suite (tiny random models)
distill smoke     # end-to-end pipeline check
```

Licensed under Apache-2.0.
