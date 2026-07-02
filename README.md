# Distill Anything

**The open-source lifecycle framework for building specialized models.**

Big models know things. Small models ship. Distill Anything covers the full loop —
**generate → clean → distill → evaluate → benchmark** — with one config schema that
scales from a MacBook to a GPU cluster.

```
Seed prompts ──▶ Teacher (Claude / GPT / local HF / Ollama)
                      │
                      ▼
              Synthetic dataset (dedup + filtering)
                      │
                      ▼
              Distillation engine
              ├─ logit KD  (white-box: forward KL / reverse KL / JSD, top-k)
              └─ seqKD     (black-box: fine-tune on teacher outputs)
                      │
                      ▼
              Evaluation (perplexity, teacher agreement)
                      │
                      ▼
              Benchmark (latency, tokens/s, memory, footprint)
```

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

- [ ] Hidden-state / feature KD with learned projectors
- [ ] Cross-tokenizer logit distillation (ULD)
- [ ] Preference distillation (teacher-as-judge → DPO)
- [ ] Multi-teacher voting and ensembling
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
