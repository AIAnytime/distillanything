<div align="center"><pre>
  ██████╗ ██╗███████╗████████╗██╗██╗     ██╗
  ██╔══██╗██║██╔════╝╚══██╔══╝██║██║     ██║
  ██║  ██║██║███████╗   ██║   ██║██║     ██║
  ██║  ██║██║╚════██║   ██║   ██║██║     ██║
  ██████╔╝██║███████║   ██║   ██║███████╗███████╗
  ╚═════╝ ╚═╝╚══════╝   ╚═╝   ╚═╝╚══════╝╚══════╝
              A N Y T H I N G
     Distill any model into one you own
</pre></div>

<p align="center"><strong>generate → distill → judge → benchmark · white-box KD + black-box seqKD · any teacher (Claude / GPT / HF / Ollama) · LLM-as-judge report card · runs on a MacBook</strong></p>

<p align="center">
  <a href="https://github.com/AIAnytime/distillanything/actions/workflows/ci.yml"><img src="https://github.com/AIAnytime/distillanything/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/distill-anything/"><img src="https://img.shields.io/pypi/v/distill-anything.svg" alt="PyPI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License: Apache 2.0"></a>
  <a href="#development"><img src="https://img.shields.io/badge/tests-36%20passing%20offline-brightgreen" alt="Tests"></a>
  <a href="#whats-real-today-vs-the-vision"><img src="https://img.shields.io/badge/status-alpha-orange" alt="Status: alpha"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-261230" alt="Code style: ruff"></a>
</p>

<p align="center">
  <a href="#get-started-60-seconds">Install</a> ·
  <a href="#proof">Proof</a> ·
  <a href="#the-report-card">Report card</a> ·
  <a href="#the-dashboard">Dashboard</a> ·
  <a href="#examples-notebook-and-tests-as-docs">Examples</a> ·
  <a href="#teachers">Teachers</a> ·
  <a href="#compared-to">Compared to</a> ·
  <a href="#roadmap">Roadmap</a>
</p>

---

Big models know things. Small models ship. Distill Anything covers the **whole distillation lifecycle** — a teacher generates your dataset, a student trains on its logits or its words, a judge scores the result blind, and a benchmark prices it — with one YAML recipe schema that scales from a 16GB MacBook to a GPU cluster.

<p align="center">
  <img src="https://raw.githubusercontent.com/AIAnytime/distillanything/main/media/demo.gif" alt="distill smoke running end-to-end on a MacBook" width="820">
  <br/><sub>Live: <code>distill smoke</code> — full logit-KD pipeline, offline, &lt;60s on a MacBook. loss 2.24 → 2.08, PASS.</sub>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/AIAnytime/distillanything/main/media/architecture.png" alt="Distill Anything architecture: seed prompts through teachers, synthetic dataset pipeline, distillation engine, evaluation, benchmark" width="100%">
  <br/><sub>The full picture. What's shipped vs planned: <a href="#whats-real-today-vs-the-vision">status table</a>.</sub>
</p>

## What it does

- **Any teacher** — `claude`, `openai:gpt-4o-mini`, `ollama:llama3.2`, or any local `hf:` model, selected by one spec string
- **Synthetic data** — `distill generate` turns seed prompts into a deduped instruction dataset, with self-instruct-style prompt expansion (`--expand 5`)
- **Quality gating** — an LLM judge scores every record 1–10 before training (`--judge claude --min-score 7`)
- **White-box logit KD** — forward KL (Hinton), reverse KL (MiniLLM-style), generalized JSD, temperature scaling, and top-k truncation so 50k-vocab KD fits in laptop memory
- **Black-box seqKD** — no logits needed; fine-tune on what an API teacher wrote
- **LLM-as-judge eval** — blind A/B, position-swapped to kill judge bias → win/tie/lose and one headline number: *quality retention*
- **Report card** — `distill report` writes a shareable `REPORT.md`: quality vs the teacher, p50/p95 latency, tokens/s, memory, $ per 1K tokens
- **LoRA / QLoRA students** — freeze the base, train adapters: 1–3B students on a 16GB laptop (`[lora]` extra); 4-bit QLoRA on CUDA (`[qlora]`)
- **Hardware-aware** — CUDA (bf16) → Apple Silicon MPS → CPU, detected automatically; teachers load in half precision on GPU/MPS
- **Dashboard** — `distill ui`: live loss curves for every run (CLI-started ones too), report cards, run comparison, and a control plane to launch/stop training and generate datasets from the browser (`[ui]` extra)

## How it works (30 seconds)

<p align="center">
  <img src="https://raw.githubusercontent.com/AIAnytime/distillanything/main/media/distil.png" alt="The pipeline: seed prompts → teacher (claude / openai / ollama / hf) with generate + expand + dedup + judge-score → curated JSONL dataset → distillation engine (logit KD for hf teachers, seqKD for any teacher) → blind position-swapped judge (win/tie/lose) → REPORT.md with quality retention, size, latency, and cost" width="560">
</p>

One `DistillConfig` YAML describes the whole run; the same recipe format works for a 135M student on a MacBook and a 7B student on an A100.

## Get started (60 seconds)

```bash
# 1 — Install
pip install distill-anything                    # core (torch, transformers)
pip install "distill-anything[anthropic]"       # optional: Claude as teacher/judge
pip install "distill-anything[openai]"          # optional: OpenAI / vLLM / Ollama teachers

# 2 — Verify your machine (tiny random models, zero downloads, <1 min)
distill smoke

# 3 — First real distillation (SmolLM2-360M → 135M, ~1GB download, runs on MPS)
git clone https://github.com/AIAnytime/distillanything && cd distillanything   # example seeds + recipes
distill generate examples/data/seed_prompts.txt \
  --teacher hf:HuggingFaceTB/SmolLM2-360M-Instruct --out data/train.jsonl --expand 5
distill train recipes/mac-small.yaml
distill report runs/mac-small --dataset data/train.jsonl \
  --teacher hf:HuggingFaceTB/SmolLM2-360M-Instruct --judge hf:HuggingFaceTB/SmolLM2-360M-Instruct
```

Prefer Claude as the teacher? `export ANTHROPIC_API_KEY=...` then swap `--teacher claude` and `distill train recipes/claude-blackbox.yaml`.

## Proof

We publish only what we've measured. Every number below reproduces on a stock Apple-Silicon MacBook with the commands shown — no cluster required.

**End-to-end pipeline, offline, under a minute** (`distill smoke`):

```
Training mode=logit device=mps steps=30 loss=forward_kl(T=2.0, alpha=0.5)
  step 10/30  loss 2.2371   →   step 30/30  loss 2.0767
PASS: loss 2.237 -> 2.077 over 30 steps
```

**Benchmark output** (`distill benchmark sshleifer/tiny-gpt2 --n-runs 3 --cost-per-hour 1.20`, measured on an M-series MacBook):

| Metric | Value |
|---|---:|
| latency_p50_s | 0.054 |
| latency_p95_s | 0.079 |
| tokens_per_s | 262.4 |
| memory_mb | 0.4 |
| cost_per_1k_tokens_usd | 0.001271 |

**A real distillation, end to end on one MacBook** — the exact Quickstart above (SmolLM2-360M teaches 135M on 60 records it generated itself, ~15 min total, all local):

| | Student (135M, distilled) | Teacher (360M) |
|---|---:|---:|
| Parameters | 134.5M | 361.8M |
| Tokens / s (MPS) | **70.5** | 56.7 |
| Latency p50 | **0.40s** | 0.76s |
| Disk | **538MB** | 1447MB |
| $ / 1K tokens @ $1.20/hr | **$0.0047** | $0.0059 |

- **Quality retention 87.5%** on 24 held-out prompts (1 win / 20 ties / 3 losses), judged blind with position swap by the teacher itself. A 360M judge is coarse — most pairs tie; point `--judge claude` at it for sharper discrimination.
- Token-level teacher agreement **81%**, perplexity 2.17 (3-example eval split — demo scale).
- The full unedited artifact: [examples/sample-report.md](examples/sample-report.md), generated by `distill report`.

This is the quickstart, not a leaderboard claim — 60 records and 8 optimizer steps prove the loop runs on a laptop, not the ceiling of the method.

**Honest-by-construction evals:** the judge sees answers blind and judges each pair twice with positions swapped — a judge that always prefers "Answer A" nets out to a tie (that exact adversarial case is in the test suite). Disagreements and unparseable verdicts count as ties, never wins. 36 tests run fully offline against tiny random models.

## The report card

The question that matters is *"did the student keep the teacher's quality at a fraction of the cost?"* — so every run can end with a one-page answer:

```bash
distill report runs/mac-small \
  --dataset data/train.jsonl \
  --teacher hf:HuggingFaceTB/SmolLM2-360M-Instruct \
  --judge claude --n 32 --cost-per-hour 1.20
```

`REPORT.md` leads with **quality retention** (how often the student matches or beats the reference), then a side-by-side efficiency table (params, tokens/s, p50/p95, memory, $/1K tokens — "3.0x smaller and 3.0x faster"), sample outputs, and the training metrics. `report.json` sits next to it for machines.

## The dashboard

```bash
pip install "distill-anything[ui]"
distill ui
```

One command starts a local dashboard (dark-first, W&B-style) over your `runs/` and `data/` directories:

- **Live training charts** — loss/KD/CE/LR stream in over SSE while a run trains. Runs started from the **CLI show up too**: the trainer writes `metrics.jsonl` + `status.json` into every run dir, and the UI just tails files.
- **Report cards** — quality retention, win/tie/lose, and the efficiency table rendered per run.
- **Compare** — overlay loss curves of up to six runs.
- **Control plane** — a New-run wizard (the form *is* the recipe YAML, saved into the run dir), teacher-driven dataset generation with judge gating, and stop buttons. One training job at a time by default — the right call on a 16GB laptop; extra submissions queue.

**Security model** (the dashboard can spawn training processes, so it gets the Jupyter treatment, not the "it's only local" treatment): binds `127.0.0.1` only by default; a per-session bearer token is always required — even on localhost — and is auto-generated and embedded in the URL it prints; Host-header allowlisting blocks DNS-rebinding; strict CSP/security headers; every client-supplied name is sandbox-resolved inside your runs/data directories; API keys for teachers are read from the environment where `distill ui` runs — the UI never asks for, stores, or returns them. Binding beyond localhost refuses to start without an explicit `--token` you chose yourself.

## Examples, notebook, and tests-as-docs

Three ways to learn the framework, all installing from PyPI (`pip install distill-anything` or `uv pip install distill-anything`):

**📓 The notebooks**

| Notebook | What it covers |
|---|---|
| [`examples/sample_notebook.ipynb`](examples/sample_notebook.ipynb) | The full lifecycle: install → offline self-test → teacher-generated dataset → logit KD → rendered report card → benchmark → swapping in Claude/GPT teachers (~15 min on a MacBook) |
| [`examples/lora_notebook.ipynb`](examples/lora_notebook.ipynb) | LoRA & QLoRA students: why adapters beat the AdamW memory wall, trainable-param collapse, merged vs adapter-only checkpoints, the CUDA-only QLoRA path, and the 1.5B-on-16GB recipe (~5 min; committed with real executed outputs) |

**🐍 The scripts** — the same flow as three standalone, heavily commented files you can copy into your own project:

| Script | What it teaches |
|---|---|
| [`examples/01_generate_dataset.py`](examples/01_generate_dataset.py) | Teacher specs, seed prompts, self-instruct expansion, dataset JSONL format |
| [`examples/02_train_student.py`](examples/02_train_student.py) | `Student.learn()`, white-box vs black-box mode selection, the KD loss knobs |
| [`examples/03_evaluate_and_report.py`](examples/03_evaluate_and_report.py) | Blind position-swapped judging, quality retention, the REPORT.md artifact |

**🧪 The tests as living docs** — the suite runs offline in seconds and doubles as executable documentation:

| Test file | Read it to learn |
|---|---|
| [`tests/test_losses.py`](tests/test_losses.py) | The KD math contracts: non-negativity, masking, top-k truncation, backprop |
| [`tests/test_judge.py`](tests/test_judge.py) | How to fake a `Teacher` for your own tests; why position-swap kills judge bias |
| [`tests/test_trainer.py`](tests/test_trainer.py) | Training end-to-end with tiny random models (the pattern for fast iteration) |
| [`tests/test_report.py`](tests/test_report.py) | The report data shape and a full build-report integration run |

```bash
git clone https://github.com/AIAnytime/distillanything && cd distillanything
pip install "distill-anything[dev]" && pytest    # green in seconds, no GPU/keys/network
```

## Teachers

One string selects any knowledge source — as teacher *or* as judge:

| Spec | Backend | KD mode |
|---|---|---|
| `hf:HuggingFaceTB/SmolLM2-360M-Instruct` | local Hugging Face model | white-box (logits) |
| `claude` / `claude:claude-opus-4-8` | Anthropic API | black-box (seqKD) |
| `openai:gpt-4o-mini` | OpenAI API (or any compatible endpoint) | black-box |
| `ollama:llama3.2` | local Ollama server | black-box |

**One rule to know:** logit KD requires student and teacher to share a tokenizer (same model family). Across tokenizers, use `mode: seqkd` — cross-tokenizer logit alignment (ULD) is on the roadmap.

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

Or skip YAML entirely:

```python
from distillanything import Student

student = Student("HuggingFaceTB/SmolLM2-135M-Instruct")
student.learn(teacher="claude", dataset="data/prompts_only.jsonl")   # generates missing responses, then trains
print(student.generate("Explain what a database index is."))
print(student.benchmark())
```

### LoRA & QLoRA — big students on small hardware

Full fine-tuning caps laptop students around ~360M (AdamW keeps two moments per
weight). LoRA freezes the base and trains ~0.5% of the params, so a **1.5B student
learns from a 3B teacher on 16GB of RAM** — see [`recipes/mac-lora.yaml`](recipes/mac-lora.yaml):

```python
student = Student("Qwen/Qwen2.5-1.5B-Instruct", lora={"r": 16, "alpha": 32})
student.learn(teacher="hf:Qwen/Qwen2.5-3B-Instruct", dataset="data/train.jsonl",
              gradient_checkpointing=True, batch_size=1, grad_accum=16)
```

Adapters merge into the base on save by default, so the output stays a plain HF
checkpoint (set `merge_lora: false` for tiny adapter-only saves — every `distill`
command loads those transparently). On CUDA, add `"qlora": True` to load the frozen
base in 4-bit ([`recipes/gpu-qlora.yaml`](recipes/gpu-qlora.yaml)); QLoRA is CUDA-only
because bitsandbytes has no Apple Silicon backend. Install: `pip install "distill-anything[lora]"`.

## When to use · When to skip

**Great fit if you…**
- want a specialized model that's 10–100x smaller than the API model you're calling today
- have (or can seed) a few hundred domain prompts and an API key or a local teacher model
- work on a laptop — everything here is sized to run and iterate on 16GB of RAM

**Skip it if you…**
- need RAG or prompt engineering, not a trained model — distillation is for when the task is stable and volume is high
- need cross-tokenizer logit KD, multimodal, or distributed training *today* (roadmap, not shipped)
- expect a hosted platform — this is a framework; you bring the compute

<a id="whats-real-today-vs-the-vision"></a>
<details>
<summary><b>What's real today vs the vision</b></summary>

The architecture diagram is the north star, not the changelog. Every box, mapped honestly:

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
| How you interact | Web UI | ✅ | `distill ui` — live runs, report cards, compare, launch/stop jobs (v0.3) |
| How you interact | REST API | 🚧 | The dashboard's token-authed local API (`/api/...`); not yet a stable public contract |

</details>

<details>
<summary><b>What's inside</b></summary>

- **`teachers/`** — one `Teacher` abstraction; local HF models expose logits (white-box), API teachers expose text (black-box). Registry resolves spec strings.
- **`losses/kd.py`** — masked, temperature-scaled (T² gradient correction) forward KL / reverse KL / generalized JSD, with top-k support truncation and automatic vocab-padding alignment.
- **`data/`** — JSONL/txt loading, chat-template rendering, prompt-masked tokenization, normalized dedup, teacher-driven generation with prompt expansion.
- **`train/trainer.py`** — hand-rolled loop (no HF Trainer): grad accumulation, cosine LR + warmup, grad clipping, KD+CE mixing, MPS-safe autocast policy.
- **`eval/judge.py`** — pairwise blind judging with position-swap debiasing; absolute 1–10 scoring for dataset gating.
- **`eval/benchmark.py`** — p50/p95 latency, throughput, peak memory, $/1K tokens.
- **`eval/report.py`** — REPORT.md + report.json builder.
- **`testing.py`** — tiny random Llama models + in-memory char tokenizer, so the whole suite runs offline.

</details>

## Compared to

Excellent distillation *trainers* exist — the gap this project fills is the **lifecycle around the training loop**:

|  | Primary focus | Data generation from API teachers | Blind LLM-judge → report card | Sized for laptops |
|---|---|:---:|:---:|:---:|
| **Distill Anything** | Full lifecycle: generate → distill → judge → benchmark | ✅ | ✅ | ✅ |
| [TRL](https://github.com/huggingface/trl) (GKD) | RL/KD training methods in the HF ecosystem | — | — | GPU-oriented |
| [DistillKit](https://github.com/arcee-ai/DistillKit) | KD training techniques | — | — | GPU-oriented |
| [EasyDistill](https://github.com/modelscope/easydistill) | KD training + data synthesis toolkit | ✅ | — | GPU-oriented |

If you already have a curated dataset and a GPU and just want a training loop, TRL's GKD trainer is great — and our loss implementations follow the same literature (Hinton KD, MiniLLM, DistiLLM).

## Roadmap

Beyond closing the 🗺️ gaps in the status table:

- [x] **LoRA/QLoRA students** — distill into 1–3B students on 16GB of RAM *(shipped in v0.2)*
- [x] **Web dashboard** — live runs, report cards, and a local control plane (`distill ui`) *(shipped in v0.3)*
- [ ] Hidden-state / feature KD with learned projectors
- [ ] Cross-tokenizer logit distillation (ULD)
- [ ] Multi-teacher voting and ensembling
- [ ] VLM, embedding, and reranker distillation
- [ ] Eval harness integration (lm-eval-harness) and regression tracking

Everything in this repo is and stays Apache-2.0. A hosted layer (GPU orchestration, dashboards, model registry) may come later — the framework is the product, not the funnel.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest            # 36 tests, fully offline (tiny random models, fake judges)
distill smoke     # end-to-end pipeline check on your hardware
```

PRs welcome — the test suite needs no GPU, no API keys, and no downloads.

## License

Apache 2.0 — see [LICENSE](LICENSE).
