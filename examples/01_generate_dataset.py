"""Step 1 of 3 — build a synthetic training dataset with a teacher model.

Install the framework from PyPI first (either works):

    pip install distill-anything            # pip
    uv pip install distill-anything         # uv

Then run:

    python examples/01_generate_dataset.py

What this does
--------------
1. Writes a handful of seed prompts to disk (bring your own domain prompts here —
   support tickets, legal questions, API docs Q&A, whatever your specialty is).
2. Resolves a *teacher* from a spec string. Everything in Distill Anything picks
   its knowledge source the same way:

       hf:HuggingFaceTB/SmolLM2-360M-Instruct   local Hugging Face model (white-box)
       claude                                   Anthropic API (needs ANTHROPIC_API_KEY
                                                and: pip install "distill-anything[anthropic]")
       openai:gpt-4o-mini                       OpenAI API (needs OPENAI_API_KEY
                                                and: pip install "distill-anything[openai]")
       ollama:llama3.2                          local Ollama server

3. Expands the seeds into more prompts (self-instruct style), asks the teacher to
   answer every prompt, dedups, and writes a training JSONL:

       {"prompt": ..., "response": ..., "teacher": ...}
"""

from pathlib import Path

from distillanything.data.generate import generate_dataset
from distillanything.teachers.registry import resolve_teacher

# --- Configuration -----------------------------------------------------------

# Swap this one string to change where knowledge comes from (see table above).
# The default is a small local model so the script runs without any API key
# (~700MB download on first run; uses Apple-Silicon MPS / CUDA automatically).
TEACHER = "hf:HuggingFaceTB/SmolLM2-360M-Instruct"

SEEDS_PATH = Path("seeds.txt")
OUT_PATH = Path("data/train.jsonl")

# How many *new* prompts the teacher should invent per seed. 0 disables expansion.
# More expansion = bigger dataset = better student, at the cost of teacher time.
EXPAND_PER_SEED = 3

# Max tokens per teacher answer. Longer answers teach more but cost more.
MAX_TOKENS = 256

# --- 1. Seed prompts ---------------------------------------------------------
# Replace these with prompts from YOUR domain — the student will specialize in
# whatever distribution you seed here.

SEEDS = """\
Explain the difference between a list and a tuple in Python.
What are three ways to reduce cloud compute costs?
Explain gradient descent using a hiking analogy.
Write a short email declining a meeting politely.
What is the difference between concurrency and parallelism?
"""

SEEDS_PATH.write_text(SEEDS)
print(f"Wrote {len(SEEDS.splitlines())} seed prompts to {SEEDS_PATH}")

# --- 2 & 3. Resolve the teacher and generate ---------------------------------

teacher = resolve_teacher(TEACHER)

records = generate_dataset(
    teacher,
    seeds_path=SEEDS_PATH,
    out_path=OUT_PATH,
    max_tokens=MAX_TOKENS,
    expand_per_seed=EXPAND_PER_SEED,
    # system="You are a concise senior engineer."   # optional teacher persona
)

print(f"\nDone: {len(records)} records in {OUT_PATH}")
print("Next: python examples/02_train_student.py")

# Tip — quality-gate the dataset with an LLM judge before training (drops weak
# teacher answers). Same thing via the CLI:
#   distill generate seeds.txt --teacher claude --judge claude --min-score 7
