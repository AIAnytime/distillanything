"""Laptop quickstart: distill SmolLM2-360M into SmolLM2-135M with white-box logit KD.

Downloads ~1GB of models on first run; trains on MPS/CUDA/CPU automatically.

    python examples/quickstart_mac.py
"""

from distillanything import Student
from distillanything.data.generate import generate_dataset
from distillanything.teachers.registry import resolve_teacher

TEACHER = "hf:HuggingFaceTB/SmolLM2-360M-Instruct"

# 1. Let the teacher manufacture a small dataset from seed prompts.
teacher = resolve_teacher(TEACHER)
generate_dataset(
    teacher,
    seeds_path="examples/data/seed_prompts.txt",
    out_path="data/train.jsonl",
    max_tokens=256,
    expand_per_seed=3,
)

# 2. Distill on logits (same tokenizer family, so white-box KD is valid).
student = Student("HuggingFaceTB/SmolLM2-135M-Instruct")
results = student.learn(
    teacher=TEACHER,
    dataset="data/train.jsonl",
    epochs=2,
    batch_size=2,
    grad_accum=8,
    output_dir="runs/mac-small",
)
print("Eval:", results.get("eval"))

# 3. Vibe-check and measure the student.
print(student.generate("Explain gradient descent using a hiking analogy."))
print(student.benchmark())
