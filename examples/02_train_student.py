"""Step 2 of 3 — distill the teacher's knowledge into a small student model.

Prerequisites:

    pip install distill-anything          # or: uv pip install distill-anything
    python examples/01_generate_dataset.py

Then run:

    python examples/02_train_student.py

The two distillation modes (picked automatically from the teacher spec)
------------------------------------------------------------------------
- WHITE-BOX ("logit" KD) — teacher runs locally, so we can match the student's
  full next-token distribution to the teacher's on every response token:

      loss = alpha * KL(student || teacher) + (1 - alpha) * cross-entropy

  Requires student and teacher to SHARE A TOKENIZER (same model family) —
  that's why this example pairs SmolLM2-135M with SmolLM2-360M.

- BLACK-BOX ("seqkd") — API teachers (claude / openai: / ollama:) expose text,
  not logits. The teacher's knowledge is already baked into the dataset it
  wrote in step 1, so the student simply fine-tunes on it (plain CE loss).
  Works across any tokenizer pair.

`Student.learn(mode="auto")` chooses: hf: teacher -> logit, API teacher -> seqkd.
"""

from distillanything import Student

# Must match step 1 — with an hf: teacher this trains white-box on logits.
TEACHER = "hf:HuggingFaceTB/SmolLM2-360M-Instruct"

# The student: same family as the teacher (shared tokenizer -> logit KD is valid).
STUDENT = "HuggingFaceTB/SmolLM2-135M-Instruct"

student = Student(STUDENT)

results = student.learn(
    teacher=TEACHER,
    dataset="data/train.jsonl",
    # --- training overrides (anything on TrainConfig/LossConfig/DataConfig) ---
    epochs=2,
    batch_size=2,       # keep small on laptops; effective batch = batch_size *
    grad_accum=8,       # grad_accum = 16 sequences per optimizer step
    lr=1e-4,
    output_dir="runs/my-student",
    # kind="reverse_kl",  # try MiniLLM-style mode-seeking KD
    # top_k=256,          # truncate KD to teacher's top-k logits (saves memory)
)

# results carries the training history and (if an eval split existed) metrics:
#   perplexity          how "surprised" the student is by held-out teacher data
#   teacher_agreement   how often student & teacher pick the same next token
print("\nFinal eval:", results.get("eval"))

# The trained model is a normal Hugging Face checkpoint — load it anywhere:
#   AutoModelForCausalLM.from_pretrained("runs/my-student")
print(student.generate("Explain gradient descent using a hiking analogy."))
print("\nNext: python examples/03_evaluate_and_report.py")
