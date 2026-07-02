"""Step 3 of 3 — judge the student against the teacher and price it: the report card.

Prerequisites:

    pip install distill-anything          # or: uv pip install distill-anything
    python examples/01_generate_dataset.py
    python examples/02_train_student.py

Then run:

    python examples/03_evaluate_and_report.py

What this does
--------------
1. Loads the distilled student from runs/my-student.
2. Generates student answers for N held-out prompts from the dataset.
3. A JUDGE compares student vs reference answers BLIND, twice per pair with
   positions swapped (so a judge biased toward "Answer A" nets out to a tie).
   Verdicts roll up into one headline number:

       quality_retention = (wins + ties) / n
       "how often is the student at least as good as the teacher?"

4. Benchmarks both models: p50/p95 latency, tokens/s, memory, disk, and
   $ per 1K generated tokens at your hardware price.
5. Writes runs/my-student/REPORT.md (share this) and report.json (parse this).

Judge choice matters: a small local judge is free but coarse (most pairs tie).
For sharp discrimination use a frontier judge — any teacher spec works:
    JUDGE = "claude"            # needs ANTHROPIC_API_KEY + [anthropic] extra
    JUDGE = "openai:gpt-4o"     # needs OPENAI_API_KEY   + [openai] extra
"""

from distillanything.eval.report import build_report

RUN_DIR = "runs/my-student"                             # from step 2
DATASET = "data/train.jsonl"                            # from step 1
TEACHER = "hf:HuggingFaceTB/SmolLM2-360M-Instruct"      # reference + benchmark rival
JUDGE = "hf:HuggingFaceTB/SmolLM2-360M-Instruct"        # swap for "claude" if you can

report_path = build_report(
    RUN_DIR,
    DATASET,
    teacher=TEACHER,
    judge=JUDGE,
    n=16,                        # held-out prompts to judge (more = tighter estimate)
    max_new_tokens=192,
    hardware_cost_per_hour=1.20,  # your $/hour -> cost per 1K tokens in the report
)

print(f"\nOpen it: {report_path}")
print((report_path).read_text()[:600])

# Same thing via the CLI:
#   distill report runs/my-student --dataset data/train.jsonl \
#     --teacher hf:HuggingFaceTB/SmolLM2-360M-Instruct --judge claude \
#     --n 32 --cost-per-hour 1.20
