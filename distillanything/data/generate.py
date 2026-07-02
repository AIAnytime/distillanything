"""Synthetic dataset generation: seed prompts -> teacher completions -> training JSONL.

This is the black-box distillation front door: point it at Claude/GPT/a local model
and it manufactures an instruction dataset for the student.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console

from distillanything.data.filters import clean_records
from distillanything.data.formats import load_records, save_records
from distillanything.teachers.base import Teacher

console = Console()

EXPAND_PROMPT = """You are helping build a training dataset. Given the topic or example prompt below, \
write {n} new, diverse, self-contained prompts a user might realistically ask on the same theme. \
Output one prompt per line with no numbering and no extra commentary.

Topic/example: {seed}"""


def expand_prompts(teacher: Teacher, seeds: list[str], per_seed: int = 5, max_tokens: int = 1024) -> list[str]:
    """Self-instruct-lite: grow seed prompts into a larger prompt set via the teacher."""
    meta = [EXPAND_PROMPT.format(n=per_seed, seed=seed) for seed in seeds]
    outputs = teacher.generate(meta, max_tokens=max_tokens)
    expanded: list[str] = []
    for output in outputs:
        expanded.extend(line.strip() for line in output.splitlines() if line.strip())
    return expanded


def generate_dataset(
    teacher: Teacher,
    seeds_path: str | Path,
    out_path: str | Path,
    system: Optional[str] = None,
    max_tokens: int = 512,
    expand_per_seed: int = 0,
    batch_size: int = 16,
) -> list[dict]:
    """Read seed prompts, (optionally) expand them, get teacher responses, write JSONL."""
    records = load_records(seeds_path)
    prompts = [r["prompt"] for r in records if r.get("prompt")]
    console.print(f"[bold]Loaded {len(prompts)} seed prompts[/] from {seeds_path}")

    if expand_per_seed > 0:
        console.print(f"Expanding prompts with teacher [cyan]{teacher.name}[/] "
                      f"({expand_per_seed} per seed)...")
        prompts = prompts + expand_prompts(teacher, prompts, per_seed=expand_per_seed)
        console.print(f"Prompt pool is now {len(prompts)}")

    dataset: list[dict] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        responses = teacher.generate(batch, system=system, max_tokens=max_tokens)
        dataset.extend(
            {"prompt": p, "response": r, "teacher": teacher.name}
            for p, r in zip(batch, responses)
        )
        console.print(f"  generated {min(start + batch_size, len(prompts))}/{len(prompts)}")

    dataset = clean_records(dataset, dedup=True, min_response_chars=1)
    save_records(dataset, out_path)
    console.print(f"[green]Wrote {len(dataset)} records[/] to {out_path}")
    return dataset
