"""The report card: one shareable REPORT.md per run answering "was it worth it?".

Pulls together LLM-as-judge quality, training metrics, and efficiency benchmarks
into a single markdown artifact (plus report.json for machines).
"""

from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path
from typing import Optional, Union

from rich.console import Console

from distillanything.data.filters import clean_records
from distillanything.data.formats import load_records
from distillanything.eval.benchmark import benchmark_model
from distillanything.eval.judge import judge_pairwise, summarize_pairwise
from distillanything.teachers.base import Teacher
from distillanything.teachers.registry import resolve_teacher

console = Console()


def _fmt(value) -> str:
    return "—" if value is None else str(value)


def render_report(data: dict) -> str:
    """Pure markdown rendering of a report dict (see build_report for the shape)."""
    lines: list[str] = []
    lines.append(f"# Distillation Report — {data.get('student_name', 'student')}")
    lines.append("")
    lines.append(f"*Generated {data.get('generated_on', date.today().isoformat())} by Distill Anything*")
    lines.append("")

    judge = data.get("judge")
    if judge and judge.get("n", 0) > 0:
        retention = judge["quality_retention"] * 100
        lines.append("## Quality (LLM-as-judge)")
        lines.append("")
        lines.append(
            f"**The student matches or beats the reference on "
            f"{retention:.0f}% of {judge['n']} held-out prompts.**"
        )
        lines.append("")
        lines.append(f"Judge: `{data.get('judge_name', '?')}`, blind A/B with position swap.")
        lines.append("")
        lines.append("| Student wins | Ties | Reference wins |")
        lines.append("|:---:|:---:|:---:|")
        lines.append(
            f"| {judge['student_wins']} ({judge['student_win_rate']:.0%}) "
            f"| {judge['ties']} ({judge['tie_rate']:.0%}) "
            f"| {judge['teacher_wins']} ({judge['teacher_win_rate']:.0%}) |"
        )
        lines.append("")

    metrics = data.get("train_eval")
    if metrics:
        lines.append("## Training metrics")
        lines.append("")
        for key, value in metrics.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    student_bench = data.get("student_benchmark")
    teacher_bench = data.get("teacher_benchmark")
    if student_bench:
        lines.append("## Efficiency")
        lines.append("")
        keys = [
            ("parameters_m", "Parameters (M)"),
            ("tokens_per_s", "Tokens / s"),
            ("latency_p50_s", "Latency p50 (s)"),
            ("latency_p95_s", "Latency p95 (s)"),
            ("memory_mb", "Memory (MB)"),
            ("disk_size_mb", "Disk size (MB)"),
            ("cost_per_1k_tokens_usd", "Cost / 1K tokens ($)"),
        ]
        if teacher_bench:
            lines.append(f"| Metric | Student | Teacher ({data.get('teacher_name', '?')}) |")
            lines.append("|---|---:|---:|")
            for key, label in keys:
                if key in student_bench or key in teacher_bench:
                    lines.append(
                        f"| {label} | {_fmt(student_bench.get(key))} | {_fmt(teacher_bench.get(key))} |"
                    )
            sp, tp = student_bench.get("parameters_m"), teacher_bench.get("parameters_m")
            ss, ts = student_bench.get("tokens_per_s"), teacher_bench.get("tokens_per_s")
            lines.append("")
            headline = []
            if sp and tp:
                headline.append(f"**{tp / sp:.1f}x smaller**")
            if ss and ts and ts > 0:
                headline.append(f"**{ss / ts:.1f}x faster**")
            if headline:
                lines.append(" and ".join(headline) + " than the teacher on this hardware.")
        else:
            lines.append("| Metric | Student |")
            lines.append("|---|---:|")
            for key, label in keys:
                if key in student_bench:
                    lines.append(f"| {label} | {_fmt(student_bench.get(key))} |")
        lines.append("")

    samples = data.get("samples") or []
    if samples:
        lines.append("## Sample outputs")
        lines.append("")
        for sample in samples:
            lines.append(f"**Prompt:** {sample['prompt']}")
            lines.append("")
            lines.append(f"> {sample['student_answer'].strip().replace(chr(10), chr(10) + '> ')}")
            lines.append("")

    lines.append("---")
    lines.append(
        f"Run dir: `{data.get('run_dir', '?')}` · dataset: `{data.get('dataset', '?')}` · "
        f"reproduce with `distill report`."
    )
    lines.append("")
    return "\n".join(lines)


def build_report(
    run_dir: str | Path,
    dataset: str | Path,
    teacher: Optional[Union[str, Teacher]] = None,
    judge: Optional[Union[str, Teacher]] = None,
    n: int = 32,
    max_new_tokens: int = 256,
    hardware_cost_per_hour: Optional[float] = None,
    benchmark_teacher: bool = True,
    seed: int = 0,
) -> Path:
    """Evaluate a distilled run and write REPORT.md + report.json into run_dir.

    Reference answers come from the dataset's ``response`` fields when present
    (i.e. what the teacher originally said); otherwise ``teacher`` is queried.
    """
    import torch

    from distillanything.hardware import best_device
    from distillanything.loading import load_model_and_tokenizer

    run_dir = Path(run_dir)
    device = best_device()

    console.print(f"Loading student from [cyan]{run_dir}[/]")
    # Handles both merged checkpoints and LoRA adapter-only saves.
    student, tokenizer = load_model_and_tokenizer(str(run_dir))
    student = student.to(device)
    student.eval()

    teacher_obj = resolve_teacher(teacher) if isinstance(teacher, str) else teacher
    judge_obj = resolve_teacher(judge) if isinstance(judge, str) else judge

    records = clean_records(load_records(dataset))
    rng = random.Random(seed)
    rng.shuffle(records)
    records = [r for r in records if r.get("prompt")][:n]
    prompts = [r["prompt"] for r in records]

    def student_answer(prompt: str) -> str:
        if tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = student.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    console.print(f"Generating student answers for {len(prompts)} prompts...")
    student_answers = [student_answer(p) for p in prompts]

    # References: dataset responses when available, else ask the teacher.
    missing = [i for i, r in enumerate(records) if not r.get("response")]
    references = [r.get("response", "") for r in records]
    if missing:
        if teacher_obj is None:
            raise ValueError(
                f"{len(missing)} records have no response and no teacher was given to generate them"
            )
        console.print(f"Generating {len(missing)} reference answers with {teacher_obj.name}...")
        generated = teacher_obj.generate([prompts[i] for i in missing], max_tokens=max_new_tokens)
        for i, text in zip(missing, generated):
            references[i] = text

    data: dict = {
        "student_name": str(run_dir),
        "teacher_name": teacher_obj.name if teacher_obj else None,
        "dataset": str(dataset),
        "run_dir": str(run_dir),
        "generated_on": date.today().isoformat(),
        "n_prompts": len(prompts),
    }

    if judge_obj is not None:
        console.print(f"Judging student vs reference with [cyan]{judge_obj.name}[/]...")
        results = judge_pairwise(judge_obj, prompts, student_answers, references)
        data["judge"] = summarize_pairwise(results)
        data["judge_name"] = judge_obj.name

    results_file = run_dir / "results.json"
    if results_file.exists():
        data["train_eval"] = json.loads(results_file.read_text()).get("eval")

    console.print("Benchmarking student...")
    data["student_benchmark"] = benchmark_model(
        student, tokenizer, hardware_cost_per_hour=hardware_cost_per_hour
    )
    if benchmark_teacher and teacher_obj is not None and teacher_obj.white_box:
        console.print("Benchmarking teacher...")
        data["teacher_benchmark"] = benchmark_model(
            teacher_obj.model, teacher_obj.tokenizer, hardware_cost_per_hour=hardware_cost_per_hour
        )

    data["samples"] = [
        {"prompt": p, "student_answer": a} for p, a in list(zip(prompts, student_answers))[:3]
    ]

    (run_dir / "report.json").write_text(json.dumps(data, indent=2, default=str))
    report_path = run_dir / "REPORT.md"
    report_path.write_text(render_report(data))
    console.print(f"[green]Wrote[/] {report_path}")
    return report_path
