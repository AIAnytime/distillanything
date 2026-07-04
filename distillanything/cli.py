"""The `distill` CLI: init -> generate -> train -> eval -> benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="distill",
    help=(
        "Distill Anything — Smaller Models. Greater Impact.\n\n"
        "Generate data from teachers, distill students, evaluate, benchmark — "
        "and watch it all live with `distill ui`."
    ),
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()


@app.command()
def init(path: str = typer.Argument("distill.yaml", help="Where to write the starter recipe")):
    """Write a starter recipe YAML you can edit and pass to `distill train`."""
    from distillanything.config import DistillConfig

    if Path(path).exists():
        console.print(f"[red]{path} already exists — not overwriting.[/]")
        raise typer.Exit(1)
    DistillConfig().to_yaml(path)
    console.print(f"[green]Wrote starter recipe to[/] {path}")
    console.print("Edit it, then run: [bold]distill train " + path + "[/]")


@app.command()
def generate(
    seeds: str = typer.Argument(..., help="Seed prompts: .txt (one per line) or .jsonl with 'prompt'"),
    out: str = typer.Option("data/train.jsonl", help="Output dataset path"),
    teacher: str = typer.Option("claude", help="Teacher spec: hf:<repo> | claude[:<model>] | openai:<m> | ollama:<m>"),
    system: Optional[str] = typer.Option(None, help="System prompt for the teacher"),
    max_tokens: int = typer.Option(512, help="Max tokens per teacher response"),
    expand: int = typer.Option(0, help="Also generate N new prompts per seed (self-instruct lite)"),
    concurrency: int = typer.Option(4, help="Parallel requests for API teachers"),
    judge: Optional[str] = typer.Option(None, help="Judge spec to quality-score records (e.g. claude)"),
    min_score: int = typer.Option(7, help="Drop records the judge scores below this (1-10)"),
):
    """Generate a training dataset from seed prompts using a teacher."""
    from distillanything.data.formats import save_records
    from distillanything.data.generate import generate_dataset
    from distillanything.teachers.registry import resolve_teacher

    teacher_obj = resolve_teacher(teacher, concurrency=concurrency)
    records = generate_dataset(
        teacher_obj, seeds, out, system=system, max_tokens=max_tokens, expand_per_seed=expand
    )
    if judge:
        from distillanything.eval.judge import filter_by_score, score_records

        judge_obj = resolve_teacher(judge, concurrency=concurrency)
        console.print(f"Quality-scoring {len(records)} records with [cyan]{judge_obj.name}[/]...")
        scored = score_records(judge_obj, records)
        kept = filter_by_score(scored, min_score)
        save_records(kept, out)
        console.print(
            f"[green]Kept {len(kept)}/{len(scored)}[/] records with judge_score >= {min_score}"
        )


@app.command()
def train(recipe: str = typer.Argument(..., help="Path to a recipe YAML (see `distill init`)")):
    """Run a full distillation from a recipe."""
    from distillanything.config import DistillConfig
    from distillanything.student import Student

    cfg = DistillConfig.from_yaml(recipe)
    student = Student(
        cfg.student.model,
        trust_remote_code=cfg.student.trust_remote_code,
        lora=cfg.student.lora,
    )
    results = student.learn(
        teacher=cfg.teacher.spec, dataset=cfg.data.path, mode=cfg.mode, config=cfg
    )
    if "eval" in results:
        console.print(f"[bold green]Done.[/] Final eval: {results['eval']}")


@app.command()
def benchmark(
    model: str = typer.Argument(..., help="HF repo or local path of the model to benchmark"),
    prompt: str = typer.Option("Explain what knowledge distillation is in two sentences."),
    max_new_tokens: int = typer.Option(128),
    n_runs: int = typer.Option(5, help="Repeat runs for p50/p95 latency"),
    cost_per_hour: Optional[float] = typer.Option(
        None, help="Hardware $/hour to compute serving cost per 1K tokens"
    ),
):
    """Measure latency (p50/p95), throughput, memory, footprint, and cost of a model."""
    from distillanything.eval.benchmark import benchmark_model
    from distillanything.loading import load_model_and_tokenizer

    loaded, tokenizer = load_model_and_tokenizer(model)
    metrics = benchmark_model(
        loaded,
        tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        n_runs=n_runs,
        hardware_cost_per_hour=cost_per_hour,
    )

    table = Table(title=f"Benchmark: {model}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in metrics.items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def chat(
    model: str = typer.Argument(..., help="HF repo or local path (e.g. your distilled run dir)"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Prompt to send"),
    max_new_tokens: int = typer.Option(256),
):
    """One-shot generation from a (distilled) model — quick vibe check."""
    from distillanything.student import Student

    student = Student(model)
    console.print(student.generate(prompt, max_new_tokens=max_new_tokens))


@app.command()
def report(
    run_dir: str = typer.Argument(..., help="Distilled run directory (output_dir of a train run)"),
    dataset: str = typer.Option(..., help="Eval dataset: JSONL with prompts (+responses as reference)"),
    teacher: Optional[str] = typer.Option(
        None, help="Teacher spec for missing references and side-by-side benchmark"
    ),
    judge: Optional[str] = typer.Option(
        None, help="Judge spec for win/tie/lose quality eval (e.g. claude)"
    ),
    n: int = typer.Option(32, help="Number of held-out prompts to evaluate"),
    max_new_tokens: int = typer.Option(256),
    cost_per_hour: Optional[float] = typer.Option(
        None, help="Hardware $/hour to compute serving cost per 1K tokens"
    ),
    benchmark_teacher: bool = typer.Option(
        True, help="Also benchmark a local (hf:) teacher for the side-by-side table"
    ),
):
    """Build REPORT.md for a run: judge-scored quality + efficiency vs the teacher."""
    from distillanything.eval.report import build_report

    build_report(
        run_dir,
        dataset,
        teacher=teacher,
        judge=judge,
        n=n,
        max_new_tokens=max_new_tokens,
        hardware_cost_per_hour=cost_per_hour,
        benchmark_teacher=benchmark_teacher,
    )


@app.command()
def ui(
    runs_dir: str = typer.Option("runs", help="Directory containing your training runs"),
    data_dir: str = typer.Option("data", help="Directory containing your JSONL datasets"),
    host: str = typer.Option("127.0.0.1", help="Bind address (non-localhost requires --token)"),
    port: int = typer.Option(7326, help="Port for the dashboard"),
    token: Optional[str] = typer.Option(
        None, help="Session token; a strong one is auto-generated if omitted"
    ),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open the browser"),
):
    """Launch the dashboard: live runs, report cards, and a control plane for jobs."""
    try:
        from distillanything.ui.server import run_server
    except ImportError:
        console.print('[red]The dashboard needs extras:[/] pip install "distill-anything[ui]"')
        raise typer.Exit(1) from None

    run_server(runs_dir, data_dir, host=host, port=port, token=token, open_browser=not no_browser)


@app.command()
def smoke():
    """End-to-end self-test with tiny random models (no downloads, <1 min)."""
    from distillanything.smoke import run_smoke

    run_smoke()


if __name__ == "__main__":
    app()
