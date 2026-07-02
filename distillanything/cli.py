"""The `distill` CLI: init -> generate -> train -> eval -> benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="distill",
    help="Distill Anything — generate data from teachers, distill students, evaluate, benchmark.",
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
):
    """Generate a training dataset from seed prompts using a teacher."""
    from distillanything.data.generate import generate_dataset
    from distillanything.teachers.registry import resolve_teacher

    teacher_obj = resolve_teacher(teacher, concurrency=concurrency)
    generate_dataset(
        teacher_obj, seeds, out, system=system, max_tokens=max_tokens, expand_per_seed=expand
    )


@app.command()
def train(recipe: str = typer.Argument(..., help="Path to a recipe YAML (see `distill init`)")):
    """Run a full distillation from a recipe."""
    from distillanything.config import DistillConfig
    from distillanything.student import Student

    cfg = DistillConfig.from_yaml(recipe)
    student = Student(cfg.student.model, trust_remote_code=cfg.student.trust_remote_code)
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
):
    """Measure latency, throughput, memory, and footprint of a model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from distillanything.eval.benchmark import benchmark_model

    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    loaded = AutoModelForCausalLM.from_pretrained(model)
    metrics = benchmark_model(loaded, tokenizer, prompt=prompt, max_new_tokens=max_new_tokens)

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
def smoke():
    """End-to-end self-test with tiny random models (no downloads, <1 min)."""
    from distillanything.smoke import run_smoke

    run_smoke()


if __name__ == "__main__":
    app()
