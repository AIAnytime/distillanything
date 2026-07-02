"""`distill smoke`: prove the whole pipeline works on this machine, offline, fast."""

from __future__ import annotations

from rich.console import Console

from distillanything.config import DistillConfig, LossConfig, TrainConfig
from distillanything.data.tokenize import SFTDataset
from distillanything.eval.benchmark import benchmark_model
from distillanything.hardware import device_summary
from distillanything.testing import tiny_records, tiny_student_and_teacher, tiny_tokenizer
from distillanything.train.trainer import DistillTrainer

console = Console()


def run_smoke() -> dict:
    console.rule("[bold]Distill Anything smoke test")
    console.print(f"Device: {device_summary()}")

    tokenizer = tiny_tokenizer()
    student, teacher = tiny_student_and_teacher(vocab_size=len(tokenizer))
    records = tiny_records(96)

    train_ds = SFTDataset(records[:80], tokenizer, max_seq_len=96)
    eval_ds = SFTDataset(records[80:], tokenizer, max_seq_len=96)

    cfg = DistillConfig(
        mode="logit",
        loss=LossConfig(kind="forward_kl", temperature=2.0, alpha=0.5),
        train=TrainConfig(
            output_dir="runs/smoke",
            lr=3e-4,
            max_steps=30,
            batch_size=8,
            grad_accum=1,
            log_every=10,
        ),
    )

    trainer = DistillTrainer(student, tokenizer, train_ds, cfg, eval_dataset=eval_ds, teacher_model=teacher)
    results = trainer.train()

    bench = benchmark_model(student, tokenizer, prompt="reverse apple1", max_new_tokens=16)
    console.print(f"Student benchmark: {bench}")

    first, last = results["train_history"][0]["loss"], results["train_history"][-1]["loss"]
    verdict = "PASS" if last < first else "FAIL"
    color = "green" if verdict == "PASS" else "red"
    console.print(f"[bold {color}]{verdict}[/]: loss {first:.3f} -> {last:.3f} over {results['steps']} steps")
    return results
