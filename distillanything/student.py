"""The high-level SDK: ``Student().learn(teacher=..., dataset=...)``.

This is the one-screen developer experience the framework is built around. It wires
together teacher resolution, (optional) synthetic data generation, tokenization,
training, evaluation, and benchmarking.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from rich.console import Console

from distillanything.config import DistillConfig, LoraSettings
from distillanything.data.filters import clean_records
from distillanything.data.formats import load_records, save_records
from distillanything.data.generate import generate_dataset
from distillanything.data.tokenize import SFTDataset
from distillanything.eval.benchmark import benchmark_model
from distillanything.hardware import device_summary
from distillanything.teachers.registry import resolve_teacher
from distillanything.train.trainer import DistillTrainer

console = Console()


def _split(records: list[dict], eval_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    if eval_fraction <= 0 or len(records) < 4:
        return records, []
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_fraction))
    return shuffled[n_eval:], shuffled[:n_eval]


class Student:
    """A small model that learns from a bigger one.

    Pass ``lora=`` (a LoraSettings or plain dict like ``{"r": 16}``) to train
    adapters instead of full weights — the practical path to 1-3B students on a
    16GB laptop. ``lora={"qlora": True}`` additionally loads the frozen base in
    4-bit (CUDA only).
    """

    def __init__(
        self,
        model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        trust_remote_code: bool = False,
        lora: LoraSettings | dict | None = None,
    ):
        from distillanything.loading import load_model_and_tokenizer, load_student_model

        if isinstance(lora, dict):
            lora = LoraSettings.model_validate(lora)
        self._lora = lora
        console.print(f"Loading student [cyan]{model}[/] on {device_summary()}")
        self.model_name = model
        if lora is not None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = load_student_model(model, lora, trust_remote_code)
        else:
            # Also handles adapter-only checkpoints (merged on load).
            self.model, self.tokenizer = load_model_and_tokenizer(model, trust_remote_code)

    def learn(
        self,
        teacher: str,
        dataset: str,
        mode: str = "auto",
        config: Optional[DistillConfig] = None,
        **overrides,
    ) -> dict:
        """Distill knowledge from ``teacher`` into this student using ``dataset``.

        mode="auto" picks logit KD for local (hf:) teachers and seqkd for API teachers.
        Extra keyword overrides are applied onto the train config (lr=..., epochs=..., etc.).
        """
        cfg = config or DistillConfig()
        cfg.teacher.spec = teacher
        cfg.student.model = self.model_name
        cfg.data.path = dataset
        if mode == "auto":
            cfg.mode = "logit" if teacher.startswith("hf:") or ":" not in teacher else "seqkd"
        else:
            cfg.mode = mode  # type: ignore[assignment]
        for key, value in overrides.items():
            if hasattr(cfg.train, key):
                setattr(cfg.train, key, value)
            elif hasattr(cfg.loss, key):
                setattr(cfg.loss, key, value)
            elif hasattr(cfg.data, key):
                setattr(cfg.data, key, value)
            else:
                raise TypeError(f"Unknown override {key!r}")

        # Reconcile LoRA settings between constructor and recipe.
        if cfg.student.lora is None and self._lora is not None:
            cfg.student.lora = self._lora
        if cfg.student.lora is not None and not hasattr(self.model, "peft_config"):
            if cfg.student.lora.qlora:
                raise RuntimeError(
                    "QLoRA must be applied at load time: Student(model, lora={'qlora': True, ...})"
                )
            from distillanything.loading import apply_lora

            self.model = apply_lora(self.model, cfg.student.lora)

        teacher_obj = resolve_teacher(cfg.teacher.spec, concurrency=cfg.teacher.concurrency)

        records = load_records(cfg.data.path)
        needs_responses = any(
            "prompt" in r and not r.get("response") and "text" not in r and "messages" not in r
            for r in records
        )
        if needs_responses:
            generated_path = Path(cfg.train.output_dir) / "generated_dataset.jsonl"
            console.print("Dataset has prompts without responses — generating with the teacher...")
            records = generate_dataset(
                teacher_obj,
                cfg.data.path,
                generated_path,
                system=cfg.teacher.system,
                max_tokens=cfg.teacher.max_tokens,
            )

        records = clean_records(records, dedup=cfg.data.dedup, min_response_chars=cfg.data.min_response_chars)
        if cfg.data.max_records:
            records = records[: cfg.data.max_records]
        train_records, eval_records = _split(records, cfg.data.eval_fraction, cfg.train.seed)

        train_ds = SFTDataset(train_records, self.tokenizer, cfg.data.max_seq_len)
        eval_ds = SFTDataset(eval_records, self.tokenizer, cfg.data.max_seq_len) if eval_records else None
        console.print(f"Dataset: {len(train_ds)} train / {len(eval_ds) if eval_ds else 0} eval examples")

        teacher_model = teacher_obj.model if teacher_obj.white_box and cfg.mode == "logit" else None
        trainer = DistillTrainer(
            self.model, self.tokenizer, train_ds, cfg, eval_dataset=eval_ds, teacher_model=teacher_model
        )
        return trainer.train()

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        import torch

        from distillanything.hardware import best_device

        device = best_device()
        self.model.to(device)
        self.model.eval()
        if self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def benchmark(self, **kwargs) -> dict:
        return benchmark_model(self.model, self.tokenizer, **kwargs)

    def save(self, path: str) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        console.print(f"[green]Saved to[/] {out}")


__all__ = ["Student", "save_records"]
