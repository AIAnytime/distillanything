"""The distillation trainer.

Two modes share one loop:
  - ``logit``: white-box KD. The (local) teacher's logits supervise every response
    position: total = alpha * KD(student, teacher) + (1 - alpha) * CE(student, labels).
  - ``seqkd``: black-box / sequence-level KD. The teacher's knowledge is already in the
    dataset text (generated responses), so the loss is plain CE (alpha is ignored).

Deliberately a hand-rolled loop rather than HF Trainer: full control over the KD
forward pass, no callback labyrinth, and predictable behavior on MPS.
"""

from __future__ import annotations

import contextlib
import json
import math
import random
import time
from pathlib import Path
from typing import Optional

import torch
from rich.console import Console
from torch.utils.data import DataLoader, Dataset

from distillanything.config import DistillConfig
from distillanything.data.tokenize import pad_collate
from distillanything.hardware import autocast_dtype, best_device
from distillanything.losses.kd import kd_loss

console = Console()


def _cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


class DistillTrainer:
    def __init__(
        self,
        student_model,
        tokenizer,
        train_dataset: Dataset,
        cfg: DistillConfig,
        eval_dataset: Optional[Dataset] = None,
        teacher_model=None,
    ):
        self.cfg = cfg
        self.device = best_device()
        self.tokenizer = tokenizer
        self.student = student_model.to(self.device)
        self.teacher = teacher_model
        if cfg.mode == "logit":
            if self.teacher is None:
                raise ValueError("logit mode requires a local teacher model (white-box KD)")
            self.teacher = self.teacher.to(self.device)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self._check_vocab_compat()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.history: list[dict] = []

    def _check_vocab_compat(self) -> None:
        sv = self.student.get_input_embeddings().weight.shape[0]
        tv = self.teacher.get_input_embeddings().weight.shape[0]
        if abs(sv - tv) > 1024:
            console.print(
                f"[yellow]Warning:[/] student vocab ({sv}) and teacher vocab ({tv}) differ "
                "substantially. Logit KD assumes a shared tokenizer — use models from the "
                "same family, or switch to mode: seqkd."
            )

    def _dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        return DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: pad_collate(batch, pad_id),
        )

    def train(self) -> dict:
        cfg = self.cfg
        torch.manual_seed(cfg.train.seed)
        random.seed(cfg.train.seed)

        loader = self._dataloader(self.train_dataset, shuffle=True)
        steps_per_epoch = max(1, math.ceil(len(loader) / cfg.train.grad_accum))
        total_steps = cfg.train.max_steps or steps_per_epoch * cfg.train.epochs
        warmup_steps = int(total_steps * cfg.train.warmup_ratio)

        optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        scheduler = _cosine_with_warmup(optimizer, warmup_steps, total_steps)
        amp_dtype = autocast_dtype(self.device)

        console.print(
            f"[bold]Training[/] mode={cfg.mode} device={self.device.type} "
            f"steps={total_steps} batch={cfg.train.batch_size}x{cfg.train.grad_accum} "
            f"loss={cfg.loss.kind}(T={cfg.loss.temperature}, alpha={cfg.loss.alpha})"
        )

        self.student.train()
        step = 0
        accumulated = 0
        running: dict[str, float] = {"total": 0.0, "kd": 0.0, "ce": 0.0, "n": 0}
        start_time = time.time()
        done = False

        for epoch in range(max(1, cfg.train.epochs if cfg.train.max_steps is None else 10**6)):
            if done:
                break
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                context = (
                    torch.autocast(self.device.type, dtype=amp_dtype)
                    if amp_dtype
                    else contextlib.nullcontext()
                )
                with context:
                    out = self.student(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    ce = out.loss
                    if cfg.mode == "logit" and cfg.loss.alpha > 0:
                        with torch.no_grad():
                            teacher_logits = self.teacher(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                            ).logits
                        kd = kd_loss(
                            out.logits,
                            teacher_logits,
                            batch["labels"],
                            kind=cfg.loss.kind,
                            temperature=cfg.loss.temperature,
                            top_k=cfg.loss.top_k,
                            jsd_beta=cfg.loss.jsd_beta,
                        )
                        loss = cfg.loss.alpha * kd + (1 - cfg.loss.alpha) * ce
                    else:
                        kd = torch.zeros((), device=self.device)
                        loss = ce

                (loss / cfg.train.grad_accum).backward()
                accumulated += 1
                running["total"] += loss.item()
                running["kd"] += kd.item()
                running["ce"] += ce.item()
                running["n"] += 1

                if accumulated % cfg.train.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), cfg.train.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1

                    if step % cfg.train.log_every == 0 or step == total_steps:
                        n = max(1, running["n"])
                        entry = {
                            "step": step,
                            "loss": running["total"] / n,
                            "kd": running["kd"] / n,
                            "ce": running["ce"] / n,
                            "lr": scheduler.get_last_lr()[0],
                            "elapsed_s": round(time.time() - start_time, 1),
                        }
                        self.history.append(entry)
                        console.print(
                            f"  step {entry['step']:>5}/{total_steps}  "
                            f"loss {entry['loss']:.4f}  kd {entry['kd']:.4f}  "
                            f"ce {entry['ce']:.4f}  lr {entry['lr']:.2e}"
                        )
                        running = {"total": 0.0, "kd": 0.0, "ce": 0.0, "n": 0}

                    if cfg.train.eval_every and step % cfg.train.eval_every == 0:
                        self.evaluate()
                        self.student.train()

                    if step >= total_steps:
                        done = True
                        break

        results = {"steps": step, "train_history": self.history}
        if self.eval_dataset is not None and len(self.eval_dataset) > 0:
            results["eval"] = self.evaluate()
        self.save(cfg.train.output_dir, results)
        return results

    @torch.no_grad()
    def evaluate(self) -> dict:
        if self.eval_dataset is None or len(self.eval_dataset) == 0:
            return {}
        self.student.eval()
        loader = self._dataloader(self.eval_dataset, shuffle=False)
        total_ce, batches = 0.0, 0
        agree_hits, agree_total = 0, 0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            total_ce += out.loss.item()
            batches += 1
            if self.teacher is not None:
                teacher_logits = self.teacher(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).logits
                v = min(out.logits.size(-1), teacher_logits.size(-1))
                mask = batch["labels"][:, 1:] != -100
                s_top = out.logits[:, :-1, :v].argmax(-1)
                t_top = teacher_logits[:, :-1, :v].argmax(-1)
                agree_hits += ((s_top == t_top) & mask).sum().item()
                agree_total += mask.sum().item()
        ce = total_ce / max(1, batches)
        metrics = {"eval_loss": round(ce, 4), "perplexity": round(math.exp(min(ce, 20.0)), 3)}
        if agree_total:
            metrics["teacher_agreement"] = round(agree_hits / agree_total, 4)
        console.print(f"[bold cyan]Eval:[/] {metrics}")
        return metrics

    def save(self, output_dir: str, results: Optional[dict] = None) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.student.save_pretrained(out)
        if self.tokenizer is not None:
            try:
                self.tokenizer.save_pretrained(out)
            except Exception:
                pass  # test tokenizers may not be fully serializable
        (out / "distill_config.json").write_text(self.cfg.model_dump_json(indent=2))
        if results is not None:
            (out / "results.json").write_text(json.dumps(results, indent=2))
        console.print(f"[green]Saved student to[/] {out}")
