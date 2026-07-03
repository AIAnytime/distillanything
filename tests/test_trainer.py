import json

import pytest

from distillanything.config import DistillConfig, LossConfig, TrainConfig
from distillanything.data.tokenize import SFTDataset
from distillanything.testing import tiny_records, tiny_student_and_teacher, tiny_tokenizer
from distillanything.train.trainer import DistillTrainer


@pytest.fixture(scope="module")
def tokenizer():
    return tiny_tokenizer()


def _cfg(mode: str, tmp_path, **loss_kw) -> DistillConfig:
    return DistillConfig(
        mode=mode,
        loss=LossConfig(**loss_kw),
        train=TrainConfig(
            output_dir=str(tmp_path / "out"),
            lr=3e-4,
            max_steps=15,
            batch_size=8,
            grad_accum=1,
            log_every=5,
        ),
    )


def test_logit_kd_loss_decreases(tokenizer, tmp_path):
    student, teacher = tiny_student_and_teacher(len(tokenizer))
    records = tiny_records(64)
    train_ds = SFTDataset(records[:56], tokenizer, max_seq_len=96)
    eval_ds = SFTDataset(records[56:], tokenizer, max_seq_len=96)

    cfg = _cfg("logit", tmp_path, kind="forward_kl", temperature=2.0, alpha=0.5)
    trainer = DistillTrainer(student, tokenizer, train_ds, cfg, eval_dataset=eval_ds, teacher_model=teacher)
    results = trainer.train()

    history = results["train_history"]
    assert history[-1]["loss"] < history[0]["loss"]
    assert "eval" in results and results["eval"]["perplexity"] > 0
    assert "teacher_agreement" in results["eval"]
    assert (tmp_path / "out" / "distill_config.json").exists()


def test_seqkd_trains_without_teacher(tokenizer, tmp_path):
    student, _ = tiny_student_and_teacher(len(tokenizer))
    train_ds = SFTDataset(tiny_records(32), tokenizer, max_seq_len=96)

    cfg = _cfg("seqkd", tmp_path, alpha=0.0)
    trainer = DistillTrainer(student, tokenizer, train_ds, cfg)
    results = trainer.train()
    assert results["steps"] == 15
    assert results["train_history"][-1]["loss"] < results["train_history"][0]["loss"]


def test_logit_mode_requires_teacher(tokenizer, tmp_path):
    student, _ = tiny_student_and_teacher(len(tokenizer))
    train_ds = SFTDataset(tiny_records(8), tokenizer, max_seq_len=96)
    with pytest.raises(ValueError, match="requires a local teacher"):
        DistillTrainer(student, tokenizer, train_ds, _cfg("logit", tmp_path))


def test_telemetry_files_written(tokenizer, tmp_path):
    student, teacher = tiny_student_and_teacher(len(tokenizer))
    records = tiny_records(64)
    train_ds = SFTDataset(records[:56], tokenizer, max_seq_len=96)
    eval_ds = SFTDataset(records[56:], tokenizer, max_seq_len=96)

    cfg = _cfg("logit", tmp_path, kind="forward_kl")
    trainer = DistillTrainer(student, tokenizer, train_ds, cfg, eval_dataset=eval_ds, teacher_model=teacher)
    results = trainer.train()

    out = tmp_path / "out"
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "completed"
    assert status["steps_completed"] == results["steps"]
    assert status["total_steps"] == 15
    assert status["device"] and status["student"] and status["mode"] == "logit"

    lines = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()]
    train_lines = [x for x in lines if x["kind"] == "train"]
    eval_lines = [x for x in lines if x["kind"] == "eval"]
    assert len(train_lines) == len(results["train_history"])
    assert train_lines[-1]["loss"] == pytest.approx(results["train_history"][-1]["loss"])
    assert len(eval_lines) == 1 and eval_lines[0]["perplexity"] > 0


def test_telemetry_status_failed_on_crash(tokenizer, tmp_path):
    student, teacher = tiny_student_and_teacher(len(tokenizer))
    train_ds = SFTDataset(tiny_records(16), tokenizer, max_seq_len=96)
    cfg = _cfg("logit", tmp_path, kind="forward_kl")

    trainer = DistillTrainer(student, tokenizer, train_ds, cfg, teacher_model=teacher)
    trainer.teacher = None  # sabotage: forward pass will raise inside the loop
    with pytest.raises(Exception):
        trainer.train()

    status = json.loads((tmp_path / "out" / "status.json").read_text())
    assert status["state"] == "failed"
    assert status["error"]


def test_reverse_kl_mode_runs(tokenizer, tmp_path):
    student, teacher = tiny_student_and_teacher(len(tokenizer))
    train_ds = SFTDataset(tiny_records(16), tokenizer, max_seq_len=96)
    cfg = _cfg("logit", tmp_path, kind="reverse_kl", top_k=16)
    cfg.train.max_steps = 3
    trainer = DistillTrainer(student, tokenizer, train_ds, cfg, teacher_model=teacher)
    results = trainer.train()
    assert results["steps"] == 3
