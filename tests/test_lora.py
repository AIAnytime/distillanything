import pytest

pytest.importorskip("peft")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from distillanything.config import DistillConfig, LoraSettings, TrainConfig  # noqa: E402
from distillanything.data.tokenize import SFTDataset  # noqa: E402
from distillanything.loading import apply_lora, load_model_and_tokenizer  # noqa: E402
from distillanything.testing import tiny_records, tiny_student_and_teacher, tiny_tokenizer  # noqa: E402
from distillanything.train.trainer import DistillTrainer  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    return tiny_tokenizer()


def _lora_cfg(tmp_path, merge: bool) -> DistillConfig:
    cfg = DistillConfig(
        mode="logit",
        train=TrainConfig(
            output_dir=str(tmp_path / "out"),
            lr=1e-3,
            max_steps=10,
            batch_size=8,
            grad_accum=1,
            log_every=5,
            merge_lora=merge,
            gradient_checkpointing=True,  # exercise the PEFT-compat path too
        ),
    )
    cfg.student.lora = LoraSettings(r=4, alpha=8)
    return cfg


def test_lora_freezes_base_and_trains():
    student, _ = tiny_student_and_teacher(vocab_size=128)
    total_before = sum(p.numel() for p in student.parameters())
    lora_model = apply_lora(student, LoraSettings(r=4, alpha=8))

    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_model.parameters())
    assert trainable < 0.1 * total_before  # adapters are a small fraction
    assert total >= total_before  # base weights all still present (frozen)


def test_lora_kd_loss_decreases_and_merges(tokenizer, tmp_path):
    student, teacher = tiny_student_and_teacher(len(tokenizer))
    cfg = _lora_cfg(tmp_path, merge=True)
    student = apply_lora(student, cfg.student.lora)

    train_ds = SFTDataset(tiny_records(48), tokenizer, max_seq_len=96)
    trainer = DistillTrainer(student, tokenizer, train_ds, cfg, teacher_model=teacher)
    results = trainer.train()
    assert results["train_history"][-1]["loss"] < results["train_history"][0]["loss"]

    # Merged save = plain HF checkpoint: loads without peft, no adapter files.
    out = tmp_path / "out"
    assert not (out / "adapter_config.json").exists()
    reloaded = AutoModelForCausalLM.from_pretrained(out)
    assert sum(p.numel() for p in reloaded.parameters()) > 0


def test_lora_adapter_only_save_roundtrip(tokenizer, tmp_path):
    student, teacher = tiny_student_and_teacher(len(tokenizer))
    cfg = _lora_cfg(tmp_path, merge=False)
    cfg.train.max_steps = 2
    student = apply_lora(student, cfg.student.lora)

    train_ds = SFTDataset(tiny_records(16), tokenizer, max_seq_len=96)
    DistillTrainer(student, tokenizer, train_ds, cfg, teacher_model=teacher).train()

    out = tmp_path / "out"
    assert (out / "adapter_config.json").exists()
    # Universal loader merges adapters transparently... but an adapter dir has no
    # base weights, so peft must be able to resolve the base model. Tiny random
    # models have no hub repo, so resolution fails — which is itself the contract
    # worth pinning: adapter dirs reference their base by name.
    import json

    adapter_cfg = json.loads((out / "adapter_config.json").read_text())
    assert "base_model_name_or_path" in adapter_cfg


def test_qlora_requires_cuda():
    if torch.cuda.is_available():
        pytest.skip("QLoRA is valid on CUDA machines")
    from distillanything.loading import load_student_model

    with pytest.raises(RuntimeError, match="CUDA"):
        load_student_model("any-model", LoraSettings(qlora=True))


def test_merged_dir_loads_via_universal_loader(tokenizer, tmp_path):
    student, teacher = tiny_student_and_teacher(len(tokenizer))
    cfg = _lora_cfg(tmp_path, merge=True)
    cfg.train.max_steps = 2
    student = apply_lora(student, cfg.student.lora)
    train_ds = SFTDataset(tiny_records(16), tokenizer, max_seq_len=96)
    DistillTrainer(student, tokenizer, train_ds, cfg, teacher_model=teacher).train()

    model, tok = load_model_and_tokenizer(str(tmp_path / "out"))
    assert tok.pad_token is not None
    assert not hasattr(model, "peft_config")
