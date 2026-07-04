"""Hidden-state KD: layer matching, projector mechanics, masking, and an
end-to-end tiny training run proving the loss trains and nothing leaks into
the saved checkpoint."""

import pytest
import torch

from distillanything.config import DistillConfig, HiddenKDConfig, LossConfig, TrainConfig
from distillanything.data.tokenize import SFTDataset
from distillanything.losses.hidden import HiddenProjectors, hidden_kd_loss, match_layers
from distillanything.testing import tiny_records, tiny_student_and_teacher, tiny_tokenizer
from distillanything.train.trainer import DistillTrainer


@pytest.fixture(scope="module")
def tokenizer():
    return tiny_tokenizer()


def test_match_layers_uniform():
    pairs = match_layers(4, 8)
    assert pairs == [(1, 2), (2, 4), (3, 6), (4, 8)]
    assert match_layers(4, 4) == [(1, 1), (2, 2), (3, 3), (4, 4)]
    # deeper student than teacher still maps into range
    pairs = match_layers(8, 4)
    assert all(1 <= t <= 4 for _, t in pairs)
    assert match_layers(4, 8, "last") == [(4, 8)]
    with pytest.raises(ValueError):
        match_layers(4, 8, "nope")


def _random_states(n_layers, batch=2, seq=6, width=8):
    return tuple(torch.randn(batch, seq, width) for _ in range(n_layers + 1))


def test_hidden_loss_masking_and_range():
    pairs = match_layers(2, 2)
    projectors = HiddenProjectors(len(pairs), 8, 12)
    s = _random_states(2, width=8)
    t = _random_states(2, width=12)
    labels = torch.full((2, 6), -100)
    labels[:, 3:] = 1  # only last three positions supervised

    loss = hidden_kd_loss(s, t, projectors, labels, pairs, metric="mse")
    assert loss.item() >= 0

    # fully masked -> zero loss, no NaN
    all_masked = torch.full((2, 6), -100)
    zero = hidden_kd_loss(s, t, projectors, all_masked, pairs, metric="mse")
    assert zero.item() == 0.0

    # changing a masked position must not change the loss
    s2 = tuple(x.clone() for x in s)
    s2[1][:, 0, :] += 100.0
    loss2 = hidden_kd_loss(s2, t, projectors, labels, pairs, metric="mse")
    assert loss2.item() == pytest.approx(loss.item(), rel=1e-5)


def test_hidden_loss_cosine_identical_states_is_zero():
    pairs = [(1, 1)]
    projectors = HiddenProjectors(1, 8, 8)
    with torch.no_grad():
        projectors.projections[0].weight.copy_(torch.eye(8))
    states = _random_states(1, width=8)
    labels = torch.ones(2, 6, dtype=torch.long)
    loss = hidden_kd_loss(states, states, projectors, labels, pairs, metric="cosine")
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_hidden_kd_trains_and_checkpoint_stays_clean(tokenizer, tmp_path):
    student, teacher = tiny_student_and_teacher(len(tokenizer))
    cfg = DistillConfig(
        mode="logit",
        loss=LossConfig(kind="forward_kl", hidden=HiddenKDConfig(weight=1.0)),
        train=TrainConfig(
            output_dir=str(tmp_path / "out"), lr=1e-3, max_steps=10, batch_size=8,
            grad_accum=1, log_every=5,
        ),
    )
    train_ds = SFTDataset(tiny_records(48), tokenizer, max_seq_len=96)
    trainer = DistillTrainer(student, tokenizer, train_ds, cfg, teacher_model=teacher)
    assert trainer.projectors is not None and len(trainer._layer_pairs) > 0

    results = trainer.train()
    history = results["train_history"]
    assert "hid" in history[0]
    assert history[-1]["loss"] < history[0]["loss"]
    assert history[-1]["hid"] < history[0]["hid"]  # projectors are learning too

    # The saved checkpoint is a plain HF model — no projector tensors inside.
    import json

    out = tmp_path / "out"
    weight_files = list(out.glob("*.safetensors")) + list(out.glob("pytorch_model*.bin"))
    assert weight_files, "no checkpoint saved"
    index = out / "model.safetensors.index.json"
    names: list[str] = []
    if index.exists():
        names = list(json.loads(index.read_text())["weight_map"])
    else:
        from safetensors import safe_open

        with safe_open(weight_files[0], framework="pt") as f:
            names = list(f.keys())
    assert names and not any("projection" in n for n in names)


def test_hidden_requires_logit_mode(tokenizer, tmp_path):
    student, _ = tiny_student_and_teacher(len(tokenizer))
    cfg = DistillConfig(
        mode="seqkd",
        loss=LossConfig(hidden=HiddenKDConfig()),
        train=TrainConfig(output_dir=str(tmp_path / "out"), max_steps=2, batch_size=4, grad_accum=1),
    )
    train_ds = SFTDataset(tiny_records(8), tokenizer, max_seq_len=96)
    trainer = DistillTrainer(student, tokenizer, train_ds, cfg)
    # seqkd has no teacher: hidden KD is silently inert rather than crashing
    assert trainer.projectors is None
    results = trainer.train()
    assert "hid" not in results["train_history"][0]
