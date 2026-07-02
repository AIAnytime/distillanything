import torch

from distillanything.losses.kd import forward_kl, generalized_jsd, kd_loss, reverse_kl


def _rand_logits(b=2, s=6, v=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, s, v, generator=g)


def test_kl_zero_when_identical():
    logits = _rand_logits()
    assert forward_kl(logits, logits).abs().max().item() < 1e-5
    assert reverse_kl(logits, logits).abs().max().item() < 1e-5
    assert generalized_jsd(logits, logits).abs().max().item() < 1e-5


def test_kl_nonnegative():
    s, t = _rand_logits(seed=1), _rand_logits(seed=2)
    assert forward_kl(s, t).min().item() >= -1e-6
    assert reverse_kl(s, t).min().item() >= -1e-6
    assert generalized_jsd(s, t).min().item() >= -1e-6


def test_jsd_symmetric_at_half():
    s, t = _rand_logits(seed=3), _rand_logits(seed=4)
    ab = generalized_jsd(s, t, beta=0.5)
    ba = generalized_jsd(t, s, beta=0.5)
    assert torch.allclose(ab, ba, atol=1e-5)


def test_kd_loss_masking_ignores_prompt_positions():
    s, t = _rand_logits(seed=5), _rand_logits(seed=6)
    labels_all = torch.randint(0, 32, (2, 6))
    labels_masked = labels_all.clone()
    labels_masked[:, :3] = -100  # mask early positions

    loss_all = kd_loss(s, t, labels_all, temperature=1.0)
    loss_masked = kd_loss(s, t, labels_masked, temperature=1.0)
    assert loss_all.item() != loss_masked.item()
    assert torch.isfinite(loss_masked)


def test_kd_loss_fully_masked_is_finite():
    s, t = _rand_logits(), _rand_logits(seed=9)
    labels = torch.full((2, 6), -100)
    assert torch.isfinite(kd_loss(s, t, labels))


def test_topk_truncation_runs_and_differs():
    s, t = _rand_logits(seed=7), _rand_logits(seed=8)
    labels = torch.randint(0, 32, (2, 6))
    full = kd_loss(s, t, labels, temperature=1.0)
    truncated = kd_loss(s, t, labels, temperature=1.0, top_k=4)
    assert torch.isfinite(truncated)
    assert full.item() != truncated.item()


def test_vocab_mismatch_is_aligned():
    s = _rand_logits(v=32)
    t = _rand_logits(v=40, seed=11)
    labels = torch.randint(0, 32, (2, 6))
    assert torch.isfinite(kd_loss(s, t, labels))


def test_kd_loss_backprops():
    s = _rand_logits(seed=12).requires_grad_(True)
    t = _rand_logits(seed=13)
    labels = torch.randint(0, 32, (2, 6))
    kd_loss(s, t, labels, kind="reverse_kl").backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
