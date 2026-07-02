"""Knowledge-distillation losses for autoregressive LMs.

All functions operate on *aligned* logits: ``student_logits`` and ``teacher_logits``
of shape ``[B, S, V]`` where position ``t`` predicts token ``t+1``. Masking uses the
label tensor convention (``-100`` = ignored position).

Notes from the literature baked in here:
  - forward KL (teacher||student) is the classic Hinton et al. objective — mode-covering,
    good when the student should imitate the full teacher distribution.
  - reverse KL (student||teacher) is mode-seeking — MiniLLM showed it often works better
    for generative students that must commit to fluent samples.
  - generalized JSD interpolates the two (DistiLLM-style skew) and is numerically safer.
  - Temperature-scaled losses are multiplied by T^2 so gradient magnitudes stay
    comparable to CE when mixing (Hinton et al., 2015).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _masked_mean(per_position: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``per_position`` [B, S] over positions where ``mask`` is True."""
    mask = mask.to(per_position.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (per_position * mask).sum() / denom


def _kl_from_logprobs(p_log: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per position, both inputs are log-probs over the last dim."""
    return (p_log.exp() * (p_log - q_log)).sum(dim=-1)


def forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL(teacher || student) per position, scaled by T^2. Shape in [B,S,V] -> out [B,S]."""
    t_log = F.log_softmax(teacher_logits / temperature, dim=-1)
    s_log = F.log_softmax(student_logits / temperature, dim=-1)
    return _kl_from_logprobs(t_log, s_log) * (temperature**2)


def reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL(student || teacher) per position, scaled by T^2."""
    t_log = F.log_softmax(teacher_logits / temperature, dim=-1)
    s_log = F.log_softmax(student_logits / temperature, dim=-1)
    return _kl_from_logprobs(s_log, t_log) * (temperature**2)


def generalized_jsd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    beta: float = 0.5,
) -> torch.Tensor:
    """Generalized Jensen-Shannon divergence with mixing weight ``beta``.

    m = beta * p_teacher + (1 - beta) * p_student
    loss = beta * KL(teacher || m) + (1 - beta) * KL(student || m)
    """
    t_log = F.log_softmax(teacher_logits / temperature, dim=-1)
    s_log = F.log_softmax(student_logits / temperature, dim=-1)
    # log of the mixture, computed stably from log-probs
    m_log = torch.logsumexp(
        torch.stack(
            [t_log + torch.log(torch.tensor(beta, device=t_log.device)),
             s_log + torch.log(torch.tensor(1.0 - beta, device=t_log.device))]
        ),
        dim=0,
    )
    jsd = beta * _kl_from_logprobs(t_log, m_log) + (1.0 - beta) * _kl_from_logprobs(s_log, m_log)
    return jsd * (temperature**2)


def _topk_truncate(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restrict both distributions to the teacher's top-k support.

    A standard approximation: renormalize over the k indices the teacher ranks highest.
    Cuts KD memory/compute from O(V) to O(k) — useful on laptops where V is 50k+.
    """
    k = min(k, teacher_logits.size(-1))
    _, idx = teacher_logits.topk(k, dim=-1)
    return student_logits.gather(-1, idx), teacher_logits.gather(-1, idx)


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    kind: str = "forward_kl",
    temperature: float = 2.0,
    top_k: Optional[int] = None,
    jsd_beta: float = 0.5,
) -> torch.Tensor:
    """Masked scalar KD loss between next-token predictions.

    ``student_logits``/``teacher_logits``: [B, S, V] raw logits (same tokenizer!).
    ``labels``: [B, S] with -100 on ignored positions (prompt + padding).
    """
    if student_logits.size(-1) != teacher_logits.size(-1):
        # Vocab padding differences (e.g. multiple-of-64 padding) are common within
        # a model family; align on the shared prefix of the vocabulary.
        v = min(student_logits.size(-1), teacher_logits.size(-1))
        student_logits = student_logits[..., :v]
        teacher_logits = teacher_logits[..., :v]

    # Position t predicts label t+1: align logits with the shifted label mask.
    s = student_logits[:, :-1, :]
    t = teacher_logits[:, :-1, :]
    mask = labels[:, 1:] != -100

    if top_k is not None:
        s, t = _topk_truncate(s, t, top_k)

    if kind == "forward_kl":
        per_pos = forward_kl(s, t, temperature)
    elif kind == "reverse_kl":
        per_pos = reverse_kl(s, t, temperature)
    elif kind == "jsd":
        per_pos = generalized_jsd(s, t, temperature, beta=jsd_beta)
    else:
        raise ValueError(f"Unknown KD loss kind: {kind!r} (expected forward_kl|reverse_kl|jsd)")

    return _masked_mean(per_pos, mask)
