"""Hidden-state (feature) knowledge distillation — the FitNets lineage.

Beyond matching the teacher's output distribution, the student matches the
teacher's *intermediate representations*: a much denser signal per token, which
is exactly what helps at small data scales. Because student and teacher have
different widths and depths, small learned linear projectors map student states
into the teacher's space; they train jointly with the student and are
**discarded at save time** — zero inference cost.

Conventions match kd.py: hidden states are input-aligned (state ``t`` encodes
token ``t``), so masking uses ``labels != -100`` directly, no shift.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def match_layers(n_student: int, n_teacher: int, strategy: str = "uniform") -> list[tuple[int, int]]:
    """Pair student transformer layers with teacher layers.

    Indices are 1-based into the ``hidden_states`` tuple (index 0 is the
    embedding output, which carries no transformer computation to distill).

    - ``uniform``: student layer i maps to the teacher layer at the same
      relative depth — the robust default in the literature (TinyBERT et al.).
    - ``last``: only the final layers are matched (cheapest, weakest signal).
    """
    if strategy == "last":
        return [(n_student, n_teacher)]
    if strategy != "uniform":
        raise ValueError(f"unknown layer-matching strategy: {strategy!r}")
    return [(i, max(1, round(i * n_teacher / n_student))) for i in range(1, n_student + 1)]


class HiddenProjectors(nn.Module):
    """One linear map per matched layer pair, student width -> teacher width.

    Train-time-only scaffolding: do NOT save these with the student.
    """

    def __init__(self, n_pairs: int, student_width: int, teacher_width: int):
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Linear(student_width, teacher_width, bias=False) for _ in range(n_pairs)
        )

    def forward(self, index: int, states: torch.Tensor) -> torch.Tensor:
        return self.projections[index](states)


def hidden_kd_loss(
    student_hidden: tuple[torch.Tensor, ...],
    teacher_hidden: tuple[torch.Tensor, ...],
    projectors: HiddenProjectors,
    labels: torch.Tensor,
    layer_pairs: list[tuple[int, int]],
    metric: str = "mse",
) -> torch.Tensor:
    """Masked feature-matching loss, averaged over matched layers.

    States are L2-normalized per token before comparison so the loss is
    invariant to layer-wise scale differences (fp16 teachers especially).
    """
    mask = labels != -100  # [B, S]
    denom = mask.sum().clamp(min=1)
    total = torch.zeros((), device=labels.device, dtype=torch.float32)
    for pair_index, (s_layer, t_layer) in enumerate(layer_pairs):
        s_state = projectors(pair_index, student_hidden[s_layer].float())
        t_state = teacher_hidden[t_layer].float()
        s_state = F.normalize(s_state, dim=-1)
        t_state = F.normalize(t_state, dim=-1)
        if metric == "cosine":
            per_pos = 1.0 - (s_state * t_state).sum(dim=-1)  # [B, S]
        elif metric == "mse":
            per_pos = (s_state - t_state).pow(2).sum(dim=-1)
        else:
            raise ValueError(f"unknown hidden-KD metric: {metric!r}")
        total = total + (per_pos * mask).sum() / denom
    return total / max(1, len(layer_pairs))


__all__ = ["match_layers", "HiddenProjectors", "hidden_kd_loss"]
