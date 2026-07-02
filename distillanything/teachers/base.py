"""Teacher abstraction: anything that can answer prompts can teach a student."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class Teacher(ABC):
    """Black-box teacher: produces text completions for prompts.

    White-box teachers (local HF models) additionally expose ``.model`` and
    ``.tokenizer`` so trainers can distill on logits.
    """

    name: str = "teacher"

    #: True when the teacher exposes logits (local model) for white-box KD.
    white_box: bool = False

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        *,
        system: Optional[str] = None,
        max_tokens: int = 512,
    ) -> list[str]:
        """Return one completion per prompt, order-aligned with the input."""
        raise NotImplementedError
