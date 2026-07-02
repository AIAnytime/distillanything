"""Teacher spec resolution: one string picks any backend.

    hf:HuggingFaceTB/SmolLM2-360M-Instruct   local white-box teacher
    claude / claude:claude-opus-4-8          Anthropic API
    openai:gpt-4o-mini                       OpenAI API
    ollama:llama3.2                          local Ollama server
"""

from __future__ import annotations

import os

from distillanything.teachers.base import Teacher

OLLAMA_DEFAULT_URL = "http://localhost:11434/v1"


def resolve_teacher(spec: str, concurrency: int = 4) -> Teacher:
    scheme, _, rest = spec.partition(":")

    if scheme == "hf":
        from distillanything.teachers.local import HFTeacher

        return HFTeacher(rest)

    if scheme == "claude":
        from distillanything.teachers.api import AnthropicTeacher

        model = rest or "claude-opus-4-8"
        return AnthropicTeacher(model=model, concurrency=concurrency)

    if scheme == "openai":
        from distillanything.teachers.api import OpenAICompatibleTeacher

        return OpenAICompatibleTeacher(model=rest, concurrency=concurrency)

    if scheme == "ollama":
        from distillanything.teachers.api import OpenAICompatibleTeacher

        base_url = os.environ.get("OLLAMA_BASE_URL", OLLAMA_DEFAULT_URL)
        return OpenAICompatibleTeacher(
            model=rest, base_url=base_url, api_key="ollama", concurrency=concurrency
        )

    # No scheme: treat as a local HF repo/path (the common case for quick experiments).
    if not rest:
        from distillanything.teachers.local import HFTeacher

        return HFTeacher(spec)

    raise ValueError(
        f"Unknown teacher spec {spec!r}. Expected hf:<repo>, claude[:<model>], "
        "openai:<model>, or ollama:<model>."
    )
