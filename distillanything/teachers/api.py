"""API teachers — black-box knowledge sources (Claude, OpenAI-compatible, Ollama).

SDKs are imported lazily so the core install stays lightweight:
``pip install "distill-anything[anthropic]"`` or ``[openai]`` as needed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from distillanything.teachers.base import Teacher


def _map_concurrently(fn, items: list[str], concurrency: int) -> list[str]:
    if concurrency <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(fn, items))


class AnthropicTeacher(Teacher):
    """Claude as a teacher via the Anthropic API."""

    def __init__(self, model: str = "claude-opus-4-8", concurrency: int = 4):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "AnthropicTeacher requires the anthropic SDK: "
                'pip install "distill-anything[anthropic]"'
            ) from e
        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"claude:{model}"
        self.concurrency = concurrency

    def generate(
        self,
        prompts: list[str],
        *,
        system: Optional[str] = None,
        max_tokens: int = 512,
    ) -> list[str]:
        def one(prompt: str) -> str:
            kwargs = {}
            if system:
                kwargs["system"] = system
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            if response.stop_reason == "refusal":
                return ""
            return "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()

        return _map_concurrently(one, prompts, self.concurrency)


class OpenAICompatibleTeacher(Teacher):
    """Any OpenAI-compatible endpoint: OpenAI, vLLM, llama.cpp server, Ollama, ..."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        concurrency: int = 4,
    ):
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "OpenAICompatibleTeacher requires the openai SDK: "
                'pip install "distill-anything[openai]"'
            ) from e
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.name = f"openai-compatible:{model}"
        self.concurrency = concurrency

    def generate(
        self,
        prompts: list[str],
        *,
        system: Optional[str] = None,
        max_tokens: int = 512,
    ) -> list[str]:
        def one(prompt: str) -> str:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=max_tokens
            )
            return (response.choices[0].message.content or "").strip()

        return _map_concurrently(one, prompts, self.concurrency)
