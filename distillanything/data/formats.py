"""Record loading/saving. A record is a plain dict in one of three shapes:

    {"prompt": str, "response": str}          instruction pair (preferred)
    {"text": str}                             raw continuation text
    {"messages": [{"role": ..., "content": ...}, ...]}   chat transcript

``.txt`` inputs are read as one *prompt* per line (no responses yet — a teacher
fills them in via `distill generate`).
"""

from __future__ import annotations

import json
from pathlib import Path


def load_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".txt":
        lines = [line.strip() for line in path.read_text().splitlines()]
        return [{"prompt": line} for line in lines if line]

    records: list[dict] = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {e}") from e
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object per line")
            records.append(record)
    return records


def save_records(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


FALLBACK_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"


def render_pair(record: dict, tokenizer) -> tuple[str, str]:
    """Render a record into (prompt_text, response_text) for tokenization.

    Uses the tokenizer's chat template when available so the student trains on
    the same format it will be served with.
    """
    if "messages" in record:
        messages = record["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("chat records must end with an assistant message")
        response = messages[-1]["content"]
        context = messages[:-1]
        if tokenizer.chat_template:
            prompt_text = tokenizer.apply_chat_template(
                context, tokenize=False, add_generation_prompt=True
            )
        else:
            rendered = "\n".join(f"{m['role']}: {m['content']}" for m in context)
            prompt_text = f"{rendered}\nassistant: "
        return prompt_text, response

    if "prompt" in record:
        prompt, response = record["prompt"], record.get("response", "")
        if tokenizer.chat_template:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = FALLBACK_TEMPLATE.format(prompt=prompt)
        return prompt_text, response

    if "text" in record:
        # Raw text: no prompt masking; the whole sequence is supervised.
        return "", record["text"]

    raise ValueError(f"Unrecognized record shape: keys={sorted(record.keys())}")
