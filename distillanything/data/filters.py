"""Dataset hygiene: dedup and quality filters applied before training."""

from __future__ import annotations

import hashlib
import re


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _record_key(record: dict) -> str:
    if "messages" in record:
        payload = " ".join(m.get("content", "") for m in record["messages"])
    else:
        payload = f"{record.get('prompt', '')} {record.get('response', '')} {record.get('text', '')}"
    return hashlib.sha256(_normalize(payload).encode()).hexdigest()


def dedup_records(records: list[dict]) -> list[dict]:
    """Exact dedup on normalized content (whitespace/case-insensitive)."""
    seen: set[str] = set()
    kept: list[dict] = []
    for record in records:
        key = _record_key(record)
        if key not in seen:
            seen.add(key)
            kept.append(record)
    return kept


def filter_short_responses(records: list[dict], min_chars: int = 1) -> list[dict]:
    """Drop records whose supervised target is trivially short (empty teacher
    outputs, refusals collapsed to '', etc.)."""

    def target_len(record: dict) -> int:
        if "messages" in record:
            return len(record["messages"][-1].get("content", ""))
        return len(record.get("response", record.get("text", "")))

    return [r for r in records if target_len(r) >= min_chars]


def clean_records(records: list[dict], dedup: bool = True, min_response_chars: int = 1) -> list[dict]:
    if dedup:
        records = dedup_records(records)
    return filter_short_responses(records, min_response_chars)
