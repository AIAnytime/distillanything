"""LLM-as-judge evaluation: the quality half of "10x cheaper without losing quality".

Two uses, one machinery:
  - pairwise: blind A/B comparison of student vs reference answers -> win/tie/lose.
    Judged twice with positions swapped to cancel position bias; the two verdicts
    must agree for a win/lose, otherwise the pair is scored a tie.
  - absolute: score a single response 1-10, used to filter synthetic datasets
    before training (quality scoring in the data pipeline).

Any Teacher can judge (claude, openai:..., ollama:..., hf:...), so the same spec
string works everywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from distillanything.teachers.base import Teacher

PAIRWISE_PROMPT = """You are an impartial judge comparing two answers to the same question. \
Judge which answer is more helpful, correct, and clear. Do not let answer length or the \
order of presentation influence you.

Question:
{prompt}

Answer A:
{a}

Answer B:
{b}

Reply with exactly one word: A, B, or TIE."""

SCORE_PROMPT = """You are grading a single answer for quality (helpfulness, correctness, clarity).

Question:
{prompt}

Answer:
{response}

Rate the answer on a 1-10 scale where 1 is useless and 10 is excellent. \
Reply with only the number."""

_VERDICT_RE = re.compile(r"\b(TIE|A|B)\b")
_SCORE_RE = re.compile(r"\b(10|[1-9])\b")


def _parse_verdict(text: str) -> Optional[str]:
    match = _VERDICT_RE.search(text.upper())
    return match.group(1) if match else None


def _parse_score(text: str) -> Optional[int]:
    match = _SCORE_RE.search(text)
    return int(match.group(1)) if match else None


@dataclass
class PairwiseResult:
    prompt: str
    student_answer: str
    reference_answer: str
    # "win" | "tie" | "lose", from the student's perspective
    verdict: str
    raw_verdicts: tuple[Optional[str], Optional[str]]


def judge_pairwise(
    judge: Teacher,
    prompts: list[str],
    student_answers: list[str],
    reference_answers: list[str],
    max_tokens: int = 16,
) -> list[PairwiseResult]:
    """Blind, position-debiased A/B judging. Returns one result per prompt."""
    if not (len(prompts) == len(student_answers) == len(reference_answers)):
        raise ValueError("prompts, student_answers, reference_answers must be the same length")

    # Round 1: student is A. Round 2: student is B. One batched call each.
    round1 = [
        PAIRWISE_PROMPT.format(prompt=p, a=s, b=r)
        for p, s, r in zip(prompts, student_answers, reference_answers)
    ]
    round2 = [
        PAIRWISE_PROMPT.format(prompt=p, a=r, b=s)
        for p, s, r in zip(prompts, student_answers, reference_answers)
    ]
    verdicts1 = [_parse_verdict(t) for t in judge.generate(round1, max_tokens=max_tokens)]
    verdicts2 = [_parse_verdict(t) for t in judge.generate(round2, max_tokens=max_tokens)]

    results = []
    for prompt, student, reference, v1, v2 in zip(
        prompts, student_answers, reference_answers, verdicts1, verdicts2
    ):
        # Map both rounds onto the student's perspective.
        pref1 = {"A": "win", "B": "lose", "TIE": "tie"}.get(v1 or "")
        pref2 = {"A": "lose", "B": "win", "TIE": "tie"}.get(v2 or "")
        if pref1 == pref2 and pref1 in ("win", "lose"):
            verdict = pref1
        else:
            # Disagreement, explicit tie, or unparseable output all count as a tie —
            # the conservative choice that never inflates the student's score.
            verdict = "tie"
        results.append(PairwiseResult(prompt, student, reference, verdict, (v1, v2)))
    return results


def summarize_pairwise(results: list[PairwiseResult]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}
    wins = sum(1 for r in results if r.verdict == "win")
    ties = sum(1 for r in results if r.verdict == "tie")
    loses = n - wins - ties
    return {
        "n": n,
        "student_wins": wins,
        "ties": ties,
        "teacher_wins": loses,
        "student_win_rate": round(wins / n, 4),
        "tie_rate": round(ties / n, 4),
        "teacher_win_rate": round(loses / n, 4),
        # "How often is the student at least as good?" — the headline number.
        "quality_retention": round((wins + ties) / n, 4),
    }


def score_records(
    judge: Teacher,
    records: list[dict],
    max_tokens: int = 8,
) -> list[dict]:
    """Absolute 1-10 quality scores for {prompt, response} records.

    Returns new records with a ``judge_score`` field (None when unparseable).
    """
    scorable = [r for r in records if r.get("prompt") and r.get("response")]
    prompts = [SCORE_PROMPT.format(prompt=r["prompt"], response=r["response"]) for r in scorable]
    outputs = judge.generate(prompts, max_tokens=max_tokens)
    scores = iter(_parse_score(t) for t in outputs)

    result = []
    for record in records:
        record = dict(record)
        if record.get("prompt") and record.get("response"):
            record["judge_score"] = next(scores)
        result.append(record)
    return result


def filter_by_score(records: list[dict], min_score: int) -> list[dict]:
    """Keep records scoring >= min_score. Unscored records (no judge_score) are kept;
    records the judge scored but couldn't be parsed (None) are dropped."""
    kept = []
    for record in records:
        if "judge_score" not in record:
            kept.append(record)
        elif record["judge_score"] is not None and record["judge_score"] >= min_score:
            kept.append(record)
    return kept
