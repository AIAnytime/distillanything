import re

from distillanything.eval.judge import (
    filter_by_score,
    judge_pairwise,
    score_records,
    summarize_pairwise,
)
from distillanything.teachers.base import Teacher


class FakeJudge(Teacher):
    """Deterministic judge: prefers whichever answer contains 'GOOD'."""

    name = "fake-judge"

    def generate(self, prompts, *, system=None, max_tokens=512):
        outputs = []
        for text in prompts:
            a = re.search(r"Answer A:\n(.*?)\n\nAnswer B:", text, re.S)
            b = re.search(r"Answer B:\n(.*?)\n\nReply", text, re.S)
            a_text = a.group(1) if a else ""
            b_text = b.group(1) if b else ""
            if "GOOD" in a_text and "GOOD" not in b_text:
                outputs.append("A")
            elif "GOOD" in b_text and "GOOD" not in a_text:
                outputs.append("B")
            else:
                outputs.append("TIE")
        return outputs


class BiasedJudge(Teacher):
    """Always says A — position swap must neutralize it to a tie."""

    name = "biased-judge"

    def generate(self, prompts, *, system=None, max_tokens=512):
        return ["A"] * len(prompts)


class GarbageJudge(Teacher):
    name = "garbage-judge"

    def generate(self, prompts, *, system=None, max_tokens=512):
        return ["no idea, sorry!"] * len(prompts)


class ScoreJudge(Teacher):
    """Scores 9 when the response contains 'GOOD', otherwise 3."""

    name = "score-judge"

    def generate(self, prompts, *, system=None, max_tokens=512):
        return ["The score is 9." if "GOOD" in p else "3" for p in prompts]


def test_consistent_judge_produces_wins_and_losses():
    prompts = ["q1", "q2"]
    students = ["GOOD answer", "weak answer"]
    references = ["weak answer", "GOOD answer"]
    results = judge_pairwise(FakeJudge(), prompts, students, references)
    assert [r.verdict for r in results] == ["win", "lose"]


def test_position_bias_is_neutralized_to_tie():
    results = judge_pairwise(BiasedJudge(), ["q"], ["s"], ["r"])
    assert results[0].verdict == "tie"


def test_unparseable_verdicts_count_as_tie():
    results = judge_pairwise(GarbageJudge(), ["q"], ["s"], ["r"])
    assert results[0].verdict == "tie"
    assert results[0].raw_verdicts == (None, None)


def test_summary_math():
    prompts = ["a", "b", "c", "d"]
    students = ["GOOD", "GOOD", "x", "y"]
    references = ["x", "y", "GOOD", "z"]
    summary = summarize_pairwise(judge_pairwise(FakeJudge(), prompts, students, references))
    assert summary["n"] == 4
    assert summary["student_wins"] == 2
    assert summary["teacher_wins"] == 1
    assert summary["ties"] == 1
    assert summary["quality_retention"] == 0.75


def test_score_and_filter_records():
    records = [
        {"prompt": "p1", "response": "GOOD stuff"},
        {"prompt": "p2", "response": "meh"},
        {"text": "raw text without prompt"},
    ]
    scored = score_records(ScoreJudge(), records)
    assert scored[0]["judge_score"] == 9
    assert scored[1]["judge_score"] == 3
    assert "judge_score" not in scored[2]

    kept = filter_by_score(scored, min_score=7)
    assert {r.get("prompt", r.get("text")) for r in kept} == {"p1", "raw text without prompt"}
