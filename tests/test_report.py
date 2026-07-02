import json

from distillanything.eval.benchmark import cost_per_1k_tokens
from distillanything.eval.report import build_report, render_report
from distillanything.teachers.base import Teacher


def test_cost_per_1k_tokens():
    # 100 tok/s at $3.60/hour -> $0.01/1K tokens
    assert cost_per_1k_tokens(100.0, 3.6) == 0.01
    assert cost_per_1k_tokens(0.0, 3.6) == float("inf")


def test_render_report_sections():
    data = {
        "student_name": "runs/test",
        "teacher_name": "hf:teacher",
        "judge_name": "fake-judge",
        "judge": {
            "n": 10,
            "student_wins": 4,
            "ties": 4,
            "teacher_wins": 2,
            "student_win_rate": 0.4,
            "tie_rate": 0.4,
            "teacher_win_rate": 0.2,
            "quality_retention": 0.8,
        },
        "train_eval": {"perplexity": 12.3},
        "student_benchmark": {"parameters_m": 135.0, "tokens_per_s": 60.0, "latency_p50_s": 0.4},
        "teacher_benchmark": {"parameters_m": 405.0, "tokens_per_s": 20.0, "latency_p50_s": 1.2},
        "samples": [{"prompt": "hi", "student_answer": "hello"}],
    }
    md = render_report(data)
    assert "matches or beats the reference on 80% of 10" in md
    assert "3.0x smaller" in md
    assert "3.0x faster" in md
    assert "perplexity" in md
    assert "Sample outputs" in md


def test_render_report_without_judge_or_teacher():
    md = render_report({"student_name": "s", "student_benchmark": {"parameters_m": 1.0}})
    assert "Efficiency" in md
    assert "Quality" not in md


class EchoTeacher(Teacher):
    name = "echo-teacher"

    def generate(self, prompts, *, system=None, max_tokens=512):
        return [f"ref: {p}" for p in prompts]


class TieJudge(Teacher):
    name = "tie-judge"

    def generate(self, prompts, *, system=None, max_tokens=512):
        return ["TIE"] * len(prompts)


def test_build_report_end_to_end(tmp_path):
    """Full report flow against a real (tiny) saved run directory."""
    from distillanything.config import DistillConfig, TrainConfig
    from distillanything.data.formats import save_records
    from distillanything.data.tokenize import SFTDataset
    from distillanything.testing import tiny_records, tiny_student_and_teacher, tiny_tokenizer
    from distillanything.train.trainer import DistillTrainer

    tokenizer = tiny_tokenizer()
    student, _ = tiny_student_and_teacher(len(tokenizer))
    run_dir = tmp_path / "run"
    cfg = DistillConfig(
        mode="seqkd",
        train=TrainConfig(output_dir=str(run_dir), max_steps=2, batch_size=4, grad_accum=1),
    )
    train_ds = SFTDataset(tiny_records(16), tokenizer, max_seq_len=96)
    DistillTrainer(student, tokenizer, train_ds, cfg).train()

    dataset_path = tmp_path / "eval.jsonl"
    save_records(tiny_records(8), dataset_path)

    report_path = build_report(
        run_dir,
        dataset_path,
        teacher=EchoTeacher(),
        judge=TieJudge(),
        n=4,
        max_new_tokens=8,
    )
    assert report_path.exists()
    md = report_path.read_text()
    assert "Quality (LLM-as-judge)" in md
    assert "Efficiency" in md

    report_json = json.loads((run_dir / "report.json").read_text())
    assert report_json["judge"]["n"] == 4
    assert report_json["judge"]["tie_rate"] == 1.0
    assert report_json["student_benchmark"]["latency_p50_s"] >= 0
