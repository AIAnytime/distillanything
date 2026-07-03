"""Filesystem-backed data source for the dashboard.

Pure functions over the runs/ and data/ directories — no HTTP here, so the whole
data layer is unit-testable with tmp_path fixtures. Every path that originates
from a client goes through :func:`safe_child`, which is the single choke point
against path traversal.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# The only files the API will ever serve out of a run directory.
RUN_FILES = frozenset(
    {
        "status.json",
        "metrics.jsonl",
        "report.json",
        "REPORT.md",
        "results.json",
        "distill_config.json",
        "recipe.yaml",
        "train.log",
    }
)

_NAME_RE = re.compile(r"^[\w][\w.-]*$")  # no leading dot/dash, no separators


class InvalidName(ValueError):
    pass


def safe_child(root: Path, name: str) -> Path:
    """Resolve ``name`` strictly inside ``root`` or raise InvalidName."""
    if not _NAME_RE.match(name):
        raise InvalidName(f"invalid name: {name!r}")
    child = (root / name).resolve()
    if not child.is_relative_to(root.resolve()):
        raise InvalidName(f"path escapes root: {name!r}")
    return child


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _last_train_metric(run_dir: Path) -> Optional[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    last = None
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
    except OSError:
        return None
    if last is None:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return None  # torn write mid-line; the next poll will see it whole


def run_summary(run_dir: Path) -> dict:
    status = _read_json(run_dir / "status.json") or {}
    report = _read_json(run_dir / "report.json") or {}
    summary = {
        "name": run_dir.name,
        "state": status.get("state", "unknown"),
        "mode": status.get("mode"),
        "student": status.get("student"),
        "teacher": status.get("teacher"),
        "total_steps": status.get("total_steps"),
        "steps_completed": status.get("steps_completed"),
        "started_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
        "error": status.get("error"),
        "last_metric": _last_train_metric(run_dir),
        "has_report": (run_dir / "report.json").exists(),
        "quality_retention": (report.get("judge") or {}).get("quality_retention"),
    }
    try:
        mtimes = [f.stat().st_mtime for f in run_dir.iterdir() if f.name in RUN_FILES]
        summary["updated_at"] = max(mtimes) if mtimes else run_dir.stat().st_mtime
    except OSError:
        summary["updated_at"] = None
    return summary


def list_runs(runs_root: Path) -> list[dict]:
    if not runs_root.exists():
        return []
    runs = []
    for child in runs_root.iterdir():
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        if not any((child / f).exists() for f in ("status.json", "distill_config.json")):
            continue
        runs.append(run_summary(child))
    runs.sort(key=lambda r: r.get("updated_at") or 0, reverse=True)
    return runs


def run_detail(runs_root: Path, name: str) -> Optional[dict]:
    run_dir = safe_child(runs_root, name)
    if not run_dir.is_dir():
        return None
    detail = run_summary(run_dir)
    detail["config"] = _read_json(run_dir / "distill_config.json")
    results = _read_json(run_dir / "results.json") or {}
    detail["eval"] = results.get("eval")
    return detail


def read_metrics(runs_root: Path, name: str) -> list[dict]:
    run_dir = safe_child(runs_root, name)
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def list_datasets(data_root: Path) -> list[dict]:
    if not data_root.exists():
        return []
    datasets = []
    for path in sorted(data_root.glob("*.jsonl")):
        n_records = 0
        try:
            with path.open() as f:
                n_records = sum(1 for line in f if line.strip())
        except OSError:
            continue
        stat = path.stat()
        datasets.append(
            {
                "name": path.name,
                "records": n_records,
                "size_bytes": stat.st_size,
                "updated_at": stat.st_mtime,
            }
        )
    return datasets


def read_dataset_records(data_root: Path, name: str, offset: int = 0, limit: int = 50) -> dict:
    path = safe_child(data_root, name)
    if path.suffix != ".jsonl" or not path.exists():
        raise FileNotFoundError(name)
    records, total = [], 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if offset <= total < offset + limit:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    records.append({"_parse_error": True})
            total += 1
    return {"name": name, "total": total, "offset": offset, "records": records}
