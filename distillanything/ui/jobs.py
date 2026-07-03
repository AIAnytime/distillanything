"""Subprocess job manager for the dashboard's control plane.

Training/generation/report jobs never run inside the web server: each is the
same CLI code path, spawned as `python -m distillanything.cli ...` in its own
process group. The server stays responsive while a model trains, a trainer
crash can't take the UI down, and everything the job writes lands on disk where
the read-only API already looks.

Security: argv is always a list (never shell=True) and every element is either
a server-built path or a value that passed Pydantic validation upstream.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    kind: str  # train | generate | report
    argv: list[str]
    log_path: Path
    run_name: Optional[str] = None
    status: str = "queued"  # queued | running | completed | failed | stopped
    created_at: str = field(default_factory=_utcnow)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    returncode: Optional[int] = None
    proc: Optional[subprocess.Popen] = None
    stop_requested: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "run_name": self.run_name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "pid": self.proc.pid if self.proc and self.status == "running" else None,
        }


def cli_argv(*args: str) -> list[str]:
    """argv for running the distill CLI with the server's own interpreter."""
    return [sys.executable, "-m", "distillanything.cli", *args]


class JobManager:
    """FIFO queue with bounded concurrency (default 1 — one training at a time
    is the right default on a 16GB laptop)."""

    def __init__(self, max_concurrent: int = 1):
        self.max_concurrent = max_concurrent
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    def submit(self, kind: str, argv: list[str], log_path: Path, run_name: str | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, argv=argv, log_path=log_path, run_name=run_name)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._start_queued_locked()
        return job

    def _start_queued_locked(self) -> None:
        running = sum(1 for j in self._jobs.values() if j.status == "running")
        for job_id in self._order:
            if running >= self.max_concurrent:
                break
            job = self._jobs[job_id]
            if job.status != "queued":
                continue
            self._start_locked(job)
            running += 1

    def _start_locked(self, job: Job) -> None:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = job.log_path.open("ab")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        try:
            job.proc = subprocess.Popen(
                job.argv,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # own process group: stop() can't hit the server
                env=env,
            )
        except OSError as exc:
            log_file.close()
            job.status = "failed"
            job.finished_at = _utcnow()
            job.log_path.write_text(f"failed to spawn: {exc}\n")
            return
        job.status = "running"
        job.started_at = _utcnow()
        if job.run_name:
            self._write_job_marker(job)
        threading.Thread(target=self._watch, args=(job, log_file), daemon=True).start()

    def _write_job_marker(self, job: Job) -> None:
        # Lets a restarted server reconcile "running" runs whose process died with it.
        marker = job.log_path.parent / "job.json"
        try:
            marker.write_text(json.dumps({"job_id": job.id, "pid": job.proc.pid, "kind": job.kind}))
        except OSError:
            pass

    def _watch(self, job: Job, log_file) -> None:
        returncode = job.proc.wait()
        log_file.close()
        with self._lock:
            job.returncode = returncode
            job.finished_at = _utcnow()
            if job.stop_requested:
                job.status = "stopped"
            elif returncode == 0:
                job.status = "completed"
            else:
                job.status = "failed"
            self._start_queued_locked()

    def stop(self, job_id: str, grace_seconds: float = 15.0) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status == "queued":
                job.status = "stopped"
                job.finished_at = _utcnow()
                return job
            if job.status != "running" or job.proc is None:
                return job
            job.stop_requested = True
            pgid = os.getpgid(job.proc.pid)
        # SIGINT first: the trainer catches KeyboardInterrupt and writes
        # status.json state=stopped before exiting.
        os.killpg(pgid, signal.SIGINT)
        deadline = time.time() + grace_seconds
        while time.time() < deadline:
            if job.proc.poll() is not None:
                return job
            time.sleep(0.2)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._lock:
            return [self._jobs[i].to_dict() for i in reversed(self._order)]

    def shutdown(self) -> None:
        with self._lock:
            running = [j for j in self._jobs.values() if j.status == "running" and j.proc]
        for job in running:
            self.stop(job.id, grace_seconds=5.0)


def reconcile_interrupted_runs(runs_root: Path) -> list[str]:
    """On server start: any run still marked running whose recorded pid is dead
    was orphaned by a previous crash — mark it interrupted so the UI is honest."""
    fixed = []
    if not runs_root.exists():
        return fixed
    for run_dir in runs_root.iterdir():
        status_path = run_dir / "status.json"
        marker_path = run_dir / "job.json"
        if not (run_dir.is_dir() and status_path.exists()):
            continue
        try:
            status = json.loads(status_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("state") != "running":
            continue
        pid = None
        if marker_path.exists():
            try:
                pid = json.loads(marker_path.read_text()).get("pid")
            except (OSError, json.JSONDecodeError):
                pid = None
        alive = False
        if pid:
            try:
                os.kill(pid, 0)
                alive = True
            except (OSError, ProcessLookupError):
                alive = False
        if not alive and pid is not None:
            status["state"] = "interrupted"
            status["finished_at"] = _utcnow()
            status_path.write_text(json.dumps(status, indent=2))
            fixed.append(run_dir.name)
    return fixed
