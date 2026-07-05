"""Filesystem → database mirroring.

The run directory stays the source of truth; sync is one-way and idempotent.
Metric rows are keyed by their line number in ``metrics.jsonl`` (``seq``), so
an incremental sync just appends the lines the database hasn't seen. A local
file *shorter* than the mirror means the run was restarted (the trainer
truncates metrics.jsonl) — the mirror is cleared and rebuilt for that run.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Optional

from distillanything.ui import runsource


def local_host_label() -> str:
    """Identifies this machine in the shared history (``DISTILL_HOST_LABEL`` to override)."""
    label = os.environ.get("DISTILL_HOST_LABEL", "").strip()
    return label or socket.gethostname().split(".")[0] or "local"


def run_signature(run_dir: Path) -> tuple:
    """Cheap change detector over the files sync reads."""
    sig = []
    for fname in ("status.json", "metrics.jsonl", "report.json", "results.json"):
        try:
            st = (run_dir / fname).stat()
            sig.append((fname, st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append((fname, None, None))
    return tuple(sig)


def sync_run(store, run_dir: Path, host: str) -> dict:
    """Mirror one run dir; returns {"metrics": n_pushed, "report": bool}."""
    name = run_dir.name
    store.upsert_run(host, name, runsource.run_summary(run_dir))

    entries = runsource.read_metrics(run_dir.parent, name)
    remote = store.metric_count(host, name)
    if len(entries) < remote:  # restart truncated the local file — rebuild mirror
        store.clear_metrics(host, name)
        remote = 0
    new = entries[remote:]
    if new:
        store.append_metrics(host, name, new, start_seq=remote)

    report_path = run_dir / "report.json"
    pushed_report = False
    if report_path.exists():
        try:
            store.upsert_report(host, name, json.loads(report_path.read_text()))
            pushed_report = True
        except (OSError, json.JSONDecodeError):
            pass  # torn mid-write; the next pass gets it whole
    return {"metrics": len(new), "report": pushed_report}


def _run_dirs(runs_root: Path):
    if not runs_root.exists():
        return
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        if any((child / f).exists() for f in ("status.json", "distill_config.json")):
            yield child


def sync_all(store, runs_root: Path, host: Optional[str] = None) -> dict:
    host = host or local_host_label()
    counts = {"runs": 0, "metrics": 0, "reports": 0}
    for run_dir in _run_dirs(runs_root):
        result = sync_run(store, run_dir, host)
        counts["runs"] += 1
        counts["metrics"] += result["metrics"]
        counts["reports"] += int(result["report"])
    return counts


class DbState:
    """Shared between the API endpoints and the sync worker. ``store`` may be
    swapped at runtime when the Settings page configures a database."""

    def __init__(self, store=None, source: Optional[str] = None, dsn: Optional[str] = None):
        self.store = store
        self.source = source  # "env" | "file" | "injected" | None
        self.dsn = dsn  # kept server-side for redacted status display only
        self.last_sync: Optional[float] = None
        self.last_error: Optional[str] = None
        self.lock = threading.Lock()

    def set_store(self, store, source: Optional[str], dsn: Optional[str] = None) -> None:
        with self.lock:
            old = self.store
            self.store, self.source, self.dsn = store, source, dsn
            self.last_error = None
        if old is not None and old is not store:
            old.close()

    def sync_now(self, runs_root: Path, host: Optional[str] = None) -> dict:
        store = self.store
        if store is None:
            raise RuntimeError("no database configured")
        try:
            counts = sync_all(store, runs_root, host=host)
        except Exception as exc:
            with self.lock:
                self.last_error = f"{type(exc).__name__}: {exc}"
            raise
        with self.lock:
            self.last_sync = time.time()
            self.last_error = None
        return counts


class SyncWorker(threading.Thread):
    """Polls run dirs and mirrors only what changed (signature-gated).
    A no-op while no store is configured, so it always runs."""

    def __init__(self, state: DbState, runs_root: Path, host: str, interval: float = 5.0):
        super().__init__(daemon=True, name="db-sync")
        self._state = state
        self._runs_root = runs_root
        self._host = host
        self._interval = interval
        self._stop = threading.Event()
        self._seen: dict[str, tuple] = {}
        self._store_id: Optional[int] = None

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.wait(self._interval):
            store = self._state.store
            if store is None:
                continue
            if id(store) != self._store_id:  # newly configured DB: full pass
                self._store_id = id(store)
                self._seen.clear()
            try:
                synced = False
                for run_dir in _run_dirs(self._runs_root):
                    sig = run_signature(run_dir)
                    if self._seen.get(run_dir.name) == sig:
                        continue
                    sync_run(store, run_dir, self._host)
                    self._seen[run_dir.name] = sig
                    synced = True
                if synced:
                    with self._state.lock:
                        self._state.last_sync = time.time()
                        self._state.last_error = None
            except Exception as exc:
                with self._state.lock:
                    self._state.last_error = f"{type(exc).__name__}: {exc}"
