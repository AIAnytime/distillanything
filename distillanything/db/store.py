"""Run-history stores: Postgres (via psycopg 3) and an in-memory fake.

Credential rules (same posture as teacher API keys):
- Resolution order: ``DISTILL_DB_URL`` env var, then ``~/.distillanything/config.json``
  (written with mode 0600 by the dashboard's Settings page).
- The connection string is never returned by any API endpoint and never sent
  to the browser; status views get a redacted ``{user, host, dbname}`` only.

Both stores implement the same duck-typed interface, keyed by
``(host_label, run_name)`` — no database ids leak into the API surface.
"""

from __future__ import annotations

import json
import os
import stat
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

ENV_VAR = "DISTILL_DB_URL"
_SCHEMES = {"postgres", "postgresql"}

SCHEMA_VERSION = 1

_DDL = [
    "CREATE TABLE IF NOT EXISTS da_meta (key text PRIMARY KEY, value text NOT NULL)",
    """
    CREATE TABLE IF NOT EXISTS da_runs (
        id bigserial PRIMARY KEY,
        host text NOT NULL,
        name text NOT NULL,
        summary jsonb NOT NULL,
        updated_at timestamptz NOT NULL DEFAULT now(),
        UNIQUE (host, name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS da_metrics (
        run_id bigint NOT NULL REFERENCES da_runs(id) ON DELETE CASCADE,
        seq integer NOT NULL,
        entry jsonb NOT NULL,
        PRIMARY KEY (run_id, seq)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS da_reports (
        run_id bigint PRIMARY KEY REFERENCES da_runs(id) ON DELETE CASCADE,
        payload jsonb NOT NULL,
        updated_at timestamptz NOT NULL DEFAULT now()
    )
    """,
]


def default_config_path() -> Path:
    return Path.home() / ".distillanything" / "config.json"


def resolve_db_url(config_path: Optional[Path] = None) -> tuple[Optional[str], Optional[str]]:
    """Return ``(url, source)`` where source is ``"env"`` or ``"file"``; (None, None) if unset."""
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        return env, "env"
    path = config_path or default_config_path()
    try:
        url = json.loads(path.read_text()).get("db_url")
    except (OSError, json.JSONDecodeError):
        return None, None
    return (url, "file") if url else (None, None)


def validate_db_url(url: str) -> None:
    """Structural check only — no network. Raises ValueError with a message
    that never echoes the URL (it may contain a password)."""
    parts = urlsplit(url)
    if parts.scheme not in _SCHEMES:
        raise ValueError("connection string must start with postgres:// or postgresql://")
    if not parts.hostname:
        raise ValueError("connection string has no hostname")


def save_db_url(url: str, config_path: Optional[Path] = None) -> None:
    validate_db_url(url)
    path = config_path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    try:
        existing = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    existing["db_url"] = url
    path.write_text(json.dumps(existing, indent=2))
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600: it's a credential


def clear_db_url(config_path: Optional[Path] = None) -> None:
    path = config_path or default_config_path()
    try:
        existing = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    existing.pop("db_url", None)
    path.write_text(json.dumps(existing, indent=2))


def redact_dsn(url: str) -> dict:
    """The only DSN-derived data that may leave the server: no password, ever."""
    parts = urlsplit(url)
    return {
        "user": parts.username,
        "host": parts.hostname,
        "dbname": (parts.path or "").lstrip("/") or None,
    }


class MemoryStore:
    """In-memory stand-in with PgStore's exact semantics; used by tests and
    injectable into the server, so the whole sync path is verifiable offline."""

    def __init__(self):
        self._runs: dict[tuple[str, str], dict] = {}
        self._metrics: dict[tuple[str, str], list[dict]] = {}
        self._reports: dict[tuple[str, str], dict] = {}
        self._lock = threading.Lock()

    def connect(self) -> None:  # parity with PgStore
        return None

    def close(self) -> None:
        return None

    def upsert_run(self, host: str, name: str, summary: dict) -> None:
        with self._lock:
            self._runs[(host, name)] = {"summary": summary, "synced_at": time.time()}

    def metric_count(self, host: str, name: str) -> int:
        with self._lock:
            return len(self._metrics.get((host, name), []))

    def append_metrics(self, host: str, name: str, entries: list[dict], start_seq: int) -> None:
        with self._lock:
            rows = self._metrics.setdefault((host, name), [])
            if start_seq != len(rows):
                raise ValueError(f"append at seq {start_seq}, have {len(rows)}")
            rows.extend(entries)

    def clear_metrics(self, host: str, name: str) -> None:
        with self._lock:
            self._metrics.pop((host, name), None)

    def upsert_report(self, host: str, name: str, payload: dict) -> None:
        with self._lock:
            self._reports[(host, name)] = payload

    def list_runs(self) -> list[dict]:
        with self._lock:
            out = [
                {"host": host, "synced_at": row["synced_at"], **row["summary"]}
                for (host, _name), row in ((k, v) for k, v in self._runs.items())
            ]
        out.sort(key=lambda r: r.get("synced_at") or 0, reverse=True)
        return out

    def get_run(self, host: str, name: str) -> Optional[dict]:
        with self._lock:
            row = self._runs.get((host, name))
            if row is None:
                return None
            return {"host": host, "synced_at": row["synced_at"], **row["summary"]}

    def get_metrics(self, host: str, name: str) -> list[dict]:
        with self._lock:
            return list(self._metrics.get((host, name), []))

    def get_report(self, host: str, name: str) -> Optional[dict]:
        with self._lock:
            return self._reports.get((host, name))

    def status(self) -> dict:
        with self._lock:
            return {
                "schema_version": SCHEMA_VERSION,
                "runs": len(self._runs),
                "metrics": sum(len(v) for v in self._metrics.values()),
            }


class PgStore:
    """Postgres/Neon store. Lazily imports psycopg so the base install never
    needs it; one shared connection guarded by a lock, with a single
    reconnect-and-retry (Neon's free tier suspends idle databases)."""

    def __init__(self, dsn: str):
        validate_db_url(dsn)
        self._dsn = dsn
        self._conn = None
        self._lock = threading.Lock()

    # -- connection plumbing -------------------------------------------------

    def _psycopg(self):
        try:
            import psycopg  # noqa: PLC0415
            from psycopg.types.json import Jsonb  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                'Postgres sync needs extras: pip install "distill-anything[db]"'
            ) from exc
        return psycopg, Jsonb

    def _open(self):
        psycopg, _ = self._psycopg()
        self._conn = psycopg.connect(self._dsn, autocommit=True, connect_timeout=15)
        return self._conn

    def _execute(self, query: str, params=(), fetch: Optional[str] = None, many: bool = False):
        psycopg, _ = self._psycopg()
        with self._lock:
            for attempt in (1, 2):
                try:
                    conn = self._conn or self._open()
                    with conn.cursor() as cur:
                        if many:
                            cur.executemany(query, params)
                        else:
                            cur.execute(query, params)
                        if fetch == "one":
                            return cur.fetchone()
                        if fetch == "all":
                            return cur.fetchall()
                    return None
                except psycopg.OperationalError:
                    self._conn = None  # stale (suspended/idle-closed); retry once fresh
                    if attempt == 2:
                        raise

    def connect(self) -> None:
        """Open a connection and migrate the schema — the app owns its tables."""
        for ddl in _DDL:
            self._execute(ddl)
        self._execute(
            "INSERT INTO da_meta (key, value) VALUES ('schema_version', %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (str(SCHEMA_VERSION),),
        )

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    # -- writes ---------------------------------------------------------------

    def upsert_run(self, host: str, name: str, summary: dict) -> None:
        _, Jsonb = self._psycopg()
        self._execute(
            "INSERT INTO da_runs (host, name, summary, updated_at) VALUES (%s, %s, %s, now()) "
            "ON CONFLICT (host, name) DO UPDATE SET summary = EXCLUDED.summary, updated_at = now()",
            (host, name, Jsonb(summary)),
        )

    def _run_id(self, host: str, name: str) -> Optional[int]:
        row = self._execute(
            "SELECT id FROM da_runs WHERE host = %s AND name = %s", (host, name), fetch="one"
        )
        return row[0] if row else None

    def metric_count(self, host: str, name: str) -> int:
        run_id = self._run_id(host, name)
        if run_id is None:
            return 0
        row = self._execute(
            "SELECT count(*) FROM da_metrics WHERE run_id = %s", (run_id,), fetch="one"
        )
        return int(row[0])

    def append_metrics(self, host: str, name: str, entries: list[dict], start_seq: int) -> None:
        if not entries:
            return
        _, Jsonb = self._psycopg()
        run_id = self._run_id(host, name)
        if run_id is None:
            raise ValueError(f"unknown run {host}/{name}")
        self._execute(
            "INSERT INTO da_metrics (run_id, seq, entry) VALUES (%s, %s, %s) "
            "ON CONFLICT (run_id, seq) DO UPDATE SET entry = EXCLUDED.entry",
            [(run_id, start_seq + i, Jsonb(entry)) for i, entry in enumerate(entries)],
            many=True,
        )

    def clear_metrics(self, host: str, name: str) -> None:
        run_id = self._run_id(host, name)
        if run_id is not None:
            self._execute("DELETE FROM da_metrics WHERE run_id = %s", (run_id,))

    def upsert_report(self, host: str, name: str, payload: dict) -> None:
        _, Jsonb = self._psycopg()
        run_id = self._run_id(host, name)
        if run_id is None:
            raise ValueError(f"unknown run {host}/{name}")
        self._execute(
            "INSERT INTO da_reports (run_id, payload, updated_at) VALUES (%s, %s, now()) "
            "ON CONFLICT (run_id) DO UPDATE SET payload = EXCLUDED.payload, updated_at = now()",
            (run_id, Jsonb(payload)),
        )

    # -- reads ----------------------------------------------------------------

    def list_runs(self) -> list[dict]:
        rows = self._execute(
            "SELECT host, summary, extract(epoch FROM updated_at) FROM da_runs "
            "ORDER BY updated_at DESC",
            fetch="all",
        )
        return [{"host": host, "synced_at": float(epoch), **summary} for host, summary, epoch in rows]

    def get_run(self, host: str, name: str) -> Optional[dict]:
        row = self._execute(
            "SELECT summary, extract(epoch FROM updated_at) FROM da_runs "
            "WHERE host = %s AND name = %s",
            (host, name),
            fetch="one",
        )
        if row is None:
            return None
        summary, epoch = row
        return {"host": host, "synced_at": float(epoch), **summary}

    def get_metrics(self, host: str, name: str) -> list[dict]:
        run_id = self._run_id(host, name)
        if run_id is None:
            return []
        rows = self._execute(
            "SELECT entry FROM da_metrics WHERE run_id = %s ORDER BY seq", (run_id,), fetch="all"
        )
        return [entry for (entry,) in rows]

    def get_report(self, host: str, name: str) -> Optional[dict]:
        run_id = self._run_id(host, name)
        if run_id is None:
            return None
        row = self._execute(
            "SELECT payload FROM da_reports WHERE run_id = %s", (run_id,), fetch="one"
        )
        return row[0] if row else None

    def status(self) -> dict:
        runs = self._execute("SELECT count(*) FROM da_runs", fetch="one")
        metrics = self._execute("SELECT count(*) FROM da_metrics", fetch="one")
        return {"schema_version": SCHEMA_VERSION, "runs": int(runs[0]), "metrics": int(metrics[0])}
