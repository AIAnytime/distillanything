"""Database mirroring: sync semantics against MemoryStore (offline), credential
handling, and — when TEST_PG_DSN is set — the same contract against a real
Postgres via PgStore."""

import json
import os
import stat

import pytest

from distillanything.db.store import (
    MemoryStore,
    clear_db_url,
    redact_dsn,
    resolve_db_url,
    save_db_url,
    validate_db_url,
)
from distillanything.db.sync import local_host_label, run_signature, sync_all, sync_run

DSN_WITH_SECRET = "postgresql://alice:hunter2-secret@db.example.com:5432/history"


def _make_run(runs_root, name, metrics=3, state="completed", report=False):
    run_dir = runs_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(
        json.dumps({"state": state, "mode": "logit", "student": "s", "teacher": "hf:t",
                    "total_steps": 10, "steps_completed": metrics})
    )
    with (run_dir / "metrics.jsonl").open("w") as f:
        for i in range(metrics):
            f.write(json.dumps({"kind": "train", "step": i + 1, "loss": 3.0 - i * 0.1}) + "\n")
    if report:
        (run_dir / "report.json").write_text(
            json.dumps({"judge": {"quality_retention": 0.792}})
        )
    return run_dir


# ------------------------------------------------------------------ sync logic


def test_sync_run_mirrors_summary_metrics_and_report(tmp_path):
    store = MemoryStore()
    run_dir = _make_run(tmp_path / "runs", "alpha", metrics=4, report=True)
    result = sync_run(store, run_dir, host="mac")

    assert result == {"metrics": 4, "report": True}
    run = store.get_run("mac", "alpha")
    assert run["state"] == "completed"
    assert run["quality_retention"] == 0.792
    assert [m["step"] for m in store.get_metrics("mac", "alpha")] == [1, 2, 3, 4]
    assert store.get_report("mac", "alpha")["judge"]["quality_retention"] == 0.792


def test_sync_is_incremental(tmp_path):
    store = MemoryStore()
    run_dir = _make_run(tmp_path / "runs", "alpha", metrics=2)
    sync_run(store, run_dir, host="mac")

    with (run_dir / "metrics.jsonl").open("a") as f:
        f.write(json.dumps({"kind": "train", "step": 3, "loss": 2.5}) + "\n")
    result = sync_run(store, run_dir, host="mac")

    assert result["metrics"] == 1  # only the new line crossed the wire
    assert [m["step"] for m in store.get_metrics("mac", "alpha")] == [1, 2, 3]


def test_restart_truncation_rebuilds_the_mirror(tmp_path):
    store = MemoryStore()
    run_dir = _make_run(tmp_path / "runs", "alpha", metrics=5)
    sync_run(store, run_dir, host="mac")

    _make_run(tmp_path / "runs", "alpha", metrics=2)  # restarted: shorter file
    sync_run(store, run_dir, host="mac")

    assert [m["step"] for m in store.get_metrics("mac", "alpha")] == [1, 2]


def test_sync_all_walks_run_dirs_and_skips_non_runs(tmp_path):
    store = MemoryStore()
    runs_root = tmp_path / "runs"
    _make_run(runs_root, "a", metrics=2)
    _make_run(runs_root, "b", metrics=3, report=True)
    (runs_root / ".jobs").mkdir()  # internal dir: never synced
    (runs_root / "not-a-run").mkdir()  # no status.json: skipped

    counts = sync_all(store, runs_root, host="mac")

    assert counts == {"runs": 2, "metrics": 5, "reports": 1}
    assert {r["name"] for r in store.list_runs()} == {"a", "b"}
    assert all(r["host"] == "mac" for r in store.list_runs())


def test_run_signature_changes_when_files_change(tmp_path):
    run_dir = _make_run(tmp_path / "runs", "alpha")
    before = run_signature(run_dir)
    with (run_dir / "metrics.jsonl").open("a") as f:
        f.write(json.dumps({"kind": "train", "step": 99, "loss": 1.0}) + "\n")
    assert run_signature(run_dir) != before


def test_host_label_env_override(monkeypatch):
    monkeypatch.setenv("DISTILL_HOST_LABEL", "colab-a100")
    assert local_host_label() == "colab-a100"


# ------------------------------------------------------------ credentials


def test_db_url_env_wins_over_file(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    save_db_url(DSN_WITH_SECRET, config)
    assert resolve_db_url(config) == (DSN_WITH_SECRET, "file")

    monkeypatch.setenv("DISTILL_DB_URL", "postgresql://env@envhost/db")
    assert resolve_db_url(config) == ("postgresql://env@envhost/db", "env")

    monkeypatch.delenv("DISTILL_DB_URL")
    clear_db_url(config)
    assert resolve_db_url(config) == (None, None)


def test_saved_credential_is_owner_only(tmp_path):
    config = tmp_path / "config.json"
    save_db_url(DSN_WITH_SECRET, config)
    mode = stat.S_IMODE(config.stat().st_mode)
    assert mode == 0o600


def test_redact_dsn_never_leaks_the_password():
    redacted = redact_dsn(DSN_WITH_SECRET)
    assert redacted == {"user": "alice", "host": "db.example.com", "dbname": "history"}
    assert "hunter2" not in json.dumps(redacted)


def test_validate_db_url_rejects_non_postgres():
    with pytest.raises(ValueError):
        validate_db_url("mysql://u:p@h/db")
    with pytest.raises(ValueError):
        validate_db_url("postgresql://")  # no host
    validate_db_url("postgres://u:p@h/db")  # both scheme spellings accepted


# ------------------------------------- the same contract on a real Postgres


@pytest.mark.skipif(not os.environ.get("TEST_PG_DSN"), reason="TEST_PG_DSN not set")
def test_pgstore_round_trip(tmp_path):
    from distillanything.db.store import PgStore

    store = PgStore(os.environ["TEST_PG_DSN"])
    store.connect()  # migrates the schema
    host = f"pytest-{os.getpid()}"
    try:
        runs_root = tmp_path / "runs"
        _make_run(runs_root, "pg-run", metrics=3, report=True)
        counts = sync_all(store, runs_root, host=host)
        assert counts == {"runs": 1, "metrics": 3, "reports": 1}

        run = store.get_run(host, "pg-run")
        assert run["state"] == "completed" and run["quality_retention"] == 0.792
        assert [m["step"] for m in store.get_metrics(host, "pg-run")] == [1, 2, 3]

        # incremental append
        with (runs_root / "pg-run" / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"kind": "train", "step": 4, "loss": 2.0}) + "\n")
        assert sync_run(store, runs_root / "pg-run", host)["metrics"] == 1
        assert store.metric_count(host, "pg-run") == 4

        # restart truncation
        _make_run(runs_root, "pg-run", metrics=2)
        sync_run(store, runs_root / "pg-run", host)
        assert [m["step"] for m in store.get_metrics(host, "pg-run")] == [1, 2]

        assert store.get_report(host, "pg-run")["judge"]["quality_retention"] == 0.792
        assert any(r["host"] == host for r in store.list_runs())
        status = store.status()
        assert status["runs"] >= 1 and status["schema_version"] == 1
    finally:
        store._execute("DELETE FROM da_runs WHERE host = %s", (host,))
        store.close()
