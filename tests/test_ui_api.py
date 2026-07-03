"""Dashboard API tests: auth, traversal, reads, SSE, and the job lifecycle.

Fully offline — runs are fixture directories, jobs are tiny `python -c` argv
stubs, no models or network involved.
"""

import json
import sys
import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from distillanything.ui.jobs import JobManager, reconcile_interrupted_runs  # noqa: E402
from distillanything.ui.server import create_app  # noqa: E402

TOKEN = "test-token-123"


def _make_run(runs_root, name, state="completed", metrics=3):
    run_dir = runs_root / name
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "state": state,
                "mode": "logit",
                "device": "cpu",
                "student": "tiny-student",
                "teacher": "hf:tiny-teacher",
                "total_steps": 15,
                "steps_completed": 15 if state == "completed" else 5,
            }
        )
    )
    (run_dir / "distill_config.json").write_text(json.dumps({"mode": "logit"}))
    with (run_dir / "metrics.jsonl").open("w") as f:
        for i in range(metrics):
            f.write(json.dumps({"kind": "train", "step": i + 1, "loss": 3.0 - i * 0.1}) + "\n")
    return run_dir


@pytest.fixture()
def env(tmp_path):
    runs_root = tmp_path / "runs"
    data_root = tmp_path / "data"
    runs_root.mkdir()
    data_root.mkdir()
    _make_run(runs_root, "run-a")
    _make_run(runs_root, "run-b", state="running")
    with (data_root / "train.jsonl").open("w") as f:
        for i in range(120):
            f.write(json.dumps({"prompt": f"p{i}", "response": f"r{i}"}) + "\n")
    manager = JobManager(max_concurrent=1)
    app = create_app(runs_root, data_root, token=TOKEN, jobs=manager, serve_static=False)
    client = TestClient(app, base_url="http://127.0.0.1")
    client.headers["Authorization"] = f"Bearer {TOKEN}"
    yield client, runs_root, data_root, manager
    manager.shutdown()


# ---------------------------------------------------------------- security


def test_api_requires_token(env):
    client, *_ = env
    bare = TestClient(client.app, base_url="http://127.0.0.1")
    assert bare.get("/api/health").status_code == 200  # health is the only open door
    assert bare.get("/api/runs").status_code == 401
    assert bare.get("/api/runs", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert bare.post("/api/jobs/train", json={}).status_code == 401


def test_token_via_query_param_for_sse(env):
    client, *_ = env
    bare = TestClient(client.app, base_url="http://127.0.0.1")
    assert bare.get(f"/api/runs?token={TOKEN}").status_code == 200


def test_host_allowlist_blocks_dns_rebinding(env):
    client, *_ = env
    evil = TestClient(client.app, base_url="http://evil.example.com")
    evil.headers["Authorization"] = f"Bearer {TOKEN}"
    assert evil.get("/api/runs").status_code == 403


def test_path_traversal_rejected(env):
    client, *_ = env
    for name in ["..", "%2e%2e", ".hidden", "-flag", "a%2fb"]:
        r = client.get(f"/api/runs/{name}")
        assert r.status_code in (400, 404), name


def test_security_headers_present(env):
    client, *_ = env
    r = client.get("/api/runs")
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["x-content-type-options"] == "nosniff"


def test_oversized_body_rejected(env):
    client, *_ = env
    r = client.post(
        "/api/jobs/train",
        content=b"{}",
        headers={"Content-Length": "9999999", "Content-Type": "application/json"},
    )
    assert r.status_code == 413


# ------------------------------------------------------------------- reads


def test_list_and_detail(env):
    client, *_ = env
    runs = client.get("/api/runs").json()
    assert {r["name"] for r in runs} == {"run-a", "run-b"}
    a = next(r for r in runs if r["name"] == "run-a")
    assert a["state"] == "completed" and a["last_metric"]["loss"] == pytest.approx(2.8)

    detail = client.get("/api/runs/run-a").json()
    assert detail["config"] == {"mode": "logit"}
    assert client.get("/api/runs/nope").status_code == 404


def test_metrics_endpoint(env):
    client, *_ = env
    metrics = client.get("/api/runs/run-a/metrics").json()
    assert len(metrics) == 3 and metrics[0]["step"] == 1


def test_report_404_when_missing(env):
    client, *_ = env
    assert client.get("/api/runs/run-a/report").status_code == 404


def test_datasets_and_pagination(env):
    client, *_ = env
    ds = client.get("/api/datasets").json()
    assert ds[0]["name"] == "train.jsonl" and ds[0]["records"] == 120

    page = client.get("/api/datasets/train.jsonl/records?offset=100&limit=50").json()
    assert page["total"] == 120 and len(page["records"]) == 20
    assert page["records"][0]["prompt"] == "p100"


def test_metrics_sse_stream_yields_lines_then_done(env):
    client, *_ = env
    lines = []
    with client.stream("GET", "/api/runs/run-a/metrics/stream") as r:
        for raw in r.iter_lines():
            if raw.startswith("data: "):
                lines.append(json.loads(raw[6:]))
            if raw.startswith("event: done"):
                break
    assert [x["step"] for x in lines] == [1, 2, 3]


# -------------------------------------------------------------------- jobs


def _wait(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_job_manager_lifecycle(tmp_path):
    m = JobManager()
    job = m.submit("train", [sys.executable, "-c", "print('hello from job')"], tmp_path / "a.log")
    assert _wait(lambda: m.get(job.id).status == "completed")
    assert m.get(job.id).returncode == 0
    assert "hello from job" in (tmp_path / "a.log").read_text()


def test_job_manager_stop_and_queue(tmp_path):
    m = JobManager(max_concurrent=1)
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    first = m.submit("train", sleeper, tmp_path / "1.log")
    second = m.submit("train", sleeper, tmp_path / "2.log")
    assert _wait(lambda: m.get(first.id).status == "running")
    assert m.get(second.id).status == "queued"

    m.stop(first.id)
    assert _wait(lambda: m.get(first.id).status == "stopped")
    assert _wait(lambda: m.get(second.id).status == "running")  # queue advanced
    m.shutdown()
    assert _wait(lambda: m.get(second.id).status == "stopped")


def test_job_manager_failed_status(tmp_path):
    m = JobManager()
    job = m.submit("train", [sys.executable, "-c", "raise SystemExit(3)"], tmp_path / "f.log")
    assert _wait(lambda: m.get(job.id).status == "failed")
    assert m.get(job.id).returncode == 3


def test_reconcile_marks_dead_running_runs(tmp_path):
    run_dir = tmp_path / "orphan"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(json.dumps({"state": "running"}))
    (run_dir / "job.json").write_text(json.dumps({"pid": 99999999, "job_id": "x", "kind": "train"}))
    fixed = reconcile_interrupted_runs(tmp_path)
    assert fixed == ["orphan"]
    assert json.loads((run_dir / "status.json").read_text())["state"] == "interrupted"


def test_train_job_endpoint_validates_and_spawns(env, monkeypatch):
    client, runs_root, data_root, manager = env
    # Replace the real CLI argv with a fast no-op so the endpoint's full flow
    # (validation -> recipe.yaml -> spawn) runs without training anything.
    import distillanything.ui.server as server_mod

    monkeypatch.setattr(server_mod, "cli_argv", lambda *a: [sys.executable, "-c", "print('ok')"])

    r = client.post(
        "/api/jobs/train",
        json={"name": "new-run", "dataset": "train.jsonl", "config": {"mode": "seqkd"}},
    )
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]
    assert _wait(lambda: manager.get(job_id).status == "completed")

    recipe = (runs_root / "new-run" / "recipe.yaml").read_text()
    assert "seqkd" in recipe and str(data_root / "train.jsonl") in recipe
    assert client.get("/api/jobs").json()[0]["id"] == job_id


def test_train_job_rejects_unknown_recipe_fields(env):
    client, *_ = env
    r = client.post(
        "/api/jobs/train",
        json={"name": "x", "dataset": "train.jsonl", "config": {"modee": "logit"}},
    )
    assert r.status_code == 422
    r = client.post("/api/jobs/train", json={"name": "x", "dataset": "missing.jsonl", "config": {}})
    assert r.status_code == 400
    r = client.post("/api/jobs/train", json={"name": "../evil", "dataset": "train.jsonl"})
    assert r.status_code == 400


def test_restart_run_from_config_snapshot(env, monkeypatch):
    client, runs_root, _, manager = env
    import distillanything.ui.server as server_mod

    monkeypatch.setattr(server_mod, "cli_argv", lambda *a: [sys.executable, "-c", "print('ok')"])

    # run-a has distill_config.json but no recipe.yaml (like a CLI-started run):
    # restart must rebuild the recipe from the snapshot. The fixture's config is
    # partial ({"mode": "logit"}) — defaults fill the rest.
    r = client.post("/api/runs/run-a/restart")
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]
    assert _wait(lambda: manager.get(job_id).status == "completed")
    recipe = (runs_root / "run-a" / "recipe.yaml").read_text()
    assert "mode: logit" in recipe and str(runs_root / "run-a") in recipe


def test_restart_rejects_running_and_missing(env):
    client, *_ = env
    assert client.post("/api/runs/run-b/restart").status_code == 409  # running
    assert client.post("/api/runs/ghost/restart").status_code == 404


def test_generate_job_validates_specs(env):
    client, *_ = env
    bad = {"name": "gen1", "teacher": "--rm -rf", "seeds_text": "hello"}
    assert client.post("/api/jobs/generate", json=bad).status_code == 422


def test_stop_unknown_job_404(env):
    client, *_ = env
    assert client.post("/api/jobs/nope/stop").status_code == 404
