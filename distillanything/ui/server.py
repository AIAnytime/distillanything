"""The dashboard server: a FastAPI app over runsource (read) + JobManager (write).

Design rules:
- Clients send *names*; the server joins them under runs_root/data_root. No client
  string is ever treated as a path or reaches a shell.
- Job request bodies validate against the same Pydantic models the CLI uses
  (DistillConfig et al., extra="forbid"), so the API can't express anything the
  recipe format can't.
- Live views are SSE tails of the files the trainer writes; the UI works the same
  whether a run was started here or from the terminal.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from distillanything import __version__
from distillanything.config import DistillConfig
from distillanything.ui import runsource
from distillanything.ui.jobs import JobManager, cli_argv, reconcile_interrupted_runs
from distillanything.ui.runsource import InvalidName, safe_child
from distillanything.ui.security import SecurityMiddleware

# Teacher/judge specs: hf:org/repo, claude[:model], openai:model, ollama:model.
# No leading dash (flag injection) and a conservative charset.
_SPEC_RE = re.compile(r"^[A-Za-z0-9][\w.:/@-]*$")

STATIC_DIR = Path(__file__).parent / "static"


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_spec(value: Optional[str]) -> Optional[str]:
    if value is not None and not _SPEC_RE.match(value):
        raise ValueError(f"invalid model spec: {value!r}")
    return value


class TrainJobRequest(_RequestModel):
    name: str
    dataset: str
    # Partial recipe (mode/teacher/student/loss/train); unknown keys are rejected
    # by DistillConfig itself. data.path and train.output_dir are server-owned.
    config: dict = Field(default_factory=dict)


class GenerateJobRequest(_RequestModel):
    name: str  # output dataset name (without .jsonl)
    teacher: str
    seeds_text: Optional[str] = None
    seeds_dataset: Optional[str] = None
    expand: int = Field(0, ge=0, le=50)
    system: Optional[str] = None
    max_tokens: int = Field(512, ge=1, le=8192)
    judge: Optional[str] = None
    min_score: int = Field(7, ge=1, le=10)
    concurrency: int = Field(4, ge=1, le=32)

    _teacher = field_validator("teacher")(_check_spec)
    _judge = field_validator("judge")(_check_spec)


class ReportJobRequest(_RequestModel):
    run: str
    dataset: str
    teacher: Optional[str] = None
    judge: Optional[str] = None
    n: int = Field(32, ge=1, le=1000)
    max_new_tokens: int = Field(256, ge=1, le=4096)
    cost_per_hour: Optional[float] = Field(None, ge=0)
    benchmark_teacher: bool = True

    _teacher = field_validator("teacher")(_check_spec)
    _judge = field_validator("judge")(_check_spec)


async def _sse_tail(path: Path, request: Request, done: callable, poll: float = 0.5):
    """Yield new lines of ``path`` as SSE events until drained AND done() is true."""
    pos = 0
    while True:
        if await request.is_disconnected():
            return
        emitted = False
        if path.exists():
            if path.stat().st_size < pos:
                pos = 0  # file was truncated (run restarted) — tail from the top
            with path.open("rb") as f:
                f.seek(pos)
                chunk = f.read()
            if chunk.endswith(b"\n"):  # only consume complete lines (torn writes)
                pos += len(chunk)
                for line in chunk.decode("utf-8", errors="replace").splitlines():
                    if line.strip():
                        emitted = True
                        yield f"data: {line}\n\n"
        if not emitted:
            if done():
                yield "event: done\ndata: {}\n\n"
                return
            yield ": ping\n\n"
        await asyncio.sleep(poll)


def create_app(
    runs_root: Path,
    data_root: Path,
    token: str,
    jobs: Optional[JobManager] = None,
    enforce_host_allowlist: bool = True,
    serve_static: bool = True,
) -> FastAPI:
    runs_root = Path(runs_root).resolve()
    data_root = Path(data_root).resolve()
    jobs = jobs or JobManager()
    jobs_log_dir = runs_root / ".jobs"

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        reconcile_interrupted_runs(runs_root)
        yield
        jobs.shutdown()

    app = FastAPI(title="Distill Anything", version=__version__, lifespan=lifespan, docs_url=None,
                  redoc_url=None, openapi_url=None)
    app.add_middleware(SecurityMiddleware, token=token, enforce_host_allowlist=enforce_host_allowlist)

    def _run_dir(name: str) -> Path:
        try:
            return safe_child(runs_root, name)
        except InvalidName as exc:
            raise HTTPException(400, str(exc)) from exc

    def _dataset_path(name: str) -> Path:
        try:
            return safe_child(data_root, name)
        except InvalidName as exc:
            raise HTTPException(400, str(exc)) from exc

    def _run_state(run_dir: Path) -> str:
        try:
            return json.loads((run_dir / "status.json").read_text()).get("state", "unknown")
        except (OSError, json.JSONDecodeError):
            return "unknown"

    def _run_active(name: str, run_dir: Path) -> bool:
        """More data may still arrive: the run is queued/running on disk, or a
        job for it is in flight (covers the model-loading gap where the trainer
        hasn't overwritten the previous terminal status yet)."""
        if _run_state(run_dir) in ("queued", "running"):
            return True
        return any(
            j["run_name"] == name and j["status"] in ("queued", "running") for j in jobs.list()
        )

    def _mark_queued(run_dir: Path, cfg: DistillConfig) -> None:
        """Written at submit time so the run appears (and streams stay open)
        before the subprocess finishes loading models. The trainer overwrites
        this with state=running when steps actually begin."""
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.jsonl").write_text("")  # drop stale curves right away
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "state": "queued",
                    "mode": cfg.mode,
                    "student": cfg.student.model,
                    "teacher": cfg.teacher.spec,
                },
                indent=2,
            )
        )

    # ------------------------------------------------------------------ reads

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/version")
    def version():
        return {"version": __version__}

    @app.get("/api/runs")
    def runs():
        return runsource.list_runs(runs_root)

    @app.get("/api/runs/{name}")
    def run(name: str):
        detail = runsource.run_detail(runs_root, _run_dir(name).name)
        if detail is None:
            raise HTTPException(404, "run not found")
        return detail

    @app.get("/api/runs/{name}/metrics")
    def metrics(name: str):
        return runsource.read_metrics(runs_root, _run_dir(name).name)

    @app.get("/api/runs/{name}/report")
    def report(name: str):
        path = _run_dir(name) / "report.json"
        if not path.exists():
            raise HTTPException(404, "no report for this run")
        return json.loads(path.read_text())

    @app.get("/api/runs/{name}/metrics/stream")
    async def metrics_stream(name: str, request: Request):
        run_dir = _run_dir(name)
        gen = _sse_tail(run_dir / "metrics.jsonl", request, done=lambda: not _run_active(name, run_dir))
        return StreamingResponse(gen, media_type="text/event-stream")

    @app.get("/api/runs/{name}/logs/stream")
    async def logs_stream(name: str, request: Request):
        run_dir = _run_dir(name)
        gen = _sse_tail(run_dir / "train.log", request, done=lambda: not _run_active(name, run_dir))
        return StreamingResponse(gen, media_type="text/event-stream")

    @app.get("/api/datasets")
    def datasets():
        return runsource.list_datasets(data_root)

    @app.get("/api/datasets/{name}/records")
    def dataset_records(name: str, offset: int = 0, limit: int = 50):
        _dataset_path(name)
        try:
            return runsource.read_dataset_records(data_root, name, offset, min(limit, 200))
        except FileNotFoundError:
            raise HTTPException(404, "dataset not found") from None

    # ------------------------------------------------------------------- jobs

    @app.get("/api/jobs")
    def list_jobs():
        return jobs.list()

    @app.post("/api/jobs/train", status_code=201)
    def submit_train(body: TrainJobRequest):
        run_dir = _run_dir(body.name)
        dataset = _dataset_path(body.dataset)
        if not dataset.exists():
            raise HTTPException(400, f"dataset not found: {body.dataset}")
        if (run_dir / "status.json").exists() and _run_state(run_dir) == "running":
            raise HTTPException(409, f"run {body.name!r} is currently running")
        try:
            cfg = DistillConfig.model_validate(body.config)
        except Exception as exc:
            raise HTTPException(422, f"invalid recipe: {exc}") from exc
        cfg.data.path = str(dataset)
        cfg.train.output_dir = str(run_dir)
        _mark_queued(run_dir, cfg)
        recipe = run_dir / "recipe.yaml"
        cfg.to_yaml(recipe)
        job = jobs.submit(
            "train", cli_argv("train", str(recipe)), log_path=run_dir / "train.log", run_name=body.name
        )
        return job.to_dict()

    @app.post("/api/runs/{name}/restart", status_code=201)
    def restart_run(name: str):
        """Re-run a finished/stopped run from its saved recipe (fresh start, same run dir)."""
        run_dir = _run_dir(name)
        if not run_dir.is_dir():
            raise HTTPException(404, "run not found")
        if _run_state(run_dir) == "running":
            raise HTTPException(409, f"run {name!r} is currently running")
        recipe = run_dir / "recipe.yaml"
        if recipe.exists():
            try:
                cfg = DistillConfig.from_yaml(recipe)
            except Exception as exc:
                raise HTTPException(422, f"saved recipe is invalid: {exc}") from exc
        else:
            # CLI-started runs have no recipe.yaml, but the trainer always saves
            # the full config snapshot — rebuild the recipe from it.
            cfg_snapshot = run_dir / "distill_config.json"
            if not cfg_snapshot.exists():
                raise HTTPException(400, "no recipe or config snapshot to restart from")
            try:
                cfg = DistillConfig.model_validate(json.loads(cfg_snapshot.read_text()))
            except Exception as exc:
                raise HTTPException(422, f"saved config is invalid: {exc}") from exc
            cfg.train.output_dir = str(run_dir)
            cfg.to_yaml(recipe)
        _mark_queued(run_dir, cfg)
        job = jobs.submit(
            "train", cli_argv("train", str(recipe)), log_path=run_dir / "train.log", run_name=name
        )
        return job.to_dict()

    @app.post("/api/jobs/generate", status_code=201)
    def submit_generate(body: GenerateJobRequest):
        out_path = _dataset_path(body.name + ".jsonl")
        if body.seeds_text is None and body.seeds_dataset is None:
            raise HTTPException(400, "provide seeds_text or seeds_dataset")
        if body.seeds_dataset is not None:
            seeds = _dataset_path(body.seeds_dataset)
            if not seeds.exists():
                raise HTTPException(400, f"seeds dataset not found: {body.seeds_dataset}")
        else:
            data_root.mkdir(parents=True, exist_ok=True)
            seeds = _dataset_path(body.name + ".seeds.txt")
            seeds.write_text(body.seeds_text)
        argv = cli_argv(
            "generate", str(seeds), "--out", str(out_path), "--teacher", body.teacher,
            "--max-tokens", str(body.max_tokens), "--expand", str(body.expand),
            "--concurrency", str(body.concurrency),
        )
        if body.system:
            argv += ["--system", body.system]
        if body.judge:
            argv += ["--judge", body.judge, "--min-score", str(body.min_score)]
        job = jobs.submit("generate", argv, log_path=jobs_log_dir / "generate.log")
        return job.to_dict()

    @app.post("/api/jobs/report", status_code=201)
    def submit_report(body: ReportJobRequest):
        run_dir = _run_dir(body.run)
        dataset = _dataset_path(body.dataset)
        if not run_dir.is_dir():
            raise HTTPException(400, f"run not found: {body.run}")
        if not dataset.exists():
            raise HTTPException(400, f"dataset not found: {body.dataset}")
        argv = cli_argv(
            "report", str(run_dir), "--dataset", str(dataset), "--n", str(body.n),
            "--max-new-tokens", str(body.max_new_tokens),
        )
        if body.teacher:
            argv += ["--teacher", body.teacher]
        if body.judge:
            argv += ["--judge", body.judge]
        if body.cost_per_hour is not None:
            argv += ["--cost-per-hour", str(body.cost_per_hour)]
        if not body.benchmark_teacher:
            argv += ["--no-benchmark-teacher"]
        job = jobs.submit("report", argv, log_path=jobs_log_dir / f"report-{body.run}.log",
                          run_name=None)
        return job.to_dict()

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str):
        job = jobs.stop(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job.to_dict()

    @app.get("/api/jobs/{job_id}/logs/stream")
    async def job_logs_stream(job_id: str, request: Request):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        gen = _sse_tail(
            job.log_path, request,
            done=lambda: (jobs.get(job_id) or job).status not in ("queued", "running"),
        )
        return StreamingResponse(gen, media_type="text/event-stream")

    # ------------------------------------------------------------------- SPA

    if serve_static and STATIC_DIR.is_dir() and (STATIC_DIR / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            candidate = (STATIC_DIR / path).resolve()
            if path and candidate.is_file() and candidate.is_relative_to(STATIC_DIR.resolve()):
                return FileResponse(candidate)
            return FileResponse(STATIC_DIR / "index.html")

    return app


def run_server(
    runs_root: str = "runs",
    data_root: str = "data",
    host: str = "127.0.0.1",
    port: int = 7326,
    token: Optional[str] = None,
    open_browser: bool = True,
) -> None:
    """Entry point used by `distill ui`."""
    import webbrowser

    import uvicorn

    from distillanything.ui.security import generate_token

    is_loopback_bind = host in {"127.0.0.1", "localhost", "::1"}
    if not is_loopback_bind and not token:
        raise SystemExit(
            "Refusing to bind non-localhost without an explicit --token. "
            "The dashboard can start training jobs; exposing it beyond this machine "
            "requires you to choose a strong token yourself."
        )
    token = token or generate_token()

    app = create_app(
        Path(runs_root), Path(data_root), token=token,
        enforce_host_allowlist=is_loopback_bind,
    )
    url = f"http://{'127.0.0.1' if is_loopback_bind else host}:{port}/?token={token}"
    # flush=True: uvicorn.run never returns, so without it the URL can sit in a
    # block buffer forever when stdout is redirected.
    print(f"\n  Distill Anything dashboard: {url}\n", flush=True)
    if not is_loopback_bind:
        print(
            "  WARNING: binding beyond localhost — anyone with this token controls training jobs.\n",
            flush=True,
        )
    if open_browser and is_loopback_bind:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=port, log_level="warning")
