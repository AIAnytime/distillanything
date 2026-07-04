import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, sse } from "../api";
import {
  ConfirmDialog,
  Retention,
  StateBadge,
  fmtNum,
  shortModel,
  usePoll,
  useToast,
} from "../components";
import { MetricChart, chartColors } from "../MetricChart";
import type { Benchmark, DatasetInfo, Job, MetricPoint, Report, RunDetail } from "../types";

function BuildReportForm({
  runName,
  defaultTeacher,
  onSubmitted,
}: {
  runName: string;
  defaultTeacher: string;
  onSubmitted: (jobId: string) => void;
}) {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [dataset, setDataset] = useState("");
  const [judge, setJudge] = useState("");
  const [teacher, setTeacher] = useState(defaultTeacher);
  const [n, setN] = useState(24);
  const [costPerHour, setCostPerHour] = useState<number | "">("");
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  useEffect(() => {
    api<DatasetInfo[]>("/api/datasets").then((ds) => {
      setDatasets(ds);
      if (ds.length) setDataset((cur) => cur || ds[0].name);
    });
  }, []);

  const submit = async () => {
    setBusy(true);
    try {
      const job = await api<Job>("/api/jobs/report", {
        method: "POST",
        body: JSON.stringify({
          run: runName,
          dataset,
          n,
          ...(judge ? { judge } : {}),
          ...(teacher ? { teacher } : {}),
          ...(costPerHour !== "" ? { cost_per_hour: costPerHour } : {}),
        }),
      });
      toast("Building the report card — the student answers every prompt, so give it a few minutes.");
      onSubmitted(job.id);
    } catch (e) {
      toast(`Report failed to start: ${(e as Error).message}`, true);
      setBusy(false);
    }
  };

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div className="card-head">Build report card</div>
      <div className="card-body" style={{ display: "grid", gap: 14 }}>
        <div className="grid-4">
          <div className="field">
            <label>Eval dataset</label>
            <select className="select" value={dataset} onChange={(e) => setDataset(e.target.value)}>
              {datasets.map((ds) => (
                <option key={ds.name} value={ds.name}>
                  {ds.name} ({ds.records})
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Judge (optional)</label>
            <input
              className="input mono"
              value={judge}
              onChange={(e) => setJudge(e.target.value)}
              placeholder="claude"
            />
            <span className="hint">blind A/B quality retention; empty = benchmarks only</span>
          </div>
          <div className="field">
            <label>Teacher (for side-by-side)</label>
            <input className="input mono" value={teacher} onChange={(e) => setTeacher(e.target.value)} />
          </div>
          <div className="field">
            <label>Prompts (n)</label>
            <input
              className="input"
              type="number"
              min={1}
              max={200}
              value={n}
              onChange={(e) => setN(Number(e.target.value))}
            />
          </div>
        </div>
        <div className="grid-4" style={{ alignItems: "end" }}>
          <div className="field">
            <label>Hardware $/hour (optional)</label>
            <input
              className="input"
              type="number"
              step="any"
              min={0}
              value={costPerHour}
              placeholder="1.20"
              onChange={(e) =>
                setCostPerHour(e.target.value === "" ? "" : Number(e.target.value))
              }
            />
            <span className="hint">prices $/1K tokens in the efficiency table</span>
          </div>
          <div>
            <button className="btn btn-primary" disabled={busy || !dataset} onClick={submit}>
              Build report
            </button>
          </div>
        </div>
        <div className="hint" style={{ color: "var(--muted)", fontSize: 12 }}>
          API judges (e.g. <code>claude</code>) read their key from the environment where{" "}
          <code>distill ui</code> runs.
        </div>
      </div>
    </div>
  );
}

function BenchTable({ report }: { report: Report }) {
  const rows: [string, keyof Benchmark, number][] = [
    ["Parameters (M)", "parameters_m", 1],
    ["Tokens / s", "tokens_per_s", 1],
    ["Latency p50 (s)", "latency_p50_s", 3],
    ["Latency p95 (s)", "latency_p95_s", 3],
    ["Memory (MB)", "memory_mb", 1],
    ["Disk (MB)", "disk_size_mb", 1],
    ["$ / 1K tokens", "cost_per_1k_tokens_usd", 5],
  ];
  const s = report.student_benchmark ?? {};
  const t = report.teacher_benchmark;
  return (
    <table className="table">
      <thead>
        <tr>
          <th>Metric</th>
          <th className="right">Student</th>
          {t && <th className="right">Teacher</th>}
        </tr>
      </thead>
      <tbody>
        {rows
          .filter(([, key]) => s[key] !== undefined || (t && t[key] !== undefined))
          .map(([label, key, digits]) => (
            <tr key={key}>
              <td>{label}</td>
              <td className="right num">{fmtNum(s[key], digits)}</td>
              {t && <td className="right num">{fmtNum(t[key], digits)}</td>}
            </tr>
          ))}
      </tbody>
    </table>
  );
}

function ReportCard({ report }: { report: Report }) {
  const judge = report.judge;
  return (
    <>
      {judge && (
        <div className="card">
          <div className="card-head">
            Quality (LLM-as-judge)
            <span className="badge">
              judge: {report.judge_name} · blind, position-swapped
            </span>
          </div>
          <div className="card-body" style={{ display: "flex", gap: 32, alignItems: "center" }}>
            <div className="stat" style={{ padding: 0 }}>
              <div className="label">Quality retention</div>
              <div className="value">
                {(judge.quality_retention * 100).toFixed(1)}%{" "}
                <small>of {judge.n} prompts</small>
              </div>
            </div>
            <div className="kv" style={{ flex: 1 }}>
              <dt>Student wins</dt>
              <dd className="num">
                {judge.student_wins} ({(judge.student_win_rate * 100).toFixed(0)}%)
              </dd>
              <dt>Ties</dt>
              <dd className="num">
                {judge.ties} ({(judge.tie_rate * 100).toFixed(0)}%)
              </dd>
              <dt>Teacher wins</dt>
              <dd className="num">
                {judge.teacher_wins} ({(judge.teacher_win_rate * 100).toFixed(0)}%)
              </dd>
            </div>
          </div>
        </div>
      )}
      {report.student_benchmark && (
        <div className="card">
          <div className="card-head">Efficiency</div>
          <BenchTable report={report} />
        </div>
      )}
      {report.samples && report.samples.length > 0 && (
        <div className="card">
          <div className="card-head">Sample outputs</div>
          <div className="card-body" style={{ display: "grid", gap: 14 }}>
            {report.samples.map((sample, i) => (
              <div key={i}>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>{sample.prompt}</div>
                <div
                  className="mono"
                  style={{
                    color: "var(--muted)",
                    borderLeft: "3px solid var(--border-strong)",
                    paddingLeft: 12,
                    whiteSpace: "pre-wrap",
                  }}
                >
                  {sample.student_answer}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}

export function RunDetailPage() {
  const { name = "" } = useParams();
  const [run, setRun] = useState<RunDetail | null>(null);
  const [points, setPoints] = useState<MetricPoint[]>([]);
  const [report, setReport] = useState<Report | null>(null);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [showLogs, setShowLogs] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  // Bumped on restart: tears down the SSE stream and re-tails from scratch.
  const [epoch, setEpoch] = useState(0);
  const [confirmRestart, setConfirmRestart] = useState(false);
  const [showReportForm, setShowReportForm] = useState(false);
  const [reportJobId, setReportJobId] = useState<string | null>(null);
  const toast = useToast();
  const logRef = useRef<HTMLDivElement>(null);
  const seenSteps = useRef<Set<string>>(new Set());

  const refresh = useCallback(async () => {
    try {
      const detail = await api<RunDetail>(`/api/runs/${name}`);
      setRun(detail);
      if (detail.has_report) {
        setReport(await api<Report>(`/api/runs/${name}/report`));
      }
    } catch (e) {
      toast(`Failed to load run: ${(e as Error).message}`, true);
    }
  }, [name, toast]);

  // Initial load: full history, then live-tail while running.
  useEffect(() => {
    seenSteps.current = new Set();
    setPoints([]);
    setLogLines([]);
    setReport(null);
    refresh();
    api<MetricPoint[]>(`/api/runs/${name}/metrics`).then((history) => {
      history.forEach((p) => seenSteps.current.add(`${p.kind}:${p.step}`));
      setPoints(history);
    });
    const closeMetrics = sse(
      `/api/runs/${name}/metrics/stream`,
      (line) => {
        try {
          const p = JSON.parse(line) as MetricPoint;
          const key = `${p.kind}:${p.step}`;
          if (!seenSteps.current.has(key)) {
            seenSteps.current.add(key);
            setPoints((prev) => [...prev, p]);
          }
        } catch {
          /* skip malformed line */
        }
      },
      () => refresh(), // run finished — pick up final state and report
    );
    return closeMetrics;
  }, [name, refresh, epoch]);

  // Keep header state fresh — catches restarts and CLI-side changes.
  usePoll(refresh, 5000, [name]);

  // Track an in-flight report job until it lands (the refresh poll then
  // picks up report.json and renders the card).
  usePoll(
    async () => {
      if (!reportJobId) return;
      const jobs = await api<Job[]>("/api/jobs");
      const job = jobs.find((j) => j.id === reportJobId);
      if (!job || job.status === "queued" || job.status === "running") return;
      setReportJobId(null);
      if (job.status === "completed") {
        toast("Report card ready");
        refresh();
      } else {
        toast(`Report build ${job.status} — see the Jobs drawer for details`, true);
      }
    },
    4000,
    [reportJobId],
  );

  useEffect(() => {
    if (!showLogs) return;
    const close = sse(`/api/runs/${name}/logs/stream`, (line) =>
      setLogLines((prev) => [...prev.slice(-500), line]),
    );
    return close;
  }, [name, showLogs]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [logLines]);

  const restartRun = async () => {
    setConfirmRestart(false);
    try {
      await api(`/api/runs/${name}/restart`, { method: "POST" });
      toast("Run restarted — model loads first, charts stream once training begins.");
      setEpoch((e) => e + 1);
    } catch (e) {
      toast(`Restart failed: ${(e as Error).message}`, true);
    }
  };

  const stopRun = async () => {
    try {
      const jobs = await api<Job[]>("/api/jobs");
      const job = jobs.find((j) => j.run_name === name && j.status === "running");
      if (!job) {
        toast("This run was started outside the dashboard — stop it with Ctrl-C in its terminal.", true);
        return;
      }
      await api(`/api/jobs/${job.id}/stop`, { method: "POST" });
      toast("Stop requested — the trainer will exit and mark the run stopped.");
    } catch (e) {
      toast(`Stop failed: ${(e as Error).message}`, true);
    }
  };

  const latest = [...points].reverse().find((p) => p.kind === "train");
  const progress =
    latest && run?.total_steps ? Math.min(100, (latest.step / run.total_steps) * 100) : null;

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <Link to="/" style={{ color: "var(--muted)", fontWeight: 400 }}>
              Runs /
            </Link>
            {name}
            {run && <StateBadge state={run.state} />}
          </h1>
          {run && (
            <div className="sub">
              {run.mode} · {shortModel(run.student)}{" "}
              <span>← {shortModel(run.teacher)}</span>
              {run.started_at && ` · started ${run.started_at}`}
            </div>
          )}
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {run && run.state !== "running" && run.state !== "queued" && (
            <button
              className="btn"
              disabled={!!reportJobId}
              onClick={() => setShowReportForm((v) => !v)}
            >
              {reportJobId ? "Building report…" : report ? "Rebuild report" : "Build report"}
            </button>
          )}
          <button className="btn" onClick={() => setShowConfig((v) => !v)}>
            Config
          </button>
          <button className="btn" onClick={() => setShowLogs((v) => !v)}>
            Logs
          </button>
          {run?.state === "running" ? (
            <button className="btn btn-danger" onClick={stopRun}>
              ■ Stop
            </button>
          ) : (
            run && (
              <button className="btn" onClick={() => setConfirmRestart(true)}>
                ↻ Run again
              </button>
            )
          )}
        </div>
      </div>

      <ConfirmDialog
        open={confirmRestart}
        title={`Re-run ${name}?`}
        confirmLabel="↻ Run again"
        onConfirm={restartRun}
        onCancel={() => setConfirmRestart(false)}
      >
        Training restarts from the saved recipe — a fresh start, not a resume. The existing
        checkpoint and metrics in <code>runs/{name}</code> will be overwritten.
      </ConfirmDialog>

      {run?.error && (
        <div className="card card-body mono" style={{ color: "var(--danger)", marginBottom: 16 }}>
          {run.error}
        </div>
      )}

      {showReportForm && run && (
        <BuildReportForm
          runName={name}
          defaultTeacher={run.teacher ?? ""}
          onSubmitted={(jobId) => {
            setReportJobId(jobId);
            setShowReportForm(false);
          }}
        />
      )}

      <div className="grid-4" style={{ marginBottom: 16 }}>
        <div className="card stat">
          <div className="label">Progress</div>
          <div className="value">
            {latest ? latest.step : (run?.steps_completed ?? "—")}
            <small>
              {" "}
              / {run?.total_steps ?? "—"}
              {progress !== null && ` (${progress.toFixed(0)}%)`}
            </small>
          </div>
        </div>
        <div className="card stat">
          <div className="label">Loss</div>
          <div className="value">{fmtNum(latest?.loss)}</div>
        </div>
        <div className="card stat">
          <div className="label">Retention</div>
          <div className="value">
            <Retention value={run?.quality_retention ?? null} />
          </div>
        </div>
        <div className="card stat">
          <div className="label">Elapsed</div>
          <div className="value">
            {latest?.elapsed_s !== undefined ? `${Math.round(latest.elapsed_s)}s` : "—"}
          </div>
        </div>
      </div>

      {showConfig && run?.config && (
        <div className="card" style={{ marginBottom: 16 }}>
          <div className="card-head">recipe (distill_config.json)</div>
          <pre className="logpane" style={{ margin: 0 }}>
            {JSON.stringify(run.config, null, 2)}
          </pre>
        </div>
      )}

      {showLogs && (
        <div className="card" style={{ marginBottom: 16 }}>
          <div className="card-head">train.log</div>
          <div className="logpane" ref={logRef}>
            {logLines.length ? logLines.join("\n") : "waiting for log output…"}
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-head">Training loss</div>
        <div className="card-body">
          <MetricChart
            points={points}
            series={[
              { key: "loss", label: "total", color: chartColors.loss },
              { key: "kd", label: "kd", color: chartColors.kd },
              { key: "ce", label: "ce", color: chartColors.ce },
            ]}
          />
        </div>
      </div>

      <div className="card">
        <div className="card-head">Learning rate</div>
        <div className="card-body">
          <MetricChart
            points={points}
            height={140}
            series={[{ key: "lr", label: "lr", color: chartColors.lr }]}
          />
        </div>
      </div>

      {run?.eval && (
        <div className="card">
          <div className="card-head">Final eval</div>
          <div className="card-body kv">
            {Object.entries(run.eval).map(([k, v]) => (
              <div key={k} style={{ display: "contents" }}>
                <dt>{k}</dt>
                <dd className="num">{String(v)}</dd>
              </div>
            ))}
          </div>
        </div>
      )}

      {report ? (
        <ReportCard report={report} />
      ) : (
        run &&
        run.state !== "running" &&
        run.state !== "queued" && (
          <div className="card card-body" style={{ color: "var(--muted)" }}>
            {reportJobId
              ? "Building the report card — it will appear here when the judge and benchmarks finish."
              : "No report card yet — click Build report above to judge this student and benchmark it against the teacher."}
          </div>
        )
      )}
    </div>
  );
}
