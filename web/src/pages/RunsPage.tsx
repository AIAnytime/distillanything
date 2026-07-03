import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import {
  EmptyState,
  Retention,
  SkeletonRows,
  StateBadge,
  fmtNum,
  shortModel,
  timeAgo,
  usePoll,
} from "../components";
import type { RunSummary } from "../types";

export function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  usePoll(async () => {
    try {
      setRuns(await api<RunSummary[]>("/api/runs"));
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, 3000);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Runs</h1>
          <div className="sub">
            Every distillation in your runs directory — including ones started from the CLI.
          </div>
        </div>
      </div>

      {error && (
        <div className="card card-body" style={{ color: "var(--danger)" }}>
          {error === "missing or invalid token"
            ? "Not authenticated. Open the dashboard through the URL printed by `distill ui` (it carries your session token)."
            : `Failed to load runs: ${error}`}
        </div>
      )}

      <div className="card">
        <table className="table">
          <thead>
            <tr>
              <th>Run</th>
              <th>Status</th>
              <th>Mode</th>
              <th>Student ← Teacher</th>
              <th className="right">Progress</th>
              <th className="right">Loss</th>
              <th className="right">Retention</th>
              <th className="right">Updated</th>
            </tr>
          </thead>
          <tbody>
            {runs === null && !error && <SkeletonRows cols={8} />}
            {runs?.map((run) => (
              <tr
                key={run.name}
                className="rowlink"
                tabIndex={0}
                onClick={() => navigate(`/runs/${run.name}`)}
                onKeyDown={(e) => e.key === "Enter" && navigate(`/runs/${run.name}`)}
              >
                <td>
                  <strong>{run.name}</strong>
                </td>
                <td>
                  <StateBadge state={run.state} />
                </td>
                <td className="mono">{run.mode ?? "—"}</td>
                <td>
                  {shortModel(run.student)}{" "}
                  <span style={{ color: "var(--muted)" }}>← {shortModel(run.teacher)}</span>
                </td>
                <td className="right num">
                  {run.last_metric && run.total_steps
                    ? `${run.last_metric.step}/${run.total_steps}`
                    : run.steps_completed && run.total_steps
                      ? `${run.steps_completed}/${run.total_steps}`
                      : "—"}
                </td>
                <td className="right num">{fmtNum(run.last_metric?.loss)}</td>
                <td className="right">
                  <Retention value={run.quality_retention} />
                </td>
                <td className="right" style={{ color: "var(--muted)" }}>
                  {timeAgo(run.updated_at)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {runs !== null && runs.length === 0 && (
          <EmptyState
            title="No runs yet"
            command="distill train recipes/mac-small.yaml"
          >
            Start one from the CLI, or click <strong>+ New run</strong> above — either way it
            shows up here live.
          </EmptyState>
        )}
      </div>
    </div>
  );
}
