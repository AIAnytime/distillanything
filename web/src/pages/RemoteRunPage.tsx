import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import { Retention, StateBadge, fmtNum, shortModel, timeAgo, usePoll } from "../components";
import { MetricChart, chartColors } from "../MetricChart";
import type { MetricPoint, RemoteRunDetail } from "../types";

/** Read-only view of a run mirrored in the database — possibly from another
 * machine (the Colab VM is long gone; its curves are not). */
export function RemoteRunPage() {
  const { host = "", name = "" } = useParams();
  const [run, setRun] = useState<RemoteRunDetail | null>(null);
  const [points, setPoints] = useState<MetricPoint[]>([]);
  const [error, setError] = useState<string | null>(null);

  usePoll(
    async () => {
      try {
        const [detail, metrics] = await Promise.all([
          api<RemoteRunDetail>(`/api/remote/runs/${host}/${name}`),
          api<MetricPoint[]>(`/api/remote/runs/${host}/${name}/metrics`),
        ]);
        setRun(detail);
        setPoints(metrics);
        setError(null);
      } catch (e) {
        setError((e as Error).message);
      }
    },
    10000,
    [host, name],
  );

  const lastEval = [...points].reverse().find((p) => p.kind === "eval");
  const judge = run?.report?.judge;

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>
            {name} <span className="badge">{host}</span>{" "}
            {run && <StateBadge state={run.state} />}
          </h1>
          <div className="sub">
            Synced history (read-only) — {shortModel(run?.student ?? null)}{" "}
            <span style={{ color: "var(--muted)" }}>← {shortModel(run?.teacher ?? null)}</span>
            {" · "}last synced {timeAgo(run?.synced_at ?? null)} ·{" "}
            <Link to="/">back to runs</Link>
          </div>
        </div>
      </div>

      {error && (
        <div className="card card-body" style={{ color: "var(--danger)" }}>
          {error}
        </div>
      )}

      <div className="grid-4">
        <div className="stat">
          <div className="label">Progress</div>
          <div className="value num">
            {run?.last_metric && run?.total_steps
              ? `${run.last_metric.step}/${run.total_steps}`
              : "—"}
          </div>
        </div>
        <div className="stat">
          <div className="label">Last loss</div>
          <div className="value num">{fmtNum(run?.last_metric?.loss)}</div>
        </div>
        <div className="stat">
          <div className="label">Perplexity</div>
          <div className="value num">{fmtNum(lastEval?.perplexity, 3)}</div>
        </div>
        <div className="stat">
          <div className="label">Quality retention</div>
          <div className="value">
            <Retention value={run?.quality_retention ?? null} />
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-head">Loss</div>
        <div className="card-body">
          <MetricChart
            points={points}
            series={[{ key: "loss", label: "loss", color: chartColors.loss }]}
          />
        </div>
      </div>

      <div className="card">
        <div className="card-head">Components</div>
        <div className="card-body">
          <MetricChart
            points={points}
            series={[
              { key: "kd", label: "kd", color: chartColors.kd },
              { key: "ce", label: "ce", color: chartColors.ce },
              { key: "hid", label: "hid", color: "#ff9bce" },
            ]}
          />
        </div>
      </div>

      {judge && (
        <div className="card">
          <div className="card-head">Report card</div>
          <div className="card-body">
            <div className="grid-4">
              <div className="stat">
                <div className="label">Retention</div>
                <div className="value">
                  <Retention value={judge.quality_retention} />
                </div>
              </div>
              <div className="stat">
                <div className="label">Wins</div>
                <div className="value num">{judge.student_wins}</div>
              </div>
              <div className="stat">
                <div className="label">Ties</div>
                <div className="value num">{judge.ties}</div>
              </div>
              <div className="stat">
                <div className="label">Losses</div>
                <div className="value num">{judge.teacher_wins}</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
