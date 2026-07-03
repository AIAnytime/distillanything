import { useEffect, useState } from "react";
import { api } from "../api";
import { EmptyState, Retention, StateBadge, fmtNum, shortModel } from "../components";
import { MetricChart, chartColors } from "../MetricChart";
import type { MetricPoint, RunSummary } from "../types";

export function ComparePage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<Record<string, MetricPoint[]>>({});

  useEffect(() => {
    api<RunSummary[]>("/api/runs").then(setRuns);
  }, []);

  useEffect(() => {
    selected.forEach((name) => {
      if (!metrics[name]) {
        api<MetricPoint[]>(`/api/runs/${name}/metrics`).then((m) =>
          setMetrics((prev) => ({ ...prev, [name]: m })),
        );
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]);

  const toggle = (name: string) =>
    setSelected((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name].slice(-6),
    );

  const chosen = selected.filter((n) => metrics[n]);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Compare runs</h1>
          <div className="sub">Overlay loss curves and line up the report numbers side by side.</div>
        </div>
      </div>

      <div className="card">
        <table className="table">
          <thead>
            <tr>
              <th style={{ width: 36 }}></th>
              <th>Run</th>
              <th>Status</th>
              <th>Student ← Teacher</th>
              <th className="right">Final loss</th>
              <th className="right">Retention</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.name} className="rowlink" onClick={() => toggle(run.name)}>
                <td>
                  <input
                    type="checkbox"
                    checked={selected.includes(run.name)}
                    onChange={() => toggle(run.name)}
                    onClick={(e) => e.stopPropagation()}
                    aria-label={`Select ${run.name}`}
                  />
                </td>
                <td>
                  <strong>{run.name}</strong>
                </td>
                <td>
                  <StateBadge state={run.state} />
                </td>
                <td>
                  {shortModel(run.student)}{" "}
                  <span style={{ color: "var(--muted)" }}>← {shortModel(run.teacher)}</span>
                </td>
                <td className="right num">{fmtNum(run.last_metric?.loss)}</td>
                <td className="right">
                  <Retention value={run.quality_retention} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {runs.length === 0 && (
          <EmptyState title="Nothing to compare yet" command="distill train recipes/mac-small.yaml" />
        )}
      </div>

      {chosen.length >= 1 && (
        <div className="card">
          <div className="card-head">Training loss — {chosen.join(" vs ")}</div>
          <div className="card-body">
            <MetricChart
              series={chosen.map((name, i) => ({
                key: "loss",
                label: name,
                color: chartColors.palette[i % chartColors.palette.length],
              }))}
              runsData={chosen.map((name) => ({ points: metrics[name] }))}
              height={280}
            />
          </div>
        </div>
      )}
    </div>
  );
}
