import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "./api";
import { StateBadge, usePoll, useToast } from "./components";
import type { Job } from "./types";

export function JobsDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const toast = useToast();

  usePoll(
    async () => {
      if (!open) return;
      try {
        setJobs(await api<Job[]>("/api/jobs"));
      } catch {
        /* server briefly unavailable; next poll retries */
      }
    },
    2500,
    [open],
  );

  if (!open) return null;

  const stop = async (job: Job) => {
    try {
      await api(`/api/jobs/${job.id}/stop`, { method: "POST" });
      toast(`Stopping ${job.kind} job…`);
    } catch (e) {
      toast(`Stop failed: ${(e as Error).message}`, true);
    }
  };

  return (
    <div className="drawer">
      <div className="card-head">
        Jobs
        <button className="icon-btn" onClick={onClose} aria-label="Close jobs drawer">
          ✕
        </button>
      </div>
      <div style={{ padding: 12 }}>
        {jobs === null && <div className="skeleton" style={{ height: 40 }} />}
        {jobs !== null && jobs.length === 0 && (
          <div className="empty" style={{ padding: "32px 8px" }}>
            <h3>No jobs this session</h3>
            Jobs started from the dashboard appear here.
          </div>
        )}
        {jobs?.map((job) => (
          <div
            key={job.id}
            className="card"
            style={{ marginBottom: 10, padding: "10px 12px", display: "grid", gap: 6 }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <strong>
                {job.kind}
                {job.run_name && (
                  <>
                    {" · "}
                    <Link to={`/runs/${job.run_name}`} onClick={onClose}>
                      {job.run_name}
                    </Link>
                  </>
                )}
              </strong>
              <StateBadge state={job.status} />
            </div>
            <div className="mono" style={{ color: "var(--muted)" }}>
              {job.id} · {job.started_at ?? job.created_at}
              {job.returncode !== null && ` · exit ${job.returncode}`}
            </div>
            {(job.status === "running" || job.status === "queued") && (
              <div>
                <button className="btn btn-danger btn-sm" onClick={() => stop(job)}>
                  Stop
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
