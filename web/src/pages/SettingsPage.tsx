import { useState } from "react";
import { api } from "../api";
import { ConfirmDialog, timeAgo, usePoll, useToast } from "../components";
import type { DbSettings } from "../types";

/** Database sync settings. The connection string is write-only: posted once,
 * stored server-side (0600 config file), and never sent back to the browser. */
export function SettingsPage() {
  const [settings, setSettings] = useState<DbSettings | null>(null);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const toast = useToast();

  const refresh = async (probe = false) => {
    try {
      setSettings(await api<DbSettings>(`/api/settings/db${probe ? "?probe=1" : ""}`));
    } catch {
      /* keep last known state */
    }
  };
  usePoll(() => refresh(), 5000);

  const save = async () => {
    if (!url.trim()) return;
    setBusy(true);
    try {
      const next = await api<DbSettings>("/api/settings/db", {
        method: "POST",
        body: JSON.stringify({ url: url.trim() }),
      });
      setSettings(next);
      setUrl("");
      toast("Database connected — schema is up to date, sync is on.");
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  };

  const syncNow = async () => {
    setBusy(true);
    try {
      const res = await api<{ synced: { runs: number; metrics: number; reports: number } }>(
        "/api/db/sync",
        { method: "POST" },
      );
      toast(
        `Synced ${res.synced.runs} runs (${res.synced.metrics} new metric rows, ${res.synced.reports} reports).`,
      );
      refresh(true);
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    setConfirmRemove(false);
    setBusy(true);
    try {
      setSettings(await api<DbSettings>("/api/settings/db", { method: "DELETE" }));
      toast("Database removed. Runs stay on disk; nothing was deleted remotely.");
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  };

  const s = settings;
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Settings</h1>
          <div className="sub">
            Local files stay the source of truth — a database is an optional mirror.
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          Database sync
          {s?.configured && (
            <span className="badge ok" style={{ marginLeft: 10 }}>
              <span className="dot" />
              connected
            </span>
          )}
        </div>
        <div className="card-body">
          <p style={{ color: "var(--muted)", marginTop: 0 }}>
            Mirror run history, metrics, and report cards to any Postgres database (works great
            with a free <span className="mono">Neon</span> project). History then survives
            ephemeral machines — a Colab VM, a rented GPU box — and every machine syncing to the
            same database shows up under <strong>Synced history</strong> on the Runs page. The
            schema is created and migrated automatically; you never run SQL.
          </p>

          {s?.configured && s.server && (
            <div className="grid-3" style={{ margin: "12px 0" }}>
              <div className="stat">
                <div className="label">Server</div>
                <div className="value" style={{ fontSize: 15 }}>
                  {s.server.host ?? "—"}
                  <small> / {s.server.dbname ?? "—"}</small>
                </div>
              </div>
              <div className="stat">
                <div className="label">Syncing as host</div>
                <div className="value" style={{ fontSize: 15 }}>
                  {s.host_label}
                  <small> ({s.source === "env" ? "env var" : "saved config"})</small>
                </div>
              </div>
              <div className="stat">
                <div className="label">Last sync</div>
                <div className="value" style={{ fontSize: 15 }}>
                  {timeAgo(s.last_sync)}
                  {s.db && <small> — {s.db.runs} runs, {s.db.metrics} metric rows</small>}
                </div>
              </div>
            </div>
          )}

          {s?.last_error && (
            <div className="card-body" style={{ color: "var(--danger)", padding: "8px 0" }}>
              {s.last_error}
            </div>
          )}

          {s?.env_locked ? (
            <p style={{ color: "var(--muted)" }}>
              Configured via the <span className="mono">DISTILL_DB_URL</span> environment
              variable — manage it from your shell, not here.
            </p>
          ) : (
            <>
              <div className="field" style={{ maxWidth: 640 }}>
                <label>
                  {s?.configured ? "Replace connection string" : "Postgres connection string"}
                </label>
                <input
                  className="input mono"
                  type="password"
                  autoComplete="off"
                  placeholder="postgresql://user:password@ep-....neon.tech/neondb?sslmode=require"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && !busy && save()}
                />
                <div className="hint">
                  Stored on the machine running <span className="mono">distill ui</span> with
                  owner-only permissions; never returned by the API. Prefer the{" "}
                  <span className="mono">DISTILL_DB_URL</span> env var on shared machines.
                </div>
              </div>
              <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
                <button className="btn btn-primary" disabled={busy || !url.trim()} onClick={save}>
                  {s?.configured ? "Save & reconnect" : "Connect & enable sync"}
                </button>
                {s?.configured && (
                  <>
                    <button className="btn" disabled={busy} onClick={syncNow}>
                      Sync now
                    </button>
                    <button
                      className="btn btn-danger"
                      disabled={busy}
                      onClick={() => setConfirmRemove(true)}
                    >
                      Remove
                    </button>
                  </>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      <ConfirmDialog
        open={confirmRemove}
        title="Remove database connection?"
        confirmLabel="Remove"
        tone="danger"
        onConfirm={remove}
        onCancel={() => setConfirmRemove(false)}
      >
        Sync stops and the saved connection string is deleted from this machine. Nothing is
        deleted from the database — reconnect any time and history picks up where it left off.
      </ConfirmDialog>
    </div>
  );
}
