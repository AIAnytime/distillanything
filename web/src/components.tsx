import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

// ---------- formatting ----------

export function fmtNum(x: number | null | undefined, digits = 4): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return x.toFixed(digits);
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 ** 2).toFixed(1)} MB`;
}

export function timeAgo(epochSeconds: number | null): string {
  if (!epochSeconds) return "—";
  const s = Math.max(0, Date.now() / 1000 - epochSeconds);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function shortModel(name: string | null): string {
  if (!name) return "—";
  return name.replace(/^hf:/, "").split("/").pop() ?? name;
}

// ---------- status ----------

const STATE_CLASS: Record<string, string> = {
  running: "live",
  completed: "ok",
  failed: "bad",
  stopped: "warn",
  interrupted: "warn",
  queued: "warn",
};

export function StateBadge({ state }: { state: string }) {
  const cls = STATE_CLASS[state] ?? "";
  return (
    <span className={`badge ${cls}`}>
      <span className={`dot ${state === "running" ? "pulse" : ""}`} />
      {state}
    </span>
  );
}

export function Retention({ value }: { value: number | null }) {
  if (value === null || value === undefined) return <span className="num">—</span>;
  const pct = value * 100;
  const cls = pct >= 80 ? "ok" : pct >= 50 ? "warn" : "bad";
  return <span className={`badge ${cls} num`}>{pct.toFixed(1)}%</span>;
}

// ---------- skeletons & empty states ----------

export function SkeletonRows({ cols, rows = 4 }: { cols: number; rows?: number }) {
  return (
    <>
      {Array.from({ length: rows }, (_, r) => (
        <tr key={r}>
          {Array.from({ length: cols }, (_, c) => (
            <td key={c}>
              <div className="skeleton" style={{ width: `${45 + ((r * 7 + c * 13) % 40)}%` }} />
            </td>
          ))}
        </tr>
      ))}
    </>
  );
}

export function EmptyState({
  title,
  children,
  command,
}: {
  title: string;
  children?: ReactNode;
  command?: string;
}) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      {children}
      {command && <code>{command}</code>}
    </div>
  );
}

// ---------- toasts ----------

interface Toast {
  id: number;
  text: string;
  error?: boolean;
}

const ToastContext = createContext<(text: string, error?: boolean) => void>(() => {});

export function useToast() {
  return useContext(ToastContext);
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);
  const push = useCallback((text: string, error = false) => {
    const id = nextId.current++;
    setToasts((t) => [...t, { id, text, error }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5000);
  }, []);
  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="toasts">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.error ? "error" : ""}`}>
            {t.text}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

// ---------- confirm dialog ----------

export function ConfirmDialog({
  open,
  title,
  confirmLabel,
  tone = "warn",
  onConfirm,
  onCancel,
  children,
}: {
  open: boolean;
  title: string;
  confirmLabel: string;
  tone?: "warn" | "danger";
  onConfirm: () => void;
  onCancel: () => void;
  children: ReactNode;
}) {
  const confirmRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    confirmRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onCancel();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onCancel]);

  if (!open) return null;
  return (
    <div className="modal-overlay" onMouseDown={(e) => e.target === e.currentTarget && onCancel()}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={title}>
        <div className="modal-head">{title}</div>
        <div className="modal-body">{children}</div>
        <div className="modal-actions">
          <button className="btn" onClick={onCancel}>
            Cancel
          </button>
          <button
            ref={confirmRef}
            className={`btn ${tone === "danger" ? "btn-danger" : "btn-warn"}`}
            onClick={onConfirm}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------- polling ----------

export function usePoll(fn: () => void | Promise<void>, ms: number, deps: unknown[] = []) {
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      if (alive) await fn();
    };
    tick();
    const id = setInterval(tick, ms);
    return () => {
      alive = false;
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}
