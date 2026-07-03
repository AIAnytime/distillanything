// API client. The server prints a one-time URL with ?token=...; we capture it,
// keep it in sessionStorage only (never localStorage — gone when the tab closes),
// and strip it from the address bar so it doesn't live in history/screenshots.

const TOKEN_KEY = "da_token";

function captureToken(): void {
  const params = new URLSearchParams(window.location.search);
  const token = params.get("token");
  if (token) {
    sessionStorage.setItem(TOKEN_KEY, token);
    params.delete("token");
    const query = params.toString();
    window.history.replaceState(
      {},
      "",
      window.location.pathname + (query ? `?${query}` : "") + window.location.hash,
    );
  }
}
captureToken();

export function getToken(): string {
  return sessionStorage.getItem(TOKEN_KEY) ?? "";
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      Authorization: `Bearer ${getToken()}`,
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

// EventSource can't set headers, so streams authenticate via query token —
// the server accepts ?token= for exactly this reason.
export function sse(path: string, onLine: (data: string) => void, onDone?: () => void): () => void {
  const sep = path.includes("?") ? "&" : "?";
  const source = new EventSource(`${path}${sep}token=${encodeURIComponent(getToken())}`);
  source.onmessage = (event) => onLine(event.data);
  source.addEventListener("done", () => {
    source.close();
    onDone?.();
  });
  source.onerror = () => {
    /* EventSource retries automatically; server closes with `done` when over */
  };
  return () => source.close();
}
