"""The dashboard's security spine.

Threat model: a single-user tool binding to localhost by default, whose API can
spawn processes — so it gets the Jupyter treatment, not the "it's only local"
treatment:

- A bearer token is ALWAYS required on /api (any local browser tab can reach
  127.0.0.1; token auth is what stands between a malicious web page and your
  process-spawning API).
- Host-header allowlist on loopback binds defeats DNS-rebinding attacks.
- Conservative security headers; request bodies are capped.
- API keys for teachers live in the server's environment only; no endpoint
  accepts, stores, or returns them.
"""

from __future__ import annotations

import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

MAX_BODY_BYTES = 1_000_000

_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


def generate_token() -> str:
    return secrets.token_urlsafe(32)


class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, token: str, enforce_host_allowlist: bool = True):
        super().__init__(app)
        if not token:
            raise ValueError("SecurityMiddleware requires a non-empty token")
        self._token = token
        self._enforce_host = enforce_host_allowlist

    def _authorized(self, request: Request) -> bool:
        header = request.headers.get("authorization", "")
        if header.startswith("Bearer "):
            return secrets.compare_digest(header[7:], self._token)
        # EventSource cannot set headers; SSE endpoints authenticate via query.
        query_token = request.query_params.get("token", "")
        return bool(query_token) and secrets.compare_digest(query_token, self._token)

    async def dispatch(self, request: Request, call_next):
        host = request.headers.get("host", "").rsplit(":", 1)[0].lower()
        if self._enforce_host and host not in LOOPBACK_HOSTS:
            return JSONResponse({"detail": "host not allowed"}, status_code=403)

        if request.url.path.startswith("/api") and request.url.path != "/api/health":
            length = request.headers.get("content-length")
            if length and int(length) > MAX_BODY_BYTES:
                return JSONResponse({"detail": "request too large"}, status_code=413)
            if not self._authorized(request):
                return JSONResponse({"detail": "missing or invalid token"}, status_code=401)

        response = await call_next(request)
        for key, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        return response
