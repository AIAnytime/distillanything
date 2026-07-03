# Dashboard frontend (contributors only)

React + TypeScript + Vite source for the `distill ui` dashboard. **Users never need
Node** — the built bundle is committed at `distillanything/ui/static/` and ships in
the wheel.

## Develop

```bash
# terminal 1: backend (token printed; the dev server proxies /api to it)
distill ui --no-browser

# terminal 2: frontend with hot reload
cd web
npm install
npm run dev          # http://localhost:5173/?token=<token printed by distill ui>
```

## Build (do this before committing frontend changes)

```bash
cd web
npm run build        # type-checks, then emits to ../distillanything/ui/static/
```

Commit the regenerated `distillanything/ui/static/` together with your `web/src`
changes — CI has no Node build step by design.

## Ground rules

- No new runtime dependencies without a strong reason: the whole app is React,
  react-router, and uPlot (~85KB gzipped total). No UI kits.
- Everything must work in dark AND light (`data-theme` on `<html>`, tokens in
  `src/theme.css`).
- The API token lives in sessionStorage only, never localStorage or the URL after
  boot (`src/api.ts` strips it).
- CSP is `default-src 'self'` — no external fonts, CDNs, or inline scripts.
