# Contributing to Distill Anything

Thanks for helping build the open-source distillation lifecycle. PRs of all sizes welcome.

## Dev setup

```bash
git clone https://github.com/AIAnytime/distillanything && cd distillanything
uv venv && uv pip install -e ".[dev]"
```

## Before you open a PR

```bash
ruff check distillanything tests   # lint
pytest                             # full suite — offline, no GPU, no API keys, no downloads
distill smoke                      # end-to-end pipeline check on your hardware
```

The test suite is deliberately hermetic: tiny random models from `distillanything/testing.py`
and fake judges stand in for real models and APIs. **Keep it that way** — a test that needs
a network connection, an API key, or a model download will not be merged. If your feature
touches an API teacher, test the logic around the call with a `Teacher` fake (see
`tests/test_judge.py` for examples).

## What makes a good contribution

- **Roadmap items** (see README) are pre-approved directions — open an issue to claim one.
- **New teachers**: subclass `Teacher`, register a spec prefix in `teachers/registry.py`,
  keep SDK imports lazy so the core install stays light.
- **New losses**: pure functions in `losses/`, operating on `[B, S, V]` logits with the
  `-100` label-mask convention, plus property tests (non-negativity, zero-at-identity,
  masking, backprop) like `tests/test_losses.py`.
- **Recipes**: sized so they run on a 16GB laptop, or clearly labeled otherwise.

## Style

- `ruff` (line length 110) — CI enforces it.
- Comments explain constraints the code can't show, not what the next line does.
- User-facing strings and errors should say what to do next, not just what went wrong.

## Releases

Maintainers cut releases from `main`. Version lives in `pyproject.toml` and
`distillanything/__init__.py`.
