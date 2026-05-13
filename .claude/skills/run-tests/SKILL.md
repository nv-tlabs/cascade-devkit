---
name: run-tests
description: Run the pytest suite via `make test`, interpret the output, and diagnose common failure modes (missing CASCADE_AV_DATASET_ROOT, ffmpeg missing, ruff not installed, workspace not synced).
---

# Run the test suite

## Procedure

1. From the repo root, run:
   ```bash
   make test
   ```
   This expands to `uv run pytest` and auto-discovers tests in both
   `tests/` (DevKit) and `tools/annotator/tests/` (annotator subproject).
2. A clean run prints `N passed in M seconds` and exits 0. The
   current passing baseline as of 2026-05-12 is **451 tests** (29 of
   which are annotator tests).
3. Failures fall into recognizable buckets — see below.

## Common failure modes

- **`uv: command not found`** — `uv` isn't installed. Install per
  <https://docs.astral.sh/uv/>, then `make install` and re-run.
- **`ModuleNotFoundError: physical_ai_av`** — the `[hf]` extra isn't
  installed but a test needs it. Run `make install` (which uses
  `uv sync --all-extras`).
- **`ffmpeg: not found` raised from inside a test** — should never
  happen. The video tests mock `subprocess.run` and `shutil.which`
  so real `ffmpeg` is never invoked. If you see this, a recent
  change broke the mocking pattern. See
  `tools/annotator/AGENTS.md` § Server-side rules.
- **`CASCADE_AV_DATASET_ROOT not set`** — only matters for tests that
  iterate the real corpus. Unit tests skip these. Set with
  `export CASCADE_AV_DATASET_ROOT=/path/to/json_annotations` before
  invoking `make test`.
- **`UserWarning: N clip_id(s) had multiple annotation files`** —
  expected on the current corpus; not a failure. Suppress in
  output by setting `CASCADE_AV_VERBOSE=0`.

## When debugging a failing test

Run with fail-fast + failed-first re-run:

```bash
uv run pytest -x --ff
```

Capture the traceback. Fix the offending code, NOT the test, unless
the test encodes a pre-change assumption that the change deliberately
invalidates — in which case update the test in the same commit and
explain why in the commit body.
