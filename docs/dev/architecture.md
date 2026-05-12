# Architecture

Design decisions, system shape, and rationale for `av-causal-dataset-tools`.

> Read this before proposing or implementing any non-trivial structural change.
> Add a new section when a decision spans multiple files or locks in a trade-off
> that future contributors would otherwise re-litigate.

## Overview

_TBD — high-level picture of what this repo does and how the pieces fit._

## Components

One subsection per top-level directory. Describe its role, its boundaries, and
how it talks to its neighbours.

### `src/`

_TBD_

### `tools/`

User-facing utilities layered on the DevKit. Each subdirectory is a
self-contained tool with its own README and (when applicable) its own
`pyproject.toml`, `package.json`, and tests. Currently:

- `tools/annotator/` — local FastAPI + React annotation tool, a uv
  workspace member exposing the `causal-av-annotate` console script.

The DevKit (`src/causal_ai_av/`) does not depend on anything in
`tools/`; the arrows point inward.

### `scripts/`

_TBD_

### `notebooks/`

_TBD_

### `tests/`

_TBD_

### `docs/`

`docs/dev/` is the project's living memory for contributors (this file,
[`contrib.md`](./contrib.md), [`changelog.md`](./changelog.md)). Reference
material for end users lives directly under `docs/` (e.g. `query_language.md`).

## Decisions

Record significant choices here as short ADR-style entries. Newest first.

### 2026-05-12 — Tools live under `tools/`

- **Context:** The annotator (a FastAPI + React app for editing
  annotation JSONs) was the first tool layered on the DevKit, and
  more user-facing utilities are anticipated (CLIs, viz helpers,
  data exporters). Without a parent grouping, every new tool would
  add another top-level directory and blur the line between the
  library (`src/`, `tests/`) and consumers of it.
- **Decision:** All user-facing utilities live under `tools/`. Each
  is a self-contained subdirectory that may or may not be a uv
  workspace member, may bring its own JS frontend, and owns its own
  tests.
- **Alternatives considered:**
  - Keep `annotator/` at the top level until a second tool appears,
    then rename. Rejected — touching the doc surface (AGENTS.md,
    README, Makefile, pyproject workspace) twice is more churn than
    moving once now while the repo is young.
  - Place tools under `src/causal_ai_av/tools/` as Python sub-packages.
    Rejected — `tools/annotator/` is a separate uv workspace member
    with its own `pyproject.toml` and JS frontend; bundling it inside
    the library package would conflate library code with tools.
- **Consequences:** `tools/` is the entry point for new utilities.
  Library code in `src/causal_ai_av/` never imports from `tools/`;
  tools may depend on the library via the workspace.

### YYYY-MM-DD — &lt;decision title&gt;

- **Context:** what forced the decision.
- **Decision:** what we chose.
- **Alternatives considered:** what we rejected and why.
- **Consequences:** what this locks in or rules out.
