# Architecture

Design decisions, system shape, and rationale for `cascade-devkit`.

> Read this before proposing or implementing any non-trivial structural change.
> Add a new section when a decision spans multiple files or locks in a trade-off
> that future contributors would otherwise re-litigate.

## Overview

`cascade_av` is a Python DevKit plus a local annotation tool, layered on
NVIDIA's *Physical AI AV Dataset*. The library reads per-clip annotation
JSONs into a typed Pydantic tree (current schema is
[2.0.0](./schema-history.md)), joins them to the parent dataset's
egomotion and video, and exposes a small DSL for querying the corpus by
entity, attribute, time, and cause. A visualization surface turns matches
into static figures or interactive widgets in notebooks. Schema-extension
data (e.g. future image-space geometry) lives in a sibling
`<stem>.extra.json` sidecar managed by `cascade_av.extensions`.

The data flow, in one sentence: **on-disk JSON → `spec` Pydantic models →
`io` loaders → `CascadeDataset` / `Sequence` → `query` DSL / engine →
`MatchSet` → `viz` figures or widgets.**

Two boundaries matter:

- **`src/cascade_av/` is the library.** It never imports from `tools/`.
- **`tools/annotator/` is a separate uv workspace member** with its own
  Python entry point (`cascade-annotate`) and a Vite + React frontend.
  It depends on the DevKit's schema; the DevKit does not depend on it.

The visualization layer is gated behind the optional `[viz]` extra so the
library stays importable in headless environments without Plotly /
ipywidgets installed. Likewise, the HuggingFace-backed data loaders live
behind the `[hf]` extra so the top-level package imports cleanly when
`physical_ai_av` is not installed.

## Components

One subsection per top-level directory. Describes its role, its boundaries,
and how it talks to its neighbours.

### `src/`

Single Python package: `cascade_av`. Sub-packages mirror the data flow.

- **`spec/`** — Pydantic v2 models for the annotation format. The current
  schema version is [2.0.0](./schema-history.md); the registry of all
  known versions lives in `cascade_av.spec.versions`. `AnnotationBundle` is
  the single entry point; every other model is reachable from it. Two
  intentional invariants:
  - `model_config = ConfigDict(extra="allow")` on every model, so the
    annotator (or any other producer) can add fields without breaking
    reads.
  - Vocabulary-typed fields (action types, agent types, light colors,
    etc.) are declared as plain `str`, not `Enum`. The on-disk corpus is
    ground truth; unknown values must round-trip rather than fail
    validation. Advisory `*Vocab` namespace classes near the bottom of
    `schema.py` enumerate the known values for autocomplete and queries.
- **`io/`** — load/save annotation files. `io/local.py` handles a
  directory of `*.json` on disk (atomic write via `os.replace`,
  filename parser, per-clip grouping). `io/hf.py` loads from a
  HuggingFace dataset repo and is gated behind the `[hf]` extra.
- **`dataset.py`** — `CascadeDataset` (a thin subclass of
  `physical_ai_av.PhysicalAIAVDatasetInterface`) and `Sequence` (one
  clip's bundle joined with egomotion + video features). If the
  `[hf]` extra is missing, `CascadeDataset` and `Sequence` import as
  `None` at the package root so non-corpus code paths still work.
- **`query/`** — the DSL and evaluator. Layered:
  - `time.py` — `Interval` and timestamp parsing.
  - `index.py` — `IdIndex`, the by-ID lookup of every addressable
    entity in a bundle (agents, actions, conditions, signal heads, …).
  - `temporal.py` / `spatial.py` — point-in-time and windowed predicates
    over agents, actions, lights, ego pose.
  - `triplets.py` — extracts `(subject, predicate, cause)` causal
    triplets from `because_of` references.
  - `constants.py` — DSL value aliases, grounded against the corpus by
    `scripts/scan_corpus_vocabulary.py`.
  - `entities.py` — DSL entity descriptors.
  - `dsl.py` — lexer + parser → AST (`And` / `Or` / `Not` /
    `EntityClause` / `AttrPredicate` / `BecauseOf` / `Within` /
    `Then` / `While`).
  - `engine.py` — AST evaluator → `MatchSet` (`Match` rows hold the
    bound subjects, the time window, and a weakref to the source
    dataset for re-hydration).
  - `context.py` — point-in-time `ContextWindow` (agents visible,
    light states) bundled around a match.
  - `api.py` — thin wrappers: `find_on_bundle`, `find_on_dataset`,
    `count_on_dataset`, `group_by_on_dataset`.
- **`viz/`** — the visualization surface, gated behind the optional
  `[viz]` extra. Contains:
  - `segments.py` — `Segment` + `annotation_to_segments`, the Python
    port of the annotator's `TimelineSegment[]` shape (so the DevKit
    and the annotator render the same thing).
  - `colors.py` — `entity_color` / `family_color`, the annotator's
    palette as hex strings.
  - `render.py` — `render_frame` (headless single-frame decode →
    `PIL.Image`) and `render_timeline` (static Plotly figure).
  - `widget.py` — `ClipPlayer`, an interactive Plotly + ipywidgets
    scrubber.
  - `carousel.py` — `build_matchset_carousel`, fans a `MatchSet` out
    into one `ClipPlayer` per match. Surfaced as `MatchSet.visualize()`.
- **`state.py`** — small utilities for point-in-time state extraction
  shared by `dataset` and `query`.

### `tools/`

User-facing utilities layered on the DevKit. Each subdirectory is a
self-contained tool with its own README and (when applicable) its own
`pyproject.toml`, `package.json`, and tests. Currently:

- `tools/annotator/` — local FastAPI + React annotation tool, a uv
  workspace member exposing the `cascade-annotate` console script.
  Its Python backend depends on `cascade_av.spec` for schema
  validation and on `cascade_av.io` for atomic file writes; the
  React frontend is its own Vite + React 19 + Tailwind v4 app under
  `tools/annotator/web/`.

The DevKit (`src/cascade_av/`) does not depend on anything in
`tools/`; the arrows point inward.

### `scripts/`

One-off Python utilities. Not packaged, not imported by the library —
just `uv run python scripts/<name>.py`. Currently:

- `build_notebooks.py` — generates the `notebooks/*.ipynb` files from
  Python sources. Run after edits to any notebook source. Use the
  `notebooks` dependency group: `uv run --group notebooks python
  scripts/build_notebooks.py`.
- `scan_corpus_vocabulary.py` — scans the corpus and prints every
  distinct value for query-relevant fields. Drives the alias tables in
  `src/cascade_av/query/constants.py`. Re-run when the corpus
  vocabulary changes.

### `notebooks/`

Six numbered notebooks (`01_quickstart` through `06_visualize`) that
double as runnable tutorials and end-to-end examples. They are
**generated artifacts**: edit the Python source in
`scripts/build_notebooks.py`, not the `.ipynb` directly. Every notebook
reads the corpus from `CASCADE_AV_DATASET_ROOT`; without it, only the
schema-only cells run.

### `tests/`

Pytest suite at the repo root, sibling to `src/`. Layout:

- `conftest.py` — shared fixtures, including `FakeVideoReader` for
  viz tests that need a stand-in for the parent dataset's video
  reader.
- `test_schema_against_corpus.py` — validates every JSON in the live
  corpus against the schema. The corpus is ground truth: if a file
  fails, the schema is wrong, not the file.
- `test_io.py` — local-disk load/save round-trips, including the
  atomic-write + `.bak` invariant.
- `test_dataset.py` — `CascadeDataset` / `Sequence` against the
  corpus, including egomotion-missing fallback.
- `test_query*.py` (`test_query`, `test_query_dsl`,
  `test_query_context`) — DSL parse, evaluate, entity-clause coupling,
  boolean composition, `because_of`, dataset-level aggregation.
- `test_sequence_visualize.py`, `test_viz_*.py` (carousel, segments,
  timeline, widget) — the visualization surface.
- `test_download_clips.py` — the HF download path; gated by the
  `[hf]` extra.

Tests that need the corpus expect `CASCADE_AV_DATASET_ROOT`. Unit tests
do not require it; they use fixtures from `conftest.py`. The annotator
also has its own test suite under `tools/annotator/tests/` (mocks
`subprocess.run` / `shutil.which`, never invokes real `ffmpeg`).

### `docs/`

`docs/dev/` is the project's living memory for contributors (this file,
[`changelog.md`](./changelog.md), [`schema-history.md`](./schema-history.md)).
The contributor workflow itself lives at the repo root in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md) so GitHub surfaces it on
issue / PR pages. User-facing reference material lives under
`docs/user/` (`query_language.md`, `visualization.md`, `annotator.md`).

## Decisions

Record significant choices here as short ADR-style entries. Newest first.
When adding a new entry, copy the [ADR template](#adr-template) at the
bottom of this file.

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
  - Place tools under `src/cascade_av/tools/` as Python sub-packages.
    Rejected — `tools/annotator/` is a separate uv workspace member
    with its own `pyproject.toml` and JS frontend; bundling it inside
    the library package would conflate library code with tools.
- **Consequences:** `tools/` is the entry point for new utilities.
  Library code in `src/cascade_av/` never imports from `tools/`;
  tools may depend on the library via the workspace.

## ADR template

Copy this template when adding a new entry above. The fenced block keeps
it from rendering as an empty decision.

```markdown
### YYYY-MM-DD — <decision title>

- **Context:** what forced the decision.
- **Decision:** what we chose.
- **Alternatives considered:** what we rejected and why.
- **Consequences:** what this locks in or rules out.
```
