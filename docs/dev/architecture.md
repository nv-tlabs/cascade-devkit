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
  The parent interface is initialized **lazily**: the constructor only
  scans local annotation JSON, and the gated-repo metadata download in
  `PhysicalAIAVDatasetInterface.__init__` is deferred to `_ensure_parent`,
  called by the feature/video entry points (`get_sequence`,
  `download_clips`). This keeps the query API (`find` / `count` /
  `group_by`, which read only `_by_clip`) zero-network and token-free; an
  HF token is required only once you reach for clip video/egomotion.
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
    `Then` / `Before` / `While` / `WhileStrict`).
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
  - `paper.py` — `render_paper_figure`, a static Plotly composition of
    zero to three explicitly selected video frames above the shared timeline
    painter, with stable per-entity track visibility switches.
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

### 2026-07-09 — Paper figures use semantic colors and focused defaults

- **Context:** Publication renders exposed four visual inconsistencies: arrow
  paths ended in dark-outlined tips, every Traffic-Light state used one red
  fill regardless of its annotation, containment rows dominated compact paper
  layouts, and raw video pre-roll could produce a negative sampled caption.
- **Decision:** Use each arrow family's color for its path, arrowhead fill, and
  arrowhead outline across all timeline-backed views. Mirror the annotator's
  per-state Green/Yellow/Red/Other palette in the shared painter. In paper mode
  only, hide all containment families by default behind an explicit
  `show_containments=True` opt-in. Reject negative paper timestamps and sample
  notebook/example frames exclusively from actual non-negative video
  timestamps.
- **Alternatives considered:** Removing the arrowhead outline was unnecessary;
  retaining a same-color outline keeps its size stable. Changing the public
  context-free `family_color("Traffic Lights", "state")` fallback was rejected
  because semantic color requires a concrete state instance. Hiding
  containment globally was rejected because interactive timelines rely on
  those rows. Clamping negative timestamps to zero was rejected because zero
  may not be an actual frame and paper mode promises no silent clamping.
- **Consequences:** Arrow styling and Light-state meaning are consistent in
  static and interactive timelines. Paper figures are less cluttered and
  never caption sampled frames below zero; callers can restore containment
  rows and arrows explicitly. Existing non-paper containment defaults remain
  unchanged.

### 2026-07-09 — Paper figures preserve complete box labels

- **Context:** The shared timeline painter optimizes interactive views by
  shortening labels to eight characters and omitting them on boxes narrower
  than 6% of clip duration. Full text remains available on hover, but
  publication exports have no hover fallback, so entity/action names appeared
  ellipsized or disappeared entirely.
- **Decision:** Add a private full-label policy to the shared painter and use
  it only from `render_paper_figure()`. With inline labels enabled, every
  visible paper box emits its complete text regardless of duration. Labels on
  the left/right half anchor toward the plot interior so time-axis edge boxes
  do not direct text outside the canvas. Keep the compact policy unchanged for
  `render_timeline`, `ClipPlayer`, and carousels.
- **Alternatives considered:** Changing the shared default was rejected
  because dense interactive timelines rely on compact labels and hover text.
  Fixed-width annotation boxes were rejected because Plotly clips their text.
  Distorting temporal box widths to make text fit was rejected because it
  would misrepresent annotation timing.
- **Consequences:** Paper-mode HTML and static exports retain every box name,
  including narrow intervals. Long labels may extend across neighboring boxes;
  callers can filter tracks/families, increase figure width, or set
  `show_inline_labels=False` for dense compositions.

### 2026-07-08 — DevKit timelines omit Influence rows and draw action causes

- **Context:** `Influence` records were rendered as both dedicated family rows
  and purple arrows, which consumed vertical space without expressing the
  action-rooted causal relation needed in publication figures. Meanwhile,
  `because_of` arrows resolved only top-level parent IDs even though valid
  references can target nested actions, light/object states, properties,
  conditions, and other segment-backed records.
- **Decision:** Keep `Influence` intact in the schema, query DSL, and public
  `annotation_to_segments()` output, but filter its rows and arrows at the
  shared DevKit timeline painter. Represent action causality with `because_of`
  arrows and resolve every visible segment-backed stable annotation ID,
  including nested records and the synthetic Ego anchor. Dangling references
  remain non-fatal and are omitted.
- **Alternatives considered:** Removing `Influence` from the schema or segment
  adapter was rejected because it would discard semantic data and break
  non-rendering consumers. Mapping every causal target to its top-level parent
  was rejected because it loses the annotated action/state/property endpoint.
  Rejecting the former `families=["influence"]` selector was avoided as an
  unnecessary hard break; it remains a documented compatibility no-op while
  the row itself is absent.
- **Consequences:** Static timelines, `ClipPlayer`, carousels, and paper figures
  share a more compact layout with no influence family. The `influence` family
  selector remains accepted as a compatibility no-op, and the arrow toggle has
  no rendered effect; `influenced_by` queries remain supported. All ID-backed
  arrow families share the expanded resolver: it follows `IdIndex` collision
  semantics and supplements visible Traffic-Light containments without
  overriding canonical indexed IDs (#10). `because_of` arrows now cover every
  non-dangling visible target and still disappear when either endpoint is
  filtered out.

### 2026-07-08 — Paper figures render frames chronologically

- **Context:** The initial paper-figure contract preserved caller order, which
  allowed a figure's frames to move backward and forward in time. Publication
  figures should read as a temporal sequence without requiring every caller to
  sort its timestamp selection first.
- **Decision:** After validating and batch-decoding the zero to three requested
  timestamps, stable-sort each timestamp/frame pair in ascending order before
  composing the figure. Duplicate timestamps remain separate adjacent frames.
  Sorting after validation preserves caller-indexed errors; direct and
  high-level rendering share the behavior.
- **Alternatives considered:** Requiring callers to pre-sort was rejected
  because it makes chronological output optional and inconsistent. Sorting
  before validation was rejected because an error such as `timestamps[1]`
  would no longer identify the caller's original value. Deduplicating was
  rejected because repeated frames may be intentional.
- **Consequences:** Every paper figure reads earliest-to-latest from left to
  right. Existing callers that intentionally supplied a nonchronological
  narrative order now receive chronological output; chronological callers and
  the executed notebook output are unchanged. This supersedes only the
  caller-order clause in the paper-figure decision below.

### 2026-07-08 — Paper figures use stable entity switches and explicit frames

- **Context:** Publication figures need several exact moments above one
  annotation timeline and must be able to omit individual entities. The
  existing `ClipPlayer` is interactive, owns a moving playhead and controls,
  and decodes only one current frame. Timeline `track_id` values cannot serve
  as entity selectors because they come from reusable `_track_index` layout
  lanes; non-overlapping entities may share one.
- **Decision:** Add an opt-in `paper_figure` visualization mode backed by the
  plain-Plotly `render_paper_figure` function. The initial implementation
  preserved zero to three caller-supplied timestamps left-to-right; that
  ordering clause is superseded by the chronological-order decision above.
  Explicit timestamps outside actual video coverage are rejected rather than
  clamped. Track visibility is grouped by entity kind and keyed by stable
  top-level annotation IDs; child rows inherit their owner, and ambiguous
  duplicate IDs are rejected. The DevKit timeline's canonical top-to-bottom
  order is Ego, Agents, Traffic Lights, Objects, Environments. The annotator UI
  owns its ordering independently.
- **Alternatives considered:** Reusing `ClipPlayer` was rejected because a
  `FigureWidget`, controls, and moving playhead are inappropriate for static
  export. Selecting `_track_index` lanes was rejected because it can hide more
  than one entity. Returning separate frame and timeline objects was rejected
  because it leaves reproducible layout composition to every caller.
- **Consequences:** Paper figures return a serializable Plotly `Figure` and
  batch-decode requested frames once. Embedded frames use bounded quality-90
  JPEG transport so notebooks do not serialize full-resolution lossless PNGs.
  An empty timestamp list remains useful for a video-free timeline-only figure.
  Omitted visibility switches default on; hiding an entity removes its parent,
  descendants, and arrows connected to hidden segments. Existing visualization
  calls retain their prior dispatch.

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
