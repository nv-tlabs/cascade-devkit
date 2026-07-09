# Visualization

`cascade_av.viz` renders any subset of a clip — a single instant, a
time range, or a `MatchSet` — as a decoded camera frame paired with
the clip's annotation timeline. The timeline carries one bar per
agent action, ego action, environment, condition, and traffic-light
state, plus three documented families of causal arrow (`because_of`,
`containment`, `action_target`). It also paints a yellow
highlight band over any match interval. Every timeline-backed view uses
the same top-to-bottom group order: **Ego, Agents, Traffic Lights,
Objects, Environments**.

This doc is the authoritative reference; the
[README's Visualization section](../../README.md#visualization) is a
short teaser pointing here.

## Install

The viz API requires the optional `[viz]` extra:

```bash
uv sync --extra viz
```

It runs in JupyterLab, VS Code / Cursor notebooks, and similar
Jupyter-protocol environments. Colab support is best-effort.

## Entry points

The common workflows hang off objects you already use.
`ds.get_sequence(clip_id)` returns a `Sequence` — the per-clip handle
that bundles the parsed annotation with camera / sensor accessors and
the `.visualize()` method:

```python
seq = ds.get_sequence(clip_id)

# Whole-clip scrubbable widget. Pass t=2.5 to open at a single instant,
# or t=(2, 5) for an explicit window.
seq.visualize()

# Single match in context — pads the match interval by `pad` seconds
# and paints `m.interval` as the yellow highlight band.
m = ds.find("agent.type = ped while ego.action = decel").matches[0]
seq.visualize(match=m, pad=1.0)

# Fan out a whole MatchSet into a carousel of mini-players (one per
# match, capped at `limit`).
ds.find("ego.action = decel because_of agent.type = ped").visualize()

# Static publication figure with selected video frames above the tracks.
seq.visualize(mode="paper_figure", timestamps=[1.0, 2.5, 4.0])
```

Filter support differs slightly by entry point. "All" below means
`arrows`, `entity_kinds`, `agent_ids`, `track_groups`, `families`, and
`track_visibility`:

| Entry point | Timeline filters |
|---|---|
| `seq.visualize(...)` | All, whenever the result includes a timeline; the `static=True` single-frame path has no tracks to filter |
| `viz.render_timeline(seq, ...)` | All |
| `viz.render_paper_figure(seq, ...)` | All |
| `ClipPlayer(seq, ...)` | All |
| `matches.visualize(...)` | `arrows`, `families` |

The segment filters AND together; the `arrows` toggle is independent
(it gates arrow *families*, not the segment filter).

## Headless rendering

For reports, doc figures, or pipelines without a Jupyter kernel, the
headless functions return plain values you can pickle, save, or
post-process:

```python
from cascade_av import viz

frame = viz.render_frame(seq, t=3.0)          # -> PIL.Image (RGB)
fig   = viz.render_timeline(seq,              # -> plotly.graph_objects.Figure
                            highlight=(2, 5))
paper = viz.render_paper_figure(              # -> plotly.graph_objects.Figure
    seq,
    timestamps=[1.0, 3.0, 5.0],
    track_visibility={"agent": {"agent_4": False}},
)
frame.save("/tmp/clip_t3.png")
fig.write_image("/tmp/timeline.png")          # needs the `kaleido` extra
paper.write_image("/tmp/paper.png")           # needs the `kaleido` extra
```

`seq.visualize(t=2.5, static=True)` is the shorthand for
`viz.render_frame(seq, 2.5)` — handy when you start interactive and
want a single still without switching modules.

## Paper figures

`viz.render_paper_figure()` composes zero to three explicitly selected
video frames side-by-side above the annotation tracks. The high-level
equivalent is `seq.visualize(mode="paper_figure", timestamps=[...])`.
Both return a plain Plotly figure rather than an interactive widget.

```python
paper = viz.render_paper_figure(
    seq,
    timestamps=[4.8, 1.2, 3.0],
    track_visibility={
        "agent": {"agent_4": False, "agent_5": False},
        "env": False,
    },
)
```

Timestamp rules are deliberately strict for reproducible figures:

- Pass zero to three finite numeric timestamps in seconds. An empty list
  creates a timeline-only figure and does not access the video.
- Frames are sorted chronologically from left to right, so the unsorted
  example above displays 1.2 s, then 3.0 s, then 4.8 s. Duplicate timestamps
  are retained as separate frames.
- Every value must lie within the video's actual timestamp coverage.
  Out-of-coverage values raise `ValueError`; they are never clamped.
- `mode="paper_figure"` uses `timestamps`, so it cannot be combined with
  `t`, `match`, `context`, or `static=True`.

When frames are requested, the sequence must have an accessible video.
They are embedded as quality-90 JPEGs capped at 1920 pixels on the longer
edge, keeping saved HTML and executed notebooks compact without changing
the source video. The timeline filters described below, plus `highlight`,
`height`, `width`, and `show_inline_labels`, are available in paper mode.

Paper figures preserve the complete name on every visible timeline box:
labels are never shortened with an ellipsis or omitted just because a box is
narrow. Boxes at either end of the time axis anchor their labels toward the
plot interior. Very dense figures can therefore have overlapping text; use
the visibility/family filters, a wider `width`, or
`show_inline_labels=False` when a label-free composition is preferable.

## Track order

All timeline-backed entry points use this canonical top-to-bottom order:

1. Ego
2. Agents
3. Traffic Lights
4. Objects
5. Environments

Filtering a group out collapses the y-axis while preserving the relative
order of the groups that remain.

## Filters

### `track_visibility` — switch whole kinds or individual entities

Type: `dict[str, bool | dict[str, bool]]` · Default: all-on

Use a boolean to switch a whole kind, or a nested mapping to switch
individual tracks by their stable, non-empty annotation IDs. Omitted kinds
and IDs default to on.

```python
# Hide all Environment tracks and two specific Agent tracks.
seq.visualize(
    track_visibility={
        "env": False,
        "agent": {"agent_4": False, "agent_5": False},
    }
)
```

| Kind | Nested selector |
|---|---|
| `"ego"` | Synthetic singleton ID `"ego"` |
| `"agent"` | `Agent.id` |
| `"light"` | `TrafficLight.id` |
| `"object"` | `TrafficObject.id` |
| `"env"` | `Environment.id` |

Hiding a top-level entity removes its parent track and every descendant
row. Causal arrows with either endpoint hidden are removed too. Unknown
kinds and IDs raise `ValueError`, which prevents a misspelled selector
from silently leaving the wrong track in a publication figure. Selecting
an ID that occurs more than once also raises: duplicate IDs are ambiguous,
so use a whole-kind switch or fix the annotation IDs first.

### `arrows` — toggle causal-arrow families

Type: `dict[str, bool]` · Default: all-on

Three documented families. Pass `False` for any you want to hide;
unmentioned keys default to on.

```python
seq.visualize(arrows={"because_of": False, "containment": False})
```

| Family | What it represents |
|---|---|
| `because_of` | causal cause arrows — `action.because_of` field on EgoAction / AgentAction |
| `containment` | spatial-containment edges (agent ∈ environment, etc.) |
| `action_target` | action → target edges |

`because_of` targets resolve to any visible segment-backed stable ID,
including nested actions, light/object states, properties, conditions, and
containment records. Dangling IDs remain non-fatal and do not paint an arrow.

`Influence` annotations remain available through the schema, query DSL, and
`viz.annotation_to_segments()`, but timeline-backed DevKit views intentionally
do not render influence rows or influence arrows. For compatibility,
`families=["influence"]` is still accepted and renders no rows.

### `entity_kinds` — whitelist segment kinds

Type: `list[str]` · Default: `None` (all kinds)

```python
viz.render_timeline(seq, entity_kinds=["agent", "ego"])
```

| Value | Selects |
|---|---|
| `"ego"` | Ego vehicle segments |
| `"agent"` | Agent segments + their sub-rows |
| `"light"` | Traffic Light segments (parent + per-signal-head) |
| `"object"` | Object segments |
| `"env"` | Environment segments |

### `agent_ids` — whitelist specific agents

Type: `list[str]` · Default: `None` (all agents) · Affects only the `agent` kind

```python
seq.visualize(agent_ids=["agent_3", "agent_7"])
```

Other kinds (env / light / object / ego) are untouched. Combine with
`entity_kinds=["agent"]` if you want *only* those specific agents and
nothing else.

### `track_groups` — whitelist category rows

Type: `list[str]` · Default: `None` (all five categories)

```python
viz.render_timeline(seq, track_groups=["Agents", "Ego"])
```

| Value | Aliases |
|---|---|
| `"Ego"` | — |
| `"Agents"` | — |
| `"Traffic Lights"` | `"Lights"` (legacy) |
| `"Objects"` | — |
| `"Environments"` | `"Env"` (legacy) |

Rows not in the whitelist drop their tick labels too — the y-axis
collapses to the visible categories.

### `families` — whitelist family-leaf rows

Type: `list[str]` · Default: `None` (all families)

```python
matches.visualize(families=["action", "condition"])
```

| Value | Where it lives |
|---|---|
| `"condition"` | Environments |
| `"physical_containment"` | Traffic Lights |
| `"signal_head"` | Traffic Lights (per signal head) |
| `"env_control"` | Traffic Lights (per signal head) |
| `"state"` | Traffic Lights (per signal head), Objects |
| `"containment"` | Objects, Agents, Ego |
| `"pose"` | Agents |
| `"action"` | Agents, Ego |
| `"property"` | Agents, Ego |

**Parent retention.** Parent entity headers (`Env Track 1`,
`Agent Track 2`, ...) are NOT named in this whitelist. They auto-
render for any entity whose sub-rows survive the filter, and **drop
completely** for entities with zero surviving sub-rows. There are no
orphan section heads.

Example: `families=["action"]` on a clip with one Env, one Ego, and
two Agents — the Env block disappears (its only sub-row is
`condition`), and the Ego + Agent blocks keep their parent headers
because each has an `action` sub-row.

## Composition

A segment paints only if it survives every supplied selector:

```
track_groups  ∧  entity_kinds  ∧  agent_ids  ∧  families  ∧  track_visibility
```

Arrows paint only if BOTH endpoints survived the segment filter.

```python
# Just agent actions for two specific agents, no Ego rows in the way,
# and no `because_of` clutter from the arrow family.
seq.visualize(
    track_groups=["Agents"],
    families=["action"],
    agent_ids=["agent_3", "agent_7"],
    arrows={"because_of": False},
)
```

## Non-filter knobs

These shape the figure without dropping content.

| Kwarg | Type | Default | What it does |
|---|---|---|---|
| `highlight` | `tuple[float, float]` | `None` | Paint a translucent yellow band across `(t0, t1)` — used to mark a match interval or window of interest. |
| `show_inline_labels` | `bool` | `True` | When `False`, suppress the in-bar text labels. Paper mode otherwise preserves every complete box label; other timeline views use compact labels plus full hover text. |
| `height` | `int` | `None` (adaptive) | Pin the total figure height in pixels. Default scales with the deepest sub-lane stack. |

### Carousel-only: `unique_clips` — one player per clip vs one per match

Type: `bool` · Default: `False` · Surface: `matches.visualize` /
`build_matchset_carousel`

By default the carousel renders **one player per match** in the
`MatchSet`. Three matches that all live in the same clip will produce
three players over that clip. Pass `unique_clips=True` to flip to
**one player per distinct clip**: matches sharing a `clip_id` are
unioned and rendered once, with the playback window covering
`[min(t0)-pad, max(t1)+pad]` and the highlight band spanning the union.
The label switches to `clip_id · N matches · [t0, t1]s` whenever a clip
has more than one match.

```python
# Three *distinct* clips — even when one of them has many matches.
matches.visualize(limit=3, unique_clips=True)
```

`limit` counts whatever the carousel is rendering: matches when
`unique_clips=False`, clips when `unique_clips=True`. The truncation
notice ("Showing 3 of 12 clips" vs ".. of 12 matches") follows the same
rule.

Use the default when you're hunting individual matches; flip the knob
when you're surveying which clips show the behaviour.

## Where this comes from in code

| File | Surface |
|---|---|
| `src/cascade_av/viz/timeline.py` | `render_timeline`, `_paint_timeline_onto` (the load-bearing painter; all segment and arrow filters resolve here) |
| `src/cascade_av/viz/paper.py` | `render_paper_figure` — validates and decodes selected frames, then composes them above the shared timeline painter |
| `src/cascade_av/viz/widget.py` | `ClipPlayer` — forwards every timeline filter to `_paint_timeline_onto` |
| `src/cascade_av/viz/carousel.py` | `build_matchset_carousel` — forwards `arrows` and `families` to each per-match `ClipPlayer` |
| `src/cascade_av/dataset.py` | `Sequence.visualize` — high-level dispatcher for frame, player, and paper-figure modes; forwards every timeline filter |

The walkthrough notebook at [`notebooks/06_visualize.ipynb`](../../notebooks/06_visualize.ipynb)
demos the interactive entry points and a three-frame, per-entity-filtered
paper figure against a real clip. The runnable
[`examples/07_visualize.py`](../../examples/07_visualize.py) also writes a
paper figure.
