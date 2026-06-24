# Visualization

`cascade_av.viz` renders any subset of a clip — a single instant, a
time range, or a `MatchSet` — as a decoded camera frame paired with
the clip's annotation timeline. The timeline carries one bar per
agent action, ego action, environment, condition, and traffic-light
state, plus four families of causal arrow (`because_of`,
`containment`, `influence`, `action_target`). It also paints a yellow
highlight band over any match interval.

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

The three common workflows hang off objects you already use.
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
```

Every filter listed below works on all four call sites:

```python
seq.visualize(...)                  # interactive ClipPlayer widget
viz.render_timeline(seq, ...)       # headless Plotly figure
ClipPlayer(seq, ...)                # the widget class directly
matches.visualize(...)              # carousel of mini-players, one per match
```

The four whitelist filters AND together; the `arrows` toggle is
independent (it gates arrow *families*, not the segment filter).

## Headless rendering

For reports, doc figures, or pipelines without a Jupyter kernel, two
functions return plain values you can pickle, save, or post-process:

```python
from cascade_av import viz

frame = viz.render_frame(seq, t=3.0)          # -> PIL.Image (RGB)
fig   = viz.render_timeline(seq,              # -> plotly.graph_objects.Figure
                            highlight=(2, 5))
frame.save("/tmp/clip_t3.png")
fig.write_image("/tmp/timeline.png")          # needs the `kaleido` extra
```

`seq.visualize(t=2.5, static=True)` is the shorthand for
`viz.render_frame(seq, 2.5)` — handy when you start interactive and
want a single still without switching modules.

## Filters

### `arrows` — toggle causal-arrow families

Type: `dict[str, bool]` · Default: all-on

Four recognized families. Pass `False` for any you want to hide;
unmentioned keys default to on.

```python
seq.visualize(arrows={"because_of": False, "containment": False})
```

| Family | What it represents |
|---|---|
| `because_of` | causal cause arrows — `action.because_of` field on EgoAction / AgentAction |
| `containment` | spatial-containment edges (agent ∈ environment, etc.) |
| `influence` | influence edges (a light influences the ego, etc.) |
| `action_target` | action → target edges |

### `entity_kinds` — whitelist segment kinds

Type: `list[str]` · Default: `None` (all kinds)

```python
viz.render_timeline(seq, entity_kinds=["agent", "ego"])
```

| Value | Selects |
|---|---|
| `"env"` | Environment segments |
| `"light"` | Traffic Light segments (parent + per-signal-head) |
| `"object"` | Object segments |
| `"agent"` | Agent segments + their sub-rows |
| `"ego"` | Ego vehicle segments |

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
| `"Environments"` | `"Env"` (legacy) |
| `"Traffic Lights"` | `"Lights"` (legacy) |
| `"Objects"` | — |
| `"Agents"` | — |
| `"Ego"` | — |

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
| `"influence"` | Agents, Ego |
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

A segment paints only if it survives all four whitelists:

```
track_groups  ∧  entity_kinds  ∧  agent_ids  ∧  families
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
| `show_inline_labels` | `bool` | `True` | When `False`, suppress the in-bar text labels. Hover tooltips still fire. Useful for screenshots / dense clips. |
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
| `src/cascade_av/viz/timeline.py` | `render_timeline`, `_paint_timeline_onto` (the load-bearing painter; all four filters resolve here) |
| `src/cascade_av/viz/widget.py` | `ClipPlayer` — forwards every filter to `_paint_timeline_onto` |
| `src/cascade_av/viz/carousel.py` | `build_matchset_carousel` — forwards `families` + `arrows` to each per-match `ClipPlayer` |
| `src/cascade_av/dataset.py` | `Sequence.visualize` — high-level dispatcher; forwards `families` + `arrows` |

The walkthrough notebook at [`notebooks/06_visualize.ipynb`](../../notebooks/06_visualize.ipynb)
demos each entry point against a real clip, with section 6 dedicated
to the `families` whitelist.
