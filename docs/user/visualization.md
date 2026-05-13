# Visualization — filter reference

`causal_ai_av.viz` renders a clip as a decoded camera frame paired with
its annotation timeline. This doc covers the **filtering** surface —
the kwargs you pass to narrow what shows up in the timeline. For the
high-level overview (entry points, install, what gets painted) start
with the [Visualization section of the README](../../README.md#visualization).

## Entry points

Every filter listed below works on all four entry points:

```python
seq.visualize(...)                  # interactive ClipPlayer widget
viz.render_timeline(seq, ...)       # headless Plotly figure
ClipPlayer(seq, ...)                # the widget class directly
matches.visualize(...)              # carousel of mini-players, one per match
```

The four whitelist filters AND together; the `arrows` toggle is
independent (it gates arrow *families*, not the segment filter).

## Filters

### `arrows` — toggle causal-arrow families

Type: `dict[str, bool]` · Default: all-on

Five recognized families. Pass `False` for any you want to hide;
unmentioned keys default to on.

```python
seq.visualize(arrows={"link_to": False, "because_of": False})
```

| Family | What it represents |
|---|---|
| `because_of` | causal cause arrows — `action.because_of` field on EgoAction / AgentAction |
| `link_to` | side-effect / linked-entity arrows — `link_to` field |
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
# and no `link_to` clutter from the arrow family.
seq.visualize(
    track_groups=["Agents"],
    families=["action"],
    agent_ids=["agent_3", "agent_7"],
    arrows={"link_to": False},
)
```

## Non-filter knobs

These shape the figure without dropping content.

| Kwarg | Type | Default | What it does |
|---|---|---|---|
| `highlight` | `tuple[float, float]` | `None` | Paint a translucent yellow band across `(t0, t1)` — used to mark a match interval or window of interest. |
| `show_inline_labels` | `bool` | `True` | When `False`, suppress the in-bar text labels. Hover tooltips still fire. Useful for screenshots / dense clips. |
| `height` | `int` | `None` (adaptive) | Pin the total figure height in pixels. Default scales with the deepest sub-lane stack. |

## Where this comes from in code

| File | Surface |
|---|---|
| `src/causal_ai_av/viz/timeline.py` | `render_timeline`, `_paint_timeline_onto` (the load-bearing painter; all four filters resolve here) |
| `src/causal_ai_av/viz/widget.py` | `ClipPlayer` — forwards every filter to `_paint_timeline_onto` |
| `src/causal_ai_av/viz/carousel.py` | `build_matchset_carousel` — forwards `families` + `arrows` to each per-match `ClipPlayer` |
| `src/causal_ai_av/dataset.py` | `Sequence.visualize` — high-level dispatcher; forwards `families` + `arrows` |

The walkthrough notebook at [`notebooks/06_visualize.ipynb`](../../notebooks/06_visualize.ipynb)
demos each entry point against a real clip, with section 6 dedicated
to the `families` whitelist.
