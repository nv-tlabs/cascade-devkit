# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless Plotly timeline renderer — the static `render_timeline` Figure.

This is the v0 of the timeline view from `meta/10_visualization_api_plan.md`.
It consumes a `Sequence`'s `AnnotationBundle`, flattens it through PR-1's
`annotation_to_segments`, and emits a `plotly.graph_objects.Figure` showing
each segment as a colored rectangle on one of five track-group rows:
**Env**, **Lights**, **Objects**, **Agents**, **Ego** — matching the
annotator's `Timeline.tsx` grouping (just collapsed to one row per group;
the per-lane lattice is intentionally squashed here, since the headless
view doesn't have to be pixel-perfect with the annotator and the same
shape works inside the PR-4 widget).

Segment colors follow PR-1's `entity_color()` palette so the headless view
and the annotator share one set of hex strings. Causal relationships
(`because_of`, `link_to`, `containment`, `influence`, `action_target`) are
drawn as upward-curving bezier `path` shapes between segment centers,
mirroring `drawBecauseOfCurve` in `tools/annotator/web/src/components/
Timeline.tsx`. Each family has its own color:

- `because_of`     → rose      (`#f43f5e`) — same hue family as the
                                              annotator's "BECAUSE" pill
                                              (the TS code uses `#b45309`;
                                              we pick a brighter rose so it
                                              reads clearly on
                                              `plotly_dark`).
- `link_to`        → teal      (`#14b8a6`)
- `containment`    → green     (`#10b981`)
- `influence`      → purple    (`#a78bfa`)
- `action_target`  → orange    (`#f97316`)

These are chosen to be high-contrast against `plotly_dark` while staying
distinguishable from each other. **Invariant:** the arrow palette must
stay disjoint from `entity_color()`'s row-fill palette — an arrow that
shares a hex with the row it crosses would visually disappear. A guard
test in `tests/test_viz_timeline.py` pins this.

`highlight=(t0, t1)` paints a translucent yellow vertical band on top of
the rows — useful when callers want to draw attention to a particular
query window. `arrows={...}` toggles arrow families individually; the
defaults are all-on. The Figure is fully static — the widget-driving
variant lands in PR-4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from causal_ai_av.viz.colors import entity_color
from causal_ai_av.viz.segments import Segment, annotation_to_segments

if TYPE_CHECKING:  # pragma: no cover — typing only
    import plotly.graph_objects as go

    from causal_ai_av.dataset import Sequence
    from causal_ai_av.spec import AnnotationBundle


# Five track-group rows, top → bottom. Order is the annotator's reading
# order (environments at the top, ego at the bottom). y indices are
# integers so we can stick to categorical-style ticks while still placing
# rect shapes with floating-point ±0.4 offsets.
_TRACK_GROUPS: tuple[str, ...] = ("Env", "Lights", "Objects", "Agents", "Ego")
_GROUP_ROW: dict[str, int] = {name: i for i, name in enumerate(_TRACK_GROUPS)}

# Arrow-family colors. See module docstring for the choice rationale.
# Invariant: values here must be disjoint from `entity_color()`'s row
# fills — guarded by `test_arrow_palette_disjoint_from_entity_palette`.
_ARROW_COLORS: dict[str, str] = {
    "because_of": "#f43f5e",
    "link_to": "#14b8a6",
    "containment": "#10b981",
    "influence": "#a78bfa",
    "action_target": "#f97316",
}

# Default arrow toggles — every family on. Callers override via the
# `arrows` kwarg; any key omitted from the user override falls back to
# this default.
_DEFAULT_ARROWS: dict[str, bool] = {name: True for name in _ARROW_COLORS}


def _track_id_to_group(track_id: str) -> str | None:
    """Map a `Segment.track_id` to one of the five group labels.

    Returns `None` for unknown prefixes — those segments get dropped from
    the figure rather than crashing.
    """
    if track_id.startswith("env_"):
        return "Env"
    if track_id.startswith("light_"):
        return "Lights"
    if track_id.startswith("obj_"):
        return "Objects"
    if track_id.startswith("agent_"):
        return "Agents"
    if track_id == "ego_act":
        return "Ego"
    return None


def _segment_kind(track_id: str) -> str | None:
    """Map a track id to the `entity_color()` palette key."""
    group = _track_id_to_group(track_id)
    if group is None:
        return None
    return {
        "Env": "env",
        "Lights": "light",
        "Objects": "object",
        "Agents": "agent",
        "Ego": "ego",
    }[group]


def _segment_center_x(seg: Segment) -> float:
    return (seg.t0 + seg.t1) / 2.0


def _segment_top_y(group: str) -> float:
    """y-coordinate of a segment's top edge (used as arrow anchor)."""
    return _GROUP_ROW[group] + 0.4


def _bezier_path(x0: float, y0: float, x1: float, y1: float) -> str:
    """Build an SVG quadratic-bezier path mirroring `drawBecauseOfCurve`.

    The annotator draws `M x1 y1 Q midX ctrlY x2 y2` where `ctrlY` is the
    minimum of the two endpoints minus 30px of canvas. We mimic the
    *shape* in data coordinates — the control point sits one full row
    above the higher of the two endpoints, so the arc reads as an upward
    curve regardless of source/target ordering.
    """
    mid_x = (x0 + x1) / 2.0
    # y grows downward in the annotator's canvas but upward in Plotly's
    # default `yaxis`. Either way, the visual is the same: the curve
    # bulges *away* from the rows. With our top-of-row anchors at
    # `row + 0.4` and a y-axis that grows upward, we lift the control
    # point *above* both endpoints by adding 0.8 to the max y.
    ctrl_y = max(y0, y1) + 0.8
    return f"M {x0} {y0} Q {mid_x} {ctrl_y} {x1} {y1}"


def _segments_by_entity_id(
    bundle: "AnnotationBundle",
    segments: list[Segment],
) -> dict[str, Segment]:
    """Build an `entity-id → Segment` lookup table.

    The annotator's TS code keeps an `entityToSeg` map keyed on each
    entity's stable id (`env_<n>`, `obj_<n>`, `light_<n>`, `agent_<n>`).
    Our `Segment.id` is synthetic — it includes a nonce — so we can't
    use it directly. Instead, we walk the bundle in the same order that
    `annotation_to_segments` does and pair each entity with the parent
    segment we just emitted.

    Only parent rows participate (environments, traffic objects, traffic
    lights, agents); subtracks resolve back to the same parent.
    """
    ann = bundle.annotation
    # Filter to parent rows in their original emission order. `meta`
    # marks them with `_objKind`.
    parents = [s for s in segments if (s.meta or {}).get("_objKind")]
    pi = 0
    by_id: dict[str, Segment] = {}

    def _next_parent_for(kind: str) -> Segment | None:
        nonlocal pi
        # Advance past any non-matching parents (defensive — should
        # always match in emission order).
        while pi < len(parents):
            seg = parents[pi]
            pi += 1
            if (seg.meta or {}).get("_objKind") == kind:
                return seg
        return None

    for env in ann.environments:
        s = _next_parent_for("environment")
        if s is not None and env.id:
            by_id.setdefault(env.id, s)
    for obj in ann.traffic_objects:
        s = _next_parent_for("traffic_object")
        if s is not None and obj.id:
            by_id.setdefault(obj.id, s)
    for light in ann.traffic_lights:
        s = _next_parent_for("traffic_light")
        if s is not None and light.id:
            by_id.setdefault(light.id, s)
    for agent in ann.agents:
        s = _next_parent_for("agent")
        if s is not None and agent.id:
            by_id.setdefault(agent.id, s)
    return by_id


def _paint_timeline_onto(
    fig: "go.Figure",
    seq: "Sequence",
    *,
    xref: str = "x",
    yref: str = "y",
    xaxis_key: str = "xaxis",
    yaxis_key: str = "yaxis",
    highlight: tuple[float, float] | None = None,
    arrows: dict[str, bool] | None = None,
) -> None:
    """Append timeline shapes for `seq` onto `fig`, on the given axes.

    Factored out of `render_timeline` so the PR-4 widget can paint the
    same timeline onto the bottom subplot of a `FigureWidget` without
    rebuilding the rasterization logic from scratch. Mutates `fig` in
    place — appends shapes to `fig.layout.shapes` and updates the
    `xaxis_key` / `yaxis_key` axis settings.

    Args:
        fig: the Plotly Figure (or FigureWidget) to paint onto.
        seq: source `Sequence`.
        xref, yref: axis references for the painted shapes
            (`"x"`/`"y"` for the primary subplot, `"x2"`/`"y2"` for the
            second subplot in a `make_subplots(rows=2, ...)` figure,
            and so on).
        xaxis_key, yaxis_key: layout keys for the axis configuration
            (`"xaxis"`/`"yaxis"` or `"xaxis2"`/`"yaxis2"`, etc.).
        highlight: optional `(t0, t1)` in seconds — translucent yellow
            band drawn behind the rows.
        arrows: optional family-on/off toggle. Recognized keys are
            `"because_of"`, `"link_to"`, `"containment"`, `"influence"`,
            `"action_target"`. Missing keys default to `True`. Unknown
            keys are ignored.

    Returns:
        None. `fig` is mutated.
    """
    bundle = seq.annotation
    duration = float(seq.duration_s) if seq.duration_s else 0.0

    # Resolve the arrow-family toggles up front.
    enabled_arrows = dict(_DEFAULT_ARROWS)
    if arrows:
        for k, v in arrows.items():
            if k in enabled_arrows:
                enabled_arrows[k] = bool(v)

    shapes: list[dict[str, Any]] = []

    # Flatten the bundle into segments via PR-1's port. Empty bundles
    # produce an empty list — we'll still emit shapes-free axis layout,
    # which is what the spec asks for.
    segments = annotation_to_segments(bundle)

    # ---------------------------------------------------------------
    # 1. Optional highlight band — drawn first so it sits *under* the
    #    segment rectangles in z-order. Plotly draws shapes in their
    #    list order; the renderer adds them to layout.shapes so the
    #    highlight band reads as a background.
    # ---------------------------------------------------------------
    if highlight is not None:
        h0, h1 = float(highlight[0]), float(highlight[1])
        if h0 > h1:
            h0, h1 = h1, h0
        shapes.append(
            {
                "type": "rect",
                "x0": h0,
                "x1": h1,
                "y0": -0.5,
                "y1": len(_TRACK_GROUPS) - 0.5,
                "xref": xref,
                "yref": yref,
                "fillcolor": "yellow",
                "opacity": 0.15,
                "line": {"width": 0},
                "layer": "below",
                # Tag for tests / inspection. Plotly preserves unknown
                # keys at the top level on the Shape object, but to
                # stay strict we keep `name` (an officially supported
                # field) as the marker.
                "name": "highlight",
            }
        )

    # ---------------------------------------------------------------
    # 2. One rectangle per segment. Drop segments whose track id
    #    doesn't map into one of the five groups (defensive — the
    #    schema currently only emits the five we know about).
    # ---------------------------------------------------------------
    for seg in segments:
        group = _track_id_to_group(seg.track_id)
        if group is None:
            continue
        kind = _segment_kind(seg.track_id)
        if kind is None:
            continue
        row = _GROUP_ROW[group]
        shapes.append(
            {
                "type": "rect",
                "x0": seg.t0,
                "x1": seg.t1,
                "y0": row - 0.4,
                "y1": row + 0.4,
                "xref": xref,
                "yref": yref,
                "fillcolor": entity_color(kind),
                "opacity": 0.85,
                "line": {
                    "width": 1.5 if seg.illegal else 0.5,
                    "color": "#fbbf24" if seg.illegal else "#0f172a",
                },
                "layer": "above",
                "name": f"segment:{seg.id}",
            }
        )

    # ---------------------------------------------------------------
    # 3. Arrows. For each family that's enabled, walk the relevant
    #    segments and emit a bezier `path` shape. The annotator
    #    resolves arrow endpoints against time-overlapping segments;
    #    we keep that simpler here — we look up the by-id entity
    #    segment, and if not found, drop the arrow rather than
    #    crash.
    # ---------------------------------------------------------------
    by_id = _segments_by_entity_id(bundle, segments)
    ann = bundle.annotation

    def _add_arrow(
        src: Segment | None,
        tgt: Segment | None,
        family: str,
    ) -> None:
        if src is None or tgt is None:
            return
        src_group = _track_id_to_group(src.track_id)
        tgt_group = _track_id_to_group(tgt.track_id)
        if src_group is None or tgt_group is None:
            return
        x0 = _segment_center_x(src)
        x1 = _segment_center_x(tgt)
        y0 = _segment_top_y(src_group)
        y1 = _segment_top_y(tgt_group)
        shapes.append(
            {
                "type": "path",
                "path": _bezier_path(x0, y0, x1, y1),
                "xref": xref,
                "yref": yref,
                "line": {"color": _ARROW_COLORS[family], "width": 2},
                "fillcolor": "rgba(0,0,0,0)",
                "layer": "above",
                "name": f"arrow:{family}:{src.id}->{tgt.id}",
            }
        )

    # 3a. because_of — read straight off the Segment dataclass.
    if enabled_arrows["because_of"]:
        for seg in segments:
            for target_id in seg.because_of:
                _add_arrow(seg, by_id.get(target_id), "because_of")

    # 3b. link_to — walk the bundle's actions and (signaling) properties.
    if enabled_arrows["link_to"]:
        # Ego actions.
        for i, act in enumerate(ann.ego_vehicle.actions):
            src = _find_seg(
                segments,
                id_starts_with="ego_act_",
                meta_key="_egoActIndex",
                meta_val=i,
            )
            for target_id in act.link_to:
                _add_arrow(src, by_id.get(target_id), "link_to")
        # Agent actions.
        for ai, agent in enumerate(ann.agents):
            for act_idx, act in enumerate(agent.actions):
                src = _find_seg(
                    segments,
                    id_starts_with=f"agent_action_{ai}_{act_idx}_",
                )
                for target_id in act.link_to:
                    _add_arrow(src, by_id.get(target_id), "link_to")
        # Ego signaling properties.
        for pi, prop in enumerate(ann.ego_vehicle.properties):
            sd = prop.signaling_details
            if sd is None:
                continue
            src = _find_seg(segments, id_starts_with=f"ego_prop_{pi}_")
            for target_id in sd.link_to:
                _add_arrow(src, by_id.get(target_id), "link_to")
        # Agent signaling properties.
        for ai, agent in enumerate(ann.agents):
            for pi, prop in enumerate(agent.properties):
                sd = prop.signaling_details
                if sd is None:
                    continue
                src = _find_seg(segments, id_starts_with=f"agent_prop_{ai}_{pi}_")
                for target_id in sd.link_to:
                    _add_arrow(src, by_id.get(target_id), "link_to")

    # 3c. containment — the annotator draws an arrow from a containment
    #     subtrack to its parent environment.
    if enabled_arrows["containment"]:
        # Ego containment.
        for ci, cont in enumerate(ann.ego_vehicle.containment):
            src = _find_seg(segments, id_starts_with=f"ego_cont_{ci}_")
            _add_arrow(src, by_id.get(cont.env_id), "containment")
        # Agent containment.
        for ai, agent in enumerate(ann.agents):
            for ci, cont in enumerate(agent.containment):
                src = _find_seg(segments, id_starts_with=f"agent_cont_{ai}_{ci}_")
                _add_arrow(src, by_id.get(cont.env_id), "containment")
        # Traffic-object containment.
        for oi, obj in enumerate(ann.traffic_objects):
            for ci, cont in enumerate(obj.containment):
                src = _find_seg(segments, id_starts_with=f"obj_cont_{oi}_{ci}_")
                _add_arrow(src, by_id.get(cont.env_id), "containment")
        # Traffic-light physical containment.
        for li, light in enumerate(ann.traffic_lights):
            for ci, cont in enumerate(light.containment):
                src = _find_seg(segments, id_starts_with=f"light_phys_cont_{li}_{ci}_")
                _add_arrow(src, by_id.get(cont.env_id), "containment")
        # Signal-head env_controlled.
        for li, light in enumerate(ann.traffic_lights):
            for hi, sh in enumerate(light.signal_heads):
                for ci, cont in enumerate(sh.env_controlled):
                    src = _find_seg(
                        segments,
                        id_starts_with=f"light_cont_{li}_{hi}_{ci}_",
                    )
                    _add_arrow(src, by_id.get(cont.env_id), "containment")

    # 3d. influence — the annotator draws an arrow from an influence
    #     subtrack to each of its influencers.
    if enabled_arrows["influence"]:
        for ii, infl in enumerate(ann.ego_vehicle.influenced_by):
            src = _find_seg(segments, id_starts_with=f"ego_infl_{ii}_")
            for influencer_id in infl.influencers:
                _add_arrow(src, by_id.get(influencer_id), "influence")
        for ai, agent in enumerate(ann.agents):
            for ii, infl in enumerate(agent.influenced_by):
                src = _find_seg(segments, id_starts_with=f"agent_infl_{ai}_{ii}_")
                for influencer_id in infl.influencers:
                    _add_arrow(src, by_id.get(influencer_id), "influence")

    # 3e. action_target — actions only (not properties).
    if enabled_arrows["action_target"]:
        for i, act in enumerate(ann.ego_vehicle.actions):
            src = _find_seg(
                segments,
                id_starts_with="ego_act_",
                meta_key="_egoActIndex",
                meta_val=i,
            )
            for target_id in act.action_target:
                _add_arrow(src, by_id.get(target_id), "action_target")
        for ai, agent in enumerate(ann.agents):
            for act_idx, act in enumerate(agent.actions):
                src = _find_seg(
                    segments,
                    id_starts_with=f"agent_action_{ai}_{act_idx}_",
                )
                for target_id in act.action_target:
                    _add_arrow(src, by_id.get(target_id), "action_target")

    # ---------------------------------------------------------------
    # 4. Axis layout — locked range, group-name ticks. Figure-wide
    #    settings (template / height / margins) are the caller's
    #    responsibility so this helper composes inside a multi-subplot
    #    figure without overwriting the host's chrome.
    #
    #    Append the new shapes after any shapes already on `fig` so
    #    callers can pre-paint playheads or other overlays without
    #    losing them.
    # ---------------------------------------------------------------
    existing_shapes = list(fig.layout.shapes or ())
    fig.update_layout(
        shapes=existing_shapes + shapes,
        **{
            xaxis_key: {
                "title": "Time (s)",
                "range": [0, duration if duration > 0 else 1.0],
                "showgrid": False,
                "zeroline": False,
            },
            yaxis_key: {
                "tickmode": "array",
                "tickvals": list(range(len(_TRACK_GROUPS))),
                "ticktext": list(_TRACK_GROUPS),
                "range": [len(_TRACK_GROUPS) - 0.5, -0.5],  # top → bottom
                "showgrid": False,
                "zeroline": False,
            },
        },
    )


def render_timeline(
    seq: "Sequence",
    *,
    highlight: tuple[float, float] | None = None,
    arrows: dict[str, bool] | None = None,
) -> "go.Figure":
    """Return a Plotly Figure showing the clip's annotation timeline.

    Rows = track groups (Environments, Lights, Objects, Agents, Ego);
    each segment is a colored rectangle via `go.layout.Shape`.
    Causal arrows are drawn as bezier `path` shapes between segment
    centers. `highlight=(t0, t1)` paints a translucent vertical band.
    `arrows={"because_of": True, ...}` toggles arrow families;
    default all-on. The Figure is fully static (no traitlets); the
    widget-driving variant lives in `viz.widget.ClipPlayer` and reuses
    the same `_paint_timeline_onto` helper.

    Args:
        seq: a `Sequence` (typically from `CausalAVDataset.get_sequence`
            or `Sequence.from_annotation`). Only `seq.annotation` and
            `seq.duration_s` are read — no video / parent dataset
            access, so this is safe in any environment.
        highlight: optional `(t0, t1)` in seconds. When set, a single
            translucent yellow vrect is drawn behind the rows.
        arrows: optional family-on/off toggle. Recognized keys are
            `"because_of"`, `"link_to"`, `"containment"`, `"influence"`,
            `"action_target"`. Missing keys default to `True`. Unknown
            keys are ignored.

    Returns:
        A `plotly.graph_objects.Figure`. The Figure has zero traces; the
        timeline is built entirely from `layout.shapes` so that the
        widget can replace just the playhead line without touching the
        rest of the figure.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    _paint_timeline_onto(
        fig,
        seq,
        xref="x",
        yref="y",
        xaxis_key="xaxis",
        yaxis_key="yaxis",
        highlight=highlight,
        arrows=arrows,
    )
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin={"l": 80, "r": 20, "t": 20, "b": 40},
    )
    return fig


def _find_seg(
    segments: list[Segment],
    *,
    id_starts_with: str,
    meta_key: str | None = None,
    meta_val: Any = None,
) -> Segment | None:
    """Find the first segment whose id starts with `id_starts_with`.

    For the ego-action case we also disambiguate by `meta[_egoActIndex]`
    because all ego-action segment ids share the `ego_act_` prefix (the
    nonce is the only differentiator — see `annotation_to_segments`).
    """
    for seg in segments:
        if not seg.id.startswith(id_starts_with):
            continue
        if meta_key is not None:
            if not seg.meta or seg.meta.get(meta_key) != meta_val:
                continue
        return seg
    return None


__all__ = ["render_timeline"]
