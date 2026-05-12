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
from causal_ai_av.viz.segments import Segment, annotation_to_segments, assign_lanes

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


# Sub-lane geometry.
#
# A group row covers `[row - _ROW_HALF_HEIGHT, row + _ROW_HALF_HEIGHT]`. With
# `lane_count` lanes inside, each lane is a horizontal stripe of height
# `2 * _ROW_HALF_HEIGHT / lane_count`. Lane 0 sits at the "top" of the
# row (which renders at the bottom of the row in data coordinates because
# `yaxis.range` is reversed in `_paint_timeline_onto`).
_ROW_HALF_HEIGHT: float = 0.4


def _lane_band(group: str, lane: int, lane_count: int) -> tuple[float, float]:
    """Return `(y0, y1)` for a segment painted at `lane` in `group`.

    `lane_count` is the deepest stack within the group — every segment in
    the same group shares the same lane height so the band tiling looks
    even.
    """
    row = _GROUP_ROW[group]
    total = 2.0 * _ROW_HALF_HEIGHT
    lane_h = total / max(lane_count, 1)
    y0 = row - _ROW_HALF_HEIGHT + lane * lane_h
    y1 = y0 + lane_h
    return y0, y1


def _segment_top_y(group: str, lane: int = 0, lane_count: int = 1) -> float:
    """y-coordinate of a segment's top edge (used as arrow anchor).

    With the yaxis range reversed (top → bottom), "top" in data
    coordinates is the *larger* y value of the lane band — that's the
    edge that visually faces away from the rows below.
    """
    _y0, y1 = _lane_band(group, lane, lane_count)
    return y1


# Inline-label font size, in points. Tuned for the 24px-per-lane
# layout from `render_timeline`'s adaptive height — at that lane
# height a `size=9` label reads cleanly without clipping vertically
# against the rectangle's top/bottom edge. Bumping this back up is
# the right knob if a caller forces a taller layout via the `height`
# kwarg.
_INLINE_LABEL_FONT_SIZE: int = 9

# Maximum characters for an inline segment label. Anything longer gets
# truncated with an ellipsis; the full label always lives in the hover
# tooltip.
_INLINE_LABEL_MAX_CHARS: int = 11

# A segment must be at least this wide (in seconds) to qualify for an
# inline label. Narrower segments suppress the inline annotation and
# rely on hover-only — otherwise the text would clip outside the
# rectangle and read as noise. Pinned by a regression test on the
# `~5 s vs ~0.05 s` divide called out in the design doc.
_INLINE_LABEL_MIN_WIDTH_S: float = 0.5


def _truncate_label(label: str, *, max_chars: int = _INLINE_LABEL_MAX_CHARS) -> str:
    """Truncate `label` to `max_chars`, appending `…` when shortened."""
    if len(label) <= max_chars:
        return label
    return label[: max(0, max_chars - 1)].rstrip() + "…"


def _xref_to_axis(xref: str) -> str:
    """Translate a shape's `xref` (`"x"`, `"x2"`, …) to a trace's `xaxis`.

    Plotly uses the same string for both — `"x"` for the primary
    subplot, `"x2"` for the second, etc. The function is a no-op alias
    but keeps the caller's intent explicit: shape `xref` and trace
    `xaxis` are semantically distinct fields even though they share a
    representation.
    """
    return xref


def _yref_to_axis(yref: str) -> str:
    """Translate a shape's `yref` to a trace's `yaxis`."""
    return yref


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
    entity_kinds: list[str] | None = None,
    agent_ids: list[str] | None = None,
    track_groups: list[str] | None = None,
) -> dict[str, int]:
    """Append timeline shapes for `seq` onto `fig`, on the given axes.

    Factored out of `render_timeline` so the PR-4 widget can paint the
    same timeline onto the bottom subplot of a `FigureWidget` without
    rebuilding the rasterization logic from scratch. Mutates `fig` in
    place — appends shapes to `fig.layout.shapes`, label annotations to
    `fig.layout.annotations`, and invisible hover-scatter traces to
    `fig.data`, then updates the `xaxis_key` / `yaxis_key` axis
    settings.

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
        entity_kinds: optional whitelist of segment kinds to draw.
            Recognized values are `"env"`, `"light"`, `"object"`,
            `"agent"`, `"ego"`. `None` = all kinds.
        agent_ids: optional whitelist of `Agent.id` values. Only
            restricts segments whose kind is `"agent"`; segments of
            other kinds are untouched. `None` = all agents.
        track_groups: optional whitelist of group rows to render.
            Recognized values are `"Env"`, `"Lights"`, `"Objects"`,
            `"Agents"`, `"Ego"`. Rows not in the whitelist drop their
            tick labels too, so the y-axis collapses to the visible
            rows. `None` = all five groups.

    Returns:
        Per-group lane counts — `{"Env": 1, "Lights": 1, "Objects": 1,
        "Agents": 3, "Ego": 1}`. Callers use this to size the figure
        proportionally to the deepest stack (see `render_timeline` and
        `widget.ClipPlayer`'s adaptive-height path). The figure itself
        is also mutated in place — shapes / annotations / hover traces
        are appended onto `fig`.

    Filter semantics:
        - The three filters AND together: a segment is drawn only if
          its group is in `track_groups`, its kind is in
          `entity_kinds`, and (for agent kinds) its agent id is in
          `agent_ids`.
        - Arrow filtering follows segment filtering: an arrow whose
          source or target segment was filtered out is dropped.
    """
    import plotly.graph_objects as go
    bundle = seq.annotation
    duration = float(seq.duration_s) if seq.duration_s else 0.0

    # Resolve the arrow-family toggles up front.
    enabled_arrows = dict(_DEFAULT_ARROWS)
    if arrows:
        for k, v in arrows.items():
            if k in enabled_arrows:
                enabled_arrows[k] = bool(v)

    # Resolve filter whitelists. `None` means "all", which we mirror by
    # building a set populated with every legal value.
    allowed_kinds: set[str] = (
        {"env", "light", "object", "agent", "ego"}
        if entity_kinds is None
        else set(entity_kinds)
    )
    allowed_groups: set[str] = (
        set(_TRACK_GROUPS) if track_groups is None else set(track_groups)
    )
    allowed_agent_ids: set[str] | None = (
        None if agent_ids is None else set(agent_ids)
    )

    shapes: list[dict[str, Any]] = []

    # Flatten the bundle into segments via PR-1's port. Empty bundles
    # produce an empty list — we'll still emit shapes-free axis layout,
    # which is what the spec asks for.
    segments = annotation_to_segments(bundle)

    # ---------------------------------------------------------------
    # Segment-level filter. Used both for the rectangle/label loop
    # and for the arrow-endpoint check.
    # ---------------------------------------------------------------
    ann_for_filter = bundle.annotation
    agent_id_by_index: dict[int, str] = {
        i: a.id for i, a in enumerate(ann_for_filter.agents) if a.id
    }

    def _segment_allowed(seg: Segment) -> bool:
        group = _track_id_to_group(seg.track_id)
        if group is None or group not in allowed_groups:
            return False
        kind = _segment_kind(seg.track_id)
        if kind is None or kind not in allowed_kinds:
            return False
        if kind == "agent" and allowed_agent_ids is not None:
            ai = (seg.meta or {}).get("_agentIndex")
            if not isinstance(ai, int):
                return False
            agent_id = agent_id_by_index.get(ai)
            if agent_id is None or agent_id not in allowed_agent_ids:
                return False
        return True

    # Cache the predicate result so arrow lookups can short-circuit
    # quickly when the source / target segment was filtered out.
    allowed_ids: set[str] = {s.id for s in segments if _segment_allowed(s)}

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
    # 2. One rectangle per segment, stacked into sub-lanes.
    #
    #    Greedy lane assignment via `assign_lanes` runs per `track_id`
    #    (mirroring the annotator's `assignTracks`). To paint inside a
    #    *group row*, we further roll up lane counts per group — the
    #    deepest stack across all of the group's track ids wins, so
    #    every segment in the same group ends up on a uniform lane
    #    grid. This is the same idea as the annotator's grouping in
    #    `Timeline.tsx`: subtracks share the parent's vertical budget.
    # ---------------------------------------------------------------
    paintable: list[tuple[Segment, str, str]] = []
    for seg in segments:
        if seg.id not in allowed_ids:
            continue
        group = _track_id_to_group(seg.track_id)
        if group is None:
            continue
        kind = _segment_kind(seg.track_id)
        if kind is None:
            continue
        paintable.append((seg, group, kind))

    seg_lane = assign_lanes([s for s, _, _ in paintable])
    # Roll lane indices up to per-group lane counts so the lane band
    # tiling is uniform within a group (otherwise neighbouring track
    # ids would paint at different y-heights and read as a ragged
    # lattice).
    group_lane_count: dict[str, int] = {g: 1 for g in _TRACK_GROUPS}
    for seg, group, _ in paintable:
        lane = seg_lane.get(seg.id, 0)
        group_lane_count[group] = max(group_lane_count[group], lane + 1)

    # Cache `(group, lane) -> (y0, y1)` for the segment loop and the
    # arrow-anchor loop.
    def _band_for(group: str, lane: int) -> tuple[float, float]:
        return _lane_band(group, lane, group_lane_count[group])

    # Collect inline annotation entries (text drawn on top of each
    # rectangle) and hover-overlay traces (invisible scatter markers at
    # segment centres carrying a tooltip with the full label, window,
    # and track id). Plotly shapes don't support hover natively; an
    # overlay trace is the standard idiom.
    annotations: list[dict[str, Any]] = []
    hover_traces: list[go.Scatter] = []
    xa = _xref_to_axis(xref)
    ya = _yref_to_axis(yref)

    for seg, group, kind in paintable:
        lane = seg_lane.get(seg.id, 0)
        y0, y1 = _band_for(group, lane)
        shapes.append(
            {
                "type": "rect",
                "x0": seg.t0,
                "x1": seg.t1,
                "y0": y0,
                "y1": y1,
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

        # Inline label — centered on the rectangle. Suppressed for
        # very narrow segments where the truncated text would clip.
        seg_width = max(seg.t1 - seg.t0, 0.0)
        if seg_width >= _INLINE_LABEL_MIN_WIDTH_S:
            annotations.append(
                {
                    "x": (seg.t0 + seg.t1) / 2.0,
                    "y": (y0 + y1) / 2.0,
                    "xref": xref,
                    "yref": yref,
                    "text": _truncate_label(seg.label),
                    "showarrow": False,
                    "font": {"size": _INLINE_LABEL_FONT_SIZE, "color": "#0f172a"},
                    "name": f"label:{seg.id}",
                }
            )

        # Hover overlay — invisible scatter marker at the segment
        # midpoint. `opacity=0` keeps it from drawing, `size=20` makes
        # the hover hitbox generous, and `hovertext` carries the full
        # tooltip body (label / window / track id).
        hover_traces.append(
            go.Scatter(
                x=[(seg.t0 + seg.t1) / 2.0],
                y=[(y0 + y1) / 2.0],
                xaxis=xa,
                yaxis=ya,
                mode="markers",
                marker={"size": 20, "opacity": 0, "color": entity_color(kind)},
                hoverinfo="text",
                hovertext=(
                    f"{seg.label}<br>"
                    f"[{seg.t0:.2f}s – {seg.t1:.2f}s]<br>"
                    f"track: {seg.track_id}"
                ),
                showlegend=False,
                name=f"hover:segment:{seg.id}",
            )
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
        # Drop the arrow if either endpoint was filtered out (per
        # `entity_kinds` / `agent_ids` / `track_groups`). Symmetric:
        # both ends must be present for the bezier to make sense.
        if src.id not in allowed_ids or tgt.id not in allowed_ids:
            return
        src_group = _track_id_to_group(src.track_id)
        tgt_group = _track_id_to_group(tgt.track_id)
        if src_group is None or tgt_group is None:
            return
        x0 = _segment_center_x(src)
        x1 = _segment_center_x(tgt)
        # Anchor each end of the bezier to its segment's lane band so
        # multi-lane stacks don't collapse arrows onto the same edge.
        src_lane = seg_lane.get(src.id, 0)
        tgt_lane = seg_lane.get(tgt.id, 0)
        y0 = _segment_top_y(src_group, src_lane, group_lane_count[src_group])
        y1 = _segment_top_y(tgt_group, tgt_lane, group_lane_count[tgt_group])
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

        # Arrowhead at the target end. Plotly's SVG `path` shape doesn't
        # render arrowheads, so we add a rotated triangle marker via a
        # `go.Scatter` trace. The angle is computed from the bezier's
        # tangent at t=1, which for a quadratic curve through control
        # point `(mid_x, ctrl_y)` is `(x1 - mid_x, y1 - ctrl_y)` in
        # data coords. With the reversed y-axis, flip dy when feeding
        # `atan2` so the triangle points along the on-screen direction.
        import math

        mid_x = (x0 + x1) / 2.0
        ctrl_y = max(y0, y1) + 0.8
        dx = x1 - mid_x
        dy = y1 - ctrl_y
        # Convert tangent vector to a Plotly `marker.angle` (degrees
        # clockwise from "north", which is +y in screen space).
        # Screen-y points "up" in Plotly's default but the timeline
        # uses a reversed yaxis, so the on-screen y direction is `-dy`
        # in data coords.
        screen_dy = -dy
        angle_rad = math.atan2(dx, screen_dy)
        angle_deg = math.degrees(angle_rad)
        hover_traces.append(
            go.Scatter(
                x=[x1],
                y=[y1],
                xaxis=xa,
                yaxis=ya,
                mode="markers",
                marker={
                    "symbol": "triangle-up",
                    "size": 10,
                    "angle": angle_deg,
                    "color": _ARROW_COLORS[family],
                    # Tailwind slate-50 outline — pops off any
                    # same-hue target row (containment-green over
                    # Env, influence-purple over Agents, etc.)
                    # without strobing the way pure white would on
                    # `plotly_dark`. Family color stays as the fill
                    # so the head is still identifiable by hue.
                    "line": {"color": "#f8fafc", "width": 1.5},
                },
                hoverinfo="text",
                hovertext=(
                    f"{family}<br>"
                    f"{src.label}<br>"
                    f"  → {tgt.label}"
                ),
                showlegend=False,
                name=f"arrowhead:{family}:{src.id}->{tgt.id}",
            )
        )

        # Mid-arc hover hotspot — invisible marker at the bezier
        # midpoint so the bezier line itself is hoverable end-to-end
        # rather than only at the arrowhead.
        bezier_mid_x = 0.25 * x0 + 0.5 * mid_x + 0.25 * x1
        bezier_mid_y = 0.25 * y0 + 0.5 * ctrl_y + 0.25 * y1
        hover_traces.append(
            go.Scatter(
                x=[bezier_mid_x],
                y=[bezier_mid_y],
                xaxis=xa,
                yaxis=ya,
                mode="markers",
                marker={"size": 18, "opacity": 0, "color": _ARROW_COLORS[family]},
                hoverinfo="text",
                hovertext=(
                    f"{family}<br>"
                    f"{src.label}<br>"
                    f"  → {tgt.label}"
                ),
                showlegend=False,
                name=f"hover:arrow:{family}:{src.id}->{tgt.id}",
            )
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
    # When `track_groups` filters the y-axis, drop unrendered rows from
    # the tick layout AND tighten the y-range to the visible band so
    # the figure doesn't render with empty rows.
    visible_rows: list[tuple[int, str]] = [
        (_GROUP_ROW[g], g) for g in _TRACK_GROUPS if g in allowed_groups
    ]
    if visible_rows:
        ticks_vals = [r for r, _ in visible_rows]
        ticks_text = [g for _, g in visible_rows]
        y_min = min(ticks_vals) - 0.5
        y_max = max(ticks_vals) + 0.5
    else:
        # Pathological: caller asked for zero groups. Keep something
        # sane so Plotly still renders a frame.
        ticks_vals = list(range(len(_TRACK_GROUPS)))
        ticks_text = list(_TRACK_GROUPS)
        y_min = -0.5
        y_max = len(_TRACK_GROUPS) - 0.5

    existing_shapes = list(fig.layout.shapes or ())
    existing_annotations = list(fig.layout.annotations or ())
    fig.update_layout(
        shapes=existing_shapes + shapes,
        annotations=existing_annotations + annotations,
        **{
            xaxis_key: {
                "title": "Time (s)",
                "range": [0, duration if duration > 0 else 1.0],
                "showgrid": False,
                "zeroline": False,
            },
            yaxis_key: {
                "tickmode": "array",
                "tickvals": ticks_vals,
                "ticktext": ticks_text,
                "range": [y_max, y_min],  # top → bottom
                "showgrid": False,
                "zeroline": False,
            },
        },
    )

    # Append the hover-overlay traces last so any pre-existing traces
    # (e.g. the widget's video-frame Image at index 0) keep their
    # positions in `fig.data`.
    for trace in hover_traces:
        fig.add_trace(trace)

    return group_lane_count


# Vertical pixels per lane band for the adaptive-height path. Drives
# both `render_timeline`'s `height` and `ClipPlayer`'s `row_heights`
# split. Tuned so a label rendered at `_INLINE_LABEL_FONT_SIZE` reads
# cleanly without clipping the row above/below.
_PX_PER_LANE: int = 24

# Minimum total timeline height in pixels. With no overlap the five
# group rows × 1 lane each would still want some breathing room; this
# is the floor before lane stacking adds more.
_MIN_TIMELINE_PX: int = 240


def _timeline_px_for(group_lane_count: dict[str, int]) -> int:
    """Compute the adaptive timeline height in pixels."""
    total_lanes = sum(group_lane_count.values()) or len(_TRACK_GROUPS)
    return max(_MIN_TIMELINE_PX, total_lanes * _PX_PER_LANE)


def render_timeline(
    seq: "Sequence",
    *,
    highlight: tuple[float, float] | None = None,
    arrows: dict[str, bool] | None = None,
    entity_kinds: list[str] | None = None,
    agent_ids: list[str] | None = None,
    track_groups: list[str] | None = None,
    height: int | None = None,
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
        entity_kinds: optional kind whitelist (`"env"`, `"light"`,
            `"object"`, `"agent"`, `"ego"`). `None` = all kinds.
        agent_ids: optional `Agent.id` whitelist. Restricts only the
            `"agent"` kind. `None` = all agents.
        track_groups: optional group-row whitelist (`"Env"`, `"Lights"`,
            `"Objects"`, `"Agents"`, `"Ego"`). Rows not in the
            whitelist drop from the y-axis layout. `None` = all groups.
        height: optional explicit pixel height. `None` (default) means
            adaptive — the height tracks the deepest sub-lane stack so
            a busy clip gets a taller timeline while a sparse one
            stays compact. Pass an int to pin a specific value; useful
            when embedding the figure in a fixed-size dashboard cell.

    Returns:
        A `plotly.graph_objects.Figure`. The visible rectangles, arrows,
        and inline labels live in `layout.shapes` + `layout.annotations`
        (so the widget can replace just the playhead line without
        touching the rest of the figure). The Figure also carries one
        invisible `go.Scatter` trace per segment/arrow purely to
        provide hover tooltips — Plotly shapes don't support hover
        natively.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    group_lane_count = _paint_timeline_onto(
        fig,
        seq,
        xref="x",
        yref="y",
        xaxis_key="xaxis",
        yaxis_key="yaxis",
        highlight=highlight,
        arrows=arrows,
        entity_kinds=entity_kinds,
        agent_ids=agent_ids,
        track_groups=track_groups,
    )
    resolved_height = (
        int(height) if height is not None else _timeline_px_for(group_lane_count)
    )
    fig.update_layout(
        template="plotly_dark",
        height=resolved_height,
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
