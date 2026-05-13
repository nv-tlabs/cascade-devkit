# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `causal_ai_av.viz.timeline.render_timeline`.

Bundles are built from scratch — no corpus dependency. We poke at the
returned Plotly Figure's `layout.shapes` since the timeline is built
entirely from shapes (no traces) per the PR-2 design.
"""

from __future__ import annotations

import plotly.graph_objects as go

from causal_ai_av.dataset import Sequence
from causal_ai_av.spec import (
    Agent,
    AgentAction,
    AgentProperty,
    AnnotationBundle,
    Condition,
    Containment,
    EgoAction,
    EgoVehicle,
    Environment,
    Influence,
    SignalingDetails,
    SilAvAnnotation,
    VideoMeta,
)
from causal_ai_av.viz import render_timeline


# ---------------------------------------------------------------------------
# Helpers — bundle construction mirrors test_viz_segments._make_bundle but
# with the relationship fields (link_to / action_target / influencers) set
# so we can exercise every arrow family.
# ---------------------------------------------------------------------------


def _make_full_bundle() -> AnnotationBundle:
    env = Environment(
        id="env_0",
        type="fst:Road",
        num_lanes=2,
        start_timestamp="0:0.0",
        end_timestamp="0:10.0",
    )
    cond = Condition(
        id="cond_0",
        env_id="env_0",
        type=["Construction Zone"],
        start_timestamp="0:1.0",
        end_timestamp="0:5.0",
    )
    ego_action = EgoAction(
        id="ego_act_0",
        type="oxd:Decelerate",
        because_of=["agent_0"],
        link_to=["light_0"],
        action_target=["agent_0"],
        illegal_flag=True,
        start_timestamp="0:2.0",
        end_timestamp="0:4.0",
    )
    ego_cont = Containment(
        id="ego_cont_0",
        env_id="env_0",
        lane_number="1",
        start_timestamp="0:0.0",
        end_timestamp="0:10.0",
    )
    ego_infl = Influence(
        id="ego_infl_0",
        influencers=["light_0"],
        start_timestamp="0:1.0",
        end_timestamp="0:3.0",
    )
    ego = EgoVehicle(actions=[ego_action], containment=[ego_cont], influenced_by=[ego_infl])

    agent_action = AgentAction(
        id="agent_0_act_0",
        action_type="Yield",
        illegal_flag=True,
        because_of=["ego"],
        link_to=["env_0"],
        action_target=["ego"],
        start_timestamp="0:1.5",
        end_timestamp="0:3.0",
    )
    agent_prop = AgentProperty(
        id="agent_0_prop_0",
        property_type="Signal",
        signaling_details=SignalingDetails(intent="left", link_to=["env_0"]),
        start_timestamp="0:1.0",
        end_timestamp="0:3.0",
    )
    agent_cont = Containment(
        id="agent_0_cont_0",
        env_id="env_0",
        lane_number="1",
        start_timestamp="0:0.5",
        end_timestamp="0:5.5",
    )
    agent_infl = Influence(
        id="agent_0_infl_0",
        influencers=["env_0"],
        start_timestamp="0:1.0",
        end_timestamp="0:2.0",
    )
    agent = Agent(
        id="agent_0",
        type="oxd:Car",
        visibility_start_timestamp="0:0.5",
        visibility_end_timestamp="0:5.5",
        actions=[agent_action],
        properties=[agent_prop],
        containment=[agent_cont],
        influenced_by=[agent_infl],
    )

    ann = SilAvAnnotation(
        environments=[env],
        conditions=[cond],
        ego_vehicle=ego,
        agents=[agent],
    )
    return AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="test_clip", duration_s=10.0),
        annotation=ann,
    )


def _seq(bundle: AnnotationBundle) -> Sequence:
    return Sequence.from_annotation(bundle)


def _shapes(fig: go.Figure) -> list[dict]:
    """Return shapes as plain dicts (Plotly serializes via `.to_dict()`)."""
    return [dict(s.to_plotly_json()) for s in fig.layout.shapes]


def _arrow_shapes(fig: go.Figure, family: str) -> list[dict]:
    return [s for s in _shapes(fig) if s.get("name", "").startswith(f"arrow:{family}:")]


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_render_timeline_returns_figure() -> None:
    fig = render_timeline(_seq(_make_full_bundle()))
    assert isinstance(fig, go.Figure)


def test_arrow_palette_disjoint_from_entity_palette() -> None:
    """Arrow-family colors must not collide with entity row-fill colors.

    A causal arrow that shares a hex with the row it crosses would
    visually disappear — e.g., a `because_of` arrow drawn over the
    Lights row when both are red. Pinned here so future palette tweaks
    cannot silently re-introduce the collision.
    """
    from causal_ai_av.viz.colors import entity_color
    from causal_ai_av.viz.timeline import _ARROW_COLORS

    entity_hexes = {
        entity_color(k) for k in ("env", "ego", "object", "agent", "light")
    }
    arrow_hexes = set(_ARROW_COLORS.values())
    overlap = arrow_hexes & entity_hexes
    assert not overlap, (
        f"arrow palette overlaps entity palette: {sorted(overlap)}"
    )


def test_render_timeline_has_segment_shapes_per_group() -> None:
    """Each populated category contributes at least one segment shape.

    With family bands, "category" means "at least one band whose tick
    label starts with `'<Category> · '`". The fixture has
    Environments, Agents, and Ego entities (no Traffic Lights /
    Objects) — so the figure paints segments on bands belonging to
    each of those three categories.
    """
    fig = render_timeline(_seq(_make_full_bundle()))
    yticks = list(fig.layout.yaxis.ticktext or ())
    tickvals = list(fig.layout.yaxis.tickvals or ())
    # Build a mapping from band row to category head ("Environments",
    # "Agents", ...). Blank tick labels (trailing sub-rows) don't
    # contribute — they share the previous band's category.
    row_to_category: dict[float, str] = {}
    last_category = ""
    for v, t in zip(tickvals, yticks, strict=False):
        if t:
            last_category = t.split(" · ", 1)[0]
        row_to_category[float(v)] = last_category

    seg_shapes = [s for s in _shapes(fig) if s.get("name", "").startswith("segment:")]
    assert seg_shapes, "expected at least one segment shape"
    categories_with_segments: set[str] = set()
    for s in seg_shapes:
        mid = (float(s["y0"]) + float(s["y1"])) / 2.0
        for row, category in row_to_category.items():
            if abs(mid - row) <= 0.4 + 1e-9:
                categories_with_segments.add(category)
                break
    assert {"Environments", "Agents", "Ego"}.issubset(categories_with_segments), (
        f"missing categories, got {categories_with_segments}"
    )


def test_render_timeline_layout_is_dark_with_locked_axes() -> None:
    fig = render_timeline(_seq(_make_full_bundle()))
    assert fig.layout.template.layout.paper_bgcolor is not None  # plotly_dark
    # Adaptive height: the figure should never collapse below the
    # 240px floor, even for a sparse clip. The exact value scales
    # with the deepest sub-lane stack so we only assert the floor
    # here. `test_explicit_height_kwarg_overrides_adaptive` covers
    # the override knob, and `test_adaptive_height_grows_with_lanes`
    # pins the lane-count scaling.
    assert fig.layout.height >= 240
    # y-axis is one tick per populated band (Environments, Ego, Agents
    # families in this fixture). Traffic Lights / Objects contribute zero
    # bands because the fixture has no light / object entities. We
    # assert by reading order and prefix rather than the full label
    # list so the test survives future tick-label rewording.
    yticks = list(fig.layout.yaxis.ticktext or ())
    assert len(yticks) > 0, "expected at least one populated band"
    # Categories appear in canonical reading order (Environments first,
    # then Agents, then Ego — no Traffic Lights / Objects since the
    # fixture has none). Trailing-sub-row labels are blank; filter
    # them out so we look at category heads only.
    categories_seen = [
        label.split(" · ")[0] for label in yticks if label
    ]
    # First non-blank band must be Environments (the fixture has at
    # least one env).
    assert categories_seen[0] == "Environments"
    # Traffic Lights and Objects don't appear — no entities of those
    # kinds in the fixture.
    assert "Traffic Lights" not in categories_seen
    assert "Objects" not in categories_seen
    # x-axis range is [0, duration].
    xrange = list(fig.layout.xaxis.range or ())
    assert xrange[0] == 0
    assert xrange[1] == 10.0


# ---------------------------------------------------------------------------
# Highlight band
# ---------------------------------------------------------------------------


def test_highlight_adds_exactly_one_vrect_style_shape() -> None:
    fig = render_timeline(_seq(_make_full_bundle()), highlight=(2.0, 5.0))
    highlights = [s for s in _shapes(fig) if s.get("name") == "highlight"]
    assert len(highlights) == 1
    band = highlights[0]
    assert float(band["x0"]) == 2.0
    assert float(band["x1"]) == 5.0
    # Fillcolor is yellow with low opacity, no border line.
    assert band["fillcolor"] == "yellow"
    assert float(band["opacity"]) == 0.15


def test_no_highlight_means_no_highlight_shape() -> None:
    fig = render_timeline(_seq(_make_full_bundle()))
    highlights = [s for s in _shapes(fig) if s.get("name") == "highlight"]
    assert highlights == []


def test_reversed_highlight_is_normalized() -> None:
    """`highlight=(5, 2)` should still produce a vrect from 2 → 5."""
    fig = render_timeline(_seq(_make_full_bundle()), highlight=(5.0, 2.0))
    band = next(s for s in _shapes(fig) if s.get("name") == "highlight")
    assert float(band["x0"]) == 2.0
    assert float(band["x1"]) == 5.0


# ---------------------------------------------------------------------------
# Arrow toggles
# ---------------------------------------------------------------------------


def test_all_arrow_families_present_by_default() -> None:
    fig = render_timeline(_seq(_make_full_bundle()))
    # The fixture wires up at least one of each family.
    for family in ("because_of", "link_to", "containment", "influence", "action_target"):
        assert len(_arrow_shapes(fig, family)) > 0, (
            f"expected at least one {family} arrow with default settings"
        )


def test_arrows_toggle_off_one_family() -> None:
    fig = render_timeline(_seq(_make_full_bundle()), arrows={"because_of": False})
    assert _arrow_shapes(fig, "because_of") == []
    # Other families are unaffected.
    assert len(_arrow_shapes(fig, "link_to")) > 0
    assert len(_arrow_shapes(fig, "containment")) > 0


def test_arrows_keep_only_one_family() -> None:
    fig = render_timeline(
        _seq(_make_full_bundle()),
        arrows={
            "because_of": True,
            "link_to": False,
            "containment": False,
            "influence": False,
            "action_target": False,
        },
    )
    assert len(_arrow_shapes(fig, "because_of")) > 0
    for family in ("link_to", "containment", "influence", "action_target"):
        assert _arrow_shapes(fig, family) == [], f"expected no {family} arrows after toggling off"


def test_arrows_unknown_keys_are_ignored() -> None:
    """Unknown arrow keys should be silently ignored, not raise."""
    fig = render_timeline(
        _seq(_make_full_bundle()), arrows={"not_a_family": True, "because_of": False}
    )
    assert _arrow_shapes(fig, "because_of") == []
    # `not_a_family` is just dropped — no crash, no arrow.
    assert _arrow_shapes(fig, "not_a_family") == []


# ---------------------------------------------------------------------------
# Empty / minimal bundles
# ---------------------------------------------------------------------------


def test_render_timeline_on_empty_bundle() -> None:
    """No entities at all → Figure returns cleanly with no segment/arrow shapes."""
    bundle = AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="empty_clip", duration_s=5.0),
    )
    fig = render_timeline(_seq(bundle))
    assert isinstance(fig, go.Figure)
    # No segment shapes and no arrow shapes.
    segment_shapes = [s for s in _shapes(fig) if s.get("name", "").startswith("segment:")]
    assert segment_shapes == []
    for family in ("because_of", "link_to", "containment", "influence", "action_target"):
        assert _arrow_shapes(fig, family) == []
    # With the family-band layout, an empty bundle has zero populated
    # bands, so the y-axis tick list is empty. The figure still
    # renders (xaxis range + plotly_dark template are set).
    yticks = list(fig.layout.yaxis.ticktext or ())
    assert yticks == []


def test_render_timeline_zero_duration_does_not_crash() -> None:
    """Bundles with no declared duration should still produce a Figure."""
    bundle = AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="no_dur"),
    )
    fig = render_timeline(_seq(bundle))
    assert isinstance(fig, go.Figure)
    # x-axis range falls back to (0, 1) to avoid Plotly's auto-range
    # squeezing the figure to a single column.
    xrange = list(fig.layout.xaxis.range or ())
    assert xrange[0] == 0
    assert xrange[1] >= 1.0


# ---------------------------------------------------------------------------
# Sub-lane assignment (feat: rich timeline)
# ---------------------------------------------------------------------------


def _segment_shapes(fig: go.Figure) -> list[dict]:
    return [s for s in _shapes(fig) if s.get("name", "").startswith("segment:")]


def _agents_row_segment_shapes(fig: go.Figure) -> list[dict]:
    """All segment shapes whose vertical mid lies inside an Agents band.

    The family-band layout means "Agents" isn't a single row anymore —
    it's a stack of bands (parent, containment, pose, influence,
    action, property). This helper walks the y-axis tick labels to
    find every band whose label starts with "Agents · " and returns
    every segment shape whose midline lies inside any of those band
    rows. Each band row covers `[row - 0.4, row + 0.4]` per the
    painter's geometry.
    """
    yticks = list(fig.layout.yaxis.ticktext or ())
    tickvals = list(fig.layout.yaxis.tickvals or ())
    agent_rows = {
        float(v) for v, t in zip(tickvals, yticks, strict=False)
        if t.startswith("Agents · ")
    }
    out = []
    for s in _segment_shapes(fig):
        mid = (float(s["y0"]) + float(s["y1"])) / 2.0
        if any(abs(mid - r) <= 0.4 + 1e-9 for r in agent_rows):
            out.append(s)
    return out


def _band_rows_for_prefix(fig: go.Figure, prefix: str) -> set[float]:
    """Return the band rows whose tick label starts with `prefix`.

    Helper for tests that need to assert "every segment sits inside
    the bands belonging to group X" without hardcoding band indices.
    """
    yticks = list(fig.layout.yaxis.ticktext or ())
    tickvals = list(fig.layout.yaxis.tickvals or ())
    return {
        float(v) for v, t in zip(tickvals, yticks, strict=False)
        if t.startswith(prefix)
    }


def _make_two_overlapping_agent_actions_bundle() -> AnnotationBundle:
    """Two agent actions on the SAME agent that overlap in time."""
    a1 = AgentAction(
        id="agent_0_act_0",
        action_type="Yield",
        start_timestamp="0:1.0",
        end_timestamp="0:5.0",
    )
    a2 = AgentAction(
        id="agent_0_act_1",
        action_type="oxd:Decelerate",
        start_timestamp="0:2.0",  # overlaps with a1
        end_timestamp="0:6.0",
    )
    agent = Agent(
        id="agent_0",
        type="oxd:Car",
        visibility_start_timestamp="0:0.5",
        visibility_end_timestamp="0:8.0",
        actions=[a1, a2],
    )
    ann = SilAvAnnotation(agents=[agent])
    return AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="overlap", duration_s=10.0),
        annotation=ann,
    )


def test_overlapping_subtracks_land_on_distinct_lanes() -> None:
    """Two overlapping agent actions paint on different lane bands
    (lane-stacked inside the `Agents · action` band). Plus the parent
    agent itself lives on the `Agents · parent` band — so at least
    two distinct y-coordinate bands appear across the agent shapes."""
    fig = render_timeline(_seq(_make_two_overlapping_agent_actions_bundle()))
    agent_shapes = _agents_row_segment_shapes(fig)
    bands = {(round(float(s["y0"]), 3), round(float(s["y1"]), 3)) for s in agent_shapes}
    # Parent agent (Agents · parent band) + two overlapping actions
    # (Agents · action band, 2 lanes inside) → at least 3 distinct
    # (y0, y1) pairs across the painted agent shapes.
    assert len(bands) >= 3, f"expected multi-band/lane stack, got {bands}"


def test_non_overlapping_subtracks_share_lane_zero() -> None:
    """Two non-overlapping actions on the same agent share a single lane."""
    a1 = AgentAction(
        id="agent_0_act_0",
        action_type="Yield",
        start_timestamp="0:1.0",
        end_timestamp="0:3.0",
    )
    a2 = AgentAction(
        id="agent_0_act_1",
        action_type="oxd:Decelerate",
        start_timestamp="0:4.0",  # no overlap with a1
        end_timestamp="0:6.0",
    )
    agent = Agent(
        id="agent_0",
        type="oxd:Car",
        visibility_start_timestamp="0:0.5",
        visibility_end_timestamp="0:8.0",
        actions=[a1, a2],
    )
    bundle = AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="no_overlap", duration_s=10.0),
        annotation=SilAvAnnotation(agents=[agent]),
    )
    fig = render_timeline(_seq(bundle))
    # The two ACTIONS occupy the same band (Agents · action) and the
    # same single lane within that band — they don't overlap each
    # other, so the band's lane count stays at 1. With family bands
    # the parent agent lives on a different band (Agents · parent),
    # so we filter to action shapes only when checking lane stacking.
    agent_shapes = _agents_row_segment_shapes(fig)
    action_bands = {
        (round(float(s["y0"]), 3), round(float(s["y1"]), 3))
        for s in agent_shapes
        if "agent_action_" in s.get("name", "")
    }
    # The two actions share one lane (since their intervals are disjoint).
    assert len(action_bands) == 1, f"expected single-lane actions, got {action_bands}"


# ---------------------------------------------------------------------------
# Inline labels
# ---------------------------------------------------------------------------


def _annotations(fig: go.Figure) -> list[dict]:
    return [dict(a.to_plotly_json()) for a in (fig.layout.annotations or ())]


def test_inline_label_present_on_long_segments_suppressed_on_short_ones() -> None:
    """Wide segment → inline annotation; sub-threshold segment → no
    inline annotation (hover-only)."""
    long_action = AgentAction(
        id="agent_0_act_0",
        action_type="Yield",
        start_timestamp="0:0.0",
        end_timestamp="0:5.0",  # 5 s wide → inline label
    )
    short_action = AgentAction(
        id="agent_0_act_1",
        action_type="oxd:Decelerate",
        start_timestamp="0:6.0",
        end_timestamp="0:6.05",  # 50 ms wide → suppress inline label
    )
    agent = Agent(
        id="agent_0",
        type="oxd:Car",
        visibility_start_timestamp="0:0.0",
        visibility_end_timestamp="0:10.0",
        actions=[long_action, short_action],
    )
    bundle = AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="labels", duration_s=10.0),
        annotation=SilAvAnnotation(agents=[agent]),
    )
    fig = render_timeline(_seq(bundle))
    label_names = {a.get("name", "") for a in _annotations(fig)}
    # The 5 s action gets an inline label, the 50 ms one does not.
    assert any(n.startswith("label:agent_action_0_0_") for n in label_names)
    assert not any(n.startswith("label:agent_action_0_1_") for n in label_names)


def test_inline_label_thresholds_match_current_constants() -> None:
    """Pin the three inline-label constants together so a tweak to
    any one is forced to update this test (and document the choice).

    The current values keep labels readable at the 24px-per-lane
    family-band layout: font 8, 8-char truncation, 0.06 fraction-of-
    duration suppression. The fraction-based threshold (vs the prior
    absolute 0.5s) lets the same segment scale sensibly across clip
    lengths — a 1.0s segment looks fine in a 5s clip and cramped in
    a 20s clip, so the threshold tracks duration.
    """
    from causal_ai_av.viz.timeline import (
        _INLINE_LABEL_FONT_SIZE,
        _INLINE_LABEL_MAX_CHARS,
        _INLINE_LABEL_MIN_WIDTH_FRAC,
        _truncate_label,
    )

    assert _INLINE_LABEL_FONT_SIZE == 8
    assert _INLINE_LABEL_MAX_CHARS == 8
    assert _INLINE_LABEL_MIN_WIDTH_FRAC == 0.06

    # Truncation cap: an 11-char label gets shortened with the new
    # 8-char cap (would have been kept whole under the old 11-char
    # cap).
    assert _truncate_label("abcdefghijk") == "abcdefg…"

    # Suppression threshold (proportional): in a 10s clip the cutoff
    # is 0.6s. A 0.5s action (5% of duration) is suppressed; a 0.8s
    # action (8% of duration) keeps its label.
    short_action = AgentAction(
        id="agent_0_act_0",
        action_type="Yield",
        start_timestamp="0:0.0",
        end_timestamp="0:0.5",  # 0.5s in 10s clip → 5% → suppressed
    )
    long_action = AgentAction(
        id="agent_0_act_1",
        action_type="oxd:Decelerate",
        start_timestamp="0:2.0",
        end_timestamp="0:2.8",  # 0.8s in 10s clip → 8% → labeled
    )
    agent = Agent(
        id="agent_0",
        type="oxd:Car",
        visibility_start_timestamp="0:0.0",
        visibility_end_timestamp="0:10.0",
        actions=[short_action, long_action],
    )
    bundle = AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="thresholds", duration_s=10.0),
        annotation=SilAvAnnotation(agents=[agent]),
    )
    fig = render_timeline(_seq(bundle))
    label_names = {a.get("name", "") for a in _annotations(fig)}
    assert not any(n.startswith("label:agent_action_0_0_") for n in label_names), (
        "0.5s segment in a 10s clip should be suppressed (5% < 6% threshold)"
    )
    assert any(n.startswith("label:agent_action_0_1_") for n in label_names), (
        "0.8s segment in a 10s clip should keep its label (8% > 6% threshold)"
    )

    # And the inline annotation that does fire uses the new font size.
    fired = [
        a for a in _annotations(fig)
        if a.get("name", "").startswith("label:agent_action_0_1_")
    ]
    assert fired, "expected the 0.8s action to fire an inline label"
    font = fired[0].get("font") or {}
    assert int(font.get("size", 0)) == _INLINE_LABEL_FONT_SIZE


# ---------------------------------------------------------------------------
# Hover trace invariant
# ---------------------------------------------------------------------------


def test_hover_trace_count_matches_segments_plus_arrow_overlays() -> None:
    """Pin the trace-count invariant: every painted segment gets one
    hover trace; every arrow contributes two (arrowhead + midpoint)."""
    fig = render_timeline(_seq(_make_full_bundle()))
    seg_hover = [
        t for t in fig.data
        if (t.name or "").startswith("hover:segment:")
    ]
    arrow_mid_hover = [
        t for t in fig.data
        if (t.name or "").startswith("hover:arrow:")
    ]
    arrowheads = [
        t for t in fig.data
        if (t.name or "").startswith("arrowhead:")
    ]
    n_segments = len(_segment_shapes(fig))
    # Every arrow family contributes one path shape per arrow.
    n_arrows = sum(
        len(_arrow_shapes(fig, family))
        for family in (
            "because_of", "link_to", "containment", "influence", "action_target",
        )
    )
    assert len(seg_hover) == n_segments
    assert len(arrow_mid_hover) == n_arrows
    assert len(arrowheads) == n_arrows


# ---------------------------------------------------------------------------
# Arrowhead presence per enabled family
# ---------------------------------------------------------------------------


def test_each_enabled_arrow_family_contributes_an_arrowhead_marker() -> None:
    """For every family with at least one arrow, there's at least one
    matching arrowhead marker trace."""
    fig = render_timeline(_seq(_make_full_bundle()))
    for family in ("because_of", "link_to", "containment", "influence", "action_target"):
        arrows = _arrow_shapes(fig, family)
        if not arrows:
            continue
        heads = [
            t for t in fig.data
            if (t.name or "").startswith(f"arrowhead:{family}:")
        ]
        assert len(heads) == len(arrows), (
            f"family={family}: {len(heads)} arrowheads but {len(arrows)} arrows"
        )


def test_disabling_arrow_family_drops_its_arrowheads() -> None:
    """Toggling `because_of` off drops its arrowhead markers, not just
    the bezier shape."""
    fig = render_timeline(_seq(_make_full_bundle()), arrows={"because_of": False})
    heads = [
        t for t in fig.data
        if (t.name or "").startswith("arrowhead:because_of:")
    ]
    assert heads == []


def test_arrowhead_has_visible_outline() -> None:
    """Each arrowhead marker carries a 1.5px Tailwind slate-50
    (`#f8fafc`) outline so the head pops off any same-hue target
    row (e.g. containment-green over the Env row). Family fill is
    preserved; only `marker.line` changes from the prior
    invisible `{"width": 0}`.
    """
    fig = render_timeline(_seq(_make_full_bundle()))
    heads = [
        t for t in fig.data if (t.name or "").startswith("arrowhead:")
    ]
    assert heads, "expected at least one arrowhead marker on the rich bundle"
    for head in heads:
        line = head.marker.line
        assert float(line.width) == 1.5, (
            f"arrowhead {head.name} has width={line.width!r}, expected 1.5"
        )
        assert line.color == "#f8fafc", (
            f"arrowhead {head.name} has color={line.color!r}, expected '#f8fafc'"
        )


# ---------------------------------------------------------------------------
# Filter kwargs
# ---------------------------------------------------------------------------


def test_entity_kinds_filter_keeps_only_requested_rows() -> None:
    """`entity_kinds=["ego"]` drops every non-Ego segment shape; the
    y-axis collapses to only Ego bands."""
    fig = render_timeline(_seq(_make_full_bundle()), entity_kinds=["ego"])
    ego_rows = _band_rows_for_prefix(fig, "Ego · ")
    yticks = list(fig.layout.yaxis.ticktext or ())
    # Every tick label belongs to the Ego group.
    for label in yticks:
        assert label.startswith("Ego · "), f"non-Ego band leaked through: {label}"
    # And every painted segment sits inside one of the Ego bands.
    for s in _segment_shapes(fig):
        mid = (float(s["y0"]) + float(s["y1"])) / 2.0
        assert any(abs(mid - r) <= 0.4 + 1e-9 for r in ego_rows), (
            f"non-Ego shape leaked through filter: {s}"
        )


def test_agent_ids_filter_keeps_only_listed_agents() -> None:
    """With two agents in the bundle, `agent_ids=[first.id]` excludes
    the second."""
    second_agent = Agent(
        id="agent_1",
        type="oxd:Car",
        visibility_start_timestamp="0:6.0",
        visibility_end_timestamp="0:9.0",
        actions=[
            AgentAction(
                id="agent_1_act_0",
                action_type="Yield",
                start_timestamp="0:7.0",
                end_timestamp="0:8.0",
            )
        ],
    )
    bundle = _make_full_bundle()
    bundle = AnnotationBundle(
        schema_version=bundle.schema_version,
        video=bundle.video,
        annotation=SilAvAnnotation(
            environments=list(bundle.annotation.environments),
            conditions=list(bundle.annotation.conditions),
            ego_vehicle=bundle.annotation.ego_vehicle,
            agents=list(bundle.annotation.agents) + [second_agent],
        ),
    )
    fig_all = render_timeline(_seq(bundle))
    fig_one = render_timeline(_seq(bundle), agent_ids=["agent_0"])
    # The unfiltered figure has more agent-row segment shapes than the
    # filtered figure (the second agent's parent + action are dropped).
    def _agent_count(fig: go.Figure) -> int:
        return len(_agents_row_segment_shapes(fig))

    assert _agent_count(fig_one) < _agent_count(fig_all)


def test_track_groups_filter_collapses_axis_to_visible_rows() -> None:
    """`track_groups=["Agents"]` keeps only Agents' family bands on
    the y-axis. With the family-band layout that's multiple ticks
    (parent / containment / influence / action / property), all
    starting with "Agents · "."""
    fig = render_timeline(_seq(_make_full_bundle()), track_groups=["Agents"])
    yticks = list(fig.layout.yaxis.ticktext or ())
    assert yticks, "expected at least one Agents band when fixture has agents"
    for label in yticks:
        assert label.startswith("Agents · "), (
            f"non-Agents band leaked through track_groups filter: {label}"
        )
    # And every painted segment sits inside one of the Agents bands.
    agent_rows = _band_rows_for_prefix(fig, "Agents · ")
    for s in _segment_shapes(fig):
        mid = (float(s["y0"]) + float(s["y1"])) / 2.0
        assert any(abs(mid - r) <= 0.4 + 1e-9 for r in agent_rows), (
            f"shape outside Agents bands: {s}"
        )


def test_filter_propagation_drops_arrows_with_filtered_endpoints() -> None:
    """When a `because_of`'s source agent is filtered out, the arrow
    is dropped from the figure."""
    # The fixture's `because_of` arrows include: ego action → agent_0
    # (source = ego, target = agent_0). Filter agents out of the figure
    # and the arrow should disappear because its target endpoint is
    # gone.
    fig = render_timeline(
        _seq(_make_full_bundle()),
        entity_kinds=["env", "light", "object", "ego"],  # no agents
    )
    assert _arrow_shapes(fig, "because_of") == []
    # And the agent-targeted action_target arrows also vanish.
    assert _arrow_shapes(fig, "action_target") == []


# ---------------------------------------------------------------------------
# Family sub-row layout (this PR)
# ---------------------------------------------------------------------------


def _yticks(fig: go.Figure) -> list[str]:
    return list(fig.layout.yaxis.ticktext or ())


def test_paint_emits_band_per_populated_family() -> None:
    """The full-bundle fixture has Environments (parent + condition),
    Ego (containment + influence + action), and Agents (parent +
    containment + influence + action + property). Every populated
    family in those categories must show up as a band on the y-axis,
    tagged with the entity's track name (or Ego singleton).
    Traffic Lights / Objects have no entities in the fixture, so they
    contribute zero bands."""
    fig = render_timeline(_seq(_make_full_bundle()))
    yticks = _yticks(fig)
    expected_present = {
        "Environments · Env Track 1",                  # env parent
        "Environments · Env Track 1 · conditions",     # env condition
        "Agents · Agent Track 1",                      # agent parent
        "Agents · Agent Track 1 · containment",
        "Agents · Agent Track 1 · influences",
        "Agents · Agent Track 1 · actions",
        "Agents · Agent Track 1 · properties",
        "Ego · containment",
        "Ego · influences",
        "Ego · actions",
    }
    missing = expected_present - set(yticks)
    assert not missing, f"missing expected bands: {missing}; got {yticks}"


def test_band_y_axis_labels_have_group_family_format() -> None:
    """Non-blank tick labels match the `"Category · ..."` shape.
    Trailing sub-rows render with a blank label so the visual grouping
    reads cleanly — those are excluded from the format check."""
    fig = render_timeline(_seq(_make_full_bundle()))
    yticks = _yticks(fig)
    assert yticks, "expected at least one band on a non-empty bundle"
    non_blank = [t for t in yticks if t]
    assert non_blank, "expected at least one labeled band"
    for label in non_blank:
        head, *_ = label.split(" · ", 1)
        # The category head must be one of the canonical labels.
        assert head in {"Environments", "Traffic Lights", "Objects", "Agents", "Ego"}, (
            f"unexpected category head in band label: {label!r}"
        )


def test_empty_bands_dropped() -> None:
    """A bundle with ONLY env entities produces a figure with bands
    from Env only; no Agent / Ego / Object / Light bands appear."""
    env = Environment(
        id="env_0",
        type="fst:Road",
        start_timestamp="0:0.0",
        end_timestamp="0:10.0",
    )
    ann = SilAvAnnotation(environments=[env])
    bundle = AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="env_only", duration_s=10.0),
        annotation=ann,
    )
    fig = render_timeline(_seq(bundle))
    yticks = _yticks(fig)
    # Exactly one band — the env parent row labeled by its track name —
    # because no other families are populated.
    assert yticks == ["Environments · Env Track 1"], yticks


def test_overlapping_subtracks_land_on_distinct_lanes_in_band() -> None:
    """Two overlapping agent actions paint on different lanes inside
    the `Agents · action` band, not bleed into other family bands."""
    fig = render_timeline(_seq(_make_two_overlapping_agent_actions_bundle()))
    action_shapes = [
        s for s in _segment_shapes(fig)
        if s.get("name", "").startswith("segment:agent_action_")
    ]
    bands = {(round(float(s["y0"]), 3), round(float(s["y1"]), 3)) for s in action_shapes}
    # Two overlapping actions → two distinct lane bands within the
    # `Agents · action` band.
    assert len(bands) == 2, f"expected 2 lanes, got {bands}"


def test_proportional_label_suppression() -> None:
    """A 1.0s segment in a 20s clip suppresses the inline label
    (1.0 / 20 = 0.05 < 0.06 threshold); the same segment in a 10s clip
    shows the inline label (1.0 / 10 = 0.10 > 0.06)."""

    def _bundle(duration: float) -> AnnotationBundle:
        act = AgentAction(
            id="agent_0_act_0",
            action_type="Yield",
            start_timestamp="0:0.0",
            end_timestamp="0:1.0",  # 1.0 s wide
        )
        agent = Agent(
            id="agent_0",
            type="oxd:Car",
            visibility_start_timestamp="0:0.0",
            visibility_end_timestamp="0:1.0",
            actions=[act],
        )
        return AnnotationBundle(
            schema_version="2.0.0",
            video=VideoMeta(clip_id=f"prop_{duration}", duration_s=duration),
            annotation=SilAvAnnotation(agents=[agent]),
        )

    fig_20 = render_timeline(_seq(_bundle(20.0)))
    fig_10 = render_timeline(_seq(_bundle(10.0)))

    def _action_label(fig: go.Figure) -> dict | None:
        for a in _annotations(fig):
            if a.get("name", "").startswith("label:agent_action_0_0_"):
                return a
        return None

    # 20s clip: 1.0 / 20 = 0.05 < 0.06 → suppressed.
    assert _action_label(fig_20) is None, (
        "1.0s segment in a 20s clip should be suppressed (5% < 6% threshold)"
    )
    # 10s clip: 1.0 / 10 = 0.10 > 0.06 → shown.
    assert _action_label(fig_10) is not None, (
        "1.0s segment in a 10s clip should be shown (10% > 6% threshold)"
    )


def test_show_inline_labels_false_suppresses_all() -> None:
    """`render_timeline(seq, show_inline_labels=False)` produces ZERO
    inline label annotations on the figure."""
    fig = render_timeline(_seq(_make_full_bundle()), show_inline_labels=False)
    labels = [
        a for a in _annotations(fig)
        if (a.get("name") or "").startswith("label:")
    ]
    assert labels == [], (
        f"expected 0 inline labels with show_inline_labels=False, got {len(labels)}"
    )


def test_arrows_target_new_band_y() -> None:
    """`because_of` arrows from agent_1's action to agent_0's parent
    anchor at the y-coordinates of the agent's action band and parent
    band respectively — NOT the legacy whole-Agents group center.

    The annotator resolves a `because_of: ["agent_0"]` reference to
    the parent agent entity (by id), so the arrow tail sits on the
    agent_1's action band and the arrowhead sits on agent_0's parent
    band. With family sub-rows those are two DIFFERENT band rows;
    the legacy renderer would have collapsed both to the same Agents
    row, which is exactly the visual regression the band layout
    fixes.
    """
    a0_act = AgentAction(
        id="agent_0_act_0",
        action_type="Yield",
        start_timestamp="0:1.0",
        end_timestamp="0:3.0",
    )
    a1_act = AgentAction(
        id="agent_1_act_0",
        action_type="oxd:Decelerate",
        because_of=["agent_0"],
        start_timestamp="0:4.0",
        end_timestamp="0:6.0",
    )
    a0 = Agent(
        id="agent_0",
        type="oxd:Car",
        visibility_start_timestamp="0:0.5",
        visibility_end_timestamp="0:3.5",
        actions=[a0_act],
    )
    # agent_1 on its own track (track_index=1) so the painter emits a
    # distinct per-entity block for it — that's the layout the arrow
    # test exercises. Passing the index via `model_validate` on a
    # dict with the schema's `_track_index` alias.
    a1 = Agent.model_validate(
        {
            "id": "agent_1",
            "type": "oxd:Car",
            "visibility_start_timestamp": "0:3.5",
            "visibility_end_timestamp": "0:6.5",
            "actions": [a1_act.model_dump()],
            "_track_index": 1,
        }
    )
    bundle = AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="arrow", duration_s=10.0),
        annotation=SilAvAnnotation(agents=[a0, a1]),
    )
    fig = render_timeline(_seq(bundle))
    yticks_text = _yticks(fig)
    yticks_vals = list(fig.layout.yaxis.tickvals or ())
    # agent_1's action lives in its own per-entity block (Agent Track
    # 2); agent_0's parent lives in Agent Track 1's parent row. The
    # arrow tail anchors at the source band; the head anchors at the
    # target band.
    src_idx = yticks_text.index("Agents · Agent Track 2 · actions")
    src_row = float(yticks_vals[src_idx])
    tgt_idx = yticks_text.index("Agents · Agent Track 1")
    tgt_row = float(yticks_vals[tgt_idx])

    arrows = _arrow_shapes(fig, "because_of")
    assert arrows, "expected at least one because_of arrow"
    # Parse the SVG bezier path: `M x0 y0 Q midX ctrlY x1 y1`.
    import re

    arrow_path = arrows[0]["path"]
    m = re.match(
        r"M\s+([-\d.]+)\s+([-\d.]+)\s+Q\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
        arrow_path,
    )
    assert m, f"could not parse bezier path: {arrow_path!r}"
    y_src = float(m.group(2))
    y_tgt = float(m.group(6))
    # Source (agent_1 action) anchors near agent_2's actions band row.
    assert abs(y_src - src_row) <= 0.4 + 1e-6, (
        f"arrow source y={y_src} not in source band "
        f"[{src_row - 0.4}, {src_row + 0.4}] (row={src_row})"
    )
    # Target (agent_0 parent) anchors near agent_1's parent band row.
    assert abs(y_tgt - tgt_row) <= 0.4 + 1e-6, (
        f"arrow target y={y_tgt} not in target band "
        f"[{tgt_row - 0.4}, {tgt_row + 0.4}] (row={tgt_row})"
    )
    # Sanity: source and target are distinct rows. If the per-entity
    # layout regressed and collapsed agent_0's and agent_1's blocks
    # onto the same row, this would silently pass.
    assert abs(src_row - tgt_row) >= 1.0, (
        "source and target bands collapsed onto the same row — "
        "per-entity layout broken"
    )
