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
    """Each of the five groups (Env, Lights, Objects, Agents, Ego) gets at
    least one segment shape when its family is populated.

    The fixture has at least one Env, one Agent, and one Ego — those are
    the three required groups.
    """
    fig = render_timeline(_seq(_make_full_bundle()))
    shapes = _shapes(fig)
    assert len(shapes) > 0
    # Group each segment shape by the row it sits on.
    rows_with_segments = set()
    for s in shapes:
        if not s.get("name", "").startswith("segment:"):
            continue
        # The y0/y1 pair encodes the row index (row ± 0.4).
        y0 = float(s["y0"])
        y1 = float(s["y1"])
        row = round((y0 + y1) / 2)
        rows_with_segments.add(row)
    # Env = 0, Agents = 3, Ego = 4 (per the _TRACK_GROUPS ordering).
    assert 0 in rows_with_segments  # Env
    assert 3 in rows_with_segments  # Agents
    assert 4 in rows_with_segments  # Ego


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
    # y-axis ticks are the five group labels, top → bottom.
    yticks = list(fig.layout.yaxis.ticktext or ())
    assert yticks == ["Env", "Lights", "Objects", "Agents", "Ego"]
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
    # Axis layout is still set up — `yticks` are still the five group labels.
    yticks = list(fig.layout.yaxis.ticktext or ())
    assert yticks == ["Env", "Lights", "Objects", "Agents", "Ego"]


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
    """All segment shapes whose vertical mid lies inside the Agents row."""
    out = []
    for s in _segment_shapes(fig):
        mid = (float(s["y0"]) + float(s["y1"])) / 2.0
        if 2.5 <= mid <= 3.5:  # Agents row is index 3.
            out.append(s)
    return out


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
    """Two overlapping agent actions paint on different lanes within
    the Agents row — count the distinct (y0, y1) bands."""
    fig = render_timeline(_seq(_make_two_overlapping_agent_actions_bundle()))
    agent_shapes = _agents_row_segment_shapes(fig)
    bands = {(round(float(s["y0"]), 3), round(float(s["y1"]), 3)) for s in agent_shapes}
    # Parent agent + two overlapping actions → at least 2 distinct bands.
    assert len(bands) >= 2, f"expected multi-lane stack, got {bands}"


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
    # Parent agent visibility spans [0.5, 8.0] which overlaps both
    # actions, so the parent + at least one action must share a lane
    # neighbour. But the two ACTIONS themselves should not need to
    # occupy distinct lanes — they don't overlap each other.
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


# ---------------------------------------------------------------------------
# Filter kwargs
# ---------------------------------------------------------------------------


def test_entity_kinds_filter_keeps_only_requested_rows() -> None:
    """`entity_kinds=["ego"]` drops every non-Ego segment shape."""
    fig = render_timeline(_seq(_make_full_bundle()), entity_kinds=["ego"])
    for s in _segment_shapes(fig):
        mid = (float(s["y0"]) + float(s["y1"])) / 2.0
        # Ego row is index 4.
        assert 3.5 <= mid <= 4.5, f"non-Ego shape leaked through filter: {s}"


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
    """`track_groups=["Agents"]` produces a y-axis with one tick label."""
    fig = render_timeline(_seq(_make_full_bundle()), track_groups=["Agents"])
    yticks = list(fig.layout.yaxis.ticktext or ())
    assert yticks == ["Agents"]
    # And every painted segment sits inside the Agents band.
    for s in _segment_shapes(fig):
        mid = (float(s["y0"]) + float(s["y1"])) / 2.0
        assert 2.5 <= mid <= 3.5


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
