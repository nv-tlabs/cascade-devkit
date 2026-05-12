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
    assert fig.layout.height == 300
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
