# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `causal_ai_av.viz.timeline.render_timeline`.

Bundles are built from scratch — no corpus dependency. We poke at the
returned Plotly Figure's `layout.shapes` since the timeline is built
entirely from shapes (no traces) per the PR-2 design.
"""

from __future__ import annotations

import re

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


# Tick-label labels are wrapped in `<span style="color:#xxx">...</span>`
# so each tick carries its category-color cue. The helpers below pull
# the plain text out for `==` comparisons and pull the color hex out
# for category-anchored tests.
_LABEL_TEXT_RE = re.compile(r"<[^>]+>")
_LABEL_COLOR_RE = re.compile(r"color\s*:\s*(#[0-9a-fA-F]{6})")


def _label_text(tick_html: str) -> str:
    """Strip `<span ...>` wrappers AND trailing non-breaking spaces
    to recover plain tick text.

    Tick labels are left-aligned by padding the right edge with
    ` ` (non-breaking space, U+00A0) so the visible text starts
    at a consistent left column. For string comparisons (`.index(...)`
    etc.) we strip both the HTML wrapper and the trailing padding.
    """
    return _LABEL_TEXT_RE.sub("", tick_html).rstrip("  ")


def _label_color(tick_html: str) -> str | None:
    """Extract the `#RRGGBB` color from a tick label's HTML span,
    or `None` if the tick is blank / unwrapped."""
    m = _LABEL_COLOR_RE.search(tick_html)
    return m.group(1) if m else None


# Inverse of `entity_color()` — for tests that need to recover the
# category from a label's color. Kept in test land so the source
# module doesn't need to expose the reverse mapping.
_HEX_TO_CATEGORY: dict[str, str] = {
    "#22c55e": "Environments",
    "#ef4444": "Traffic Lights",
    "#f59e0b": "Objects",
    "#a855f7": "Agents",
    "#3b82f6": "Ego",
}


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

    With short-form tick labels, per-row → category lookup goes via
    the `entity_block:<Category>:<idx>` shapes (each band tick row
    falls inside the y-range of exactly one such block).
    """
    fig = render_timeline(_seq(_make_full_bundle()))
    seg_shapes = [s for s in _shapes(fig) if s.get("name", "").startswith("segment:")]
    assert seg_shapes, "expected at least one segment shape"
    # Collect every entity_block shape with its category + y-range.
    block_ranges: list[tuple[str, float, float]] = []
    for s in _shapes(fig):
        name = s.get("name") or ""
        if not name.startswith("entity_block:"):
            continue
        category = name.split(":")[1]
        block_ranges.append((category, float(s["y0"]), float(s["y1"])))
    categories_with_segments: set[str] = set()
    for s in seg_shapes:
        mid = (float(s["y0"]) + float(s["y1"])) / 2.0
        for category, lo, hi in block_ranges:
            if lo <= mid <= hi:
                categories_with_segments.add(category)
                break
    assert {"Environments", "Agents", "Ego"}.issubset(categories_with_segments), (
        f"missing categories, got {categories_with_segments}"
    )


def test_render_timeline_layout_is_light_with_locked_axes() -> None:
    fig = render_timeline(_seq(_make_full_bundle()))
    # Light template (`plotly_white`) — paper background is white,
    # which makes the figure suitable for Jupyter notebooks (light
    # bg by default) and paper figures.
    assert fig.layout.template.layout.paper_bgcolor is not None  # plotly_white
    # Adaptive height: the figure should never collapse below the
    # 240px floor, even for a sparse clip. The exact value scales
    # with the deepest sub-lane stack so we only assert the floor
    # here. `test_explicit_height_kwarg_overrides_adaptive` covers
    # the override knob, and `test_adaptive_height_grows_with_lanes`
    # pins the lane-count scaling.
    assert fig.layout.height >= 240
    # y-axis is one tick per populated band (Environments, Ego, Agents
    # families in this fixture). Traffic Lights / Objects contribute zero
    # bands because the fixture has no light / object entities. With
    # short-form tick labels we resolve category via the per-entity
    # block shapes, not by parsing the tick text.
    yticks = list(fig.layout.yaxis.ticktext or ())
    assert len(yticks) > 0, "expected at least one populated band"
    # Categories appear in canonical reading order (Environments first,
    # then Agents, then Ego — no Traffic Lights / Objects since the
    # fixture has none). Pull the order from the `entity_block:*` shape
    # names which preserve y-axis order.
    block_categories: list[str] = []
    for s in _shapes(fig):
        name = s.get("name") or ""
        if not name.startswith("entity_block:"):
            continue
        cat = name.split(":")[1]
        if not block_categories or block_categories[-1] != cat:
            block_categories.append(cat)
    assert block_categories[0] == "Environments"
    assert "Traffic Lights" not in block_categories
    assert "Objects" not in block_categories
    assert {"Environments", "Agents", "Ego"}.issubset(set(block_categories))
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
    # renders (xaxis range + plotly_white template are set).
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


def _category_y_range(fig: go.Figure, category: str) -> tuple[float, float] | None:
    """Return `(y_lo, y_hi)` covering every `entity_block:<category>:*`
    shape on the figure, or `None` if no such block exists.

    With short-form tick labels, the y-axis prefix can no longer be
    used to anchor "Agents rows" / "Ego rows" — the per-entity block
    shapes are the canonical signal for "which band rows belong to
    category X". This helper aggregates the block shapes' y-ranges
    into one bounding range per category for the filter-test set.
    """
    block_y_ranges = [
        (float(s["y0"]), float(s["y1"]))
        for s in _shapes(fig)
        if (s.get("name") or "").startswith(f"entity_block:{category}:")
    ]
    if not block_y_ranges:
        return None
    lo = min(y0 for y0, _ in block_y_ranges)
    hi = max(y1 for _, y1 in block_y_ranges)
    return lo, hi


def _agents_row_segment_shapes(fig: go.Figure) -> list[dict]:
    """All segment shapes whose vertical mid lies inside any Agents
    entity block (parent + containment + influence + action + property
    rows for every agent)."""
    y_range = _category_y_range(fig, "Agents")
    if y_range is None:
        return []
    lo, hi = y_range
    out = []
    for s in _segment_shapes(fig):
        mid = (float(s["y0"]) + float(s["y1"])) / 2.0
        if lo <= mid <= hi:
            out.append(s)
    return out


def _band_rows_for_category(fig: go.Figure, category: str) -> set[float]:
    """Return every band tickval whose row sits inside `category`'s
    entity blocks. With short-form tick labels, segment-row → category
    lookup goes through the entity_block shapes (which DO carry the
    category in their `name`)."""
    y_range = _category_y_range(fig, category)
    if y_range is None:
        return set()
    lo, hi = y_range
    yticks = list(fig.layout.yaxis.ticktext or ())
    tickvals = list(fig.layout.yaxis.tickvals or ())
    return {
        float(v) for v, _t in zip(tickvals, yticks, strict=False)
        if lo <= float(v) <= hi
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
    """Each arrowhead marker carries a 1.5px Tailwind slate-900
    (`#0f172a`) outline so the head pops off any same-hue target
    row on the `plotly_white` template. Family fill is preserved;
    only `marker.line` changes from the prior invisible
    `{"width": 0}`.
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
        assert line.color == "#0f172a", (
            f"arrowhead {head.name} has color={line.color!r}, expected '#0f172a'"
        )


# ---------------------------------------------------------------------------
# Filter kwargs
# ---------------------------------------------------------------------------


def test_entity_kinds_filter_keeps_only_requested_rows() -> None:
    """`entity_kinds=["ego"]` drops every non-Ego segment shape; the
    y-axis collapses to only Ego bands.

    With short-form tick labels, "is this an Ego row?" is decided by
    asking the per-entity background band (the `entity_block:Ego:0`
    shape) — the only block on the figure when filtered to Ego only.
    """
    fig = render_timeline(_seq(_make_full_bundle()), entity_kinds=["ego"])
    block_names = {
        s.get("name") for s in _shapes(fig)
        if (s.get("name") or "").startswith("entity_block:")
    }
    # Only the Ego block survives.
    assert block_names == {"entity_block:Ego:0"}, (
        f"non-Ego entity blocks leaked through: {block_names}"
    )
    # Every painted segment sits inside the Ego block.
    ego_range = _category_y_range(fig, "Ego")
    assert ego_range is not None
    lo, hi = ego_range
    for s in _segment_shapes(fig):
        mid = (float(s["y0"]) + float(s["y1"])) / 2.0
        assert lo <= mid <= hi, (
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
    (parent + containment + influence + action + property). The
    per-entity background band is the canonical "this row belongs to
    Agents" signal."""
    fig = render_timeline(_seq(_make_full_bundle()), track_groups=["Agents"])
    yticks = list(fig.layout.yaxis.ticktext or ())
    assert yticks, "expected at least one Agents band when fixture has agents"
    # Only Agents entity blocks survive.
    block_names = {
        s.get("name") for s in _shapes(fig)
        if (s.get("name") or "").startswith("entity_block:")
    }
    assert all(n.startswith("entity_block:Agents:") for n in block_names), (
        f"non-Agents blocks leaked through: {block_names}"
    )
    # And every painted segment sits inside an Agents block.
    agents_range = _category_y_range(fig, "Agents")
    assert agents_range is not None
    lo, hi = agents_range
    for s in _segment_shapes(fig):
        mid = (float(s["y0"]) + float(s["y1"])) / 2.0
        assert lo <= mid <= hi, f"shape outside Agents bands: {s}"


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
    """Return tick labels with the `<span>` color wrapper stripped.

    Tests that need the raw HTML (e.g. category-color anchoring) read
    `fig.layout.yaxis.ticktext` directly.
    """
    return [_label_text(t) for t in (fig.layout.yaxis.ticktext or ())]


def _yticks_raw(fig: go.Figure) -> list[str]:
    return list(fig.layout.yaxis.ticktext or ())


def test_paint_emits_band_per_populated_family() -> None:
    """The full-bundle fixture has Environments (parent + condition),
    Ego (containment + influence + action), and Agents (parent +
    containment + influence + action + property). Every populated
    family in those categories must show up as a band on the y-axis.

    Parent rows carry the entity track name; sub-rows carry just the
    family leaf (the visual grouping is handled by the per-entity
    background band painted in `_paint_timeline_onto`). Traffic Lights
    / Objects have no entities in the fixture, so they contribute zero
    bands."""
    fig = render_timeline(_seq(_make_full_bundle()))
    yticks = _yticks(fig)
    expected_present = {
        "Env Track 1",      # env parent
        "Agent Track 1",    # agent parent
        # Sub-row family heads — same leaf words show up in multiple
        # categories; the per-entity background band carries the
        # category cue.
        "conditions",
        "containment",
        "influences",
        "actions",
        "properties",
    }
    missing = expected_present - set(yticks)
    assert not missing, f"missing expected bands: {missing}; got {yticks}"


def test_band_y_axis_labels_use_short_form() -> None:
    """Non-blank tick labels are either:
      - a parent row `"<Kind> Track N"` (e.g. "Env Track 1"), or
      - a family leaf (e.g. "conditions", "actions"), or
      - a Lights per-SH head `"sh<i> · <leaf>"`.
    They MUST NOT carry the `"<Category> · "` prefix that the
    pre-PR-39 layout used — that prefix is now redundant with the
    per-entity background band."""
    fig = render_timeline(_seq(_make_full_bundle()))
    yticks = _yticks(fig)
    assert yticks, "expected at least one band on a non-empty bundle"
    non_blank = [t for t in yticks if t]
    assert non_blank, "expected at least one labeled band"
    forbidden_prefixes = (
        "Environments · ",
        "Traffic Lights · ",
        "Objects · ",
        "Agents · ",
        "Ego · ",
    )
    for label in non_blank:
        for prefix in forbidden_prefixes:
            assert not label.startswith(prefix), (
                f"label still carries redundant category prefix: {label!r}"
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
    # because no other families are populated. Short-form label: the
    # background band carries the category cue, so the tick is just
    # the entity track name.
    assert yticks == ["Env Track 1"], yticks


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
    # target band. The "actions" tick label appears twice (once per
    # agent block), so we anchor on the parent labels (which are
    # unique) and resolve the actions row by offset from there.
    a1_parent = yticks_text.index("Agent Track 2")
    # First non-blank label below the parent is the first family head
    # ("actions" — only family populated for these agents).
    src_idx = a1_parent + 1
    assert yticks_text[src_idx] == "actions", (
        f"expected 'actions' under Agent Track 2, got {yticks_text[src_idx]!r}"
    )
    src_row = float(yticks_vals[src_idx])
    tgt_idx = yticks_text.index("Agent Track 1")
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


# ---------------------------------------------------------------------------
# Per-entity bands (Category > Entity > Family > Sub-row)
# ---------------------------------------------------------------------------


def test_per_entity_bands() -> None:
    """A bundle with 3 agents on 3 distinct tracks produces 3 separate
    "Agent Track N" blocks on the y-axis, each with its own family
    sub-rows. The actions sub-row label repeats per block (one
    "actions" tick per agent); the parent rows carry the entity
    track name so the user can disambiguate blocks."""

    def _agent(idx: int) -> Agent:
        return Agent.model_validate(
            {
                "id": f"agent_{idx}",
                "type": "oxd:Car",
                "visibility_start_timestamp": "0:0.0",
                "visibility_end_timestamp": "0:5.0",
                "actions": [
                    AgentAction(
                        id=f"agent_{idx}_act_0",
                        action_type="Yield",
                        start_timestamp="0:1.0",
                        end_timestamp="0:2.0",
                    ).model_dump()
                ],
                "_track_index": idx,
            }
        )

    bundle = AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="per_entity", duration_s=10.0),
        annotation=SilAvAnnotation(agents=[_agent(0), _agent(1), _agent(2)]),
    )
    fig = render_timeline(_seq(bundle))
    yticks = _yticks(fig)
    # Three "Agent Track N" parents must show up.
    for n in (1, 2, 3):
        assert f"Agent Track {n}" in yticks, (
            f"missing parent for agent {n}; got {yticks}"
        )
    # And exactly three "actions" sub-rows (one per agent block).
    assert yticks.count("actions") == 3, (
        f"expected 3 'actions' sub-rows, got {yticks.count('actions')} in {yticks}"
    )


def test_singleton_ego_has_no_entity_label() -> None:
    """Ego is a singleton — its tick labels are bare leaves
    (`"containment"`, `"influences"`, `"actions"`, `"properties"`)
    with no per-track tag and no category prefix. The Ego background
    band carries the category cue.

    We anchor on the Ego entity_block shape's row range to find the
    Ego rows, then assert each tick label is a bare leaf.
    """
    fig = render_timeline(_seq(_make_full_bundle()))
    yticks = _yticks(fig)
    yvals = list(fig.layout.yaxis.tickvals or ())
    # Locate Ego rows via its block shape.
    ego_shape = next(
        (s for s in _shapes(fig) if s.get("name") == "entity_block:Ego:0"),
        None,
    )
    assert ego_shape is not None, "expected an Ego entity_block shape"
    y_lo = float(ego_shape["y0"])
    y_hi = float(ego_shape["y1"])
    ego_labels = [
        t for t, v in zip(yticks, yvals, strict=False)
        if y_lo <= float(v) <= y_hi and t
    ]
    assert ego_labels, "expected at least one labeled Ego sub-row"
    legal_leaves = {"containment", "influences", "actions", "properties"}
    for label in ego_labels:
        assert label in legal_leaves, (
            f"Ego sub-row label {label!r} not a known leaf"
        )


def test_multi_track_index_produces_multiple_sub_rows() -> None:
    """An Agent containing 3 containments at `_cont_track_index =
    0, 1, 2` produces 3 sub-rows under its containment family band:
    the first labeled "containment", the next two blank."""
    cont0 = Containment(
        id="cont0", env_id="env_0", lane_number="1",
        start_timestamp="0:0.0", end_timestamp="0:2.0",
    )
    cont1 = Containment.model_validate({
        "id": "cont1", "env_id": "env_0", "lane_number": "2",
        "start_timestamp": "0:0.0", "end_timestamp": "0:2.0",
        "_cont_track_index": 1,
    })
    cont2 = Containment.model_validate({
        "id": "cont2", "env_id": "env_0", "lane_number": "3",
        "start_timestamp": "0:0.0", "end_timestamp": "0:2.0",
        "_cont_track_index": 2,
    })
    env = Environment(
        id="env_0", type="fst:Road",
        start_timestamp="0:0.0", end_timestamp="0:10.0",
    )
    agent = Agent(
        id="agent_0", type="oxd:Car",
        visibility_start_timestamp="0:0.0", visibility_end_timestamp="0:5.0",
        containment=[cont0, cont1, cont2],
    )
    bundle = AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="sub_rows", duration_s=10.0),
        annotation=SilAvAnnotation(environments=[env], agents=[agent]),
    )
    fig = render_timeline(_seq(bundle))
    yticks = _yticks(fig)
    # Find the index of the first labeled "containment" row under
    # Agent Track 1 and verify the next two are blank — matches the
    # annotator's `i === 0 ? 'containment' : ''` labeling rule.
    parent_idx = yticks.index("Agent Track 1")
    cont_idx = parent_idx + 1
    assert yticks[cont_idx] == "containment", (
        f"expected 'containment' under Agent Track 1, got {yticks[cont_idx]!r}"
    )
    assert yticks[cont_idx + 1] == "", (
        f"expected blank trailing sub-row label, got {yticks[cont_idx + 1]!r}"
    )
    assert yticks[cont_idx + 2] == "", (
        f"expected blank trailing sub-row label, got {yticks[cont_idx + 2]!r}"
    )


def test_trailing_sub_row_labels_are_blank() -> None:
    """The first sub-row of each family carries the family leaf
    label; subsequent sub-rows render with `""`. This matches the
    annotator's behavior — the family name is shown ONCE per group.
    """
    cond0 = Condition(
        id="c0", env_id="env_0", type=["Construction Zone"],
        start_timestamp="0:0.0", end_timestamp="0:2.0",
    )
    cond1 = Condition.model_validate({
        "id": "c1", "env_id": "env_0", "type": ["Construction Zone"],
        "start_timestamp": "0:0.0", "end_timestamp": "0:2.0",
        "_cond_track_index": 1,
    })
    env = Environment(
        id="env_0", type="fst:Road",
        start_timestamp="0:0.0", end_timestamp="0:10.0",
    )
    bundle = AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="cond_sub_rows", duration_s=10.0),
        annotation=SilAvAnnotation(
            environments=[env], conditions=[cond0, cond1],
        ),
    )
    fig = render_timeline(_seq(bundle))
    yticks = _yticks(fig)
    # Two condition sub-rows under Env Track 1: first labeled
    # "conditions", second blank.
    parent_idx = yticks.index("Env Track 1")
    cond_idx = parent_idx + 1
    assert yticks[cond_idx] == "conditions", (
        f"expected 'conditions' under Env Track 1, got {yticks[cond_idx]!r}"
    )
    assert yticks[cond_idx + 1] == "", (
        f"trailing sub-row not blank: {yticks[cond_idx + 1]!r}"
    )


def test_segment_fill_uses_family_color() -> None:
    """Per-family bar colors come from `family_color(category, family)`,
    NOT the entity base color. A containment shape and an action shape
    on the same agent have distinct fills."""
    bundle = _make_full_bundle()
    fig = render_timeline(_seq(bundle))
    shapes_by_prefix: dict[str, str] = {}
    for s in _shapes(fig):
        name = s.get("name", "")
        if not name.startswith("segment:"):
            continue
        sid = name[len("segment:"):]
        for prefix in (
            "agent_action_", "agent_cont_", "agent_prop_", "agent_infl_",
            "ego_act_", "ego_cont_", "ego_infl_",
            "cond_",
        ):
            if sid.startswith(prefix):
                shapes_by_prefix.setdefault(prefix, s["fillcolor"])
                break

    # Agent containment vs. action — distinct hues per annotator's
    # subColor palette.
    assert shapes_by_prefix["agent_cont_"] == "#22c55e"
    assert shapes_by_prefix["agent_action_"] == "#c084fc"
    assert shapes_by_prefix["agent_prop_"] == "#d8b4fe"
    assert shapes_by_prefix["agent_infl_"] == "#f97316"
    # Ego subtracks have their own palette.
    assert shapes_by_prefix["ego_act_"] == "#3b82f6"
    assert shapes_by_prefix["ego_cont_"] == "#22c55e"
    assert shapes_by_prefix["ego_infl_"] == "#f97316"
    # Env condition.
    assert shapes_by_prefix["cond_"] == "#06b6d4"
    # Distinct fills are present on the figure.
    assert (
        shapes_by_prefix["agent_cont_"] != shapes_by_prefix["agent_action_"]
    )


def test_track_groups_filter_accepts_legacy_short_names() -> None:
    """The `track_groups` kwarg accepts BOTH the new category labels
    ("Environments", "Traffic Lights") and the PR-37 legacy short
    names ("Env", "Lights") for backward compat."""
    fig_new = render_timeline(_seq(_make_full_bundle()), track_groups=["Agents"])
    fig_legacy = render_timeline(_seq(_make_full_bundle()), track_groups=["Agents"])
    # Sanity baseline — same call, same output.
    assert _yticks(fig_new) == _yticks(fig_legacy)
    # Filter to Environments via the new name AND the legacy "Env"
    # alias; both must produce the same y-axis tick list.
    fig_envs_new = render_timeline(
        _seq(_make_full_bundle()), track_groups=["Environments"]
    )
    fig_envs_legacy = render_timeline(
        _seq(_make_full_bundle()), track_groups=["Env"]
    )
    assert _yticks(fig_envs_new) == _yticks(fig_envs_legacy)
    assert _yticks(fig_envs_new), "expected at least one Environments band"


def test_band_tick_text_matches_annotator_terse_style() -> None:
    """Tick labels use the annotator's terse leaf words: plural
    "conditions"/"actions"/"properties"/"influences"/"states", and
    short "TL control" / "signal head" / "containment" for the
    paren-disambiguated families."""
    from causal_ai_av.spec import (
        LightStates,
        ObjectStateEntry,
        SignalHead,
        TrafficLight,
        TrafficObject,
    )

    env = Environment(
        id="env_0", type="fst:Road",
        start_timestamp="0:0.0", end_timestamp="0:10.0",
    )
    cond = Condition(
        id="cond_0", env_id="env_0", type=["Construction Zone"],
        start_timestamp="0:1.0", end_timestamp="0:5.0",
    )
    obj = TrafficObject(
        id="obj_0", type="oxd:Cone",
        visibility_start_timestamp="0:0.0", visibility_end_timestamp="0:5.0",
        state_sequence=[
            ObjectStateEntry(
                id="obj_state_0", motion_state="Stationary",
                start_timestamp="0:0.0", end_timestamp="0:5.0",
            )
        ],
    )
    light = TrafficLight(
        id="light_0",
        visibility_start_timestamp="0:0.0", visibility_end_timestamp="0:5.0",
        containment=[
            Containment(
                id="light_phys_0", env_id="env_0",
                start_timestamp="0:0.0", end_timestamp="0:5.0",
            )
        ],
        signal_heads=[
            SignalHead(
                id="sh_0",
                start_timestamp="0:0.0", end_timestamp="0:5.0",
                state_sequence=[
                    LightStates(
                        id="ls_0", type="Steady", color="red",
                        start_timestamp="0:0.0", end_timestamp="0:3.0",
                    )
                ],
                env_controlled=[
                    Containment(
                        id="sh_cont_0", env_id="env_0",
                        start_timestamp="0:0.0", end_timestamp="0:3.0",
                    )
                ],
            ),
        ],
    )
    agent = Agent(
        id="agent_0", type="oxd:Car",
        visibility_start_timestamp="0:0.0", visibility_end_timestamp="0:5.0",
        actions=[AgentAction(
            id="agent_0_act_0", action_type="Yield",
            start_timestamp="0:1.0", end_timestamp="0:2.0",
        )],
        properties=[AgentProperty(
            id="agent_0_prop_0", property_type="Parked",
            start_timestamp="0:1.0", end_timestamp="0:2.0",
        )],
        influenced_by=[Influence(
            id="agent_0_infl_0", influencers=["env_0"],
            start_timestamp="0:1.0", end_timestamp="0:2.0",
        )],
    )
    ann = SilAvAnnotation(
        environments=[env], conditions=[cond],
        traffic_objects=[obj], traffic_lights=[light],
        agents=[agent],
    )
    bundle = AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="terse", duration_s=10.0),
        annotation=ann,
    )
    fig = render_timeline(_seq(bundle))
    yticks = set(_yticks(fig))
    # Spot check each terse leaf rewrite — short-form, no category /
    # entity prefix. The per-entity background band carries the
    # grouping cue, and the parent rows (e.g. "Light Track 1") sit
    # above each block's sub-rows.
    assert "Env Track 1" in yticks
    assert "Object Track 1" in yticks
    assert "Light Track 1" in yticks
    assert "conditions" in yticks
    assert "states" in yticks
    assert "TL control" in yticks
    # Lights per-signal-head bands keep the sh tag (a single Light
    # can host multiple signal heads, so it's the only in-block
    # disambiguator).
    assert "sh0 · signal head" in yticks
    assert "sh0 · containment" in yticks
    # `states` shows up for both Objects and Lights · sh0 — both
    # render as the bare leaf. Tick repetition is fine here; the
    # background band tells you which block each "states" belongs to.
    assert "actions" in yticks
    assert "properties" in yticks
    assert "influences" in yticks


# ---------------------------------------------------------------------------
# Per-entity background bands (this PR)
# ---------------------------------------------------------------------------


def _entity_block_shapes(fig: go.Figure) -> list[dict]:
    return [
        s for s in _shapes(fig)
        if (s.get("name") or "").startswith("entity_block:")
    ]


def test_entity_block_shape_per_entity() -> None:
    """One `entity_block:<Category>:<idx>` shape per `(category,
    entity_idx)` block. Full-bundle fixture has Environments[env_0],
    Agents[agent_0], and Ego — three blocks total."""
    fig = render_timeline(_seq(_make_full_bundle()))
    block_names = {
        s.get("name") for s in _entity_block_shapes(fig)
    }
    expected = {
        "entity_block:Environments:0",
        "entity_block:Agents:0",
        "entity_block:Ego:0",
    }
    assert block_names == expected, (
        f"expected exactly {expected} blocks, got {block_names}"
    )


def test_entity_block_spans_all_its_bands() -> None:
    """The Agent block's y-range covers the agent_0 parent row + every
    sub-row (containment / influence / action / property), and extends
    0.5 units past the first/last band so the lane band edges are
    fully enclosed.

    Strategy: enumerate every segment shape belonging to agent_0 and
    assert each one's y-mid sits inside the Agent block's y-range.
    """
    fig = render_timeline(_seq(_make_full_bundle()))
    block = next(
        s for s in _entity_block_shapes(fig)
        if s.get("name") == "entity_block:Agents:0"
    )
    y0 = float(block["y0"])
    y1 = float(block["y1"])
    # Every agent_0 segment rectangle must sit inside the block.
    agent_segments = [
        s for s in _segment_shapes(fig)
        if (s.get("name") or "").startswith("segment:agent_")
    ]
    assert agent_segments, "expected at least one agent segment in fixture"
    for s in agent_segments:
        y_mid = (float(s["y0"]) + float(s["y1"])) / 2.0
        assert y0 <= y_mid <= y1, (
            f"agent segment {s.get('name')!r} y_mid={y_mid} fell outside "
            f"Agent block [{y0}, {y1}]"
        )
    # The block extends exactly 0.5 past the outermost band rows (the
    # `_lane_band` half-height). Pick the outer envelope from the
    # agent segments themselves.
    agent_y_lo = min(float(s["y0"]) for s in agent_segments)
    agent_y_hi = max(float(s["y1"]) for s in agent_segments)
    # `_ROW_HALF_HEIGHT` = 0.4; the block's outer edge is 0.5 past the
    # band integer row, which is `_ROW_HALF_HEIGHT + 0.1` past any
    # segment edge. The exact distance from segment edge to block
    # edge varies with lane count, but the block must contain the
    # segment envelope with at least one band-half-height of margin.
    assert y0 <= agent_y_lo, (
        f"block y0={y0} above first agent segment y0={agent_y_lo}"
    )
    assert y1 >= agent_y_hi, (
        f"block y1={y1} below last agent segment y1={agent_y_hi}"
    )


def test_entity_block_layer_is_below() -> None:
    """Entity blocks paint with `layer="below"` so the highlight band,
    segment rectangles, and arrows draw on top."""
    fig = render_timeline(_seq(_make_full_bundle()))
    for s in _entity_block_shapes(fig):
        assert s.get("layer") == "below", (
            f"entity_block {s.get('name')!r} not on 'below' layer"
        )


def test_entity_block_tint_alternates_within_category() -> None:
    """Within a category, adjacent entity blocks use alternating
    opacities so the user can visually separate Agent Track 1 from
    Agent Track 2. The tint is a *neutral* white (not category-hued)
    — the category cue lives on the tick label color."""

    def _agent(idx: int) -> Agent:
        return Agent.model_validate(
            {
                "id": f"agent_{idx}",
                "type": "oxd:Car",
                "visibility_start_timestamp": "0:0.0",
                "visibility_end_timestamp": "0:5.0",
                "actions": [
                    AgentAction(
                        id=f"agent_{idx}_act_0",
                        action_type="Yield",
                        start_timestamp="0:1.0",
                        end_timestamp="0:2.0",
                    ).model_dump()
                ],
                "_track_index": idx,
            }
        )

    bundle = AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="alt_tint", duration_s=10.0),
        annotation=SilAvAnnotation(agents=[_agent(0), _agent(1)]),
    )
    fig = render_timeline(_seq(bundle))
    a0 = next(
        s for s in _entity_block_shapes(fig)
        if s.get("name") == "entity_block:Agents:0"
    )
    a1 = next(
        s for s in _entity_block_shapes(fig)
        if s.get("name") == "entity_block:Agents:1"
    )
    # Same neutral slate-900 base, alternating opacity → different
    # fillcolor strings. On the light `plotly_white` template, a
    # dark tint reads as a faint gray on white.
    assert a0["fillcolor"] != a1["fillcolor"], (
        "adjacent Agent blocks share the same fillcolor — "
        "alternating opacity broke"
    )
    # Tailwind slate-900 triple: (15, 23, 42).
    assert "15, 23, 42" in a0["fillcolor"] or "15,23,42" in a0["fillcolor"]
    assert "15, 23, 42" in a1["fillcolor"] or "15,23,42" in a1["fillcolor"]


def test_tick_label_per_row_alignment() -> None:
    """Per-row-type alignment: parent rows are centered (visible
    text sits in the middle of the longest-label column), sub-rows
    are left-aligned (visible text starts at the same left column
    as every other sub-row). Padding is U+00A0 (non-breaking space)
    so SVG renders the padding width without collapsing.

    Concretely, for a fixture whose longest tick text is "Agent
    Track 1" (13 chars):
      - Sub-rows ("conditions"=10, "actions"=7, etc.) pad with the
        full deficit → inner width = 13 for every sub-row.
      - Parent rows ("Env Track 1" = 11) pad with `(13 - 11) // 2 =
        1` trailing nbsp → inner width = 12, less than the sub-row
        width because the visible text now sits CENTERED inside
        the column.
    """
    fig = render_timeline(_seq(_make_full_bundle()))
    raw = _yticks_raw(fig)
    yticks = _yticks(fig)
    non_blank = [(plain, html) for plain, html in zip(yticks, raw, strict=False) if plain]
    assert non_blank, "expected at least one non-blank tick"

    parent_inner_lens: set[int] = set()
    subrow_inner_lens: set[int] = set()
    parents_seen = 0
    subrows_seen = 0
    nbsp = " "
    for plain, html in non_blank:
        inner = _LABEL_TEXT_RE.sub("", html)
        # Padding MUST be U+00A0, never a regular space (regular
        # spaces collapse in SVG text rendering).
        if len(inner) > len(plain):
            tail = inner[len(plain):]
            assert all(ch == nbsp for ch in tail), (
                f"tick {html!r} padded with non-nbsp whitespace"
            )
        is_parent = plain.endswith(" Track 1") or plain.endswith(" Track 2")
        if is_parent:
            parent_inner_lens.add(len(inner))
            parents_seen += 1
        else:
            subrow_inner_lens.add(len(inner))
            subrows_seen += 1

    assert subrows_seen > 0, "expected at least one sub-row tick"
    # Every sub-row shares the same width (left-aligned).
    assert len(subrow_inner_lens) == 1, (
        f"sub-row tick widths not equalized: {sorted(subrow_inner_lens)}"
    )
    if parents_seen > 0:
        # Parent rows are centered → pad with at most half the
        # deficit → inner width strictly less than the sub-row
        # width on the full fixture.
        max_sub = max(subrow_inner_lens)
        for w in parent_inner_lens:
            assert w <= max_sub, (
                f"parent inner width {w} > sub-row width {max_sub} — "
                "parent should pad with at most half the deficit"
            )


def test_tick_labels_wrap_text_in_category_colored_span() -> None:
    """Every non-blank tick label wraps its text in an HTML
    `<span style="color:#xxx">...</span>` whose color matches the
    category's `entity_color()` base hex. The color is the visual
    cue identifying which category a row belongs to."""
    fig = render_timeline(_seq(_make_full_bundle()))
    raw = _yticks_raw(fig)
    # Every non-blank tick carries a `<span style="color:#xxx">` wrap.
    non_blank = [t for t in raw if t]
    assert non_blank, "expected at least one non-blank tick label"
    for t in non_blank:
        color = _label_color(t)
        assert color is not None, f"tick {t!r} missing color wrapper"
        assert color in _HEX_TO_CATEGORY, (
            f"tick {t!r} has unknown category color {color!r}"
        )
    # Sanity check: each tick's color maps to a known category.
    seen_categories: set[str] = set()
    for t in non_blank:
        color = _label_color(t)
        assert color is not None  # narrowed above
        seen_categories.add(_HEX_TO_CATEGORY[color])
    # Full-bundle fixture exercises Environments, Agents, and Ego.
    assert {"Environments", "Agents", "Ego"}.issubset(seen_categories), (
        f"expected E/A/Ego categories represented in tick colors, got "
        f"{seen_categories}"
    )


def test_tick_label_color_matches_category_for_parent_and_sub_rows() -> None:
    """All ticks within one entity block share the same color —
    parent row AND every sub-row paint with the category hue. This
    is the load-bearing visual: the colored label is what tells the
    reader "this row belongs to Agents" once the band itself goes
    soft / neutral."""
    fig = render_timeline(_seq(_make_full_bundle()))
    raw = _yticks_raw(fig)
    yticks = [_label_text(t) for t in raw]
    # Find the Agent block's parent row + each sub-row leaf.
    parent_idx = yticks.index("Agent Track 1")
    parent_color = _label_color(raw[parent_idx])
    assert parent_color == "#a855f7", (
        f"Agent parent color {parent_color!r} ≠ #a855f7 (agent purple)"
    )
    # Every Agents sub-row leaf must share the same purple. Walk down
    # from the parent until the tick color changes — that's the
    # boundary where we cross into the next entity block. The label
    # text alone isn't a reliable end-of-block signal because Ego
    # has no parent row (its sub-rows lead the block), so a "next
    # parent" sentinel can't fire.
    agent_leaves_seen = 0
    for i in range(parent_idx + 1, len(yticks)):
        text = yticks[i]
        if not text:
            continue  # blank trailing sub-row inside this block
        color = _label_color(raw[i])
        if color != "#a855f7":
            break  # crossed into the next entity block
        agent_leaves_seen += 1
    # `_make_full_bundle` puts one containment, one influence, one
    # action, and one property on agent_0 — four sub-row family heads
    # all colored purple. (Trailing sub-rows are blank and skipped.)
    assert agent_leaves_seen >= 4, (
        f"expected ≥4 purple Agents sub-rows, saw {agent_leaves_seen}"
    )


def test_entity_block_skipped_when_category_filtered_out() -> None:
    """`track_groups=["Ego"]` produces only an Ego entity block — no
    Agent or Environment blocks appear."""
    fig = render_timeline(
        _seq(_make_full_bundle()), track_groups=["Ego"]
    )
    block_names = {s.get("name") for s in _entity_block_shapes(fig)}
    assert block_names == {"entity_block:Ego:0"}, (
        f"expected only Ego block, got {block_names}"
    )


def test_entity_block_spans_label_margin_via_paper_xref() -> None:
    """The block rect uses `xref="paper"` with a negative left edge
    so it walks INTO the y-tick-label margin, clustering the labels
    along with the bars. Mirrors the annotator's soft-background
    pattern: the entity tint reads behind both the row labels and
    the segment rectangles. Right edge sits at `x=1.0` (plot right
    edge in paper coords)."""
    fig = render_timeline(_seq(_make_full_bundle()))
    for s in _entity_block_shapes(fig):
        assert s.get("xref") == "paper", (
            f"block {s.get('name')!r} xref={s.get('xref')!r}, expected 'paper'"
        )
        assert float(s["x0"]) < 0.0, (
            f"block {s.get('name')!r} x0={s['x0']} must extend into the "
            f"y-tick-label margin (negative paper-x)"
        )
        assert abs(float(s["x1"]) - 1.0) < 1e-6, (
            f"block {s.get('name')!r} x1={s['x1']} ≠ 1.0 (plot right edge)"
        )


def test_entity_block_gap_between_adjacent_blocks() -> None:
    """Adjacent entity blocks leave a visible y-gap between their
    rects so the user reads the "this entity ended, next entity
    starts" cue — the annotator's "visual break" between entities.
    The gap exists by construction: each block's y-extent is inset
    to `±_ROW_HALF_HEIGHT` (0.4) instead of `±0.5`, so the dark
    template bg shows through a 0.2-row strip between blocks."""

    def _agent(idx: int) -> Agent:
        return Agent.model_validate(
            {
                "id": f"agent_{idx}",
                "type": "oxd:Car",
                "visibility_start_timestamp": "0:0.0",
                "visibility_end_timestamp": "0:5.0",
                "actions": [
                    AgentAction(
                        id=f"agent_{idx}_act_0",
                        action_type="Yield",
                        start_timestamp="0:1.0",
                        end_timestamp="0:2.0",
                    ).model_dump()
                ],
                "_track_index": idx,
            }
        )

    bundle = AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="gap", duration_s=10.0),
        annotation=SilAvAnnotation(agents=[_agent(0), _agent(1)]),
    )
    fig = render_timeline(_seq(bundle))
    blocks = _entity_block_shapes(fig)
    # Sort by y0 so adjacent-in-y blocks are adjacent in the list.
    blocks_by_y = sorted(blocks, key=lambda s: float(s["y0"]))
    for prev, nxt in zip(blocks_by_y, blocks_by_y[1:], strict=False):
        gap = float(nxt["y0"]) - float(prev["y1"])
        assert gap > 0.0, (
            f"no gap between {prev.get('name')!r} (y1={prev['y1']}) and "
            f"{nxt.get('name')!r} (y0={nxt['y0']})"
        )
