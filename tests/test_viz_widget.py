# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `causal_ai_av.viz.widget.ClipPlayer`.

No real video decode — `seq.video.decode_images_from_timestamps` is
replaced with a stub that returns a fixed-shape uint8 numpy array.
The bundle is built from scratch via the same Pydantic models the
timeline tests use, so the corpus is not required.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import pytest

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
from causal_ai_av.viz import ClipPlayer, render_timeline


# ---------------------------------------------------------------------------
# Fixtures — bundle + a Sequence whose `video` is mocked to a fake reader.
# ---------------------------------------------------------------------------

# `_FakeVideoReader` was lifted into `tests/conftest.py` so the PR-5
# carousel tests can reuse it. The leading-underscore alias is kept
# here for backwards compatibility with the existing test code below.
from tests.conftest import FakeVideoReader as _FakeVideoReader  # noqa: E402


def _make_full_bundle() -> AnnotationBundle:
    """Same fixture shape as test_viz_timeline._make_full_bundle.

    Has at least one of every arrow family populated so we can exercise
    propagation through `_paint_timeline_onto`.
    """
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
        link_to=["env_0"],
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
        influencers=["env_0"],
        start_timestamp="0:1.0",
        end_timestamp="0:3.0",
    )
    ego = EgoVehicle(
        actions=[ego_action], containment=[ego_cont], influenced_by=[ego_infl]
    )

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


def _seq_with_fake_video(bundle: AnnotationBundle | None = None) -> Sequence:
    bundle = bundle if bundle is not None else _make_full_bundle()
    seq = Sequence.from_annotation(bundle)
    # Sequence.video is a property that hits the parent dataset; with
    # no parent it raises. Inject a `_cameras` view that returns the
    # fake reader for the canonical camera. The simplest stub: monkey-
    # patch the `video` property at the instance level by binding a
    # descriptor-less attribute that bypasses the class property.
    reader = _FakeVideoReader()
    # `Sequence.video` is a property — assigning to `seq.video` on the
    # instance is blocked because the property has no setter. Replace
    # the underlying `_cameras` slot with a dict-like that satisfies
    # the property's contract instead.
    seq._cameras = {seq.annotation_camera: reader}  # type: ignore[assignment]
    return seq


# ---------------------------------------------------------------------------
# Helpers — shape inspection mirrors test_viz_timeline.
# ---------------------------------------------------------------------------


def _shapes(fig: go.Figure) -> list[dict]:
    return [dict(s.to_plotly_json()) for s in fig.layout.shapes]


def _bottom_subplot_shapes(fig: go.Figure) -> list[dict]:
    """Shapes whose axes reference the second (timeline) subplot."""
    return [s for s in _shapes(fig) if s.get("xref") == "x2"]


# ---------------------------------------------------------------------------
# 1. Smoke
# ---------------------------------------------------------------------------


def test_clipplayer_constructs_without_raising() -> None:
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq)
    assert isinstance(player._fig, go.FigureWidget)
    # The video Image trace is added before `_paint_timeline_onto`, so
    # it sits at `data[0]`; the hover-overlay scatter traces emitted by
    # the painter follow. `_apply_t` writes to `data[0]` and relies on
    # that invariant, so pin both.
    assert len(player._fig.data) >= 1
    assert player._fig.data[0].type == "image"


# ---------------------------------------------------------------------------
# 2. Default range
# ---------------------------------------------------------------------------


def test_default_t_start_and_t_end_track_sequence_duration() -> None:
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq)
    assert player._t_start == 0.0
    assert player._t_end == seq.duration_s
    assert player.t == 0.0


# ---------------------------------------------------------------------------
# 3. Custom range
# ---------------------------------------------------------------------------


def test_custom_t_start_t_end_pin_slider_range() -> None:
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq, t_start=2.0, t_end=5.0)
    assert player._slider.min == 2.0
    assert player._slider.max == 5.0
    assert player.t == 2.0


# ---------------------------------------------------------------------------
# 4. Seek
# ---------------------------------------------------------------------------


def test_seek_updates_t_and_decodes_at_expected_timestamp() -> None:
    seq = _seq_with_fake_video()
    reader: _FakeVideoReader = seq._cameras[seq.annotation_camera]  # type: ignore[assignment]
    player = ClipPlayer(seq)
    # Construction itself decodes one frame at t_start.
    construction_calls = len(reader.calls)
    player.seek(3.5)
    assert player.t == 3.5
    assert len(reader.calls) > construction_calls
    # The most recent decode was for t=3.5s → 3_500_000 microseconds.
    assert reader.calls[-1].tolist() == [3_500_000]


# ---------------------------------------------------------------------------
# 5. Out-of-range seek
# ---------------------------------------------------------------------------


def test_seek_outside_window_raises() -> None:
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq, t_start=1.0, t_end=4.0)
    with pytest.raises(ValueError):
        player.seek(0.5)
    with pytest.raises(ValueError):
        player.seek(4.5)


# ---------------------------------------------------------------------------
# 6. Highlight propagation
# ---------------------------------------------------------------------------


def test_highlight_kwarg_paints_a_band_on_the_timeline_subplot() -> None:
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq, highlight=(2.0, 5.0))
    bands = [
        s
        for s in _bottom_subplot_shapes(player._fig)
        if s.get("name") == "highlight"
    ]
    assert len(bands) == 1
    band = bands[0]
    assert float(band["x0"]) == 2.0
    assert float(band["x1"]) == 5.0


# ---------------------------------------------------------------------------
# 7. Arrow toggle propagation
# ---------------------------------------------------------------------------


def test_arrows_toggle_propagates_into_widget_subplot() -> None:
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq, arrows={"because_of": False})
    because_of = [
        s
        for s in _bottom_subplot_shapes(player._fig)
        if s.get("name", "").startswith("arrow:because_of:")
    ]
    assert because_of == []
    # Other families still present.
    link_to = [
        s
        for s in _bottom_subplot_shapes(player._fig)
        if s.get("name", "").startswith("arrow:link_to:")
    ]
    assert len(link_to) > 0


# ---------------------------------------------------------------------------
# 8. Timeline-painter parity
# ---------------------------------------------------------------------------


def test_widget_timeline_shape_count_matches_render_timeline_plus_playhead() -> None:
    """Pin the factoring: the bottom subplot's shape count equals the
    static `render_timeline` shape count, minus the playhead line that
    only the widget adds."""
    seq = _seq_with_fake_video()
    static = render_timeline(seq)
    player = ClipPlayer(seq)
    static_count = len(_shapes(static))
    # All shapes on the widget that reference the timeline subplot
    # (`x2` / `y2`) — that's the timeline shapes + the playhead line.
    bottom = _bottom_subplot_shapes(player._fig)
    assert len(bottom) == static_count + 1
    # And exactly one of those is the playhead.
    playheads = [s for s in bottom if s.get("name") == "playhead"]
    assert len(playheads) == 1


# ---------------------------------------------------------------------------
# 9. `_repr_mimebundle_` returns both keys
# ---------------------------------------------------------------------------


def test_repr_mimebundle_has_html_and_widget_view_keys() -> None:
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq)
    bundle: dict[str, Any] = player._repr_mimebundle_()
    assert "text/html" in bundle
    assert "application/vnd.jupyter.widget-view+json" in bundle
    # HTML snapshot should be a non-empty string.
    assert isinstance(bundle["text/html"], str)
    assert len(bundle["text/html"]) > 0
    # widget-view JSON must include the model_id so Jupyter can find
    # the live widget.
    widget_view = bundle["application/vnd.jupyter.widget-view+json"]
    assert isinstance(widget_view, dict)
    assert "model_id" in widget_view
