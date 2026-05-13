# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `cascade_av.viz.widget.ClipPlayer`.

No real video decode — `seq.video.decode_images_from_timestamps` is
replaced with a stub that returns a fixed-shape uint8 numpy array.
The bundle is built from scratch via the same Pydantic models the
timeline tests use, so the corpus is not required.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import pytest

from cascade_av.dataset import Sequence
from cascade_av.spec import (
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
from cascade_av.viz import ClipPlayer, render_timeline


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
    """Shapes whose axes reference the second (timeline) subplot.

    Identified by yref="y2" (the bottom subplot's y-axis) rather
    than xref, because entity-block bands use `xref="paper"` so they
    can reach into the y-tick-label margin while still being
    pinned to the timeline subplot's y-axis.
    """
    return [s for s in _shapes(fig) if s.get("yref") == "y2"]


# ---------------------------------------------------------------------------
# 1. Smoke
# ---------------------------------------------------------------------------


def test_clipplayer_constructs_without_raising() -> None:
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq)
    assert isinstance(player._fig, go.FigureWidget)
    # Since PR-44 the video frame lives at `layout.images[0]` (was
    # `data[0]` — a `go.Image` trace bound to the top subplot). The
    # layout-image approach lets the video reach into the y-tick-label
    # margin (paper xref allows negative paper-x). `_apply_t` mutates
    # the layout image's `source` per tick.
    assert player._fig.layout.images, "expected video as layout.images[0]"
    assert player._fig.layout.images[0].source, (
        "layout.images[0].source must be a JPEG data URI"
    )


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


# ---------------------------------------------------------------------------
# 10. Filter kwarg propagation
# ---------------------------------------------------------------------------


def test_filter_kwargs_forwarded_to_timeline_subplot() -> None:
    """`ClipPlayer(seq, entity_kinds=["agent"])` paints only Agent-band
    segment shapes on the timeline subplot. The y-axis collapses to
    just Agents bands.

    With short-form tick labels (PR-39), the canonical signal for
    "this row belongs to Agents" is the per-entity background block
    shape (`entity_block:Agents:<idx>`). Anchor on those to find
    the y-range every Agents band occupies on the timeline subplot.
    """
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq, entity_kinds=["agent"])
    bottom_shapes = _bottom_subplot_shapes(player._fig)
    agent_block_ranges = [
        (float(s["y0"]), float(s["y1"]))
        for s in bottom_shapes
        if (s.get("name") or "").startswith("entity_block:Agents:")
    ]
    assert agent_block_ranges, (
        f"expected at least one Agents entity_block on the bottom subplot; "
        f"got {[s.get('name') for s in bottom_shapes]!r}"
    )
    lo = min(y0 for y0, _ in agent_block_ranges)
    hi = max(y1 for _, y1 in agent_block_ranges)
    seg_shapes = [
        s for s in bottom_shapes
        if s.get("name", "").startswith("segment:")
    ]
    assert seg_shapes, "expected at least one Agents-band segment shape"
    for s in seg_shapes:
        mid = (float(s["y0"]) + float(s["y1"])) / 2.0
        assert lo <= mid <= hi, (
            f"non-Agent shape leaked through filter: {s}"
        )


def test_families_kwarg_forwarded_to_timeline_subplot() -> None:
    """`ClipPlayer(seq, families=["action"])` forwards the whitelist to
    `_paint_timeline_onto`. The bottom subplot's segment shapes survive
    only inside the Agents entity-block band (the fixture's Agent block
    has an `action` sub-row); the Env block drops entirely because its
    only family is `condition`.
    """
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq, families=["action"])
    bottom_shapes = _bottom_subplot_shapes(player._fig)
    block_names = {
        s.get("name") for s in bottom_shapes
        if (s.get("name") or "").startswith("entity_block:")
    }
    assert "entity_block:Environments:0" not in block_names, (
        f"Env block should drop when only `action` family is allowed; "
        f"got blocks {block_names}"
    )
    assert any(
        (n or "").startswith("entity_block:Agents:") for n in block_names
    ), (
        f"Agents entity block should survive (it has an action sub-row); "
        f"got blocks {block_names}"
    )


# ---------------------------------------------------------------------------
# 11. Adaptive height — grows with deepest sub-lane stack
# ---------------------------------------------------------------------------


def _bundle_with_n_overlapping_agent_actions(n: int) -> AnnotationBundle:
    """Construct a bundle whose Agents row stacks `n` overlapping actions
    on a single agent track. Adds the parent agent on top of the actions
    so the resulting lane count is `n + 1`."""
    actions = [
        AgentAction(
            id=f"agent_0_act_{i}",
            action_type="Yield",
            # All overlap on [1.0, 5.0].
            start_timestamp="0:1.0",
            end_timestamp="0:5.0",
        )
        for i in range(n)
    ]
    agent = Agent(
        id="agent_0",
        type="oxd:Car",
        visibility_start_timestamp="0:0.0",
        visibility_end_timestamp="0:10.0",
        actions=actions,
    )
    return AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="adaptive_lanes", duration_s=10.0),
        annotation=SilAvAnnotation(agents=[agent]),
    )


def test_adaptive_height_grows_with_lanes() -> None:
    """A bundle with N stacked agent-action lanes produces a figure
    whose total height grows with the deepest stack. Pins the
    24-px-per-lane scaling so any regression to a fixed-height
    layout is caught.

    Uses `render_timeline` (headless) so the comparison isn't
    masked by the widget's fixed-pixel video pane / chrome. The
    widget's height tracks the same lane budget through the
    shared `_timeline_px_for` helper.

    With the family-band layout, the lane budget is per band rather
    than per group — so a busy clip with N overlapping agent actions
    grows the `Agents · action` band's lane count to N, while the
    other (sparse) bands stay at 1 lane each. The total lane count
    still scales linearly with N.
    """
    from cascade_av.viz.timeline import _PX_PER_LANE

    # Baseline: 1 agent parent + 1 agent action = 2 bands × 1 lane =
    # 2 lanes. Clamps to the 240px floor.
    baseline_fig = render_timeline(
        _seq_with_fake_video(_bundle_with_n_overlapping_agent_actions(0))
    )
    baseline_height = int(baseline_fig.layout.height)

    # 16 overlapping actions: Agents · parent band (1 lane) + Agents
    # · action band (16 lanes) = 17 total lanes, well above the floor.
    busy_fig = render_timeline(
        _seq_with_fake_video(_bundle_with_n_overlapping_agent_actions(16))
    )
    busy_height = int(busy_fig.layout.height)

    # At least 4 lanes' worth of growth (24 * 4 = 96 px). Robust to
    # small floor / rounding interactions, but tight enough to catch
    # a regression to fixed-height layout.
    assert busy_height >= baseline_height + 4 * _PX_PER_LANE, (
        f"adaptive height regression: baseline={baseline_height} "
        f"busy={busy_height} (expected delta >= {4 * _PX_PER_LANE})"
    )


def test_explicit_height_kwarg_overrides_adaptive() -> None:
    """Passing `height=N` to `ClipPlayer` pins `fig.layout.height` to
    exactly N, regardless of the adaptive computation."""
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq, height=900)
    assert int(player._fig.layout.height) == 900


def test_render_timeline_explicit_height_kwarg_overrides_adaptive() -> None:
    """`render_timeline(seq, height=N)` pins `fig.layout.height` to N."""
    seq = _seq_with_fake_video()
    fig = render_timeline(seq, height=600)
    assert int(fig.layout.height) == 600


# ---------------------------------------------------------------------------
# 12. Play tick decodes the frame exactly once
# ---------------------------------------------------------------------------


def test_play_tick_decodes_once() -> None:
    """A slider value change fires exactly ONE
    `decode_images_from_timestamps` call. The prior `_on_play_change`
    duplicated the apply path (Play → slider → _apply_t, then Play
    → _apply_t directly), causing every Play tick to decode the
    same frame twice. The fix drops the second call.

    Mirror the Play observer's event flow rather than poking the
    slider directly, so this guards the actual path the bug lived
    on.
    """
    seq = _seq_with_fake_video()
    reader: _FakeVideoReader = seq._cameras[seq.annotation_camera]  # type: ignore[assignment]
    player = ClipPlayer(seq)

    # Drain the construction decode so we count only Play-tick
    # decodes.
    reader.calls.clear()

    # Synthesize the event ipywidgets would dispatch when Play
    # advances to t=1.5s (Play tracks integer ms internally).
    new_value_ms = 1_500
    player._on_play_change({"new": new_value_ms})

    # The slider observer chain should have decoded the frame at
    # t=1.5s exactly once. Two decodes would mean the duplicate
    # `_apply_t` call has been re-introduced.
    assert len(reader.calls) == 1, (
        f"expected 1 decode per Play tick, got {len(reader.calls)}"
    )
    assert reader.calls[0].tolist() == [1_500_000]
    # And the slider mirror landed.
    assert player._slider.value == 1.5


# ---------------------------------------------------------------------------
# 13. Frame transport — JPEG data-URI via `source=` (PR #32 / fix/viz-frame-transport-jpeg)
# ---------------------------------------------------------------------------


def _decode_jpeg_data_uri_to_array(data_uri: str):
    """Round-trip a `data:image/jpeg;base64,...` URI back to a numpy array."""
    import base64
    import io

    import numpy as np
    from PIL import Image

    assert data_uri.startswith("data:image/jpeg;base64,"), data_uri[:40]
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw))
    return np.asarray(img)


def test_frame_transport_is_base64_jpeg() -> None:
    """The layout-image carries a base64 JPEG data URI via its
    `source` attribute. PR-44 moved the video from a `go.Image`
    TRACE at `data[0]` to a `go.layout.Image` at
    `layout.images[0]`; PR-33's JPEG transport is otherwise
    untouched (encoding still happens once per tick into a base64
    data URI, keeping the per-tick Comm payload at ~1.8MB).
    """
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq)
    src = player._fig.layout.images[0].source
    assert isinstance(src, str)
    assert src.startswith("data:image/jpeg;base64,"), src[:40]


class _NoisyFakeVideoReader:
    """FakeVideoReader variant whose frames are random noise.

    The vanilla `FakeVideoReader` returns all-zero frames with a single
    non-zero pixel — that pattern compresses to identical JPEG bytes at
    quality 20 and quality 90, so we cannot tell the kwarg apart. A
    noise-filled frame has enough high-frequency content that the
    rate / distortion knob actually moves the encoded size.
    """

    def __init__(self, height: int = 240, width: int = 320, seed: int = 0) -> None:
        import numpy as np

        self.height = height
        self.width = width
        self.calls: list = []
        rng = np.random.default_rng(seed)
        self._frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    def decode_images_from_timestamps(self, t_us):  # type: ignore[no-untyped-def]
        import numpy as np

        self.calls.append(np.asarray(t_us).copy())
        n = len(t_us)
        # Stack the same noise frame N times — quality is what we're
        # measuring, not per-call uniqueness.
        frames = np.broadcast_to(self._frame, (n, self.height, self.width, 3)).copy()
        return frames, np.asarray(t_us, dtype=np.int64)


def test_frame_quality_kwarg_changes_payload_size() -> None:
    """Higher `frame_quality` produces a strictly longer data URI than
    a much lower one. Loose bound (>=2x) so encoder-version drift
    doesn't flake — but tight enough to catch a regression that wires
    the kwarg to a no-op.
    """
    seq = _seq_with_fake_video()
    # JPEG-quality is a rate/distortion knob: the all-zero frames from
    # the default FakeVideoReader compress to the same bytes at every
    # quality, so swap in a noisy reader before constructing the players.
    seq._cameras[seq.annotation_camera] = _NoisyFakeVideoReader()  # type: ignore[index]
    low = ClipPlayer(seq, frame_quality=20)
    high = ClipPlayer(seq, frame_quality=90)
    low_len = len(low._fig.layout.images[0].source)
    high_len = len(high._fig.layout.images[0].source)
    assert high_len >= 2 * low_len, (
        f"frame_quality=90 should produce a much larger payload than "
        f"frame_quality=20: got high={high_len} low={low_len}"
    )


def test_frame_max_dim_kwarg_downscales() -> None:
    """`frame_max_dim=N` caps the longer side of the frame at `N` pixels
    before JPEG encoding. Decode the resulting data URI back to a numpy
    array and assert the cap held.
    """
    seq = _seq_with_fake_video()
    # Replace the fake reader with one whose frame is comfortably above
    # 320px so the downscale path actually runs (default fake reader is
    # 8x12).
    seq._cameras[seq.annotation_camera] = _FakeVideoReader(  # type: ignore[index]
        height=720, width=1280
    )
    player = ClipPlayer(seq, frame_max_dim=320)
    arr = _decode_jpeg_data_uri_to_array(player._fig.layout.images[0].source)
    h, w = arr.shape[:2]
    assert max(h, w) == 320, (
        f"expected max(H, W) == 320 after frame_max_dim cap, got {(h, w)}"
    )


def test_clipplayer_forwards_show_inline_labels() -> None:
    """`ClipPlayer(seq, show_inline_labels=False)` produces ZERO inline
    label annotations on the timeline subplot."""
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq, show_inline_labels=False)
    label_annotations = [
        a for a in (player._fig.layout.annotations or ())
        if (a.name or "").startswith("label:")
    ]
    assert label_annotations == [], (
        f"expected 0 inline labels with show_inline_labels=False, "
        f"got {len(label_annotations)}"
    )


def test_clipplayer_playhead_spans_full_band_range() -> None:
    """The playhead line spans the painter's computed y-range, NOT
    the legacy `[-0.5, len(_TRACK_GROUPS)-0.5]` constant. With the
    family-band layout the band count varies per bundle — a bundle
    with N bands gives the playhead a `[-0.5, N - 0.5]` range."""
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq)
    # Find the playhead shape.
    playhead = player._fig.layout.shapes[player._playhead_index]
    # Compute expected band count from a parallel render_timeline call.
    fig = render_timeline(seq)
    yticks = list(fig.layout.yaxis.ticktext or ())
    expected_n_bands = len(yticks)
    assert expected_n_bands > 0, "fixture must produce at least one band"
    # Playhead y0 / y1 — `_paint_timeline_onto` reports `(y_min, y_max)`,
    # widget assigns `y0 = y_min, y1 = y_max`.
    assert float(playhead.y0) == -0.5
    assert float(playhead.y1) == expected_n_bands - 0.5


def test_play_tick_updates_layout_image_source() -> None:
    """A Play tick swaps the layout-image's `source` data URI on
    `layout.images[0]`. Pins the JPEG transport (data URI prefix +
    mutation per tick) AND the surgical playhead invariant from
    PR #30. PR-44 moved the video from `data[0]` (trace) to
    `layout.images[0]` (layout-level image with paper-coord
    positioning).
    """
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq)

    # Capture the construction-time data URI + playhead handle so we
    # can detect mutation.
    before_source = player._fig.layout.images[0].source
    before_playhead_x = float(player._fig.layout.shapes[player._playhead_index].x0)

    # Synthesize a Play tick that drives the slider to t=1.5s; the
    # slider observer calls `_apply_t`.
    player._on_play_change({"new": 1_500})

    after_source = player._fig.layout.images[0].source
    after_playhead_x = float(player._fig.layout.shapes[player._playhead_index].x0)

    assert isinstance(after_source, str)
    assert after_source.startswith("data:image/jpeg;base64,")
    assert after_source != before_source, (
        "`_apply_t` did not update `layout.images[0].source` on tick"
    )
    # Surgical-playhead invariant: the shape's x coordinates moved with
    # the tick, no full layout rebuild.
    assert after_playhead_x != before_playhead_x
    assert abs(after_playhead_x - 1.5) < 1e-9


def test_layout_image_sizing_is_stretch() -> None:
    """The video layout-image uses `sizing="stretch"` — fills the box
    exactly with no internal letterboxing. PR-44 used `"contain"` and
    that recreated a visible gap between the video bottom and the
    timeline top whenever the box's aspect didn't match the source
    16:9 (which is most cell widths).
    """
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq)
    img = player._fig.layout.images[0]
    assert img.sizing == "stretch", (
        f"layout image sizing={img.sizing!r}, expected 'stretch' "
        f"(any other value reintroduces the letterbox gap)"
    )


def test_default_fig_width_pinned() -> None:
    """`fig.layout.width` is explicitly pinned to a wide default so
    the FigureWidget renders at a useful size. `autosize=True` +
    `ipywidgets.Layout(width="100%")` alone don't reliably expand
    a FigureWidget past Plotly's ~700px first-render default.
    """
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq)
    width = player._fig.layout.width
    assert width is not None and width >= 1200, (
        f"fig.layout.width={width!r} — needs to be pinned to a wide "
        f"default so the widget actually expands in Jupyter cells"
    )


def test_width_kwarg_overrides_default() -> None:
    """Callers can pass `width=N` to pin a specific figure width."""
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq, width=900)
    assert player._fig.layout.width == 900


def test_yaxis_tickfont_is_monospace() -> None:
    """The timeline's y-axis tick font family is a monospace
    fontstack so the nbsp-based left-alignment in `_band_label_html`
    produces visually aligned columns. With Plotly's default
    proportional font, two labels with the same char count have
    different pixel widths → they don't align even though the math
    pads to equal char counts.
    """
    seq = _seq_with_fake_video()
    player = ClipPlayer(seq)
    family = (player._fig.layout.yaxis2.tickfont.family or "").lower()
    # Accept any monospace-coded font name in the stack.
    assert any(
        kw in family
        for kw in ("monospace", "menlo", "consolas", "sfmono", "mono")
    ), (
        f"yaxis2.tickfont.family={family!r} is not monospace; "
        f"nbsp padding will not produce aligned label columns"
    )
