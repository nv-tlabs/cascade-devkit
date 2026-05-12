# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `causal_ai_av.viz` — segments port, entity colors, and the
headless `render_frame` Pillow path.

Bundles are built from scratch — no corpus dependency.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

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
    SilAvAnnotation,
    VideoMeta,
)
from causal_ai_av.viz import (
    Segment,
    annotation_to_segments,
    entity_color,
    render_frame,
)
from causal_ai_av.viz.segments import _parse_ts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bundle() -> AnnotationBundle:
    """A small but representative bundle covering the families PR-1 ports.

    Includes: one environment, one condition, one ego action (illegal!), one
    agent with one action labeled "Yield" + an illegal flag, one influence,
    one property, ego containment.
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
    ego = EgoVehicle(actions=[ego_action], containment=[ego_cont])

    agent_action = AgentAction(
        id="agent_0_act_0",
        action_type="Yield",
        illegal_flag=True,
        because_of=["ego"],
        start_timestamp="0:1.5",
        end_timestamp="0:3.0",
    )
    agent_prop = AgentProperty(
        id="agent_0_prop_0",
        property_type="Parked",
        start_timestamp="0:1.0",
        end_timestamp="0:3.0",
    )
    agent_infl = Influence(
        id="agent_0_infl_0",
        influencers=["light_0"],
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


# ---------------------------------------------------------------------------
# _parse_ts — TS parity
# ---------------------------------------------------------------------------


def test_parse_ts_matches_ts_parseTs() -> None:
    # Happy path: M:S.D → seconds.
    assert _parse_ts("0:5.4") == 5.4
    assert _parse_ts("1:30.0") == 90.0
    # The TS port returns 0.0 (not None) on empty / unparseable strings.
    assert _parse_ts("") == 0.0
    assert _parse_ts(None) == 0.0  # type: ignore[arg-type]
    assert _parse_ts("garbage") == 0.0


# ---------------------------------------------------------------------------
# annotation_to_segments — coverage of the ported families
# ---------------------------------------------------------------------------


def test_segments_cover_expected_families() -> None:
    bundle = _make_bundle()
    segs = annotation_to_segments(bundle)
    assert len(segs) > 0
    # Every Segment is the dataclass with the contract we promise.
    for s in segs:
        assert isinstance(s, Segment)
        assert s.t0 <= s.t1, f"segment {s.id} has t0 > t1"

    # We expect at least one of each family in the fixture bundle.
    track_ids = {s.track_id for s in segs}
    assert "env_0" in track_ids  # environment + condition both live here
    assert "ego_act" in track_ids
    assert any(tid.startswith("agent_") for tid in track_ids)


def test_segments_track_id_per_family() -> None:
    bundle = _make_bundle()
    segs = annotation_to_segments(bundle)
    by_id = {s.id.split("_")[0]: s for s in segs if not s.id.startswith("agent_")}
    # Environment segments live on `env_<track_idx>` tracks.
    env_seg = next(s for s in segs if s.id.startswith("env_"))
    assert env_seg.track_id.startswith("env_")
    # Condition segments inherit the parent env's track.
    cond_seg = next(s for s in segs if s.id.startswith("cond_"))
    assert cond_seg.track_id == env_seg.track_id
    # Ego action lives on `ego_act`.
    ego_act = next(s for s in segs if s.id.startswith("ego_act_"))
    assert ego_act.track_id == "ego_act"
    # Ego containment also lives on `ego_act` per the TS port.
    ego_cont = next(s for s in segs if s.id.startswith("ego_cont_"))
    assert ego_cont.track_id == "ego_act"
    # Agent action lives on `agent_<n>`.
    agent_act = next(s for s in segs if s.id.startswith("agent_action_"))
    assert agent_act.track_id.startswith("agent_")
    # The parent agent segment also lives on the same agent track.
    agent_parent = next(s for s in segs if s.id.startswith("agent_0_"))
    assert agent_parent.track_id == agent_act.track_id

    del by_id  # silence unused-binding lint without mutating asserts above


def test_agent_yield_illegal_flag_propagates() -> None:
    """A specific assertion matching the TS port — illegal_flag must round-trip."""
    bundle = _make_bundle()
    segs = annotation_to_segments(bundle)
    yield_segs = [
        s for s in segs if s.label == "Yield" and s.id.startswith("agent_action_")
    ]
    assert len(yield_segs) == 1
    s = yield_segs[0]
    assert s.illegal is True
    assert s.because_of == ("ego",)
    # t0 < t1 — the fixture sets 0:1.5 → 0:3.0
    assert s.t0 == pytest.approx(1.5)
    assert s.t1 == pytest.approx(3.0)


def test_segment_is_frozen() -> None:
    """`Segment` is `frozen=True` — attempted mutation raises."""
    s = Segment(id="x", track_id="env_0", label="L", t0=0.0, t1=1.0)
    with pytest.raises(Exception):  # FrozenInstanceError is a subclass.
        s.label = "Y"  # type: ignore[misc]


def test_segments_on_empty_bundle() -> None:
    """No environments / agents / ego actions → empty list, no exception."""
    bundle = AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="empty_clip"),
    )
    assert annotation_to_segments(bundle) == []


# ---------------------------------------------------------------------------
# entity_color — palette parity with index.css
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("env", "#22c55e"),
        ("ego", "#3b82f6"),
        ("object", "#f59e0b"),
        ("agent", "#a855f7"),
        ("light", "#ef4444"),
    ],
)
def test_entity_color_known_kinds(kind: str, expected: str) -> None:
    got = entity_color(kind)
    assert got == expected
    # Sanity: hex string, 7 chars including the leading '#'.
    assert len(got) == 7
    assert got.startswith("#")
    int(got[1:], 16)  # parses as hex


def test_entity_color_unknown_kind_raises() -> None:
    with pytest.raises(KeyError):
        entity_color("not-a-real-kind")


# ---------------------------------------------------------------------------
# render_frame — smoke test with a mocked video reader
# ---------------------------------------------------------------------------


def test_render_frame_returns_pil_image() -> None:
    """Mock `seq.video.decode_images_from_timestamps` to skip real decode."""
    h, w = 32, 48
    fake_frame = np.zeros((h, w, 3), dtype=np.uint8)
    fake_frame[..., 0] = 200  # tint so we can spot the data round-trip

    mock_video = MagicMock()
    mock_video.decode_images_from_timestamps.return_value = (
        np.stack([fake_frame], axis=0),
        None,
    )
    mock_seq = MagicMock()
    mock_seq.video = mock_video

    img = render_frame(mock_seq, t=2.5)

    # Returns a PIL image of the right size & mode.
    from PIL import Image as _Image

    assert isinstance(img, _Image.Image)
    assert img.size == (w, h)
    assert img.mode == "RGB"

    # Decoder was asked for the right microsecond timestamp.
    args, _kwargs = mock_video.decode_images_from_timestamps.call_args
    requested = args[0]
    assert requested.dtype == np.int64
    assert requested.shape == (1,)
    assert int(requested[0]) == int(round(2.5 * 1_000_000))


def test_render_frame_accepts_but_ignores_overlays() -> None:
    """`overlays` kwarg is part of the v0 signature but a no-op."""
    fake_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    mock_seq = MagicMock()
    mock_seq.video.decode_images_from_timestamps.return_value = (
        np.stack([fake_frame], axis=0),
        None,
    )

    img = render_frame(mock_seq, t=0.0, overlays={"boxes": True, "keypoints": True})

    from PIL import Image as _Image

    assert isinstance(img, _Image.Image)


def test_render_frame_raises_clear_error_when_video_missing() -> None:
    """A Sequence without a usable `.video` must produce a clear RuntimeError."""
    mock_seq = MagicMock()
    # Make `.video` access raise (mirrors the real failure path when the
    # parent dataset is absent or the clip has no canonical camera).
    type(mock_seq).video = property(
        fget=lambda self: (_ for _ in ()).throw(KeyError("no video reader"))
    )

    with pytest.raises(RuntimeError, match="video"):
        render_frame(mock_seq, t=1.0)
