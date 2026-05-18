# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `cascade_av.viz` — segments port, entity colors, and the
headless `render_frame` Pillow path.

Bundles are built from scratch — no corpus dependency.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

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
    SilAvAnnotation,
    VideoMeta,
)
from cascade_av.viz import (
    Segment,
    annotation_to_segments,
    entity_color,
    render_frame,
)
from cascade_av.viz.segments import _parse_ts


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
        schema_version="2.1.0",
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


def test_parse_ts_is_anchored_like_query_parser() -> None:
    # Locked: the historical `re.search` divergence (segments accepted
    # leading/trailing garbage, query rejected it) is gone. Both
    # `_parse_ts` and `parse_timestamp` now share the same anchored regex
    # via the shared `parse_timestamp_or` helper.
    assert _parse_ts("foo 1:30.0") == 0.0
    assert _parse_ts("1:30.0 bar") == 0.0
    assert _parse_ts(" 1:30.0") == 0.0


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


def test_segment_family_defaults_to_parent() -> None:
    """`Segment.family` defaults to "parent" so historical
    constructions without an explicit family keep working."""
    s = Segment(id="x", track_id="env_0", label="L", t0=0.0, t1=1.0)
    assert s.family == "parent"


def test_segment_carries_family() -> None:
    """`annotation_to_segments` tags every emitted segment with the
    correct family. One assertion per family the fixture exercises;
    families not present in the fixture (signal_head / state /
    env_control / physical_containment) are covered by the lights
    fixture below.
    """
    bundle = _make_bundle()
    segs = annotation_to_segments(bundle)

    def _by_prefix(prefix: str) -> list[Segment]:
        return [s for s in segs if s.id.startswith(prefix)]

    # Env parent.
    envs = _by_prefix("env_")
    assert envs and all(s.family == "parent" for s in envs)
    # Condition.
    conds = _by_prefix("cond_")
    assert conds and all(s.family == "condition" for s in conds)
    # Ego action.
    ego_acts = _by_prefix("ego_act_")
    assert ego_acts and all(s.family == "action" for s in ego_acts)
    # Ego containment.
    ego_conts = _by_prefix("ego_cont_")
    assert ego_conts and all(s.family == "containment" for s in ego_conts)
    # Agent parent — there's exactly one in the fixture, under the
    # `agent_0_` prefix (no extra subtrack suffix).
    agent_parents = [s for s in segs if s.id.startswith("agent_0_") and s.family == "parent"]
    assert agent_parents, "expected one agent parent segment"
    # Agent action.
    agent_acts = _by_prefix("agent_action_")
    assert agent_acts and all(s.family == "action" for s in agent_acts)
    # Agent property.
    agent_props = _by_prefix("agent_prop_")
    assert agent_props and all(s.family == "property" for s in agent_props)
    # Agent influence.
    agent_infls = _by_prefix("agent_infl_")
    assert agent_infls and all(s.family == "influence" for s in agent_infls)


def test_segment_carries_family_for_lights() -> None:
    """Per-signal-head light families: signal_head, state, env_control,
    plus physical_containment (at the light's top level)."""
    from cascade_av.spec import (
        LightStates,
        SignalHead,
        TrafficLight,
    )

    env = Environment(
        id="env_0",
        type="fst:Road",
        start_timestamp="0:0.0",
        end_timestamp="0:10.0",
    )
    light_phys = Containment(
        id="light_phys_0",
        env_id="env_0",
        start_timestamp="0:0.0",
        end_timestamp="0:5.0",
    )
    sh_state = LightStates(
        id="sh_state_0",
        type="Steady",
        color="red",
        start_timestamp="0:0.0",
        end_timestamp="0:3.0",
    )
    sh_env_cont = Containment(
        id="sh_cont_0",
        env_id="env_0",
        start_timestamp="0:0.0",
        end_timestamp="0:3.0",
    )
    sh = SignalHead(
        id="sh_0",
        start_timestamp="0:0.0",
        end_timestamp="0:5.0",
        state_sequence=[sh_state],
        env_controlled=[sh_env_cont],
    )
    light = TrafficLight(
        id="light_0",
        visibility_start_timestamp="0:0.0",
        visibility_end_timestamp="0:5.0",
        containment=[light_phys],
        signal_heads=[sh],
    )
    bundle = AnnotationBundle(
        schema_version="2.1.0",
        video=VideoMeta(clip_id="light_clip", duration_s=10.0),
        annotation=SilAvAnnotation(environments=[env], traffic_lights=[light]),
    )
    segs = annotation_to_segments(bundle)
    by_id_prefix = {
        "light_0_": "parent",
        "light_phys_cont_0_0_": "physical_containment",
        "light_sh_0_0_": "signal_head",
        "light_state_0_0_0_": "state",
        "light_cont_0_0_0_": "env_control",
    }
    for prefix, expected_family in by_id_prefix.items():
        match = [s for s in segs if s.id.startswith(prefix)]
        assert match, f"no segment with prefix {prefix!r} in {[s.id for s in segs]!r}"
        assert match[0].family == expected_family, (
            f"prefix {prefix}: family={match[0].family!r} expected {expected_family!r}"
        )
    # Per-head light segments carry `_sh_index` in meta so the painter
    # can group them by signal head.
    sh_segs = [s for s in segs if s.id.startswith("light_sh_")]
    assert sh_segs[0].meta is not None and sh_segs[0].meta.get("_sh_index") == 0


def test_segment_carries_family_for_traffic_objects() -> None:
    """Traffic-object parent + state + containment families."""
    from cascade_av.spec import ObjectStateEntry, TrafficObject

    env = Environment(
        id="env_0",
        type="fst:Road",
        start_timestamp="0:0.0",
        end_timestamp="0:10.0",
    )
    obj_cont = Containment(
        id="obj_cont_0",
        env_id="env_0",
        lane_number="1",
        start_timestamp="0:0.0",
        end_timestamp="0:5.0",
    )
    obj_state = ObjectStateEntry(
        id="obj_state_0",
        motion_state="Stationary",
        start_timestamp="0:0.0",
        end_timestamp="0:5.0",
    )
    obj = TrafficObject(
        id="obj_0",
        type="oxd:Cone",
        visibility_start_timestamp="0:0.0",
        visibility_end_timestamp="0:5.0",
        state_sequence=[obj_state],
        containment=[obj_cont],
    )
    bundle = AnnotationBundle(
        schema_version="2.1.0",
        video=VideoMeta(clip_id="obj_clip", duration_s=10.0),
        annotation=SilAvAnnotation(environments=[env], traffic_objects=[obj]),
    )
    segs = annotation_to_segments(bundle)
    expected = {
        "obj_0_": "parent",
        "obj_state_0_0_": "state",
        "obj_cont_0_0_": "containment",
    }
    for prefix, family in expected.items():
        match = [s for s in segs if s.id.startswith(prefix)]
        assert match, f"no segment with prefix {prefix!r}"
        assert match[0].family == family


def test_segments_on_empty_bundle() -> None:
    """No environments / agents / ego actions → empty list, no exception."""
    bundle = AnnotationBundle(
        schema_version="2.1.0",
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
# family_color — per-family bar palette + entity-base fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category,family,expected",
    [
        # Environments
        ("Environments", "condition", "#06b6d4"),
        # Ego
        ("Ego", "containment", "#22c55e"),
        ("Ego", "influence", "#f97316"),
        ("Ego", "action", "#3b82f6"),
        ("Ego", "property", "#93c5fd"),
        # Agents
        ("Agents", "containment", "#22c55e"),
        ("Agents", "pose", "#14b8a6"),
        ("Agents", "influence", "#f97316"),
        ("Agents", "action", "#c084fc"),
        ("Agents", "property", "#d8b4fe"),
        # Objects
        ("Objects", "containment", "#22c55e"),
        ("Objects", "state", "#d97706"),
        # Traffic Lights
        ("Traffic Lights", "physical_containment", "#22c55e"),
        ("Traffic Lights", "signal_head", "#f97316"),
        ("Traffic Lights", "env_control", "#22c55e"),
        ("Traffic Lights", "state", "#dc2626"),
    ],
)
def test_family_color_palette_matches_annotator(
    category: str, family: str, expected: str
) -> None:
    """Pin every `(category, family) → hex` to the annotator's
    `subColor` table in `Timeline.tsx:990-1068`. Any palette edit must
    update this table and document why."""
    from cascade_av.viz import family_color

    assert family_color(category, family) == expected


@pytest.mark.parametrize(
    "category,group_kind",
    [
        ("Environments", "env"),
        ("Ego", "ego"),
        ("Objects", "object"),
        ("Agents", "agent"),
        ("Traffic Lights", "light"),
    ],
)
def test_family_color_fallback_to_entity_color(
    category: str, group_kind: str
) -> None:
    """For families with no family-specific color (notably the `parent`
    row), `family_color` falls back to `entity_color(group_kind)` so
    parent bars keep their entity hue."""
    from cascade_av.viz import family_color

    assert family_color(category, "parent") == entity_color(group_kind)
    # An unknown family also falls through to the entity base — this
    # keeps callers stable when a future SegmentFamily lands before its
    # palette entry does.
    assert family_color(category, "totally-new-family") == entity_color(group_kind)


def test_family_color_unknown_category_raises() -> None:
    """Unknown categories raise — silently mis-spelling a label should
    not silently mis-color a row."""
    from cascade_av.viz import family_color

    with pytest.raises(KeyError):
        family_color("NotACategory", "containment")


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
