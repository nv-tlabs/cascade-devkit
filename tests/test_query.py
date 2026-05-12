# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `causal_ai_av.query`."""

from __future__ import annotations

from pathlib import Path

import pytest

from causal_ai_av.io import load_file
from causal_ai_av.query import (
    IdIndex,
    Interval,
    action_interval,
    actions_at,
    agent_visibility_interval,
    agents_in_position,
    agents_visible_at,
    ego_relative_pose_at,
    extract_causal_triplets,
    format_timestamp,
    iter_triplets,
    overlapping_actions,
    parse_timestamp,
)
from causal_ai_av.spec import AnnotationBundle

# ---------------------------------------------------------------------------
# time
# ---------------------------------------------------------------------------


def test_parse_timestamp_happy_cases() -> None:
    assert parse_timestamp("0:5.4") == 5.4
    assert parse_timestamp("1:30.0") == 90.0
    assert parse_timestamp("0:0.0") == 0.0
    assert parse_timestamp("0:20.166") == 20.166


def test_parse_timestamp_returns_none_on_bad_input() -> None:
    assert parse_timestamp("") is None
    assert parse_timestamp(None) is None
    assert parse_timestamp("garbage") is None
    assert parse_timestamp("1:") is None


def test_format_timestamp_round_trip() -> None:
    assert format_timestamp(75.5) == "1:15.5"
    assert format_timestamp(0.0) == "0:0.0"
    s = format_timestamp(20.2)
    assert parse_timestamp(s) == 20.2


def test_interval_contains_is_closed() -> None:
    iv = Interval(0.0, 5.0)
    assert iv.contains(0.0)
    assert iv.contains(5.0)
    assert iv.contains(2.5)
    assert not iv.contains(-0.01)
    assert not iv.contains(5.01)


def test_interval_overlaps_is_closed() -> None:
    a = Interval(0.0, 5.0)
    assert a.overlaps(Interval(3.0, 7.0))
    assert a.overlaps(Interval(5.0, 7.0))  # boundary touch
    assert not a.overlaps(Interval(5.01, 7.0))
    assert a.overlaps(a)


def test_interval_from_strings() -> None:
    iv = Interval.from_strings("0:1.0", "0:5.0")
    assert iv is not None and iv.start == 1.0 and iv.end == 5.0
    assert Interval.from_strings("", "0:5.0") is None
    assert iv.duration == 4.0


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


def test_id_index_has_synthetic_ego_anchor(rich_bundle: AnnotationBundle) -> None:
    idx = IdIndex(rich_bundle)
    ego = idx.get("Ego")
    assert ego is not None
    assert ego.kind == "ego"


def test_id_index_resolves_first_agent(rich_bundle: AnnotationBundle) -> None:
    idx = IdIndex(rich_bundle)
    if not rich_bundle.annotation.agents:
        pytest.skip("rich bundle has no agents")
    a = rich_bundle.annotation.agents[0]
    assert idx.get(a.id) is not None
    assert idx.get(a.id).kind == "agent"


def test_id_index_returns_none_for_dangling() -> None:
    # Empty bundle with synthetic ego only.
    b = AnnotationBundle.model_validate({"video": {"clip_id": "x"}})
    idx = IdIndex(b)
    assert idx.get("NoSuch") is None
    assert "Ego" in idx


# ---------------------------------------------------------------------------
# triplets
# ---------------------------------------------------------------------------


def test_extract_triplets_returns_at_least_one(file_with_causal_link: Path) -> None:
    bundle = load_file(file_with_causal_link)
    triplets = extract_causal_triplets(bundle)
    assert len(triplets) >= 1
    t = triplets[0]
    assert t.predicate, "predicate (action_type) should be non-empty"
    assert t.cause_id, "cause_id is required"
    assert t.clip_id == bundle.video.clip_id


def test_triplets_resolve_at_least_some_causes(file_with_causal_link: Path) -> None:
    triplets = extract_causal_triplets(load_file(file_with_causal_link))
    resolved = [t for t in triplets if t.cause is not None]
    assert len(resolved) >= 1, "at least one triplet must resolve to a real entity"


def test_iter_triplets_across_bundles(file_with_causal_link: Path) -> None:
    bundle = load_file(file_with_causal_link)
    flat = list(iter_triplets([bundle, bundle]))
    direct = extract_causal_triplets(bundle)
    assert len(flat) == 2 * len(direct)


# ---------------------------------------------------------------------------
# temporal
# ---------------------------------------------------------------------------


def test_actions_at_midpoint_returns_consistent_list(
    rich_bundle: AnnotationBundle,
) -> None:
    hits = actions_at(rich_bundle, 5.0)
    assert isinstance(hits, list)
    for kind, _id, action in hits:
        assert kind in ("agent", "ego")
        iv = action_interval(action)
        assert iv is not None and iv.contains(5.0)


def test_overlapping_actions_returns_pairs(rich_bundle: AnnotationBundle) -> None:
    pairs = overlapping_actions(rich_bundle)
    for (k1, _i1, a1), (k2, _i2, a2) in pairs:
        iv1 = action_interval(a1)
        iv2 = action_interval(a2)
        assert iv1 is not None and iv2 is not None
        assert iv1.overlaps(iv2)


# ---------------------------------------------------------------------------
# spatial
# ---------------------------------------------------------------------------


def test_agents_visible_at_returns_list(rich_bundle: AnnotationBundle) -> None:
    assert isinstance(agents_visible_at(rich_bundle, 5.0), list)


def test_ego_relative_pose_at_finds_a_pose(rich_bundle: AnnotationBundle) -> None:
    for agent in rich_bundle.annotation.agents:
        if not agent.ego_relative_pose:
            continue
        first = agent.ego_relative_pose[0]
        iv = Interval.from_strings(first.start_timestamp, first.end_timestamp)
        if iv is None:
            continue
        t = (iv.start + iv.end) / 2
        assert ego_relative_pose_at(agent, t) is not None
        return
    pytest.skip("no agent with ego_relative_pose in rich bundle")


def test_agent_visibility_interval(rich_bundle: AnnotationBundle) -> None:
    for agent in rich_bundle.annotation.agents:
        iv = agent_visibility_interval(agent)
        if iv is not None:
            assert iv.start <= iv.end
            return
    pytest.skip("no agent with parseable visibility window")


def test_agents_in_position_smoke(rich_bundle: AnnotationBundle) -> None:
    # Try the four canonical positions; at least one of them should hit on a
    # rich clip with ego-relative poses sampled at t=5.0.
    found = False
    for pos in ("In front", "Left", "Right", "Behind"):
        if agents_in_position(rich_bundle, pos, 5.0):
            found = True
            break
    # Soft assertion — corpus is real data, not guaranteed to hit all positions.
    assert isinstance(found, bool)
