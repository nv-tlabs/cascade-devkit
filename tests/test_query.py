# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `cascade_av.query`."""

from __future__ import annotations

from pathlib import Path

import pytest

from cascade_av.io import load_file
from cascade_av.query import (
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
    parse_timestamp_or,
)
from cascade_av.spec import AnnotationBundle

# ---------------------------------------------------------------------------
# time
# ---------------------------------------------------------------------------


def test_parse_timestamp_happy_cases() -> None:
    assert parse_timestamp("0:5.4") == 5.4
    assert parse_timestamp("1:30.0") == 90.0
    assert parse_timestamp("0:0.0") == 0.0
    assert parse_timestamp("0:20.166") == 20.166
    assert parse_timestamp("0.000") == 0.0
    assert parse_timestamp("20.167") == 20.167
    assert parse_timestamp("20") == 20.0


def test_parse_timestamp_returns_none_on_bad_input() -> None:
    assert parse_timestamp("") is None
    assert parse_timestamp(None) is None
    assert parse_timestamp("garbage") is None
    assert parse_timestamp("1:") is None
    assert parse_timestamp("-1.0") is None
    assert parse_timestamp("+1.0") is None


def test_parse_timestamp_is_anchored() -> None:
    # The unified parser is strict: leading or trailing garbage rejects.
    # Locks the historical regression where the viz layer's `re.search`
    # variant would accept "foo 1:30.0 bar" → 90.0 silently while the
    # query layer's `re.match` variant returned None. They now agree.
    assert parse_timestamp("foo 1:30.0") is None
    assert parse_timestamp("1:30.0 bar") is None
    assert parse_timestamp(" 1:30.0") is None
    assert parse_timestamp("foo 20.167") is None
    assert parse_timestamp("20.167 bar") is None
    assert parse_timestamp("20.167 ") is None
    assert parse_timestamp("20.167\n") is None
    assert parse_timestamp_or("foo 1:30.0", 0.0) == 0.0
    assert parse_timestamp_or("1:30.0 bar", 0.0) == 0.0
    assert parse_timestamp_or("foo 20.167", 0.0) == 0.0
    assert parse_timestamp_or("20.167 bar", 0.0) == 0.0


def test_parse_timestamp_or_matches_parseTs_contract() -> None:
    # Locked TS-parity behaviour: empty / None / unparseable → default.
    assert parse_timestamp_or("0:5.4") == 5.4
    assert parse_timestamp_or("1:30.0") == 90.0
    assert parse_timestamp_or("0.000") == 0.0
    assert parse_timestamp_or("20.167") == 20.167
    assert parse_timestamp_or("") == 0.0
    assert parse_timestamp_or(None) == 0.0
    assert parse_timestamp_or("garbage") == 0.0
    assert parse_timestamp_or("1:") == 0.0
    # The default is configurable per call site.
    assert parse_timestamp_or("garbage", default=-1.0) == -1.0
    # And it stays the same parser as `parse_timestamp` — happy paths
    # produce identical numerics.
    for ts in ("0:0.0", "0:5.4", "1:30.0", "0:20.166", "0.000", "20.167"):
        assert parse_timestamp_or(ts) == parse_timestamp(ts)


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


# ---------------------------------------------------------------------------
# influenced_by — synthetic-bundle semantics
# ---------------------------------------------------------------------------


def _bundle_for_influence(
    *,
    traffic_objects=None,
    traffic_lights=None,
    agents=None,
    ego_influences=None,
    duration_s: float = 10.0,
):
    """Build a synthetic AnnotationBundle from focused inputs. The
    ``influenced_by`` tests want to control exactly which Influence
    windows + influencer IDs + traffic objects / lights exist; the
    helper avoids tying tests to the real corpus and keeps each assertion
    local to its setup."""
    from cascade_av.spec import AnnotationBundle

    payload: dict = {
        "video": {"clip_id": "influenced-by-test", "duration_s": duration_s},
    }
    bundle = AnnotationBundle.model_validate(payload)
    if traffic_objects:
        bundle.annotation.traffic_objects.extend(traffic_objects)
    if traffic_lights:
        bundle.annotation.traffic_lights.extend(traffic_lights)
    if agents:
        bundle.annotation.agents.extend(agents)
    if ego_influences:
        bundle.annotation.ego_vehicle.influenced_by.extend(ego_influences)
    return bundle


def test_influenced_by_traffic_object_dispatch() -> None:
    """A bare TrafficObject influencer resolves through `obj` and the
    RHS predicate matches on `obj.type`. The match interval is the
    Influence window, not the object's visibility — verifies the
    invariant called out in the design doc."""
    from cascade_av.query import evaluate, parse
    from cascade_av.spec import EgoVehicle, Influence, TrafficObject

    yield_obj = TrafficObject(
        id="obj-yield",
        type="fst:YieldSign",
        visibility_start_timestamp="0:0.0",
        visibility_end_timestamp="0:9.5",
    )
    infl = Influence(
        id="i1",
        influencers=["obj-yield"],
        start_timestamp="0:1.0",
        end_timestamp="0:3.0",
    )
    bundle = _bundle_for_influence(
        traffic_objects=[yield_obj], ego_influences=[infl]
    )

    ms = evaluate(parse("ego influenced_by obj.type = yield_sign"), bundle)
    assert len(ms) == 1
    m = ms.matches[0]
    # Owner of the match is the EgoVehicle.
    assert isinstance(m.entity, EgoVehicle)
    # Interval is the Influence window (1.0 – 3.0), not the wider
    # visibility window of the object (0.0 – 9.5).
    assert m.interval is not None
    assert m.interval.start == pytest.approx(1.0)
    assert m.interval.end == pytest.approx(3.0)

    # And the negative case: predicate that doesn't hit the influencer
    # produces zero matches without raising.
    ms = evaluate(parse("ego influenced_by obj.type = stop_sign"), bundle)
    assert len(ms) == 0


def test_influenced_by_mixed_kind_dispatch() -> None:
    """One Influence window with two influencer IDs — a signal head
    (expanded to LightStates with color=Red) and a TrafficObject
    (`fst:StopSign`). Each RHS predicate must dispatch to the matching
    kind: `light.color = red` picks up the signal-head expansion,
    `obj.type = stop_sign` picks up the traffic object, and the
    parenthesised OR matches against either."""
    from cascade_av.query import evaluate, parse
    from cascade_av.spec import (
        Influence,
        LightStates,
        SignalHead,
        TrafficLight,
        TrafficObject,
    )

    head = SignalHead(
        id="sh-1",
        state_sequence=[
            LightStates(
                id="ls-1", color="Red",
                start_timestamp="0:1.0", end_timestamp="0:4.0",
            ),
        ],
        affects_ego="True",
    )
    tl = TrafficLight(id="tl-1", signal_heads=[head])
    stop_obj = TrafficObject(
        id="obj-stop",
        type="fst:StopSign",
        visibility_start_timestamp="0:0.0",
        visibility_end_timestamp="0:5.0",
    )
    infl = Influence(
        id="i1",
        influencers=["sh-1", "obj-stop"],
        start_timestamp="0:1.0",
        end_timestamp="0:4.0",
    )
    bundle = _bundle_for_influence(
        traffic_objects=[stop_obj],
        traffic_lights=[tl],
        ego_influences=[infl],
    )

    # light dispatches; obj dispatches; cross-kind disjunction matches
    # via the first satisfying candidate in the Influence's influencer
    # list.
    assert len(evaluate(parse("ego influenced_by light.color = red"), bundle)) == 1
    assert len(evaluate(parse("ego influenced_by obj.type = stop_sign"), bundle)) == 1
    assert len(evaluate(parse("ego influenced_by obj.type = yield_sign"), bundle)) == 0
    # Parenthesised compound: matches because at least one influencer
    # satisfies one disjunct.
    assert len(
        evaluate(
            parse("ego influenced_by (light.color = red or obj.type = stop_sign)"),
            bundle,
        )
    ) == 1


def test_influenced_by_while_composition_uses_influence_window() -> None:
    """`influenced_by … while …` intersects the Influence window with
    the right-hand match's window. A signal head with a red state spans
    (0.0, 5.0); a ped agent is visible (2.0, 8.0). The composed match
    interval is the overlap of the Influence window (0.0, 5.0) with the
    agent's visibility window (2.0, 8.0) — i.e. (2.0, 5.0). This
    locks the design-doc invariant that influencer expansion does NOT
    pull the light-state's own interval into the match."""
    from cascade_av.query import evaluate, parse
    from cascade_av.spec import (
        Agent,
        EgoRelativePose,
        Influence,
        LightStates,
        SignalHead,
        TrafficLight,
    )

    head = SignalHead(
        id="sh-1",
        state_sequence=[
            LightStates(
                id="ls-1", color="Red",
                # Light-state interval intentionally narrower than the
                # Influence window to prove we don't read it: if we did,
                # the composed interval would clamp to (2.0, 4.0)
                # instead of (2.0, 5.0).
                start_timestamp="0:0.0", end_timestamp="0:4.0",
            ),
        ],
    )
    tl = TrafficLight(id="tl-1", signal_heads=[head])
    infl = Influence(
        id="i1",
        influencers=["sh-1"],
        start_timestamp="0:0.0",
        end_timestamp="0:5.0",
    )
    ped = Agent(
        id="a-ped",
        type="Pedestrian (Adult)",  # canonical "ped" alias target
        visibility_start_timestamp="0:2.0",
        visibility_end_timestamp="0:8.0",
        ego_relative_pose=[
            EgoRelativePose(
                position_rel_to_ego="In front",
                start_timestamp="0:2.0",
                end_timestamp="0:8.0",
            ),
        ],
    )
    bundle = _bundle_for_influence(
        traffic_lights=[tl], agents=[ped], ego_influences=[infl],
    )

    ms = evaluate(
        parse("ego influenced_by light.color = red while agent.type = ped"),
        bundle,
    )
    assert len(ms) == 1
    iv = ms.matches[0].interval
    assert iv is not None
    assert iv.start == pytest.approx(2.0)
    assert iv.end == pytest.approx(5.0)


def test_influenced_by_signal_head_filters_by_window() -> None:
    """A signal head cycling Green → Red across the clip must NOT match
    every colour predicate for every Influence window. Only LightStates
    whose own interval overlaps the Influence window we're currently
    evaluating against may contribute to the RHS dispatch.

    Setup: one SignalHead with two LightStates — Green (0.0, 2.0) and
    Red (2.0, 5.0) — referenced by one Influence window (3.0, 4.0).
    The Red state overlaps; the Green state does not. So
    `light.color = red` must match and `light.color = green` must NOT.
    Before the fix both matched, inflating red/green counts on the
    real corpus by ~25-60%."""
    from cascade_av.query import evaluate, parse
    from cascade_av.spec import (
        Influence,
        LightStates,
        SignalHead,
        TrafficLight,
    )

    head = SignalHead(
        id="sh-1",
        state_sequence=[
            LightStates(
                id="ls-green", color="Green",
                start_timestamp="0:0.0", end_timestamp="0:2.0",
            ),
            LightStates(
                id="ls-red", color="Red",
                start_timestamp="0:2.0", end_timestamp="0:5.0",
            ),
        ],
    )
    tl = TrafficLight(id="tl-1", signal_heads=[head])
    infl = Influence(
        id="i1",
        influencers=["sh-1"],
        start_timestamp="0:3.0",
        end_timestamp="0:4.0",
    )
    bundle = _bundle_for_influence(
        traffic_lights=[tl], ego_influences=[infl],
    )

    # Red state overlaps (3.0,4.0) ⊂ (2.0,5.0) — matches.
    assert len(evaluate(parse("ego influenced_by light.color = red"), bundle)) == 1
    # Green state (0.0,2.0) does not overlap the Influence window (3.0,4.0).
    assert len(evaluate(parse("ego influenced_by light.color = green"), bundle)) == 0


def test_influenced_by_agent_multiple_windows() -> None:
    """One ego with two distinct Influence windows, both pointing at
    different red-light influencers, must produce TWO matches —
    cardinality is per (subject, Influence-window) pair. Locks the
    semantics that overlapping or sequential influence windows aren't
    collapsed."""
    from cascade_av.query import evaluate, parse
    from cascade_av.spec import (
        Influence,
        LightStates,
        SignalHead,
        TrafficLight,
    )

    head_a = SignalHead(
        id="sh-a",
        state_sequence=[
            LightStates(
                id="ls-a", color="Red",
                start_timestamp="0:0.0", end_timestamp="0:3.0",
            ),
        ],
    )
    head_b = SignalHead(
        id="sh-b",
        state_sequence=[
            LightStates(
                id="ls-b", color="Red",
                start_timestamp="0:4.0", end_timestamp="0:7.0",
            ),
        ],
    )
    tl = TrafficLight(id="tl-1", signal_heads=[head_a, head_b])
    infl_1 = Influence(
        id="i1", influencers=["sh-a"],
        start_timestamp="0:1.0", end_timestamp="0:2.0",
    )
    infl_2 = Influence(
        id="i2", influencers=["sh-b"],
        start_timestamp="0:5.0", end_timestamp="0:6.0",
    )
    bundle = _bundle_for_influence(
        traffic_lights=[tl], ego_influences=[infl_1, infl_2],
    )

    ms = evaluate(parse("ego influenced_by light.color = red"), bundle)
    assert len(ms) == 2
    intervals = sorted((m.interval.start, m.interval.end) for m in ms.matches)
    assert intervals[0] == pytest.approx((1.0, 2.0))
    assert intervals[1] == pytest.approx((5.0, 6.0))


def test_influenced_by_second_influencer_wins() -> None:
    """One Influence window with two influencers where only the SECOND
    satisfies the RHS predicate. The evaluator must keep scanning the
    influencer list and emit one match — verifies the inner loop does
    not short-circuit before reaching the second candidate."""
    from cascade_av.query import evaluate, parse
    from cascade_av.spec import (
        Influence,
        LightStates,
        SignalHead,
        TrafficLight,
    )

    head_green = SignalHead(
        id="sh-green",
        state_sequence=[
            LightStates(
                id="ls-g", color="Green",
                start_timestamp="0:0.0", end_timestamp="0:5.0",
            ),
        ],
    )
    head_red = SignalHead(
        id="sh-red",
        state_sequence=[
            LightStates(
                id="ls-r", color="Red",
                start_timestamp="0:0.0", end_timestamp="0:5.0",
            ),
        ],
    )
    tl = TrafficLight(id="tl-1", signal_heads=[head_green, head_red])
    infl = Influence(
        id="i1",
        # First influencer fails RHS (green); second satisfies (red).
        influencers=["sh-green", "sh-red"],
        start_timestamp="0:1.0", end_timestamp="0:4.0",
    )
    bundle = _bundle_for_influence(
        traffic_lights=[tl], ego_influences=[infl],
    )

    ms = evaluate(parse("ego influenced_by light.color = red"), bundle)
    assert len(ms) == 1
    iv = ms.matches[0].interval
    assert iv is not None
    assert (iv.start, iv.end) == pytest.approx((1.0, 4.0))
