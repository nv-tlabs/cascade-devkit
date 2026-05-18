# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `cascade_av.validate`.

One test per rule covers a clean (no-issue) and a dirty (issue-emitting)
case. Aggregator tests confirm `validate()` chains the rules and returns
the empty list for a fully-valid bundle.

Fixtures are built in-memory — no corpus required.
"""

from __future__ import annotations

import pytest

from cascade_av.spec import (
    Agent,
    AgentAction,
    AgentProperty,
    AnnotationBundle,
    Condition,
    EgoAction,
    EgoVehicle,
    Environment,
    Influence,
    SignalHead,
    SignalingDetails,
    SilAvAnnotation,
    TrafficLight,
    TrafficObject,
    VideoMeta,
)
from cascade_av.validate import (
    HARD_RULES,
    Issue,
    agent_requires_type,
    condition_requires_type,
    ego_has_at_least_one_action,
    environment_requires_type,
    id_references_resolve,
    timestamps_in_video_range,
    traffic_object_requires_type,
    validate,
)


def _clean_bundle() -> AnnotationBundle:
    """A fully-valid bundle: one ego action, no other entities.

    The minimum bundle that satisfies every HARD_RULE. Tests that want to
    exercise one rule's negative path mutate a copy of this fixture so the
    other rules stay quiet.
    """
    return AnnotationBundle(
        video=VideoMeta(clip_id="clip-xyz", duration_s=10.0),
        annotation=SilAvAnnotation(
            ego_vehicle=EgoVehicle(
                actions=[EgoAction(id="EA1", type="Drive Straight")],
            ),
        ),
    )


# -----------------------------------------------------------------------------
# environment_requires_type
# -----------------------------------------------------------------------------

def test_environment_requires_type_clean_bundle_returns_no_issues() -> None:
    bundle = _clean_bundle()
    bundle.annotation.environments.append(
        Environment(id="Env1", type="Road", start_timestamp="0:0.0", end_timestamp="0:5.0")
    )
    assert environment_requires_type(bundle) == []


def test_environment_requires_type_empty_string_errors() -> None:
    bundle = _clean_bundle()
    bundle.annotation.environments.append(Environment(id="Env1", type=""))
    issues = environment_requires_type(bundle)
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].rule == "environment_requires_type"
    assert issues[0].entity_id == "Env1"
    assert issues[0].field == "type"
    assert issues[0].entity_path == "annotation.environments[0]"


def test_environment_requires_type_whitespace_errors() -> None:
    bundle = _clean_bundle()
    bundle.annotation.environments.append(Environment(id="EnvWS", type="   "))
    issues = environment_requires_type(bundle)
    assert len(issues) == 1
    assert issues[0].entity_id == "EnvWS"


# -----------------------------------------------------------------------------
# condition_requires_type
# -----------------------------------------------------------------------------

def test_condition_requires_type_empty_list_errors() -> None:
    bundle = _clean_bundle()
    bundle.annotation.conditions.append(Condition(id="C1", type=[]))
    issues = condition_requires_type(bundle)
    assert len(issues) == 1
    assert issues[0].rule == "condition_requires_type"
    assert issues[0].entity_id == "C1"
    assert issues[0].entity_path == "annotation.conditions[0]"


def test_condition_requires_type_list_of_empty_strings_errors() -> None:
    bundle = _clean_bundle()
    bundle.annotation.conditions.append(Condition(id="C2", type=["", "  "]))
    issues = condition_requires_type(bundle)
    assert len(issues) == 1
    assert issues[0].entity_id == "C2"


def test_condition_requires_type_with_label_passes() -> None:
    bundle = _clean_bundle()
    bundle.annotation.conditions.append(Condition(id="C3", type=["Clear"]))
    assert condition_requires_type(bundle) == []


# -----------------------------------------------------------------------------
# agent_requires_type
# -----------------------------------------------------------------------------

def test_agent_requires_type_empty_errors() -> None:
    bundle = _clean_bundle()
    bundle.annotation.agents.append(Agent(id="A1", type=""))
    issues = agent_requires_type(bundle)
    assert len(issues) == 1
    assert issues[0].rule == "agent_requires_type"
    assert issues[0].entity_id == "A1"


def test_agent_requires_type_with_type_passes() -> None:
    bundle = _clean_bundle()
    bundle.annotation.agents.append(Agent(id="A2", type="oxd:Car"))
    assert agent_requires_type(bundle) == []


# -----------------------------------------------------------------------------
# traffic_object_requires_type
# -----------------------------------------------------------------------------

def test_traffic_object_requires_type_empty_errors() -> None:
    bundle = _clean_bundle()
    bundle.annotation.traffic_objects.append(TrafficObject(id="T1", type=""))
    issues = traffic_object_requires_type(bundle)
    assert len(issues) == 1
    assert issues[0].rule == "traffic_object_requires_type"
    assert issues[0].entity_id == "T1"


def test_traffic_object_requires_type_with_type_passes() -> None:
    bundle = _clean_bundle()
    bundle.annotation.traffic_objects.append(TrafficObject(id="T2", type="oxd:Cone"))
    assert traffic_object_requires_type(bundle) == []


# -----------------------------------------------------------------------------
# ego_has_at_least_one_action
# -----------------------------------------------------------------------------

def test_ego_has_at_least_one_action_empty_actions_errors() -> None:
    bundle = AnnotationBundle(
        video=VideoMeta(clip_id="empty-ego", duration_s=10.0),
        annotation=SilAvAnnotation(ego_vehicle=EgoVehicle(actions=[])),
    )
    issues = ego_has_at_least_one_action(bundle)
    assert len(issues) == 1
    assert issues[0].rule == "ego_has_at_least_one_action"
    assert issues[0].entity_path == "annotation.ego_vehicle"
    assert issues[0].entity_id is None
    assert issues[0].field == "actions"


def test_ego_has_at_least_one_action_with_action_passes() -> None:
    bundle = _clean_bundle()
    assert ego_has_at_least_one_action(bundle) == []


# -----------------------------------------------------------------------------
# id_references_resolve
# -----------------------------------------------------------------------------

def test_id_references_resolve_dangling_because_of_errors() -> None:
    bundle = _clean_bundle()
    # Ego action's because_of points at a non-existent id.
    bundle.annotation.ego_vehicle.actions[0].because_of = ["Ghost"]
    issues = id_references_resolve(bundle)
    assert len(issues) == 1
    assert issues[0].rule == "id_references_resolve"
    assert issues[0].field == "because_of"
    assert "Ghost" in issues[0].message
    assert issues[0].entity_path == "annotation.ego_vehicle.actions[0]"


def test_id_references_resolve_ego_keyword_is_valid_target() -> None:
    bundle = _clean_bundle()
    bundle.annotation.agents.append(
        Agent(
            id="A1",
            type="oxd:Car",
            actions=[AgentAction(id="AA1", action_type="Drive", because_of=["Ego"])],
        )
    )
    assert id_references_resolve(bundle) == []


def test_id_references_resolve_resolves_to_agent_action_id() -> None:
    """An agent action's id is itself a valid reference target."""
    bundle = _clean_bundle()
    bundle.annotation.agents.append(
        Agent(
            id="A1",
            type="oxd:Car",
            actions=[AgentAction(id="AA1", action_type="Drive")],
        )
    )
    # Ego action references the nested agent-action id by name.
    bundle.annotation.ego_vehicle.actions[0].because_of = ["AA1"]
    assert id_references_resolve(bundle) == []


def test_id_references_resolve_resolves_to_ego_action_id() -> None:
    """The forward direction: an agent action references an ego action id."""
    bundle = _clean_bundle()  # ego action id is "EA1"
    bundle.annotation.agents.append(
        Agent(
            id="A1",
            type="oxd:Car",
            actions=[
                AgentAction(
                    id="AA1",
                    action_type="Drive",
                    because_of=["EA1"],
                )
            ],
        )
    )
    assert id_references_resolve(bundle) == []


def test_id_references_resolve_influencers_dangling_errors() -> None:
    bundle = _clean_bundle()
    bundle.annotation.ego_vehicle.influenced_by.append(
        Influence(id="I1", influencers=["Phantom"])
    )
    issues = id_references_resolve(bundle)
    assert any(i.field == "influencers" and "Phantom" in i.message for i in issues)


def test_id_references_resolve_blank_entries_ignored() -> None:
    bundle = _clean_bundle()
    # Empty-string entries are UI fill-state, not violations.
    bundle.annotation.ego_vehicle.actions[0].because_of = [""]
    assert id_references_resolve(bundle) == []


def test_id_references_resolve_signaling_details_link_to_dangling_errors() -> None:
    """`AgentAction.signaling_details.link_to` is a real reference field.

    A `Signal*` action without a resolved target is the upstream bug the
    reviewer flagged: pre-fix, this case round-tripped through `validate`
    as ``ok=True``.
    """
    bundle = _clean_bundle()
    bundle.annotation.agents.append(
        Agent(
            id="A1",
            type="oxd:Car",
            actions=[
                AgentAction(
                    id="AA1",
                    action_type="Signal",
                    signaling_details=SignalingDetails(link_to=["Ghost"]),
                )
            ],
        )
    )
    issues = id_references_resolve(bundle)
    assert len(issues) == 1
    assert issues[0].rule == "id_references_resolve"
    assert issues[0].field == "link_to"
    assert "Ghost" in issues[0].message
    assert "signaling_details" in issues[0].entity_path


def test_id_references_resolve_property_signaling_details_link_to_dangling_errors() -> None:
    """Same coverage for `AgentProperty.signaling_details.link_to`."""
    bundle = _clean_bundle()
    bundle.annotation.agents.append(
        Agent(
            id="A1",
            type="oxd:Car",
            properties=[
                AgentProperty(
                    id="AP1",
                    property_type="SignalLeft",
                    signaling_details=SignalingDetails(link_to=["Ghost"]),
                )
            ],
        )
    )
    issues = id_references_resolve(bundle)
    assert any(
        i.field == "link_to" and "Ghost" in i.message and "properties" in i.entity_path
        for i in issues
    )


def test_id_references_resolve_signal_head_influenced_agent_ids_dangling_errors() -> None:
    """`SignalHead.influenced_agent_ids` names the agents this head controls."""
    bundle = _clean_bundle()
    bundle.annotation.traffic_lights.append(
        TrafficLight(
            id="TL1",
            signal_heads=[
                SignalHead(id="SH1", influenced_agent_ids=["Phantom"])
            ],
        )
    )
    issues = id_references_resolve(bundle)
    assert any(
        i.field == "influenced_agent_ids" and "Phantom" in i.message for i in issues
    )


def test_id_references_resolve_signal_head_influenced_agent_ids_resolves_to_agent() -> None:
    bundle = _clean_bundle()
    bundle.annotation.agents.append(Agent(id="A1", type="oxd:Car"))
    bundle.annotation.traffic_lights.append(
        TrafficLight(
            id="TL1",
            signal_heads=[SignalHead(id="SH1", influenced_agent_ids=["A1"])],
        )
    )
    assert id_references_resolve(bundle) == []


# -----------------------------------------------------------------------------
# timestamps_in_video_range
# -----------------------------------------------------------------------------

def test_timestamps_in_video_range_clean_passes() -> None:
    bundle = _clean_bundle()
    bundle.annotation.environments.append(
        Environment(
            id="Env1",
            type="Road",
            start_timestamp="0:0.0",
            end_timestamp="0:5.0",
        )
    )
    assert timestamps_in_video_range(bundle) == []


def test_timestamps_in_video_range_zero_duration_skips_upper_bound() -> None:
    """`duration_s <= 0.0` (failed/missing ffprobe) disables the upper bound.

    Otherwise every populated timestamp on a probe-failed clip floods the
    issues panel and blocks "Mark complete" — see `_check_window` docstring.
    Start ≤ end and parseability checks still run; this asserts only the
    upper-bound check is skipped.
    """
    bundle = AnnotationBundle(
        video=VideoMeta(clip_id="c", duration_s=0.0),
        annotation=SilAvAnnotation(
            ego_vehicle=EgoVehicle(
                actions=[
                    EgoAction(
                        id="EA1",
                        type="Drive",
                        # Both well past the (unknown) duration; must NOT error.
                        start_timestamp="0:5.0",
                        end_timestamp="0:6.0",
                    )
                ]
            )
        ),
    )
    issues = timestamps_in_video_range(bundle)
    assert not [i for i in issues if "outside the clip range" in i.message]


def test_timestamps_in_video_range_zero_duration_still_checks_start_le_end() -> None:
    """Zero-duration only relaxes the upper bound, not the ordering check."""
    bundle = AnnotationBundle(
        video=VideoMeta(clip_id="c", duration_s=0.0),
        annotation=SilAvAnnotation(
            ego_vehicle=EgoVehicle(
                actions=[
                    EgoAction(
                        id="EA1",
                        type="Drive",
                        start_timestamp="0:6.0",
                        end_timestamp="0:5.0",  # before start
                    )
                ]
            )
        ),
    )
    issues = timestamps_in_video_range(bundle)
    assert any("is after" in i.message for i in issues)


def test_timestamps_in_video_range_past_duration_errors() -> None:
    bundle = _clean_bundle()  # duration_s = 10.0
    bundle.annotation.environments.append(
        Environment(
            id="Env1",
            type="Road",
            start_timestamp="0:0.0",
            end_timestamp="0:15.0",  # past 10.0
        )
    )
    issues = timestamps_in_video_range(bundle)
    assert len(issues) == 1
    assert issues[0].rule == "timestamps_in_video_range"
    assert issues[0].field == "end_timestamp"
    assert issues[0].entity_id == "Env1"


def test_timestamps_in_video_range_start_after_end_errors() -> None:
    bundle = _clean_bundle()  # duration_s = 10.0
    bundle.annotation.environments.append(
        Environment(
            id="Env1",
            type="Road",
            start_timestamp="0:5.0",
            end_timestamp="0:2.0",  # before start
        )
    )
    issues = timestamps_in_video_range(bundle)
    assert any(
        i.rule == "timestamps_in_video_range" and "is after" in i.message for i in issues
    )


def test_timestamps_in_video_range_unparseable_errors() -> None:
    bundle = _clean_bundle()
    bundle.annotation.environments.append(
        Environment(
            id="Env1",
            type="Road",
            start_timestamp="not a time",
            end_timestamp="0:5.0",
        )
    )
    issues = timestamps_in_video_range(bundle)
    assert any(
        i.rule == "timestamps_in_video_range" and "unparseable" in i.message for i in issues
    )


def test_timestamps_in_video_range_blank_intervals_ignored() -> None:
    """Empty start/end (open intervals during edit) do not emit issues."""
    bundle = _clean_bundle()
    bundle.annotation.environments.append(
        Environment(id="Env1", type="Road", start_timestamp="", end_timestamp="")
    )
    assert timestamps_in_video_range(bundle) == []


# -----------------------------------------------------------------------------
# validate aggregator
# -----------------------------------------------------------------------------

def test_validate_clean_bundle_returns_empty_list() -> None:
    assert validate(_clean_bundle()) == []


def test_validate_aggregates_all_rules() -> None:
    """Every hard rule fires AND issues appear in `HARD_RULES + SOFT_RULES` order.

    The `validate()` docstring sells stable order; this locks it. A reorder
    that breaks UI grouping in PR 2 would otherwise pass unnoticed.
    """
    from cascade_av.validate import SOFT_RULES

    bundle = AnnotationBundle(
        video=VideoMeta(clip_id="messy", duration_s=10.0),
        annotation=SilAvAnnotation(
            environments=[Environment(id="Env1", type="")],  # rule 1
            conditions=[Condition(id="C1", type=[])],  # rule 2
            agents=[Agent(id="A1", type="")],  # rule 3
            traffic_objects=[TrafficObject(id="T1", type="")],  # rule 4
            ego_vehicle=EgoVehicle(  # rule 5: no actions; rule 6/7 via influence
                influenced_by=[
                    Influence(
                        id="I1",
                        influencers=["DanglingGhost"],
                        start_timestamp="0:0.0",
                        end_timestamp="0:99.0",  # past duration
                    )
                ],
            ),
        ),
    )
    issues = validate(bundle)

    rule_ids = {i.rule for i in issues}
    expected = {r.__name__ for r in HARD_RULES}
    assert expected.issubset(rule_ids), f"missing rules: {expected - rule_ids}"

    # Every emitted issue is an Issue instance with severity=error here.
    for i in issues:
        assert isinstance(i, Issue)
        assert i.severity == "error"

    # Stable order: the rule-id sequence (with duplicates removed in
    # first-seen order) matches the HARD_RULES + SOFT_RULES rule names.
    seen: list[str] = []
    for i in issues:
        if i.rule not in seen:
            seen.append(i.rule)
    expected_order = [r.__name__ for r in (*HARD_RULES, *SOFT_RULES) if r.__name__ in seen]
    assert seen == expected_order, (
        f"issue order broke stable contract; got {seen}, expected {expected_order}"
    )


def test_validate_orders_hard_rules_before_soft_rules() -> None:
    """Stable order: hard rules first (in HARD_RULES order), then soft rules."""
    bundle = AnnotationBundle(
        video=VideoMeta(clip_id="c", duration_s=10.0),
        annotation=SilAvAnnotation(
            environments=[Environment(id="Env1", type="")],
            ego_vehicle=EgoVehicle(actions=[EgoAction(id="EA1", type="Drive")]),
        ),
    )
    issues = validate(bundle)
    # Only environment_requires_type trips here.
    assert len(issues) == 1
    assert issues[0].rule == "environment_requires_type"


def test_issue_is_frozen_dataclass() -> None:
    """Issue is immutable; downstream consumers can hash / safely re-emit."""
    issue = Issue(
        severity="error",
        entity_path="annotation.environments[0]",
        entity_id="Env1",
        field="type",
        rule="environment_requires_type",
        message="x",
    )
    with pytest.raises(Exception):  # FrozenInstanceError, but stay flexible
        issue.message = "mutated"  # type: ignore[misc]
