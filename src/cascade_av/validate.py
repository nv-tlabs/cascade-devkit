# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Annotation-bundle validation rules for the 'Mark complete' workflow.

A bundle is "complete" when every rule in :data:`HARD_RULES` returns an
empty list. The annotator's UI calls :func:`validate` via the new
``POST /api/clips/{id}/annotations/validate`` endpoint and blocks the
"Mark complete" action until ``ok=True``.

Rules are pure functions: ``rule(bundle) -> list[Issue]``. Add new rules
to :data:`HARD_RULES` or :data:`SOFT_RULES`. Hard rules block completion;
soft rules are warnings (shown but non-blocking).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Literal

from cascade_av.query.time import parse_timestamp
from cascade_av.spec import AnnotationBundle

Severity = Literal["error", "warning"]

# Small slack applied when comparing parsed timestamps to the video duration.
# The on-disk format carries one-decimal `M:S.D` strings, so a clip whose true
# duration rounds to e.g. `0:10.0` can carry annotations stamped right at
# `0:10.0` against a probed `duration_s = 9.997…`. The epsilon prevents that
# rounding from flagging perfectly aligned end-of-clip annotations.
_DURATION_EPS = 0.001


@dataclass(frozen=True)
class Issue:
    """One validation finding."""

    severity: Severity
    entity_path: str  # e.g. "annotation.environments[2]"
    entity_id: str | None  # e.g. "Env3" — None for clip-level issues
    field: str | None  # e.g. "type" — None when whole-entity
    rule: str  # stable id, e.g. "environment_requires_type"
    message: str  # human

    def to_dict(self) -> dict[str, object]:
        """Wire-shape view (JSON-serialisable)."""
        return asdict(self)


Rule = Callable[[AnnotationBundle], list[Issue]]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _is_blank(s: str | None) -> bool:
    """True if `s` is None/empty/whitespace-only."""
    return s is None or not s.strip()


def _all_blank_list(items: list[str]) -> bool:
    """True if `items` is empty OR every element is blank."""
    return not items or all(_is_blank(x) for x in items)


# -----------------------------------------------------------------------------
# Hard rules — emit `error`s that block "Mark complete"
# -----------------------------------------------------------------------------

def environment_requires_type(bundle: AnnotationBundle) -> list[Issue]:
    """Every Environment must carry a non-blank `type`."""
    issues: list[Issue] = []
    for i, env in enumerate(bundle.annotation.environments):
        if _is_blank(env.type):
            issues.append(
                Issue(
                    severity="error",
                    entity_path=f"annotation.environments[{i}]",
                    entity_id=env.id or None,
                    field="type",
                    rule="environment_requires_type",
                    message=f"Environment {env.id!r} is missing a `type`.",
                )
            )
    return issues


def condition_requires_type(bundle: AnnotationBundle) -> list[Issue]:
    """Every Condition must carry at least one non-blank `type` label."""
    issues: list[Issue] = []
    for i, cond in enumerate(bundle.annotation.conditions):
        if _all_blank_list(cond.type):
            issues.append(
                Issue(
                    severity="error",
                    entity_path=f"annotation.conditions[{i}]",
                    entity_id=cond.id or None,
                    field="type",
                    rule="condition_requires_type",
                    message=f"Condition {cond.id!r} is missing a `type`.",
                )
            )
    return issues


def agent_requires_type(bundle: AnnotationBundle) -> list[Issue]:
    """Every Agent must carry a non-blank `type`."""
    issues: list[Issue] = []
    for i, agent in enumerate(bundle.annotation.agents):
        if _is_blank(agent.type):
            issues.append(
                Issue(
                    severity="error",
                    entity_path=f"annotation.agents[{i}]",
                    entity_id=agent.id or None,
                    field="type",
                    rule="agent_requires_type",
                    message=f"Agent {agent.id!r} is missing a `type`.",
                )
            )
    return issues


def traffic_object_requires_type(bundle: AnnotationBundle) -> list[Issue]:
    """Every TrafficObject must carry a non-blank `type`."""
    issues: list[Issue] = []
    for i, obj in enumerate(bundle.annotation.traffic_objects):
        if _is_blank(obj.type):
            issues.append(
                Issue(
                    severity="error",
                    entity_path=f"annotation.traffic_objects[{i}]",
                    entity_id=obj.id or None,
                    field="type",
                    rule="traffic_object_requires_type",
                    message=f"TrafficObject {obj.id!r} is missing a `type`.",
                )
            )
    return issues


def ego_has_at_least_one_action(bundle: AnnotationBundle) -> list[Issue]:
    """`ego_vehicle.actions` must be a non-empty list.

    Hard requirement per the "Mark complete" spec — every clip must record at
    least one ego action before it can be marked complete.
    """
    if bundle.annotation.ego_vehicle.actions:
        return []
    return [
        Issue(
            severity="error",
            entity_path="annotation.ego_vehicle",
            entity_id=None,
            field="actions",
            rule="ego_has_at_least_one_action",
            message="Ego vehicle must record at least one action.",
        )
    ]


def _collect_entity_ids(bundle: AnnotationBundle) -> set[str]:
    """Every id any reference field is allowed to point at.

    Includes the literal string ``"Ego"`` since the existing frontend allows
    `because_of` / `link_to` / `action_target` lists to name the ego vehicle
    directly. Empty-string ids (an artefact of partially-filled UI rows) are
    excluded — referencing an empty id is never well-defined.
    """
    ids: set[str] = {"Ego"}
    ann = bundle.annotation

    def _add(value: str | None) -> None:
        if value:
            ids.add(value)

    for env in ann.environments:
        _add(env.id)
    for cond in ann.conditions:
        _add(cond.id)
    for obj in ann.traffic_objects:
        _add(obj.id)
        for c in obj.containment:
            _add(c.id)
        for s in obj.state_sequence:
            _add(s.id)
    for tl in ann.traffic_lights:
        _add(tl.id)
        for c in tl.containment:
            _add(c.id)
        for head in tl.signal_heads:
            _add(head.id)
            for state in head.state_sequence:
                _add(state.id)
            for c in head.env_controlled:
                _add(c.id)
    for agent in ann.agents:
        _add(agent.id)
        for action in agent.actions:
            _add(action.id)
        for prop in agent.properties:
            _add(prop.id)
        for c in agent.containment:
            _add(c.id)
        for inf in agent.influenced_by:
            _add(inf.id)
    for action in ann.ego_vehicle.actions:
        _add(action.id)
    for prop in ann.ego_vehicle.properties:
        _add(prop.id)
    for c in ann.ego_vehicle.containment:
        _add(c.id)
    for inf in ann.ego_vehicle.influenced_by:
        _add(inf.id)

    return ids


def id_references_resolve(bundle: AnnotationBundle) -> list[Issue]:
    """Every id mentioned in a cross-entity reference field must resolve.

    Reference fields scanned:
      * Agent/Ego ``EgoAction.because_of / .link_to / .action_target``
      * Agent ``AgentAction.because_of / .link_to / .action_target``
      * ``Influence.influencers`` (under ``agent.influenced_by`` and
        ``ego_vehicle.influenced_by``)

    The literal string ``"Ego"`` is always a valid target. Empty-string
    references are silently ignored — they're UI fill-state, not a violation.
    """
    known = _collect_entity_ids(bundle)
    issues: list[Issue] = []

    def _check(refs: list[str], source_path: str, source_id: str | None, field: str) -> None:
        for ref in refs:
            if not ref:
                # blank entry — ignore, not a dangling reference
                continue
            if ref in known:
                continue
            issues.append(
                Issue(
                    severity="error",
                    entity_path=source_path,
                    entity_id=source_id,
                    field=field,
                    rule="id_references_resolve",
                    message=f"Reference {ref!r} from `{field}` does not resolve to any entity.",
                )
            )

    ann = bundle.annotation
    for i, action in enumerate(ann.ego_vehicle.actions):
        base = f"annotation.ego_vehicle.actions[{i}]"
        _check(action.because_of, base, action.id or None, "because_of")
        _check(action.link_to, base, action.id or None, "link_to")
        _check(action.action_target, base, action.id or None, "action_target")
    for i, inf in enumerate(ann.ego_vehicle.influenced_by):
        base = f"annotation.ego_vehicle.influenced_by[{i}]"
        _check(inf.influencers, base, inf.id or None, "influencers")
    for ai, agent in enumerate(ann.agents):
        for j, action in enumerate(agent.actions):
            base = f"annotation.agents[{ai}].actions[{j}]"
            _check(action.because_of, base, action.id or None, "because_of")
            _check(action.link_to, base, action.id or None, "link_to")
            _check(action.action_target, base, action.id or None, "action_target")
        for j, inf in enumerate(agent.influenced_by):
            base = f"annotation.agents[{ai}].influenced_by[{j}]"
            _check(inf.influencers, base, inf.id or None, "influencers")

    return issues


def _check_window(
    issues: list[Issue],
    start_ts: str | None,
    end_ts: str | None,
    *,
    entity_path: str,
    entity_id: str | None,
    duration_s: float,
    start_field: str = "start_timestamp",
    end_field: str = "end_timestamp",
) -> None:
    """Append timestamp issues for one (start, end) window in place."""
    max_t = duration_s + _DURATION_EPS

    def _check_one(raw: str | None, field: str) -> float | None:
        # Treat empty / missing as "no constraint here" — many open intervals
        # are emitted by the annotator UI before both ends are filled in.
        if raw is None or not raw.strip():
            return None
        parsed = parse_timestamp(raw)
        if parsed is None:
            issues.append(
                Issue(
                    severity="error",
                    entity_path=entity_path,
                    entity_id=entity_id,
                    field=field,
                    rule="timestamps_in_video_range",
                    message=f"Timestamp {raw!r} on `{field}` is unparseable.",
                )
            )
            return None
        if parsed < 0 or parsed > max_t:
            issues.append(
                Issue(
                    severity="error",
                    entity_path=entity_path,
                    entity_id=entity_id,
                    field=field,
                    rule="timestamps_in_video_range",
                    message=(
                        f"Timestamp {raw!r} on `{field}` is outside the clip range "
                        f"[0, {duration_s}]."
                    ),
                )
            )
        return parsed

    start = _check_one(start_ts, start_field)
    end = _check_one(end_ts, end_field)
    if start is not None and end is not None and start > end:
        issues.append(
            Issue(
                severity="error",
                entity_path=entity_path,
                entity_id=entity_id,
                field=start_field,
                rule="timestamps_in_video_range",
                message=(
                    f"`{start_field}` ({start_ts!r}) is after `{end_field}` ({end_ts!r})."
                ),
            )
        )


def timestamps_in_video_range(bundle: AnnotationBundle) -> list[Issue]:
    """Every interval-bearing entity's timestamps lie inside ``[0, duration_s]``.

    Walks every entity carrying a (start, end) pair — Environments, Conditions,
    Agents (visibility + every nested action/property/pose/containment/influence),
    TrafficObjects (visibility + nested state/containment), TrafficLights
    (visibility + nested signal-head state/containment), EgoVehicle's actions,
    properties, containments and influences.

    Unparseable timestamps share the rule id (single "timestamps" surface for
    the UI) but emit a distinct ``unparseable`` message so the human review can
    distinguish them from out-of-range values.
    """
    duration_s = bundle.video.duration_s
    issues: list[Issue] = []
    ann = bundle.annotation

    for i, env in enumerate(ann.environments):
        _check_window(
            issues,
            env.start_timestamp,
            env.end_timestamp,
            entity_path=f"annotation.environments[{i}]",
            entity_id=env.id or None,
            duration_s=duration_s,
        )
    for i, cond in enumerate(ann.conditions):
        _check_window(
            issues,
            cond.start_timestamp,
            cond.end_timestamp,
            entity_path=f"annotation.conditions[{i}]",
            entity_id=cond.id or None,
            duration_s=duration_s,
        )
    for i, obj in enumerate(ann.traffic_objects):
        _check_window(
            issues,
            obj.visibility_start_timestamp,
            obj.visibility_end_timestamp,
            entity_path=f"annotation.traffic_objects[{i}]",
            entity_id=obj.id or None,
            duration_s=duration_s,
            start_field="visibility_start_timestamp",
            end_field="visibility_end_timestamp",
        )
        for j, c in enumerate(obj.containment):
            _check_window(
                issues,
                c.start_timestamp,
                c.end_timestamp,
                entity_path=f"annotation.traffic_objects[{i}].containment[{j}]",
                entity_id=c.id or None,
                duration_s=duration_s,
            )
        for j, s in enumerate(obj.state_sequence):
            _check_window(
                issues,
                s.start_timestamp,
                s.end_timestamp,
                entity_path=f"annotation.traffic_objects[{i}].state_sequence[{j}]",
                entity_id=s.id or None,
                duration_s=duration_s,
            )
    for i, tl in enumerate(ann.traffic_lights):
        _check_window(
            issues,
            tl.visibility_start_timestamp,
            tl.visibility_end_timestamp,
            entity_path=f"annotation.traffic_lights[{i}]",
            entity_id=tl.id or None,
            duration_s=duration_s,
            start_field="visibility_start_timestamp",
            end_field="visibility_end_timestamp",
        )
        for j, c in enumerate(tl.containment):
            _check_window(
                issues,
                c.start_timestamp,
                c.end_timestamp,
                entity_path=f"annotation.traffic_lights[{i}].containment[{j}]",
                entity_id=c.id or None,
                duration_s=duration_s,
            )
        for j, head in enumerate(tl.signal_heads):
            _check_window(
                issues,
                head.start_timestamp,
                head.end_timestamp,
                entity_path=f"annotation.traffic_lights[{i}].signal_heads[{j}]",
                entity_id=head.id or None,
                duration_s=duration_s,
            )
            for k, state in enumerate(head.state_sequence):
                _check_window(
                    issues,
                    state.start_timestamp,
                    state.end_timestamp,
                    entity_path=(
                        f"annotation.traffic_lights[{i}].signal_heads[{j}]"
                        f".state_sequence[{k}]"
                    ),
                    entity_id=state.id or None,
                    duration_s=duration_s,
                )
            for k, c in enumerate(head.env_controlled):
                _check_window(
                    issues,
                    c.start_timestamp,
                    c.end_timestamp,
                    entity_path=(
                        f"annotation.traffic_lights[{i}].signal_heads[{j}]"
                        f".env_controlled[{k}]"
                    ),
                    entity_id=c.id or None,
                    duration_s=duration_s,
                )
    for i, agent in enumerate(ann.agents):
        _check_window(
            issues,
            agent.visibility_start_timestamp,
            agent.visibility_end_timestamp,
            entity_path=f"annotation.agents[{i}]",
            entity_id=agent.id or None,
            duration_s=duration_s,
            start_field="visibility_start_timestamp",
            end_field="visibility_end_timestamp",
        )
        for j, action in enumerate(agent.actions):
            _check_window(
                issues,
                action.start_timestamp,
                action.end_timestamp,
                entity_path=f"annotation.agents[{i}].actions[{j}]",
                entity_id=action.id or None,
                duration_s=duration_s,
            )
        for j, prop in enumerate(agent.properties):
            _check_window(
                issues,
                prop.start_timestamp,
                prop.end_timestamp,
                entity_path=f"annotation.agents[{i}].properties[{j}]",
                entity_id=prop.id or None,
                duration_s=duration_s,
            )
        for j, pose in enumerate(agent.ego_relative_pose):
            _check_window(
                issues,
                pose.start_timestamp,
                pose.end_timestamp,
                entity_path=f"annotation.agents[{i}].ego_relative_pose[{j}]",
                entity_id=None,
                duration_s=duration_s,
            )
        for j, c in enumerate(agent.containment):
            _check_window(
                issues,
                c.start_timestamp,
                c.end_timestamp,
                entity_path=f"annotation.agents[{i}].containment[{j}]",
                entity_id=c.id or None,
                duration_s=duration_s,
            )
        for j, inf in enumerate(agent.influenced_by):
            _check_window(
                issues,
                inf.start_timestamp,
                inf.end_timestamp,
                entity_path=f"annotation.agents[{i}].influenced_by[{j}]",
                entity_id=inf.id or None,
                duration_s=duration_s,
            )
    for i, action in enumerate(ann.ego_vehicle.actions):
        _check_window(
            issues,
            action.start_timestamp,
            action.end_timestamp,
            entity_path=f"annotation.ego_vehicle.actions[{i}]",
            entity_id=action.id or None,
            duration_s=duration_s,
        )
    for i, prop in enumerate(ann.ego_vehicle.properties):
        _check_window(
            issues,
            prop.start_timestamp,
            prop.end_timestamp,
            entity_path=f"annotation.ego_vehicle.properties[{i}]",
            entity_id=prop.id or None,
            duration_s=duration_s,
        )
    for i, c in enumerate(ann.ego_vehicle.containment):
        _check_window(
            issues,
            c.start_timestamp,
            c.end_timestamp,
            entity_path=f"annotation.ego_vehicle.containment[{i}]",
            entity_id=c.id or None,
            duration_s=duration_s,
        )
    for i, inf in enumerate(ann.ego_vehicle.influenced_by):
        _check_window(
            issues,
            inf.start_timestamp,
            inf.end_timestamp,
            entity_path=f"annotation.ego_vehicle.influenced_by[{i}]",
            entity_id=inf.id or None,
            duration_s=duration_s,
        )

    return issues


HARD_RULES: tuple[Rule, ...] = (
    environment_requires_type,
    condition_requires_type,
    agent_requires_type,
    traffic_object_requires_type,
    ego_has_at_least_one_action,
    id_references_resolve,
    timestamps_in_video_range,
)

# Soft rules are warnings that surface to the UI but do not block completion.
# Empty for now; populate as classification rules emerge from real annotator
# usage.
SOFT_RULES: tuple[Rule, ...] = ()


def validate(bundle: AnnotationBundle) -> list[Issue]:
    """Run every rule and return the merged issue list (errors then warnings).

    Pure function — does not mutate ``bundle``. Order is stable: hard rules
    in :data:`HARD_RULES` order, then soft rules in :data:`SOFT_RULES` order.
    """
    issues: list[Issue] = []
    for rule in HARD_RULES:
        issues.extend(rule(bundle))
    for rule in SOFT_RULES:
        issues.extend(rule(bundle))
    return issues


__all__ = [
    "HARD_RULES",
    "Issue",
    "Rule",
    "SOFT_RULES",
    "Severity",
    "agent_requires_type",
    "condition_requires_type",
    "ego_has_at_least_one_action",
    "environment_requires_type",
    "id_references_resolve",
    "timestamps_in_video_range",
    "traffic_object_requires_type",
    "validate",
]
