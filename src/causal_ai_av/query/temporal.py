"""Temporal queries over `AnnotationBundle`."""

from __future__ import annotations

from typing import Iterable, TypeVar

from causal_ai_av.query.time import Interval
from causal_ai_av.spec import AgentAction, AnnotationBundle, EgoAction

_T = TypeVar("_T")


def action_interval(action: AgentAction | EgoAction) -> Interval | None:
    """The `[start, end]` interval of an action, or `None` if timestamps are missing."""
    return Interval.from_strings(action.start_timestamp, action.end_timestamp)


def actions_at(
    bundle: AnnotationBundle, t: float
) -> list[tuple[str, str, AgentAction | EgoAction]]:
    """Every action active at clip-relative time `t` (seconds).

    Returns tuples of `(subject_kind, subject_id, action)` where `subject_kind`
    is `"agent"` or `"ego"`.
    """
    out: list[tuple[str, str, AgentAction | EgoAction]] = []
    for agent in bundle.annotation.agents:
        for action in agent.actions:
            iv = action_interval(action)
            if iv and iv.contains(t):
                out.append(("agent", agent.id, action))
    for action in bundle.annotation.ego_vehicle.actions:
        iv = action_interval(action)
        if iv and iv.contains(t):
            out.append(("ego", "Ego", action))
    return out


def overlapping_actions(
    bundle: AnnotationBundle,
) -> list[tuple[tuple[str, str, AgentAction | EgoAction], tuple[str, str, AgentAction | EgoAction]]]:
    """All distinct pairs of actions whose intervals overlap.

    Pairs are unordered (each pair appears once); same-subject pairs are
    included — they represent concurrent sub-actions on the same actor.
    """
    flat: list[tuple[str, str, AgentAction | EgoAction, Interval]] = []
    for agent in bundle.annotation.agents:
        for action in agent.actions:
            iv = action_interval(action)
            if iv:
                flat.append(("agent", agent.id, action, iv))
    for action in bundle.annotation.ego_vehicle.actions:
        iv = action_interval(action)
        if iv:
            flat.append(("ego", "Ego", action, iv))

    out = []
    for i, (k1, s1, a1, iv1) in enumerate(flat):
        for k2, s2, a2, iv2 in flat[i + 1 :]:
            if iv1.overlaps(iv2):
                out.append(((k1, s1, a1), (k2, s2, a2)))
    return out


def filter_active_at(
    items: Iterable[_T],
    t: float,
    *,
    start_attr: str = "start_timestamp",
    end_attr: str = "end_timestamp",
) -> list[_T]:
    """Items whose `[start, end]` interval contains `t`.

    Pulls interval bounds off each item using `getattr(item, start_attr)` /
    `getattr(item, end_attr)`. Items with unparseable timestamps are dropped.
    """
    out: list[_T] = []
    for item in items:
        iv = Interval.from_strings(getattr(item, start_attr), getattr(item, end_attr))
        if iv is not None and iv.contains(t):
            out.append(item)
    return out


def filter_active_in_range(
    items: Iterable[_T],
    t1: float,
    t2: float,
    *,
    start_attr: str = "start_timestamp",
    end_attr: str = "end_timestamp",
) -> list[_T]:
    """Items whose `[start, end]` interval overlaps `[t1, t2]`.

    Pulls interval bounds off each item using `getattr(item, start_attr)` /
    `getattr(item, end_attr)`. Items with unparseable timestamps are dropped.
    """
    window = Interval(t1, t2)
    out: list[_T] = []
    for item in items:
        iv = Interval.from_strings(getattr(item, start_attr), getattr(item, end_attr))
        if iv is not None and iv.overlaps(window):
            out.append(item)
    return out
