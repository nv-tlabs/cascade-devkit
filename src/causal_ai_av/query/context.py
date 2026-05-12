# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context windows — *what else was happening* during a match.

Given a clip-relative `[t0, t1]` window (typically a `Match.interval`),
`context_at` returns every annotated entity that overlaps the window:
visible agents and their active actions, ego actions, environments,
conditions, traffic-light states, and persistent traffic objects.

Built on the per-kind helpers in `temporal` and `spatial` — no engine
changes required.
"""

from __future__ import annotations

from dataclasses import dataclass

from causal_ai_av.query.spatial import agent_visibility_interval
from causal_ai_av.query.temporal import filter_active_in_range
from causal_ai_av.query.time import Interval
from causal_ai_av.spec import (
    Agent,
    AgentAction,
    AnnotationBundle,
    Condition,
    EgoAction,
    Environment,
    LightStates,
    SignalHead,
    TrafficObject,
)


@dataclass(frozen=True)
class AgentInWindow:
    """An agent that was visible at some point during the window."""

    agent: Agent
    visibility: Interval
    """Intersection of the agent's overall visibility window with the query window."""

    actions: list[AgentAction]
    """Actions of this agent whose interval overlaps the query window."""


@dataclass(frozen=True)
class LightStateInWindow:
    """A signal-head state entry whose interval overlaps the window."""

    head: SignalHead
    state: LightStates


@dataclass(frozen=True)
class ContextWindow:
    """Everything that was happening in a clip during a given window."""

    clip_id: str
    interval: Interval
    agents: list[AgentInWindow]
    ego_actions: list[EgoAction]
    environments: list[Environment]
    conditions: list[Condition]
    light_states: list[LightStateInWindow]
    traffic_objects: list[TrafficObject]


def context_at(bundle: AnnotationBundle, interval: Interval) -> ContextWindow:
    """Snapshot every entity in `bundle` that overlaps `interval`.

    `interval` is in clip-relative seconds, the same time base as
    `Match.interval`. Entities are included if any of their annotated
    time ranges (visibility window, action interval, state-sequence
    entry, …) overlap `interval`.

    Traffic objects without a `state_sequence` (persistent signs,
    barriers) are always included — they're considered present for the
    whole clip.
    """
    t0, t1 = interval.start, interval.end

    agents: list[AgentInWindow] = []
    for ag in bundle.annotation.agents:
        vis = agent_visibility_interval(ag)
        if vis is None or not vis.overlaps(interval):
            continue
        clipped = Interval(max(vis.start, t0), min(vis.end, t1))
        active = filter_active_in_range(ag.actions, t0, t1)
        agents.append(AgentInWindow(agent=ag, visibility=clipped, actions=active))

    ego_actions = filter_active_in_range(bundle.annotation.ego_vehicle.actions, t0, t1)
    environments = filter_active_in_range(bundle.annotation.environments, t0, t1)
    conditions = filter_active_in_range(bundle.annotation.conditions, t0, t1)

    light_states: list[LightStateInWindow] = []
    for tl in bundle.annotation.traffic_lights:
        for head in tl.signal_heads:
            for state in filter_active_in_range(head.state_sequence, t0, t1):
                light_states.append(LightStateInWindow(head=head, state=state))

    traffic_objects: list[TrafficObject] = []
    for obj in bundle.annotation.traffic_objects:
        if not obj.state_sequence:
            traffic_objects.append(obj)
            continue
        if filter_active_in_range(obj.state_sequence, t0, t1):
            traffic_objects.append(obj)

    return ContextWindow(
        clip_id=bundle.video.clip_id,
        interval=interval,
        agents=agents,
        ego_actions=ego_actions,
        environments=environments,
        conditions=conditions,
        light_states=light_states,
        traffic_objects=traffic_objects,
    )


__all__ = [
    "AgentInWindow",
    "LightStateInWindow",
    "ContextWindow",
    "context_at",
]
