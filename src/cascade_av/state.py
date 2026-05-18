# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derived snapshot types — "what is active in the scene at time `t` (or in `[t1, t2]`)".

These are plain frozen dataclasses, *not* Pydantic models, because they are
computed views over an `AnnotationBundle` rather than serialized state. The
authoritative on-disk schema lives in `cascade_av.spec`.

The point-in-time snapshot is `SequenceState`; the window aggregate is
`SequenceStateRange`. Both are populated by `Sequence.state_at(t, t_end=None)`.

Active-at-t semantics used to populate these (see `CascadeDataset` for the
implementation):

- Actions, properties, containments, influences, conditions, environments,
  agent visibility, traffic-object visibility, traffic-light visibility:
  `Interval.from_strings(start, end).contains(t)`.
- Triplets: a triplet is active at `t` when its subject-action interval
  contains `t`.

Range counterparts use `Interval.overlaps(Interval(t1, t2))` instead of
`contains`.

Bounding-box geometry is no longer part of the base schema (schema_version
2.1.0+). When a future bbox extension lands, its per-frame data will be
attached to the bundle via :meth:`AnnotationBundle.ext` rather than being
mirrored on these snapshot dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cascade_av.query import CausalTriplet
from cascade_av.spec import (
    Agent,
    AgentAction,
    AgentProperty,
    Condition,
    Containment,
    EgoAction,
    EgoRelativePose,
    Environment,
    Influence,
    TrafficLight,
    TrafficObject,
)

if TYPE_CHECKING:  # pragma: no cover — typing only
    # `EgomotionState` lives in the parent DevKit which is an optional
    # install-time dependency. Avoid importing it at module top so this
    # module is usable without the `[hf]` extra. The string-quoted
    # annotations below resolve via this import under type-checkers;
    # `from __future__ import annotations` keeps them as strings at
    # runtime so the `else: EgomotionState = Any` fallback that used to
    # live here is unnecessary.
    from physical_ai_av.egomotion import EgomotionState


# -----------------------------------------------------------------------------
# Point-in-time snapshot
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ActorState:
    """An agent's active state at one instant in time."""

    agent: Agent
    actions: list[AgentAction]               # actions whose [start, end] contains t
    pose_rel_to_ego: EgoRelativePose | None  # pose entry whose interval contains t


@dataclass(frozen=True, slots=True)
class EgoState:
    """The ego vehicle's active state at one instant in time."""

    actions: list[EgoAction]
    properties: list[AgentProperty]
    containment: list[Containment]
    influenced_by: list[Influence]
    pose: "EgomotionState | None"            # interpolated pose at t, or None


@dataclass(frozen=True, slots=True)
class SequenceState:
    """Everything active in the scene at instant `t`."""

    t: float
    clip_id: str
    ego: EgoState
    actors: list[ActorState]
    environments: list[Environment]
    conditions: list[Condition]
    traffic_objects: list[TrafficObject]
    traffic_lights: list[TrafficLight]
    triplets: list[CausalTriplet]


# -----------------------------------------------------------------------------
# Range aggregate — "alive at any point in [t1, t2]"
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ActorStateRange:
    """An agent's aggregate state over a window."""

    agent: Agent
    actions: list[AgentAction]               # interval overlaps [t1, t2]
    pose_rel_to_ego: list[EgoRelativePose]   # entries whose interval overlaps [t1, t2]


@dataclass(frozen=True, slots=True)
class EgoStateRange:
    """The ego vehicle's aggregate state over a window."""

    actions: list[EgoAction]
    properties: list[AgentProperty]
    containment: list[Containment]
    influenced_by: list[Influence]
    pose_trajectory: "list[EgomotionState]"  # sampled across [t1, t2]


@dataclass(frozen=True, slots=True)
class SequenceStateRange:
    """Everything alive at any point in `[t_start, t_end]`."""

    t_start: float
    t_end: float
    clip_id: str
    ego: EgoStateRange
    actors: list[ActorStateRange]
    environments: list[Environment]
    conditions: list[Condition]
    traffic_objects: list[TrafficObject]
    traffic_lights: list[TrafficLight]
    triplets: list[CausalTriplet]


__all__ = [
    "ActorState",
    "ActorStateRange",
    "EgoState",
    "EgoStateRange",
    "SequenceState",
    "SequenceStateRange",
]
