# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spatial queries over `AnnotationBundle`.

This corpus's spatial grounding is qualitative: agents carry an
`ego_relative_pose` series describing where they are *relative to the ego
vehicle* at each time window (e.g. `position_rel_to_ego="In front"`). Hard
metric positions used to live in `bounding_boxes` but were removed from the
base schema in 2.1.0; image-space geometry will return as a schema extension
(see :mod:`cascade_av.extensions`).
"""

from __future__ import annotations

from cascade_av.query.time import Interval
from cascade_av.spec import Agent, AnnotationBundle, EgoRelativePose


def agent_visibility_interval(agent: Agent) -> Interval | None:
    """The agent's overall visibility window in clip-relative seconds."""
    return Interval.from_strings(
        agent.visibility_start_timestamp, agent.visibility_end_timestamp
    )


def agents_visible_at(bundle: AnnotationBundle, t: float) -> list[Agent]:
    """Every agent whose visibility window covers clip-time `t` (seconds)."""
    out: list[Agent] = []
    for agent in bundle.annotation.agents:
        iv = agent_visibility_interval(agent)
        if iv and iv.contains(t):
            out.append(agent)
    return out


def ego_relative_pose_at(agent: Agent, t: float) -> EgoRelativePose | None:
    """The first `ego_relative_pose` entry whose interval covers `t`."""
    for pose in agent.ego_relative_pose:
        iv = Interval.from_strings(pose.start_timestamp, pose.end_timestamp)
        if iv and iv.contains(t):
            return pose
    return None


def agents_in_position(
    bundle: AnnotationBundle, position: str, at_time: float
) -> list[Agent]:
    """Agents whose `position_rel_to_ego` equals `position` at clip-time `at_time`.

    `position` is a free-form string — compared against the corpus values such
    as `"In front"`, `"Left"`, `"Right"`, `"Behind"`. The match is exact.
    """
    out: list[Agent] = []
    for agent in agents_visible_at(bundle, at_time):
        pose = ego_relative_pose_at(agent, at_time)
        if pose and pose.position_rel_to_ego == position:
            out.append(agent)
    return out
