# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Causal-triplet extraction.

A triplet is the smallest unit of relational structure in this corpus:

    (subject_action, predicate, cause)

where:
- `subject` is the action whose `because_of` list contained `cause_id`,
- `predicate` is the subject's action type (`action_type` for agents,
  `type` for ego), and
- `cause` is the entity referenced (or `None` if the reference dangles).

One `because_of` entry produces exactly one triplet. An action with
`because_of = ["A", "B"]` produces two triplets — one per cause.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from causal_ai_av.query.index import EGO_ID, IdIndex, Subject
from causal_ai_av.query.time import Interval
from causal_ai_av.spec import AnnotationBundle


@dataclass(frozen=True, slots=True)
class CausalTriplet:
    """One `because_of` link, resolved against the rest of the bundle."""

    clip_id: str
    subject: Subject  # the *agent* or *ego* that performed the action
    subject_action: Subject  # the action object itself (kind: agent_action | ego_action)
    predicate: str  # action type string (e.g. `fst:Yield`)
    cause_id: str  # raw `because_of` entry, before resolution
    cause: Subject | None  # resolved cause, or `None` if the reference dangles
    interval: Interval | None  # subject action's `[start, end]`


def extract_causal_triplets(bundle: AnnotationBundle) -> list[CausalTriplet]:
    """Walk every `because_of` link in a bundle and emit one triplet per link."""
    idx = IdIndex(bundle)
    out: list[CausalTriplet] = []
    clip_id = bundle.video.clip_id

    # Agent actions — subject is the parent Agent, action is the AgentAction.
    for agent in bundle.annotation.agents:
        subject = Subject("agent", agent.id, agent)
        for action in agent.actions:
            if not action.because_of:
                continue
            subject_action = Subject("agent_action", action.id, action)
            iv = Interval.from_strings(action.start_timestamp, action.end_timestamp)
            for cause_id in action.because_of:
                out.append(
                    CausalTriplet(
                        clip_id=clip_id,
                        subject=subject,
                        subject_action=subject_action,
                        predicate=action.action_type,
                        cause_id=cause_id,
                        cause=idx.get(cause_id),
                        interval=iv,
                    )
                )

    # Ego actions — subject is the synthetic "Ego" anchor.
    ego_subject = Subject("ego", EGO_ID, bundle.annotation.ego_vehicle)
    for action in bundle.annotation.ego_vehicle.actions:
        if not action.because_of:
            continue
        subject_action = Subject("ego_action", action.id, action)
        iv = Interval.from_strings(action.start_timestamp, action.end_timestamp)
        for cause_id in action.because_of:
            out.append(
                CausalTriplet(
                    clip_id=clip_id,
                    subject=ego_subject,
                    subject_action=subject_action,
                    predicate=action.type,
                    cause_id=cause_id,
                    cause=idx.get(cause_id),
                    interval=iv,
                )
            )

    return out


def iter_triplets(bundles: Iterable[AnnotationBundle]) -> Iterator[CausalTriplet]:
    """Flatten triplets across many bundles."""
    for b in bundles:
        yield from extract_causal_triplets(b)
