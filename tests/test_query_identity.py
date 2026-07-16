# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identity projection and optional strict-identity query semantics."""

from __future__ import annotations

from cascade_av.dataset import CascadeDataset, Sequence
from cascade_av.query import IDLESS_RECORD_ID, Match, evaluate
from cascade_av.query.api import (
    count_on_dataset,
    find_on_bundle,
    find_on_dataset,
    group_by_on_dataset,
)
from cascade_av.query.dsl import parse
from cascade_av.query.index import EGO_ID
from cascade_av.spec import AnnotationBundle


def _identity_bundle() -> AnnotationBundle:
    return AnnotationBundle.model_validate(
        {
            "video": {"clip_id": "identity-test", "duration_s": 10.0},
            "annotation": {
                "ego_vehicle": {
                    "driving_judgment": "good",
                    "influenced_by": [
                        {
                            "id": "ego-influence",
                            "influencers": ["obj-1"],
                            "start_timestamp": "0:1.0",
                            "end_timestamp": "0:2.0",
                        },
                        {
                            # Deliberately id-less: default evaluation keeps it.
                            "influencers": ["obj-1"],
                            "start_timestamp": "0:3.0",
                            "end_timestamp": "0:4.0",
                        },
                    ],
                },
                "agents": [
                    {
                        "id": "agent-1",
                        "type": "Pedestrian",
                        "visibility_start_timestamp": "0:0.0",
                        "visibility_end_timestamp": "0:9.0",
                        "actions": [
                            {
                                "id": "action-1",
                                "action_type": "oxd:Walk",
                                "start_timestamp": "0:1.0",
                                "end_timestamp": "0:2.0",
                            },
                            {
                                # Same predicate as action-1, but no ID.
                                "action_type": "oxd:Walk",
                                "start_timestamp": "0:3.0",
                                "end_timestamp": "0:4.0",
                            },
                            {
                                # Exercises the motivating id-less because_of LHS.
                                "action_type": "fst:Yield",
                                "because_of": ["obj-1"],
                                "start_timestamp": "0:5.0",
                                "end_timestamp": "0:6.0",
                            },
                        ],
                    },
                    {
                        # Bare influenced_by owners are candidates too.
                        "type": "oxd:Vehicle",
                        "influenced_by": [
                            {
                                "id": "agent-influence",
                                "influencers": ["obj-1"],
                                "start_timestamp": "0:1.0",
                                "end_timestamp": "0:2.0",
                            }
                        ],
                    },
                ],
                "traffic_objects": [
                    {
                        "id": "obj-1",
                        "type": "fst:StopSign",
                        "visibility_start_timestamp": "0:0.0",
                        "visibility_end_timestamp": "0:10.0",
                    }
                ],
            },
        }
    )


class _DatasetHarness:
    """Small weak-referenceable stand-in for the dataset API wrappers."""

    def __init__(self, bundle: AnnotationBundle) -> None:
        self._by_clip = {bundle.video.clip_id: (None, None, bundle)}


class _SequenceHarness:
    def __init__(self, bundle: AnnotationBundle) -> None:
        self.annotation = bundle
        self._parent = None


def test_match_record_ids_project_scalars_and_recursive_tuples() -> None:
    bundle = _identity_bundle()
    identified, idless = bundle.annotation.agents[0].actions[:2]
    ego = bundle.annotation.ego_vehicle
    obj = bundle.annotation.traffic_objects[0]

    match = Match(
        bundle.video.clip_id,
        ((identified, idless), (ego, (obj, identified))),
        interval=None,
    )
    assert IDLESS_RECORD_ID is None
    assert match.record_ids == (
        "action-1",
        IDLESS_RECORD_ID,
        EGO_ID,
        "obj-1",
        "action-1",
    )
    assert Match(bundle.video.clip_id, bundle, None).record_ids == ("identity-test",)


def test_strict_identity_filters_idless_direct_candidates_default_off() -> None:
    bundle = _identity_bundle()
    expr = parse("agent.action.type = walk")

    default = evaluate(expr, bundle)
    explicit_default = evaluate(expr, bundle, strict_identity=False)
    strict = evaluate(expr, bundle, strict_identity=True)

    assert default == explicit_default
    assert [m.record_ids for m in default.matches] == [
        ("action-1",),
        (IDLESS_RECORD_ID,),
    ]
    assert [m.record_ids for m in strict.matches] == [("action-1",)]

    # Clip and ego are intentional roots without schema `id` fields; strict
    # mode must not remove them.
    ego_matches = evaluate(parse("ego.judgment = good"), bundle, strict_identity=True)
    assert [m.record_ids for m in ego_matches.matches] == [(EGO_ID,)]


def test_strict_identity_reaches_nested_because_of_and_within_candidates() -> None:
    bundle = _identity_bundle()

    nested = "agent(type = ped and action(type = yield))"
    assert len(find_on_bundle(bundle, nested)) == 1
    assert len(find_on_bundle(bundle, nested, strict_identity=True)) == 0

    causal = "agent.action.type = yield because_of obj.type = stop_sign"
    default_causal = find_on_bundle(bundle, causal)
    assert [m.record_ids for m in default_causal.matches] == [
        (IDLESS_RECORD_ID,),
        ("obj-1",),
    ]
    assert not find_on_bundle(bundle, causal, strict_identity=True)

    windowed = "within agent.action.type = yield: obj.type = stop_sign"
    assert len(find_on_bundle(bundle, windowed)) == 1
    assert not find_on_bundle(bundle, windowed, strict_identity=True)


def test_strict_identity_filters_influence_records_and_idless_owners() -> None:
    bundle = _identity_bundle()

    ego_query = parse("ego influenced_by obj.type = stop_sign")
    assert len(evaluate(ego_query, bundle)) == 2
    strict_ego = evaluate(ego_query, bundle, strict_identity=True)
    assert len(strict_ego) == 1
    assert strict_ego.matches[0].record_ids == (EGO_ID,)

    agent_query = parse("agent influenced_by obj.type = stop_sign")
    default_agent = evaluate(agent_query, bundle)
    assert [m.record_ids for m in default_agent.matches] == [(IDLESS_RECORD_ID,)]
    assert not evaluate(agent_query, bundle, strict_identity=True)


def test_strict_identity_propagates_through_public_query_surfaces() -> None:
    bundle = _identity_bundle()
    dataset = _DatasetHarness(bundle)
    sequence = _SequenceHarness(bundle)
    query = "agent.action.type = yield"

    assert len(find_on_dataset(dataset, query)) == 1
    assert not find_on_dataset(dataset, query, strict_identity=True)
    assert count_on_dataset(dataset, query) == 1
    assert count_on_dataset(dataset, query, strict_identity=True) == 0
    assert group_by_on_dataset(dataset, query, "agent.action.type") == {"fst:Yield": 1}
    assert group_by_on_dataset(dataset, query, "agent.action.type", strict_identity=True) == {}

    # Exercise the thin dataset/sequence methods without constructing the HF
    # parent; each method only delegates to the wrappers above.
    assert len(CascadeDataset.find(dataset, query)) == 1
    assert not CascadeDataset.find(dataset, query, strict_identity=True)
    assert CascadeDataset.count(dataset, query, strict_identity=True) == 0
    assert CascadeDataset.group_by(dataset, query, "agent.action.type", strict_identity=True) == {}
    assert CascadeDataset.histogram(dataset, query, "agent.action.type", strict_identity=True) == {}
    assert len(Sequence.find(sequence, query)) == 1
    assert not Sequence.find(sequence, query, strict_identity=True)
