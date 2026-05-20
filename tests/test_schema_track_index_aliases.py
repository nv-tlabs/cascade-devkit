# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin the family-specific underscore-prefixed UI hint keys to typed fields.

The upstream `sil-dense-annotation-tool 0.4.5` emits a different
underscore prefix per nested family:

    Agent / Environment / TrafficObject / TrafficLight   -> _track_index
    Containment                                          -> _cont_track_index
    AgentProperty / EgoProperty                          -> _prop_track_index
    Condition                                            -> _cond_track_index
    Influence (ego + agent)                              -> _influence_track_index
    SignalState                                          -> _state_track_index
    SignalHead                                           -> _state_lane_count

Each must bind to a typed Pydantic field so callers can read it via the
Python attribute (not via `model_extra`) and so dump emits the correct
prefix back to disk. Regression for issues #124 + #125.
"""
from __future__ import annotations

from cascade_av.spec import AnnotationBundle
from cascade_av.spec.schema import (
    Agent,
    AgentProperty,
    Containment,
    Environment,
    SignalHead,
)


def test_containment_track_index_binds_to_cont_track_index_alias() -> None:
    cont = Containment.model_validate({"id": "c1", "_cont_track_index": 7})
    assert cont.track_index == 7
    assert cont.model_extra == {}  # nothing fell through to extras
    dumped = cont.model_dump(by_alias=True, exclude_none=True)
    assert dumped.get("_cont_track_index") == 7
    assert "_track_index" not in dumped  # the old (wrong) alias is gone


def test_agent_property_track_index_binds_to_prop_track_index_alias() -> None:
    prop = AgentProperty.model_validate({"id": "p1", "_prop_track_index": 3})
    assert prop.track_index == 3
    assert prop.model_extra == {}
    dumped = prop.model_dump(by_alias=True, exclude_none=True)
    assert dumped.get("_prop_track_index") == 3


def test_signal_head_state_lane_count_binds_to_state_lane_count_alias() -> None:
    sh = SignalHead.model_validate({"id": "sh1", "_state_lane_count": 2})
    assert sh.state_lane_count == 2
    assert sh.model_extra == {}
    dumped = sh.model_dump(by_alias=True, exclude_none=True)
    assert dumped.get("_state_lane_count") == 2


def test_agent_top_level_track_index_still_uses_plain_track_index_alias() -> None:
    # Agents (and Environment/TrafficObject/TrafficLight) keep the bare
    # `_track_index` prefix — only the *nested* families switched. Guard
    # against an accidental cross-rename.
    agent = Agent.model_validate({"id": "a1", "_track_index": 5})
    assert agent.track_index == 5
    dumped = agent.model_dump(by_alias=True, exclude_none=True)
    assert dumped.get("_track_index") == 5
    env = Environment.model_validate({"id": "e1", "_track_index": 2})
    assert env.track_index == 2
    assert env.model_dump(by_alias=True, exclude_none=True).get("_track_index") == 2


def test_round_trip_through_bundle_preserves_nested_family_aliases() -> None:
    src = {
        "video": {"clip_id": "clip", "fps": 30.0, "duration_s": 1.0},
        "schema_version": "2.0.0",
        "annotation": {
            "environments": [
                {"id": "Environment1", "type": "oxd:Road", "_track_index": 0}
            ],
            "agents": [
                {
                    "id": "Agent1",
                    "type": "oxd:Car",
                    "_track_index": 0,
                    "containment": [
                        {
                            "id": "AgentContainment1",
                            "env_id": "Environment1",
                            "_cont_track_index": 4,
                        }
                    ],
                    "properties": [
                        {
                            "id": "AgentProperty1",
                            "property_type": "Aggressive",
                            "_prop_track_index": 1,
                        }
                    ],
                }
            ],
            "traffic_lights": [
                {
                    "id": "TL1",
                    "signal_heads": [
                        {"id": "SH1", "_state_lane_count": 3}
                    ],
                }
            ],
        },
    }
    bundle = AnnotationBundle.model_validate(src)
    cont = bundle.annotation.agents[0].containment[0]
    prop = bundle.annotation.agents[0].properties[0]
    sh = bundle.annotation.traffic_lights[0].signal_heads[0]
    assert cont.track_index == 4
    assert prop.track_index == 1
    assert sh.state_lane_count == 3
    # None of these are sitting in model_extra anymore
    assert cont.model_extra == {}
    assert prop.model_extra == {}
    assert sh.model_extra == {}
    # Round-tripped dump emits the right per-family alias
    dumped = bundle.model_dump(mode="json", by_alias=True, exclude_none=True)
    dumped_cont = dumped["annotation"]["agents"][0]["containment"][0]
    dumped_prop = dumped["annotation"]["agents"][0]["properties"][0]
    dumped_sh = dumped["annotation"]["traffic_lights"][0]["signal_heads"][0]
    assert dumped_cont.get("_cont_track_index") == 4
    assert "_track_index" not in dumped_cont
    assert dumped_prop.get("_prop_track_index") == 1
    assert dumped_sh.get("_state_lane_count") == 3
