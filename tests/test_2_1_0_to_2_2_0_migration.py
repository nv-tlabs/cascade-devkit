# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the 2.1.0 → 2.2.0 migration step.

Covers both the single-hop case (a freshly-migrated 2.1.0 bundle) and the
two-hop chain (2.0.0 → 2.1.0 → 2.2.0 in one CLI invocation, exercising
``_migration_path`` BFS).
"""

from __future__ import annotations

import json
from pathlib import Path

from cascade_av.cli.migrate import (
    MIGRATIONS,
    _migrate_2_1_0_to_2_2_0,
    _migration_path,
    main,
)
from cascade_av.io.local import _sidecar_path_for
from cascade_av.spec import CURRENT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Registry hygiene
# ---------------------------------------------------------------------------


def test_migration_pair_is_registered() -> None:
    assert ("2.1.0", "2.2.0") in MIGRATIONS


def test_migration_path_walks_two_hops_from_2_0_0() -> None:
    path = _migration_path("2.0.0", "2.2.0")
    assert path == [("2.0.0", "2.1.0"), ("2.1.0", "2.2.0")]


def test_current_schema_is_2_2_0() -> None:
    assert CURRENT_SCHEMA_VERSION == "2.2.0"


# ---------------------------------------------------------------------------
# Function-level: _migrate_2_1_0_to_2_2_0 on raw dicts
# ---------------------------------------------------------------------------


def _bundle_2_1_0_with_indices() -> dict[str, object]:
    return {
        "schema_version": "2.1.0",
        "video": {"clip_id": "x", "fps": 30.0, "duration_s": 5.0},
        "annotation": {
            "environments": [
                {"id": "Env1", "type": "oxd:Road", "_track_index": 0,
                 "start_timestamp": "0:0.0", "end_timestamp": "0:5.0"},
                {"id": "Env2", "type": "oxd:Road", "_track_index": 1,
                 "start_timestamp": "0:0.0", "end_timestamp": "0:5.0"},
            ],
            "conditions": [
                {"id": "C1", "env_id": "Env1", "type": ["Clear"],
                 "_track_index": 0, "_cond_track_index": 0,
                 "start_timestamp": "0:0.0", "end_timestamp": "0:5.0"},
            ],
            "traffic_lights": [
                {"id": "L1", "type": "fst:RegularTrafficLight",
                 "_track_index": 0,
                 "visibility_start_timestamp": "0:0.0",
                 "visibility_end_timestamp": "0:5.0",
                 "signal_heads": [
                     {"id": "H1", "state_sequence": [
                         {"id": "S1", "_state_track_index": 0,
                          "start_timestamp": "0:0.0",
                          "end_timestamp": "0:5.0"},
                     ]},
                 ]},
            ],
            "agents": [
                {"id": "A1", "amount": "Single", "type": "oxd:Car",
                 "_track_index": 0,
                 "visibility_start_timestamp": "0:0.0",
                 "visibility_end_timestamp": "0:5.0",
                 "actions": [],
                 "containment": [
                     {"id": "Co1", "env_id": "Env1",
                      "_track_index": 0, "_cont_track_index": 0,
                      "start_timestamp": "0:0.0", "end_timestamp": "0:5.0"},
                 ],
                 "influenced_by": [
                     {"id": "I1", "influencers": ["L1"],
                      "_influence_track_index": 0,
                      "start_timestamp": "0:0.0",
                      "end_timestamp": "0:5.0"},
                 ],
                 "properties": [
                     {"id": "P1", "property_type": "Parked",
                      "_prop_track_index": 0,
                      "start_timestamp": "0:0.0",
                      "end_timestamp": "0:5.0"},
                 ]},
            ],
        },
    }


def test_migration_lifts_indices_into_sidecar() -> None:
    main_dict, sidecar = _migrate_2_1_0_to_2_2_0(_bundle_2_1_0_with_indices())
    assert main_dict["schema_version"] == "2.2.0"
    # No inline indices remain anywhere in the main payload.
    main_json = json.dumps(main_dict)
    for name in (
        "_track_index", "_cond_track_index", "_cont_track_index",
        "_state_track_index", "_influence_track_index", "_prop_track_index",
    ):
        assert name not in main_json, f"{name} leaked into main JSON"
    # And every index lands in the right sub-map of the ui payload.
    ui = sidecar["ui/1.0"]
    assert ui["track_index"] == {
        "Env1": 0, "Env2": 1, "C1": 0, "L1": 0, "A1": 0, "Co1": 0,
    }
    assert ui["cond_track_index"] == {"C1": 0}
    assert ui["state_track_index"] == {"S1": 0}
    assert ui["cont_track_index"] == {"Co1": 0}
    assert ui["influence_track_index"] == {"I1": 0}
    assert ui["prop_track_index"] == {"P1": 0}


def test_migration_without_any_indices_emits_no_sidecar() -> None:
    bare = {
        "schema_version": "2.1.0",
        "video": {"clip_id": "x", "fps": 30.0, "duration_s": 1.0},
        "annotation": {"environments": [], "agents": []},
    }
    main_dict, sidecar = _migrate_2_1_0_to_2_2_0(bare)
    assert main_dict["schema_version"] == "2.2.0"
    assert sidecar == {}


def test_migration_entity_missing_id_skips_index_silently() -> None:
    """If an entity has no `id`, its index has nowhere stable to land.

    The corpus's typed schema requires `id` on the entities that carry a
    `track_index`, but nested models (Containment, LightStates, etc.) had
    `id = ""` as default — a future bundle without an id should not crash
    the migration; the index is simply dropped.
    """
    payload = {
        "schema_version": "2.1.0",
        "annotation": {
            "agents": [
                {"id": "A1", "amount": "Single", "type": "oxd:Car",
                 "visibility_start_timestamp": "0:0.0",
                 "visibility_end_timestamp": "0:5.0",
                 "actions": [],
                 "containment": [
                     # no `id` field at all
                     {"env_id": "Env1", "_cont_track_index": 7,
                      "start_timestamp": "0:0.0", "end_timestamp": "0:5.0"},
                 ]}
            ]
        },
    }
    main_dict, sidecar = _migrate_2_1_0_to_2_2_0(payload)
    # The inline field is gone (so it does not survive into 2.2.0),
    # and the sidecar carries no orphan key.
    ag = main_dict["annotation"]["agents"][0]
    assert "_cont_track_index" not in ag["containment"][0]
    assert sidecar == {}


# ---------------------------------------------------------------------------
# CLI-level: two-hop chain on a real file
# ---------------------------------------------------------------------------


def _write_bundle(path: Path, bundle: dict[str, object]) -> None:
    path.write_text(json.dumps(bundle, indent=2))


def test_cli_two_hop_2_0_0_to_2_2_0_writes_both_extensions(tmp_path: Path) -> None:
    """Single CLI invocation against a 2.0.0 file with inline bboxes AND
    inline indices migrates through 2.1.0 to 2.2.0, with both extensions
    landing in the sidecar.
    """
    src = tmp_path / "twohop.json"
    _write_bundle(src, {
        "schema_version": "2.0.0",
        "video": {"clip_id": "twohop", "fps": 30.0, "duration_s": 5.0},
        "annotation": {
            "agents": [{
                "id": "A1", "amount": "Single", "type": "oxd:Car",
                "_track_index": 2,
                "visibility_start_timestamp": "0:0.0",
                "visibility_end_timestamp": "0:5.0",
                "actions": [],
                "bounding_boxes": [
                    {"timestamp": "0:0.0", "bounding_boxes": [
                        {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2,
                         "object_id": "A1", "object_class": "oxd:Car"},
                    ]},
                ],
            }],
            "traffic_objects": [],
            "traffic_lights": [],
            "environments": [],
            "conditions": [],
        },
    })

    assert main([str(src)]) == 0
    written_main = json.loads(src.read_text())
    assert written_main["schema_version"] == "2.2.0"
    # Both bbox and indices removed from main.
    agent = written_main["annotation"]["agents"][0]
    assert "_track_index" not in agent
    assert "bounding_boxes" not in agent

    sidecar = _sidecar_path_for(src)
    assert sidecar.exists()
    sidecar_data = json.loads(sidecar.read_text())
    ext = sidecar_data["extensions"]
    assert "bbox/1.0" in ext
    assert "ui/1.0" in ext
    assert ext["ui/1.0"]["track_index"] == {"A1": 2}
    assert "A1" in ext["bbox/1.0"]["agents"]


def test_cli_idempotent_on_already_2_2_0(tmp_path: Path) -> None:
    """Running the CLI twice on the same file is a no-op the second time."""
    src = tmp_path / "idem.json"
    _write_bundle(src, {
        "schema_version": "2.1.0",
        "video": {"clip_id": "i", "fps": 30.0, "duration_s": 1.0},
        "annotation": {
            "environments": [{
                "id": "Env1", "type": "oxd:Road",
                "_track_index": 1,
                "start_timestamp": "0:0.0", "end_timestamp": "0:1.0",
            }],
        },
    })

    assert main([str(src)]) == 0
    first_main = src.read_text()
    first_sidecar = _sidecar_path_for(src).read_text()

    assert main([str(src)]) == 0
    assert src.read_text() == first_main
    assert _sidecar_path_for(src).read_text() == first_sidecar


def test_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    src = tmp_path / "dr.json"
    payload = _bundle_2_1_0_with_indices()
    _write_bundle(src, payload)
    before = src.read_bytes()
    assert main([str(src), "--dry-run"]) == 0
    assert src.read_bytes() == before
    assert not _sidecar_path_for(src).exists()
