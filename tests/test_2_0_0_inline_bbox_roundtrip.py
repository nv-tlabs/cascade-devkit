# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression guard: inline `bounding_boxes` in a 2.0.0 file must round-trip
through ``cascade_av.io.load_file`` / ``save_file`` without being silently
dropped, even though the typed field is gone in 2.1.0.

The contract is load-bearing: it lets users open un-migrated 2.0.0 corpus
files (e.g. in the annotator) without losing data. If this test fails, the
soft-warn-and-accept design breaks and we'd need to hard-reject 2.0.0
before opening — see plan
``/home/horde/.claude/plans/regarding-the-boundingboxes-my-woolly-plum.md``.
The mechanism: Pydantic v2 ``extra="allow"`` puts unknown keys in
``__pydantic_extra__``, which ``model_dump_json`` includes in its output.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from cascade_av.io import load_file, save_file


def _bundle_with_inline_bbox() -> dict[str, object]:
    """Construct a 2.0.0-shape bundle whose agents carry inline bboxes."""
    return {
        "schema_version": "2.0.0",
        "video": {"clip_id": "regression", "fps": 30.0, "duration_s": 1.0},
        "annotation": {
            "agents": [
                {
                    "id": "agent_0",
                    "type": "oxd:Pedestrian",
                    "bounding_boxes": [
                        {
                            "timestamp": "0:00.000",
                            "bounding_boxes": [
                                {
                                    "x": 0.10,
                                    "y": 0.20,
                                    "w": 0.30,
                                    "h": 0.40,
                                    "object_id": "agent_0",
                                    "object_class": "oxd:Pedestrian",
                                },
                            ],
                        },
                        {
                            "timestamp": "0:00.100",
                            "bounding_boxes": [],
                        },
                    ],
                },
            ],
        },
    }


def test_2_0_0_inline_bboxes_survive_load_save_reload(tmp_path: Path) -> None:
    src = tmp_path / "old.json"
    src.write_text(json.dumps(_bundle_with_inline_bbox()))

    # Suppress the expected DeprecationWarning so other test runs don't flag it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        bundle = load_file(src)

    out = tmp_path / "out.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        save_file(bundle, out)

    written = json.loads(out.read_text())
    agents = written["annotation"]["agents"]
    assert len(agents) == 1
    # The full bbox payload — wrapped frames + per-frame boxes — survives.
    assert agents[0]["bounding_boxes"][0]["timestamp"] == "0:00.000"
    boxes = agents[0]["bounding_boxes"][0]["bounding_boxes"]
    assert len(boxes) == 1
    assert boxes[0] == {
        "x": 0.10,
        "y": 0.20,
        "w": 0.30,
        "h": 0.40,
        "object_id": "agent_0",
        "object_class": "oxd:Pedestrian",
    }
    # Empty bbox frame still present (preserves frame-time skeleton).
    assert agents[0]["bounding_boxes"][1]["timestamp"] == "0:00.100"
    assert agents[0]["bounding_boxes"][1]["bounding_boxes"] == []


def test_2_0_0_schema_version_round_trips_verbatim(tmp_path: Path) -> None:
    """A 2.0.0 file does not get silently upgraded to 2.1.0 on save."""
    src = tmp_path / "v2.json"
    src.write_text(json.dumps(_bundle_with_inline_bbox()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        bundle = load_file(src)
    out = tmp_path / "out.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        save_file(bundle, out)
    assert json.loads(out.read_text())["schema_version"] == "2.0.0"
