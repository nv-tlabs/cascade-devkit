# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `causal_ai_av.io`."""

from __future__ import annotations

import json
from pathlib import Path

from causal_ai_av.io import (
    group_by_clip_id,
    iter_dir,
    load_dir,
    load_file,
    parse_filename,
    save_file,
)

CORPUS = Path("/home/horde/01_json_annotations")


def test_load_file_smoke(rich_path: Path) -> None:
    bundle = load_file(rich_path)
    assert bundle.video.clip_id
    assert bundle.schema_version == "2.0.0"


def test_parse_filename_real_corpus() -> None:
    sample = next(CORPUS.glob("*.json"))
    annotation_id, clip_id = parse_filename(sample)
    # Both halves are non-empty and the second half matches video.clip_id.
    bundle = load_file(sample)
    assert annotation_id
    assert clip_id == bundle.video.clip_id


def test_round_trip_preserves_keysets(tmp_path: Path, rich_path: Path) -> None:
    bundle = load_file(rich_path)
    out = tmp_path / "out.json"
    save_file(bundle, out)
    original = json.loads(rich_path.read_text())
    dumped = json.loads(out.read_text())
    assert set(original.keys()) == set(dumped.keys())
    assert set(original["annotation"].keys()) == set(dumped["annotation"].keys())


def test_load_dir_loads_everything() -> None:
    bundles = load_dir(CORPUS)
    assert len(bundles) >= 300
    assert all(b.schema_version == "2.0.0" for b in bundles)


def test_iter_dir_is_lazy() -> None:
    count = 0
    for _ in iter_dir(CORPUS, on_error="skip"):
        count += 1
        if count >= 5:
            break
    assert count == 5


def test_group_by_clip_id_partitions_input() -> None:
    bundles = load_dir(CORPUS)
    grouped = group_by_clip_id(bundles)
    assert sum(len(v) for v in grouped.values()) == len(bundles)
    # Every clip_id we saw appears as a key
    seen_ids = {b.video.clip_id for b in bundles}
    assert set(grouped.keys()) == seen_ids


def test_hf_adapter_importable() -> None:
    """Smoke check that the optional HF adapter resolves with the `[hf]` extra."""
    from causal_ai_av.io.hf import CausalAnnotationsHfRepo  # noqa: F401
