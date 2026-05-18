# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``cascade-migrate``."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from cascade_av.cli.migrate import (
    MIGRATIONS,
    _migration_path,
    main,
)
from cascade_av.io.local import _sidecar_path_for
from cascade_av.spec import CURRENT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bundle_2_0_0_with_bboxes() -> dict[str, object]:
    return {
        "schema_version": "2.0.0",
        "video": {"clip_id": "with-bboxes", "fps": 30.0, "duration_s": 1.0},
        "annotation": {
            "agents": [
                {
                    "id": "agent_0",
                    "type": "oxd:Pedestrian",
                    "bounding_boxes": [
                        {
                            "timestamp": "0:00.000",
                            "bounding_boxes": [
                                {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4,
                                 "object_id": "agent_0",
                                 "object_class": "oxd:Pedestrian"},
                            ],
                        },
                    ],
                },
                {"id": "agent_1", "type": "oxd:Car"},  # no bboxes
            ],
            "traffic_objects": [
                {
                    "id": "obj_0",
                    "type": "fst:StopSign",
                    "bounding_boxes": [
                        {"timestamp": "0:00.500", "bounding_boxes": []},
                    ],
                },
            ],
            "traffic_lights": [],
        },
    }


def _bundle_2_0_0_no_bboxes() -> dict[str, object]:
    return {
        "schema_version": "2.0.0",
        "video": {"clip_id": "no-bboxes", "fps": 30.0, "duration_s": 1.0},
        "annotation": {"agents": [], "traffic_objects": [], "traffic_lights": []},
    }


def _write_bundle(path: Path, bundle: dict[str, object]) -> None:
    path.write_text(json.dumps(bundle))


# ---------------------------------------------------------------------------
# Registry + chain
# ---------------------------------------------------------------------------


def test_migrations_registry_contains_seed_pair() -> None:
    assert ("2.0.0", "2.1.0") in MIGRATIONS


def test_migration_path_handles_no_op() -> None:
    assert _migration_path("2.1.0", "2.1.0") == []


def test_migration_path_walks_adjacent_pair() -> None:
    assert _migration_path("2.0.0", "2.1.0") == [("2.0.0", "2.1.0")]


def test_migration_path_returns_none_for_unreachable() -> None:
    assert _migration_path("9.9.9", "2.1.0") is None


# ---------------------------------------------------------------------------
# CLI behaviour — single file
# ---------------------------------------------------------------------------


def test_migrate_bbox_bearing_file_writes_sidecar(tmp_path: Path) -> None:
    src = tmp_path / "f.json"
    _write_bundle(src, _bundle_2_0_0_with_bboxes())
    sidecar = _sidecar_path_for(src)
    assert not sidecar.exists()

    exit_code = main([str(src), "-v"])
    assert exit_code == 0

    written_main = json.loads(src.read_text())
    # cascade-migrate walks the chain all the way to CURRENT_SCHEMA_VERSION
    # — through 2.1.0 (bbox extraction) and 2.2.0 (ui-index extraction).
    assert written_main["schema_version"] == CURRENT_SCHEMA_VERSION
    for agent in written_main["annotation"]["agents"]:
        assert "bounding_boxes" not in agent
    for obj in written_main["annotation"]["traffic_objects"]:
        assert "bounding_boxes" not in obj

    assert sidecar.exists()
    written_sidecar = json.loads(sidecar.read_text())
    assert written_sidecar["schema_version"] == CURRENT_SCHEMA_VERSION
    assert written_sidecar["main_file"] == "f.json"
    bbox = written_sidecar["extensions"]["bbox/1.0"]
    assert "agent_0" in bbox["agents"]
    assert "agent_1" not in bbox["agents"]  # had no bboxes
    assert "obj_0" in bbox["traffic_objects"]
    assert "traffic_lights" not in bbox  # all empty groups dropped


def test_migrate_file_without_bboxes_skips_sidecar(tmp_path: Path) -> None:
    src = tmp_path / "g.json"
    _write_bundle(src, _bundle_2_0_0_no_bboxes())

    exit_code = main([str(src), "-v"])
    assert exit_code == 0
    assert json.loads(src.read_text())["schema_version"] == CURRENT_SCHEMA_VERSION
    assert not _sidecar_path_for(src).exists(), (
        "no sidecar should be written when there is nothing to migrate"
    )


def test_migrate_already_current_is_no_op(tmp_path: Path) -> None:
    src = tmp_path / "h.json"
    bundle = _bundle_2_0_0_no_bboxes()
    bundle["schema_version"] = CURRENT_SCHEMA_VERSION
    _write_bundle(src, bundle)
    before_bytes = src.read_bytes()

    exit_code = main([str(src), "-v"])
    assert exit_code == 0
    assert src.read_bytes() == before_bytes


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    src = tmp_path / "i.json"
    _write_bundle(src, _bundle_2_0_0_with_bboxes())

    assert main([str(src)]) == 0
    first_main = src.read_text()
    first_sidecar = _sidecar_path_for(src).read_text()

    assert main([str(src)]) == 0
    assert src.read_text() == first_main
    assert _sidecar_path_for(src).read_text() == first_sidecar


# ---------------------------------------------------------------------------
# CLI behaviour — output-dir, dry-run, check, force
# ---------------------------------------------------------------------------


def test_migrate_output_dir_does_not_touch_source(tmp_path: Path) -> None:
    src = tmp_path / "src.json"
    _write_bundle(src, _bundle_2_0_0_with_bboxes())
    out = tmp_path / "out"

    exit_code = main([str(src), "--output-dir", str(out)])
    assert exit_code == 0
    # Source untouched.
    assert json.loads(src.read_text())["schema_version"] == "2.0.0"
    # Output present.
    out_main = out / "src.json"
    out_sidecar = out / "src.extra.json"
    assert json.loads(out_main.read_text())["schema_version"] == CURRENT_SCHEMA_VERSION
    assert "bbox/1.0" in json.loads(out_sidecar.read_text())["extensions"]


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    src = tmp_path / "j.json"
    _write_bundle(src, _bundle_2_0_0_with_bboxes())
    before = src.read_bytes()

    exit_code = main([str(src), "--dry-run", "-v"])
    assert exit_code == 0
    assert src.read_bytes() == before
    assert not _sidecar_path_for(src).exists()


def test_check_returns_one_when_migration_needed(tmp_path: Path) -> None:
    src = tmp_path / "k.json"
    _write_bundle(src, _bundle_2_0_0_no_bboxes())
    assert main([str(src), "--check"]) == 1
    # Nothing written even though it "would" migrate.
    assert json.loads(src.read_text())["schema_version"] == "2.0.0"


def test_check_returns_zero_when_all_current(tmp_path: Path) -> None:
    src = tmp_path / "l.json"
    bundle = _bundle_2_0_0_no_bboxes()
    bundle["schema_version"] = CURRENT_SCHEMA_VERSION
    _write_bundle(src, bundle)
    assert main([str(src), "--check"]) == 0


# ---------------------------------------------------------------------------
# CLI behaviour — sidecar interactions
# ---------------------------------------------------------------------------


def test_preexisting_unknown_sidecar_key_is_preserved(tmp_path: Path) -> None:
    src = tmp_path / "m.json"
    _write_bundle(src, _bundle_2_0_0_with_bboxes())
    sidecar = _sidecar_path_for(src)
    sidecar.write_text(json.dumps({
        "schema_version": "2.0.0",
        "main_file": "m.json",
        "extensions": {"alien/2.0": {"keep": True}},
    }))

    assert main([str(src)]) == 0
    written_sidecar = json.loads(sidecar.read_text())
    assert "alien/2.0" in written_sidecar["extensions"]
    assert written_sidecar["extensions"]["alien/2.0"] == {"keep": True}
    assert "bbox/1.0" in written_sidecar["extensions"]


def test_preexisting_bbox_sidecar_key_refuses_without_force(tmp_path: Path) -> None:
    src = tmp_path / "n.json"
    _write_bundle(src, _bundle_2_0_0_with_bboxes())
    sidecar = _sidecar_path_for(src)
    sidecar.write_text(json.dumps({
        "schema_version": "2.0.0",
        "main_file": "n.json",
        "extensions": {"bbox/1.0": {"stale": "payload"}},
    }))

    exit_code = main([str(src)])
    assert exit_code == 2  # error
    # Sidecar must not have been overwritten.
    assert json.loads(sidecar.read_text())["extensions"]["bbox/1.0"] == {
        "stale": "payload",
    }
    # Main must not have been migrated either.
    assert json.loads(src.read_text())["schema_version"] == "2.0.0"


def test_force_overwrites_existing_sidecar_key(tmp_path: Path) -> None:
    src = tmp_path / "o.json"
    _write_bundle(src, _bundle_2_0_0_with_bboxes())
    sidecar = _sidecar_path_for(src)
    sidecar.write_text(json.dumps({
        "schema_version": "2.0.0",
        "main_file": "o.json",
        "extensions": {"bbox/1.0": {"stale": "payload"}, "alien/2.0": {"keep": 1}},
    }))

    assert main([str(src), "--force"]) == 0
    rewritten = json.loads(sidecar.read_text())
    # bbox/1.0 got replaced...
    assert "stale" not in json.dumps(rewritten["extensions"]["bbox/1.0"])
    # ...but the unknown key alien/2.0 was preserved.
    assert rewritten["extensions"]["alien/2.0"] == {"keep": 1}


# ---------------------------------------------------------------------------
# CLI behaviour — directory inputs + atomicity hint
# ---------------------------------------------------------------------------


def test_directory_input_processes_every_main_json_but_skips_sidecars(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path / "a.json", _bundle_2_0_0_with_bboxes())
    _write_bundle(tmp_path / "b.json", _bundle_2_0_0_no_bboxes())
    # A stray sidecar shaped like a main file should not be processed.
    _write_bundle(tmp_path / "c.extra.json", {"schema_version": "0.0.0"})

    exit_code = main([str(tmp_path), "-v"])
    assert exit_code == 0
    assert json.loads((tmp_path / "a.json").read_text())["schema_version"] == CURRENT_SCHEMA_VERSION
    assert json.loads((tmp_path / "b.json").read_text())["schema_version"] == CURRENT_SCHEMA_VERSION
    # The stray "main_file"-shaped sidecar is left alone.
    assert json.loads((tmp_path / "c.extra.json").read_text())["schema_version"] == "0.0.0"


def test_missing_schema_version_field_errors_gracefully(tmp_path: Path) -> None:
    src = tmp_path / "bad.json"
    src.write_text(json.dumps({"video": {"clip_id": "x"}}))
    assert main([str(src)]) == 2


def test_empty_input_directory_exits_with_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main([str(empty)]) == 1
    out = capsys.readouterr().out
    assert "no annotation JSON" in out


# ---------------------------------------------------------------------------
# Round-trip via the library on a migrated pair
# ---------------------------------------------------------------------------


def test_migrated_pair_loads_through_library_and_preserves_sidecar(
    tmp_path: Path,
) -> None:
    """End-to-end: after migration, ``cascade_av.io.load_file`` reads the
    main, picks up the sidecar, exposes the bbox payload as raw passthrough
    (no extension registered), and re-saves without loss.
    """
    from cascade_av.extensions.registry import _reset_for_tests
    from cascade_av.io.local import _reset_warn_cache_for_tests, load_file, save_file

    _reset_for_tests()
    _reset_warn_cache_for_tests()

    src = tmp_path / "round.json"
    _write_bundle(src, _bundle_2_0_0_with_bboxes())
    assert main([str(src)]) == 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = load_file(src)
    assert bundle.schema_version == CURRENT_SCHEMA_VERSION
    # No bbox extension is registered in-tree; it round-trips as raw payload.
    assert "bbox/1.0" in bundle._sidecar_raw

    out = tmp_path / "round2.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save_file(bundle, out)

    out_sidecar = json.loads(_sidecar_path_for(out).read_text())
    assert "bbox/1.0" in out_sidecar["extensions"]
    bbox_after = out_sidecar["extensions"]["bbox/1.0"]
    bbox_before = json.loads(_sidecar_path_for(src).read_text())["extensions"]["bbox/1.0"]
    assert bbox_after == bbox_before
