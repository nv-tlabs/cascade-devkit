# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`cascade_av.extensions.ui` — the ``ui/1.0`` round-trip.

The extension is the only home for timeline-layout indices in schema 2.2.0;
without it, the annotator's row/sub-lane assignments would not survive a
load → save cycle. The four behaviours that must hold:

1. Round-trip: ``save_file → load_file`` preserves the indices.
2. Stale-key sweep: deleting an entity drops its index automatically on
   the next ``dump()``.
3. ``get_or_create`` is idempotent and visible to subsequent reads.
4. Unknown sidecar keys (a future or third-party extension) round-trip
   verbatim alongside ``ui/1.0`` — no clobbering on save.
"""

from __future__ import annotations

import json
from pathlib import Path

from cascade_av.extensions.ui import UiExtension, UiIndexes
from cascade_av.io.local import (
    _reset_warn_cache_for_tests,
    _sidecar_path_for,
    load_file,
    save_file,
)
from cascade_av.spec import (
    Agent,
    AnnotationBundle,
    Environment,
    SilAvAnnotation,
    VideoMeta,
)


def _make_bundle() -> AnnotationBundle:
    return AnnotationBundle(
        video=VideoMeta(clip_id="rt"),
        annotation=SilAvAnnotation(
            environments=[Environment(
                id="Env1", type="oxd:Road",
                start_timestamp="0:0.0", end_timestamp="0:5.0",
            )],
            agents=[Agent(
                id="A1", type="oxd:Car",
                visibility_start_timestamp="0:0.0",
                visibility_end_timestamp="0:5.0",
            )],
        ),
    )


def test_get_or_create_returns_empty_indexer_on_fresh_bundle() -> None:
    bundle = _make_bundle()
    ui = UiIndexes.get_or_create(bundle)
    assert ui.get("Env1", "track_index") is None


def test_get_or_create_is_idempotent_and_shared() -> None:
    bundle = _make_bundle()
    ui_a = UiIndexes.get_or_create(bundle)
    ui_a.set("A1", "track_index", 7)
    ui_b = UiIndexes.get_or_create(bundle)
    # Same instance — second call must not wipe what the first wrote.
    assert ui_b is ui_a
    assert ui_b.get("A1", "track_index") == 7


def test_save_and_load_preserves_indices(tmp_path: Path) -> None:
    _reset_warn_cache_for_tests()
    bundle = _make_bundle()
    ui = UiIndexes.get_or_create(bundle)
    ui.set("Env1", "track_index", 2)
    ui.set("A1", "track_index", 3)

    path = tmp_path / "rt.json"
    save_file(bundle, path)
    # Sidecar must exist with the ui/1.0 payload.
    sidecar = _sidecar_path_for(path)
    assert sidecar.exists(), "ui/1.0 indices should produce a sidecar"
    sidecar_data = json.loads(sidecar.read_text())
    ui_payload = sidecar_data["extensions"]["ui/1.0"]
    assert ui_payload["track_index"] == {"Env1": 2, "A1": 3}

    # Round-trip: reload, ui-extension hydrates onto bundle._extensions.
    reloaded = load_file(path)
    ui2 = UiIndexes.get_or_create(reloaded)
    assert ui2.get("Env1", "track_index") == 2
    assert ui2.get("A1", "track_index") == 3


def test_dump_prunes_stale_keys(tmp_path: Path) -> None:
    """An entity deleted from the bundle must not leave its index in the
    sidecar on the next save. This is the load-bearing stale-key invariant
    documented in :mod:`cascade_av.extensions.ui`.
    """
    _reset_warn_cache_for_tests()
    bundle = _make_bundle()
    ui = UiIndexes.get_or_create(bundle)
    ui.set("Env1", "track_index", 0)
    ui.set("A1", "track_index", 0)
    ui.set("Ghost", "track_index", 99)  # Ghost has no matching entity.

    path = tmp_path / "stale.json"
    save_file(bundle, path)
    sidecar_data = json.loads(_sidecar_path_for(path).read_text())
    ui_payload = sidecar_data["extensions"]["ui/1.0"]
    # Ghost gets pruned — only live ids survive.
    assert "Ghost" not in ui_payload["track_index"]
    assert set(ui_payload["track_index"]) == {"Env1", "A1"}


def test_dump_returns_none_when_no_live_indices(tmp_path: Path) -> None:
    """Bundle whose only indices belong to deleted entities → no sidecar.

    Confirms the dump-time sweep collapses to ``None`` (and therefore the
    sidecar file is not written at all) when the live entity set is empty
    or when every recorded id has been removed.
    """
    _reset_warn_cache_for_tests()
    bundle = _make_bundle()
    ui = UiIndexes.get_or_create(bundle)
    ui.set("Ghost", "track_index", 1)  # no matching live entity

    ext = UiExtension()
    assert ext.dump(bundle) is None


def test_unknown_sidecar_keys_round_trip_alongside_ui(tmp_path: Path) -> None:
    """A sidecar that already has an unknown extension key — say a future
    'audio/0.5' that no extension is registered for — must survive a save
    that also writes ui/1.0. This mirrors the bbox-style passthrough
    invariant tested in test_extensions.py.
    """
    _reset_warn_cache_for_tests()
    bundle = _make_bundle()
    UiIndexes.get_or_create(bundle).set("Env1", "track_index", 0)
    # Pre-stage an unknown sidecar entry.
    bundle._sidecar_raw["audio/0.5"] = {"placeholder": True}

    path = tmp_path / "mixed.json"
    save_file(bundle, path)
    sidecar_data = json.loads(_sidecar_path_for(path).read_text())
    keys = sidecar_data["extensions"].keys()
    assert "ui/1.0" in keys
    assert "audio/0.5" in keys
    assert sidecar_data["extensions"]["audio/0.5"] == {"placeholder": True}


def test_set_with_none_value_removes_entry() -> None:
    bundle = _make_bundle()
    ui = UiIndexes.get_or_create(bundle)
    ui.set("Env1", "track_index", 5)
    assert ui.get("Env1", "track_index") == 5
    ui.set("Env1", "track_index", None)
    assert ui.get("Env1", "track_index") is None


def test_unknown_index_key_raises() -> None:
    bundle = _make_bundle()
    ui = UiIndexes.get_or_create(bundle)
    try:
        ui.set("Env1", "not_a_real_key", 0)
    except ValueError as exc:
        assert "not_a_real_key" in str(exc)
    else:
        raise AssertionError("expected ValueError on unknown index key")


def test_load_coerces_string_int_values(tmp_path: Path) -> None:
    """A buggy producer might write `"3"` instead of `3`. The loader should
    accept and coerce rather than crash, so legacy sidecars don't brick the
    annotator. Invalid (non-int-coercible) values are silently skipped."""
    _reset_warn_cache_for_tests()
    bundle = _make_bundle()
    UiIndexes.get_or_create(bundle).set("Env1", "track_index", 0)

    path = tmp_path / "coerce.json"
    save_file(bundle, path)

    # Mutate the sidecar to inject a stringified value and a junk value.
    sidecar_path = _sidecar_path_for(path)
    sidecar_data = json.loads(sidecar_path.read_text())
    sidecar_data["extensions"]["ui/1.0"]["track_index"]["Env1"] = "3"
    sidecar_data["extensions"]["ui/1.0"]["track_index"]["A1"] = "junk"
    sidecar_path.write_text(json.dumps(sidecar_data))

    reloaded = load_file(path)
    ui = UiIndexes.get_or_create(reloaded)
    assert ui.get("Env1", "track_index") == 3
    assert ui.get("A1", "track_index") is None
