# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the schema-extension surface and its sidecar round-trip."""

from __future__ import annotations

import json
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from cascade_av.extensions import Extension, register
from cascade_av.extensions.registry import _reset_for_tests
from cascade_av.io.local import (
    _reset_warn_cache_for_tests,
    _sidecar_path_for,
    load_file,
    save_file,
)
from cascade_av.spec import AnnotationBundle


@pytest.fixture(autouse=True)
def _clean_registry_and_warn_cache() -> None:
    """Each test starts with an empty registry and warn-once cache."""
    _reset_for_tests()
    _reset_warn_cache_for_tests()
    yield
    _reset_for_tests()
    _reset_warn_cache_for_tests()


class _EchoExtension(Extension):
    """Minimal extension: stores its payload verbatim under ``bundle._extensions``.

    Used to lock the contract that registered extensions can claim a
    sidecar key and round-trip arbitrary JSON data through it.
    """

    key = "echo/1.0"
    schema_versions = ("2.0.0",)

    def load(self, bundle: AnnotationBundle, ext_data: dict[str, Any]) -> None:
        bundle._extensions[self.key] = dict(ext_data)

    def dump(self, bundle: AnnotationBundle) -> dict[str, Any] | None:
        return bundle._extensions.get(self.key)


def _new_bundle() -> AnnotationBundle:
    return AnnotationBundle.model_validate({
        "schema_version": "2.0.0",
        "video": {"clip_id": "test-clip", "fps": 30.0, "duration_s": 1.0},
    })


def test_sidecar_path_strips_last_suffix_only() -> None:
    assert _sidecar_path_for(Path("foo.json")).name == "foo.extra.json"
    assert _sidecar_path_for(Path("a/b/foo.bar.json")).name == "foo.bar.extra.json"


def test_load_file_without_sidecar_is_unchanged(tmp_path: Path) -> None:
    """A main JSON with no sibling sidecar loads exactly as before."""
    main = tmp_path / "x.json"
    bundle = _new_bundle()
    save_file(bundle, main)
    # No sidecar was written because there is nothing to write.
    assert not _sidecar_path_for(main).exists()
    reloaded = load_file(main)
    assert reloaded._sidecar_raw == {}
    assert reloaded._extensions == {}


def test_registered_extension_round_trips_payload(tmp_path: Path) -> None:
    register(_EchoExtension())
    bundle = _new_bundle()
    bundle._extensions["echo/1.0"] = {"hello": "world", "n": 7}

    main = tmp_path / "round.json"
    save_file(bundle, main)

    sidecar = _sidecar_path_for(main)
    assert sidecar.exists(), "sidecar should be written when an extension dumps data"
    sidecar_data = json.loads(sidecar.read_text())
    assert sidecar_data["main_file"] == "round.json"
    assert sidecar_data["schema_version"] == "2.0.0"
    assert sidecar_data["extensions"] == {"echo/1.0": {"hello": "world", "n": 7}}

    reloaded = load_file(main)
    assert reloaded.ext("echo/1.0") == {"hello": "world", "n": 7}
    assert reloaded._sidecar_raw == {}


def test_extension_dump_returning_none_omits_sidecar_key(tmp_path: Path) -> None:
    """An extension that finds nothing to dump must not produce a sidecar entry."""
    register(_EchoExtension())
    bundle = _new_bundle()
    # Do NOT populate _extensions: dump returns None.
    main = tmp_path / "empty.json"
    save_file(bundle, main)
    assert not _sidecar_path_for(main).exists()


def test_unknown_sidecar_key_round_trips_and_warns_once(tmp_path: Path) -> None:
    """Sidecar entries with no registered handler survive load → save, warning once."""
    main = tmp_path / "unk.json"
    sidecar = _sidecar_path_for(main)
    save_file(_new_bundle(), main)
    sidecar.write_text(json.dumps({
        "schema_version": "2.0.0",
        "main_file": main.name,
        "extensions": {"future_ext/1.0": {"opaque": [1, 2, 3]}},
    }))

    with pytest.warns(UserWarning, match=r"future_ext/1\.0"):
        reloaded = load_file(main)
    # A second load fires the warning at most zero additional times.
    with warnings_record() as caught:
        load_file(main)
    assert not [w for w in caught if "future_ext/1.0" in str(w.message)]

    assert reloaded._sidecar_raw == {"future_ext/1.0": {"opaque": [1, 2, 3]}}

    out = tmp_path / "rewritten.json"
    save_file(reloaded, out)
    out_sidecar = _sidecar_path_for(out)
    assert out_sidecar.exists()
    rewritten = json.loads(out_sidecar.read_text())
    assert rewritten["extensions"] == {"future_ext/1.0": {"opaque": [1, 2, 3]}}


def test_registered_extension_overwrites_unknown_key_on_save(tmp_path: Path) -> None:
    """If a key was unclaimed at load time then an extension is registered,
    saving again replaces the raw passthrough with the extension's own dump.
    """
    main = tmp_path / "merge.json"
    sidecar = _sidecar_path_for(main)
    save_file(_new_bundle(), main)
    sidecar.write_text(json.dumps({
        "schema_version": "2.0.0",
        "main_file": main.name,
        "extensions": {"echo/1.0": {"old": True}},
    }))

    bundle = load_file(main)
    assert bundle._sidecar_raw == {"echo/1.0": {"old": True}}

    register(_EchoExtension())
    bundle._extensions["echo/1.0"] = {"old": False, "new": "value"}

    save_file(bundle, main)
    rewritten = json.loads(sidecar.read_text())
    assert rewritten["extensions"] == {"echo/1.0": {"old": False, "new": "value"}}


def test_sidecar_unknown_payload_preserved_when_extension_dumps_alongside(
    tmp_path: Path,
) -> None:
    """Mix: one known extension + one unknown key in the same sidecar."""
    register(_EchoExtension())
    main = tmp_path / "mix.json"
    sidecar = _sidecar_path_for(main)
    save_file(_new_bundle(), main)
    sidecar.write_text(json.dumps({
        "schema_version": "2.0.0",
        "main_file": main.name,
        "extensions": {
            "echo/1.0": {"a": 1},
            "alien/2.0": {"keep_me": True},
        },
    }))

    with pytest.warns(UserWarning, match=r"alien/2\.0"):
        bundle = load_file(main)
    assert bundle.ext("echo/1.0") == {"a": 1}
    assert bundle._sidecar_raw == {"alien/2.0": {"keep_me": True}}

    save_file(bundle, main)
    rewritten = json.loads(sidecar.read_text())
    assert rewritten["extensions"] == {
        "echo/1.0": {"a": 1},
        "alien/2.0": {"keep_me": True},
    }


def test_private_attrs_do_not_leak_into_main_json(tmp_path: Path) -> None:
    """The private extension/sidecar attrs must never appear in `model_dump_json`."""
    register(_EchoExtension())
    bundle = _new_bundle()
    bundle._extensions["echo/1.0"] = {"x": 1}
    bundle._sidecar_raw["alien/2.0"] = {"y": 2}

    main = tmp_path / "leak.json"
    save_file(bundle, main)
    raw_main = json.loads(main.read_text())
    assert "_extensions" not in raw_main
    assert "_sidecar_raw" not in raw_main
    assert "extensions" not in raw_main


def test_ext_returns_none_when_unregistered() -> None:
    bundle = _new_bundle()
    assert bundle.ext("never/registered") is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def warnings_record():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        yield recorded
