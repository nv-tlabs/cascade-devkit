# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `cascade_av.io`."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cascade_av.io import (
    group_by_clip_id,
    iter_dir,
    load_dir,
    load_file,
    parse_filename,
    save_file,
)

_root = os.environ.get("CASCADE_AV_DATASET_ROOT")
CORPUS = Path(_root) if _root else None

pytestmark = pytest.mark.skipif(
    CORPUS is None or not CORPUS.is_dir(),
    reason="set CASCADE_AV_DATASET_ROOT to run corpus-backed tests",
)


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
    from cascade_av.io.hf import CausalAnnotationsHfRepo  # noqa: F401


# ---------------------------------------------------------------------------
# HF adapter — autodetect + split-aware loaders. All exercised against
# monkey-patched parent class methods so the tests never touch the network.
# `_make_hf_repo` is the shared no-network constructor.
# ---------------------------------------------------------------------------
import pytest  # noqa: E402  — local imports keep the network-free tests grouped


def _make_hf_repo(monkeypatch, *, repo_files=None, **kwargs):
    """Construct a `CausalAnnotationsHfRepo` with the parent class network
    calls stubbed out.

    `repo_files` controls what `HfApi.list_repo_files` returns during
    autodetect; pass `None` to make autodetect raise (covering the
    "HF unreachable → fall back to root" path).
    """
    from huggingface_hub import HfApi
    from physical_ai_av.utils.hf_interface import HfRepoInterface

    def fake_init(self, *, repo_id, repo_type, revision=None, **_kw):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.revision = revision

    monkeypatch.setattr(HfRepoInterface, "__init__", fake_init)

    if repo_files is None:
        def boom(*a, **kw):
            raise RuntimeError("simulated HF failure")
        monkeypatch.setattr(HfApi, "list_repo_files", boom)
    else:
        monkeypatch.setattr(
            HfApi, "list_repo_files", lambda self, *a, **kw: list(repo_files),
        )

    from cascade_av.io.hf import CausalAnnotationsHfRepo
    return CausalAnnotationsHfRepo("fake/repo", **kwargs)


def test_hf_autodetect_picks_data_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Files prefixed with `data/` → autodetect picks `data`."""
    repo = _make_hf_repo(
        monkeypatch,
        repo_files=[
            "LICENSE",
            "README.md",
            "data/batch_00001/foo.json",
            "data/dataset_split.yaml",
        ],
    )
    assert repo.path_in_repo == "data"


def test_hf_autodetect_falls_back_to_root_when_no_data_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No `data/` prefix → autodetect returns `""` (legacy / flat layouts)."""
    repo = _make_hf_repo(
        monkeypatch,
        repo_files=["LICENSE", "README.md", "annotation_a.json"],
    )
    assert repo.path_in_repo == ""


def test_hf_autodetect_falls_back_on_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any HF API failure during autodetect → fall back to root rather
    than propagating an opaque error. Callers can still pass an explicit
    `path_in_repo=` if needed."""
    repo = _make_hf_repo(monkeypatch, repo_files=None)
    assert repo.path_in_repo == ""


def test_hf_explicit_path_in_repo_bypasses_autodetect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing `path_in_repo=` explicitly skips the `list_repo_files`
    round-trip — locks the "autodetect is free when you don't need it"
    contract."""
    from huggingface_hub import HfApi
    from physical_ai_av.utils.hf_interface import HfRepoInterface

    monkeypatch.setattr(
        HfRepoInterface, "__init__",
        lambda self, *, repo_id, repo_type, revision=None, **_kw: setattr(
            self, "repo_id", repo_id,
        ) or setattr(self, "repo_type", repo_type) or setattr(
            self, "revision", revision,
        ),
    )
    called: list[bool] = []
    monkeypatch.setattr(
        HfApi, "list_repo_files",
        lambda self, *a, **kw: (called.append(True), [])[1],
    )

    from cascade_av.io.hf import CausalAnnotationsHfRepo
    repo = CausalAnnotationsHfRepo("fake/repo", path_in_repo="custom")
    assert repo.path_in_repo == "custom"
    assert called == [], (
        "autodetect should not run when path_in_repo is supplied explicitly"
    )


def test_hf_available_splits_parses_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """`available_splits()` parses `dataset_split.yaml` and returns
    `{name: [sub_splits]}` ordered by manifest position."""
    yaml_path = tmp_path / "dataset_split.yaml"
    yaml_path.write_text(
        "- name: cascade-v0.1\n"
        "  date: 2026-05-14\n"
        "  train:\n"
        "    - a.json\n"
        "    - b.json\n"
        "  validation:\n"
        "    - c.json\n"
        "- name: cascade-v0.2\n"
        "  date: 2026-06-01\n"
        "  train:\n"
        "    - d.json\n"
        "  validation:\n"
        "    - e.json\n"
        "  test:\n"
        "    - f.json\n"
    )

    repo = _make_hf_repo(monkeypatch, repo_files=["data/dataset_split.yaml"])

    from physical_ai_av.utils.hf_interface import HfRepoInterface
    monkeypatch.setattr(
        HfRepoInterface, "download_file",
        lambda self, repo_path: str(yaml_path),
    )

    splits = repo.available_splits()
    assert splits == {
        "cascade-v0.1": ["train", "validation"],
        "cascade-v0.2": ["train", "validation", "test"],
    }


def test_hf_iter_split_yields_bundles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rich_path: Path,
) -> None:
    """`iter_split(name, split)` resolves each filename in the manifest
    via `load_annotation`, threading `path_in_repo` onto the request."""
    yaml_path = tmp_path / "dataset_split.yaml"
    yaml_path.write_text(
        "- name: cascade-v0.1\n"
        "  date: 2026-05-14\n"
        "  train:\n"
        "    - batch_00001/sample.json\n"
        "  validation: []\n"
    )
    requested: list[str] = []

    def fake_download_file(self, repo_path):
        requested.append(repo_path)
        if repo_path.endswith("dataset_split.yaml"):
            return str(yaml_path)
        if repo_path == "data/batch_00001/sample.json":
            return str(rich_path)
        raise FileNotFoundError(repo_path)

    repo = _make_hf_repo(monkeypatch, repo_files=["data/dataset_split.yaml"])

    from physical_ai_av.utils.hf_interface import HfRepoInterface
    monkeypatch.setattr(HfRepoInterface, "download_file", fake_download_file)

    bundles = list(repo.iter_split("cascade-v0.1", "train"))
    assert len(bundles) == 1
    assert bundles[0].schema_version == "2.0.0"
    # Locked: path_in_repo got prepended to the filename from the YAML.
    assert "data/batch_00001/sample.json" in requested


def test_hf_iter_split_raises_on_unknown_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    yaml_path = tmp_path / "dataset_split.yaml"
    yaml_path.write_text("- name: cascade-v0.1\n  train: []\n")
    repo = _make_hf_repo(monkeypatch, repo_files=["data/dataset_split.yaml"])
    from physical_ai_av.utils.hf_interface import HfRepoInterface
    monkeypatch.setattr(
        HfRepoInterface, "download_file",
        lambda self, repo_path: str(yaml_path),
    )
    with pytest.raises(KeyError, match="cascade-bogus"):
        list(repo.iter_split("cascade-bogus", "train"))


def test_hf_iter_split_raises_on_unknown_sub_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    yaml_path = tmp_path / "dataset_split.yaml"
    yaml_path.write_text("- name: cascade-v0.1\n  train: []\n")
    repo = _make_hf_repo(monkeypatch, repo_files=["data/dataset_split.yaml"])
    from physical_ai_av.utils.hf_interface import HfRepoInterface
    monkeypatch.setattr(
        HfRepoInterface, "download_file",
        lambda self, repo_path: str(yaml_path),
    )
    with pytest.raises(KeyError, match="test"):
        list(repo.iter_split("cascade-v0.1", "test"))


def test_round_trip_preserves_eventful_reason(tmp_path: Path) -> None:
    """A fabricated bundle carrying the post-2.0.0 `eventful_reason` /
    `eventful_reason_other` fields survives a `save_file` → `load_file`
    round-trip. In-memory only — no corpus needed (though the module-wide
    skipif still gates this on `CASCADE_AV_DATASET_ROOT` being set)."""
    from cascade_av.spec.schema import (
        AnnotationBundle,
        EventfulReasonVocab,
        SilAvAnnotation,
        VideoMeta,
    )

    bundle = AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="eventful-rt", fps=30.0, duration_s=10.0),
        annotation=SilAvAnnotation(
            eventful=True,
            eventful_reason=EventfulReasonVocab.EGO_ADAPTS,
            eventful_reason_other="x",
        ),
    )

    out = tmp_path / "eventful.json"
    save_file(bundle, out)
    reloaded = load_file(out)

    assert reloaded.annotation.eventful_reason == EventfulReasonVocab.EGO_ADAPTS
    assert reloaded.annotation.eventful_reason_other == "x"


def test_condition_type_accepts_bare_string() -> None:
    """Some corpora write `conditions[].type` as a bare string ("Construction
    Zone") rather than the canonical list shape. The schema coerces it so
    downstream consumers always see a list."""
    from cascade_av.spec.schema import Condition

    c_str = Condition.model_validate({"id": "c1", "type": "Construction Zone"})
    assert c_str.type == ["Construction Zone"]

    # Canonical list shape still works.
    c_list = Condition.model_validate({"id": "c2", "type": ["Clear", "Wet"]})
    assert c_list.type == ["Clear", "Wet"]

    # Empty / missing defaults to [].
    c_empty = Condition.model_validate({"id": "c3"})
    assert c_empty.type == []
