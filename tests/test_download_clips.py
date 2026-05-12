# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `CausalAVDataset.download_clips` — argument plumbing only.

Network calls are stubbed: we replace the parent's
`download_clip_features` with a recorder and assert what we pass it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from causal_ai_av.dataset import CausalAVDataset

CORPUS = Path("/home/horde/01_json_annotations")


@pytest.fixture
def patched_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    from causal_ai_av import dataset as ds_mod

    class _StubParent:
        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr(ds_mod, "PhysicalAIAVDatasetInterface", _StubParent)


def _record_download(ds: CausalAVDataset, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def fake(clip_ids, features=None, **kwargs):
        calls.append({"clip_ids": list(clip_ids), "features": features, "kwargs": kwargs})

    monkeypatch.setattr(ds, "download_clip_features", fake)
    return calls


def test_download_clips_defaults_to_all_clips_and_canonical_camera(
    patched_parent: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = CausalAVDataset(CORPUS)
    calls = _record_download(ds, monkeypatch)

    ds.download_clips()

    assert len(calls) == 1
    assert calls[0]["clip_ids"] == ds.list_sequences()
    # Default pulls the annotation camera + the minimum extras for
    # `get_sequence` (egomotion).
    assert calls[0]["features"] == [ds.annotation_camera, "egomotion"]


def test_download_clips_takes_explicit_list(
    patched_parent: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = CausalAVDataset(CORPUS)
    calls = _record_download(ds, monkeypatch)
    chosen = ds.list_sequences()[:3]

    ds.download_clips(chosen, features=["CAMERA_FRONT_WIDE_120FOV", "CAMERA_REAR"])

    assert calls[0]["clip_ids"] == chosen
    assert calls[0]["features"] == ["CAMERA_FRONT_WIDE_120FOV", "CAMERA_REAR"]


def test_download_clips_rejects_unknown_clip(
    patched_parent: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = CausalAVDataset(CORPUS)
    _record_download(ds, monkeypatch)

    with pytest.raises(KeyError, match="not in dataset"):
        ds.download_clips(["not-a-real-clip-id"])
