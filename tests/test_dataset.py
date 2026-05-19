# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for `CascadeDataset` and `Sequence`.

Per the user's constraint, this is intentionally a thin suite (~5 tests). We
exercise the public surface that is unlikely to change before the annotator
subsystem lands; we do *not* write per-method unit coverage.

Network behavior is sidestepped: the parent `PhysicalAIAVDatasetInterface.__init__`
is monkeypatched to a no-op for tests that need a constructed dataset, and
`Sequence` is exercised via its `from_annotation` constructor when no parent
dataset is needed at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cascade_av.dataset import CascadeDataset, Sequence
from cascade_av.io import load_file
from cascade_av.spec import AnnotationBundle
from cascade_av.state import SequenceState, SequenceStateRange
from tests.conftest import FakeVideoReader  # noqa: E402 — used by the visualize smoke test

CORPUS = Path("/home/horde/01_json_annotations")


@pytest.fixture
def patched_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass the parent `PhysicalAIAVDatasetInterface.__init__` network calls."""
    from physical_ai_av import PhysicalAIAVDatasetInterface

    def _noop(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Mimic just enough state for the dataset to be usable in tests.
        self.is_offline_mode = True

    monkeypatch.setattr(PhysicalAIAVDatasetInterface, "__init__", _noop)


def test_no_arg_constructor_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        CascadeDataset()


def test_non_dir_path_raises_file_not_found(tmp_path: Path) -> None:
    bogus = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        CascadeDataset(bogus)


def test_local_dir_construction(patched_parent: None) -> None:
    ds = CascadeDataset(CORPUS)
    assert len(ds.list_sequences()) > 0
    assert ds.annotation_camera == "camera_front_wide_120fov"


def test_sequence_from_annotation_round_trip(patched_parent: None) -> None:
    ds = CascadeDataset(CORPUS)
    clip_id = ds.list_sequences()[0]
    # `get_sequence` would trigger `parent.get_clip_feature("egomotion")` —
    # which we've stubbed out. Instead, exercise the annotation-only path via
    # `from_annotation`, which is the documented test/notebook constructor.
    path, _, _ = ds._by_clip[clip_id]
    seq = Sequence.from_annotation(load_file(path))
    assert isinstance(seq.annotation, AnnotationBundle)
    assert seq.clip_id == clip_id
    assert isinstance(seq.description, str)


def test_flat_corpus_yields_none_batch(patched_parent: None) -> None:
    """When an annotation lives directly under the user-supplied root, its
    `batch` should be `None` — not the root directory's basename."""
    ds = CascadeDataset(CORPUS)
    batches = [batch for _, batch, _ in ds._by_clip.values()]
    assert any(b is None for b in batches)


def test_state_at_instant_and_range() -> None:
    # Pick an arbitrary corpus file — the largest, so we know it has content.
    path = max(CORPUS.glob("*.json"), key=lambda p: p.stat().st_size)
    seq = Sequence.from_annotation(load_file(path))

    instant = seq.state_at(5.0)
    assert isinstance(instant, SequenceState)
    assert instant.clip_id == seq.clip_id

    window = seq.state_at(0.0, 5.0)
    assert isinstance(window, SequenceStateRange)
    assert window.clip_id == seq.clip_id


def test_visualize_smoke_returns_clipplayer() -> None:
    """The placeholder dict-shape return was replaced in PR-6 by a
    polymorphic dispatcher into ``cascade_av.viz``. The detailed
    dispatch matrix (scalar / tuple / match / context / static + the
    error paths) lives in ``tests/test_sequence_visualize.py``; this
    smoke test only pins that the call now produces a ``ClipPlayer``
    via the lazy ``viz`` import.
    """
    import numpy as np

    from cascade_av.viz import ClipPlayer

    path = max(CORPUS.glob("*.json"), key=lambda p: p.stat().st_size)
    seq = Sequence.from_annotation(load_file(path))
    # ``Sequence.video`` would normally require a parent dataset; the
    # widget tests stub the lazy ``_cameras`` slot to inject a fake
    # reader. Same trick here so ClipPlayer's initial decode succeeds.
    # ``Sequence.visualize`` clamps ``[t_start, t_end]`` to the video's
    # actual timestamp coverage; configure the fake's timestamps to
    # span the full ``seq.duration_s`` so the clamp is a no-op and the
    # "whole-clip" assertion below stays meaningful.
    fake_ts = np.arange(0, int(seq.duration_s * 1_000_000), 33_333, dtype=np.int64)
    seq._cameras = {  # type: ignore[assignment]
        seq.annotation_camera: FakeVideoReader(timestamps=fake_ts),
    }

    player = seq.visualize()
    assert isinstance(player, ClipPlayer)
    # The "no args" path opens over the whole clip (within the fake's
    # configured coverage, which we set to span the full duration).
    assert player._t_start == 0.0
    assert player._t_end == pytest.approx(fake_ts[-1] / 1_000_000, abs=1e-3)


# -----------------------------------------------------------------------------
# Egomotion: graceful missing-cache behavior (fix/sequence-egomotion-graceful)
#
# The egomotion chunk lives in the parent HF dataset and isn't always cached
# locally; `download_clips` only pulls per-clip video. Before the fix, the
# eager `parent.get_clip_feature(clip_id, "egomotion")` in `Sequence.__init__`
# raised `FileNotFoundError` and sank construction — taking the viz carousel
# down with it. After the fix:
#   * construction swallows `FileNotFoundError`, leaves the slot `None`,
#   * the `egomotion_interpolator` property raises a clear `RuntimeError`
#     when actually accessed,
#   * the viz path (`Sequence.video`, `state_at`, …) keeps working.
# -----------------------------------------------------------------------------


def test_sequence_init_swallows_egomotion_filenotfound(
    patched_parent: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache-miss on the egomotion chunk leaves `_egomotion_interpolator = None`
    rather than aborting `Sequence` construction."""
    from physical_ai_av import PhysicalAIAVDatasetInterface

    def _raise_missing(self, clip_id: str, feature: str):  # type: ignore[no-untyped-def]
        raise FileNotFoundError(
            "filename='labels/egomotion/egomotion.chunk_X.zip' "
            "not found in cache; set `maybe_stream=True` ..."
        )

    monkeypatch.setattr(
        PhysicalAIAVDatasetInterface, "get_clip_feature", _raise_missing
    )

    ds = CascadeDataset(CORPUS)
    clip_id = ds.list_sequences()[0]
    # Must not raise — the fix.
    seq = ds.get_sequence(clip_id)
    assert seq._egomotion_interpolator is None
    assert seq.clip_id == clip_id


def test_egomotion_interpolator_raises_runtime_when_uncached(
    patched_parent: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the chunk stayed un-cached, accessing the property raises a
    clear `RuntimeError` naming the clip id and pointing at `download_clips`."""
    from physical_ai_av import PhysicalAIAVDatasetInterface

    def _raise_missing(self, clip_id: str, feature: str):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("not in cache")

    monkeypatch.setattr(
        PhysicalAIAVDatasetInterface, "get_clip_feature", _raise_missing
    )

    ds = CascadeDataset(CORPUS)
    clip_id = ds.list_sequences()[0]
    seq = ds.get_sequence(clip_id)
    with pytest.raises(RuntimeError, match=r"egomotion data not loaded"):
        _ = seq.egomotion_interpolator
    # The same guard reaches `seq.egomotion`, which delegates to the
    # interpolator via `.values`.
    with pytest.raises(RuntimeError, match=r"download_clips"):
        _ = seq.egomotion
    # And the clip id is named in the error so the user knows which
    # clip to re-fetch.
    with pytest.raises(RuntimeError, match=clip_id):
        _ = seq.egomotion_interpolator


def test_egomotion_interpolator_retries_after_cache_populated(
    patched_parent: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the chunk lands in cache between construction and access, the
    property's re-attempt path picks it up — no need to rebuild the Sequence."""
    from physical_ai_av import PhysicalAIAVDatasetInterface

    calls = {"n": 0}
    sentinel = object()

    def _stub(self, clip_id: str, feature: str):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise FileNotFoundError("not in cache yet")
        return sentinel

    monkeypatch.setattr(PhysicalAIAVDatasetInterface, "get_clip_feature", _stub)

    ds = CascadeDataset(CORPUS)
    clip_id = ds.list_sequences()[0]
    seq = ds.get_sequence(clip_id)
    assert seq._egomotion_interpolator is None  # eager load swallowed
    assert seq.egomotion_interpolator is sentinel  # re-attempt succeeded
    # Subsequent access doesn't re-call the parent.
    assert seq.egomotion_interpolator is sentinel
    assert calls["n"] == 2


def test_viz_path_works_when_egomotion_missing(
    patched_parent: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load-bearing one: a Sequence with `_egomotion_interpolator is None`
    can still drive the viz API. `Sequence.video` and `Sequence.visualize`
    never touch egomotion."""
    from physical_ai_av import PhysicalAIAVDatasetInterface

    from cascade_av.viz import ClipPlayer

    def _raise_missing(self, clip_id: str, feature: str):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("not in cache")

    monkeypatch.setattr(
        PhysicalAIAVDatasetInterface, "get_clip_feature", _raise_missing
    )

    ds = CascadeDataset(CORPUS)
    clip_id = ds.list_sequences()[0]
    seq = ds.get_sequence(clip_id)
    assert seq._egomotion_interpolator is None
    # Stub `seq._cameras` so `seq.video` doesn't ask the parent for a
    # `SeekVideoReader` (the carousel tests use the same trick).
    seq._cameras = {seq.annotation_camera: FakeVideoReader()}  # type: ignore[assignment]
    player = seq.visualize()
    assert isinstance(player, ClipPlayer)
    # And `state_at` — which calls `_interpolate_ego_pose` internally —
    # silently degrades to no ego pose rather than re-raising.
    s = seq.state_at(0.0)
    assert isinstance(s, SequenceState)
    assert s.ego.pose is None


def test_egomotion_happy_path_caches_eager_load(
    patched_parent: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `parent.get_clip_feature` succeeds, the slot is populated at
    construction time and the property returns it without re-calling the
    parent — pinning the unchanged happy-path behavior."""
    from physical_ai_av import PhysicalAIAVDatasetInterface

    calls = {"n": 0}
    sentinel = object()

    def _stub(self, clip_id: str, feature: str):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return sentinel

    monkeypatch.setattr(PhysicalAIAVDatasetInterface, "get_clip_feature", _stub)

    ds = CascadeDataset(CORPUS)
    clip_id = ds.list_sequences()[0]
    seq = ds.get_sequence(clip_id)
    assert seq._egomotion_interpolator is sentinel
    assert seq.egomotion_interpolator is sentinel
    assert calls["n"] == 1  # cached — no re-fetch on property access
