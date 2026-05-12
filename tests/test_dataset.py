# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for `CausalAVDataset` and `Sequence`.

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

from causal_ai_av.dataset import CausalAVDataset, Sequence
from causal_ai_av.io import load_file
from causal_ai_av.spec import AnnotationBundle
from causal_ai_av.state import SequenceState, SequenceStateRange
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
        CausalAVDataset()


def test_non_dir_path_raises_file_not_found(tmp_path: Path) -> None:
    bogus = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        CausalAVDataset(bogus)


def test_local_dir_construction(patched_parent: None) -> None:
    ds = CausalAVDataset(CORPUS)
    assert len(ds.list_sequences()) > 0
    assert ds.annotation_camera == "camera_front_wide_120fov"


def test_sequence_from_annotation_round_trip(patched_parent: None) -> None:
    ds = CausalAVDataset(CORPUS)
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
    ds = CausalAVDataset(CORPUS)
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
    polymorphic dispatcher into ``causal_ai_av.viz``. The detailed
    dispatch matrix (scalar / tuple / match / context / static + the
    error paths) lives in ``tests/test_sequence_visualize.py``; this
    smoke test only pins that the call now produces a ``ClipPlayer``
    via the lazy ``viz`` import.
    """
    from causal_ai_av.viz import ClipPlayer

    path = max(CORPUS.glob("*.json"), key=lambda p: p.stat().st_size)
    seq = Sequence.from_annotation(load_file(path))
    # ``Sequence.video`` would normally require a parent dataset; the
    # widget tests stub the lazy ``_cameras`` slot to inject a fake
    # reader. Same trick here so ClipPlayer's initial decode succeeds.
    seq._cameras = {seq.annotation_camera: FakeVideoReader()}  # type: ignore[assignment]

    player = seq.visualize()
    assert isinstance(player, ClipPlayer)
    # The "no args" path opens over the whole clip.
    assert player._t_start == 0.0
    assert player._t_end == seq.duration_s
