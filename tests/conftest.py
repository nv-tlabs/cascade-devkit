# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from cascade_av.io import load_file
from cascade_av.spec import AnnotationBundle


def _corpus_root() -> Path | None:
    """Resolve the annotation-corpus root from `CASCADE_AV_DATASET_ROOT`,
    matching the env-var contract documented in AGENTS.md and `.env.example`.
    Returns ``None`` when unset so fixtures can `pytest.skip` cleanly
    instead of failing with a hard-to-diagnose path error.
    """
    root = os.environ.get("CASCADE_AV_DATASET_ROOT")
    if not root:
        return None
    return Path(root)


CORPUS = _corpus_root()


class FakeVideoReader:
    """Stand-in for the parent dataset's `SeekVideoReader`.

    `decode_images_from_timestamps` returns a `(H, W, 3)` uint8 array per
    requested timestamp and records each call so tests can assert on
    the exact `t_us` argument the widget passed in.

    Originally lived inside ``tests/test_viz_widget.py`` (PR #24); lifted
    here so the PR-5 carousel tests can reuse the same fake without
    re-creating it. The widget tests still import this name as
    ``_FakeVideoReader`` via a module-level alias.
    """

    def __init__(
        self,
        height: int = 8,
        width: int = 12,
        timestamps: np.ndarray | None = None,
    ) -> None:
        self.height = height
        self.width = width
        self.calls: list[np.ndarray] = []
        # Frame timestamps the reader "exposes" — drives the widget's
        # range-clamping in `_decode_frame`. Default mirrors a 30fps,
        # 10s clip that starts at exactly 0us, which keeps every
        # pre-existing test that constructs a bare `FakeVideoReader()`
        # green. Pass an explicit array (e.g., starting at `16389us`)
        # to exercise the GOP-offset path.
        if timestamps is None:
            timestamps = np.arange(0, 10 * 30, dtype=np.int64) * 33333
        self.timestamps = np.asarray(timestamps, dtype=np.int64)

    def decode_images_from_timestamps(
        self, t_us: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(np.asarray(t_us).copy())
        n = len(t_us)
        # Encode the timestamp into the first pixel so tests can tell
        # which frame was decoded.
        frames = np.zeros((n, self.height, self.width, 3), dtype=np.uint8)
        for i, ts in enumerate(t_us):
            frames[i, 0, 0, 0] = int(ts) & 0xFF
        return frames, np.asarray(t_us, dtype=np.int64)


def _require_corpus() -> Path:
    if CORPUS is None:
        pytest.skip("set CASCADE_AV_DATASET_ROOT to run corpus-backed tests")
    if not CORPUS.is_dir():
        pytest.skip(f"CASCADE_AV_DATASET_ROOT does not exist: {CORPUS}")
    return CORPUS


@pytest.fixture(scope="session")
def rich_path() -> Path:
    """The richest corpus file by size — guaranteed to have many features."""
    corpus = _require_corpus()
    return max(corpus.glob("*.json"), key=lambda p: p.stat().st_size)


@pytest.fixture(scope="session")
def rich_bundle(rich_path: Path) -> AnnotationBundle:
    return load_file(rich_path)


@pytest.fixture(scope="session")
def file_with_causal_link() -> Path:
    """First corpus file that has at least one `because_of` link."""
    corpus = _require_corpus()
    for p in sorted(corpus.glob("*.json")):
        data = json.loads(p.read_text())
        ann = data.get("annotation", {})
        for action in ann.get("ego_vehicle", {}).get("actions", []):
            if action.get("because_of"):
                return p
        for agent in ann.get("agents", []):
            for action in agent.get("actions", []):
                if action.get("because_of"):
                    return p
    pytest.skip("No corpus file with because_of links found")
