# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from causal_ai_av.io import load_file
from causal_ai_av.spec import AnnotationBundle

CORPUS = Path("/home/horde/01_json_annotations")


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

    def __init__(self, height: int = 8, width: int = 12) -> None:
        self.height = height
        self.width = width
        self.calls: list[np.ndarray] = []

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


@pytest.fixture(scope="session")
def rich_path() -> Path:
    """The richest corpus file by size — guaranteed to have many features."""
    return max(CORPUS.glob("*.json"), key=lambda p: p.stat().st_size)


@pytest.fixture(scope="session")
def rich_bundle(rich_path: Path) -> AnnotationBundle:
    return load_file(rich_path)


@pytest.fixture(scope="session")
def file_with_causal_link() -> Path:
    """First corpus file that has at least one `because_of` link."""
    for p in sorted(CORPUS.glob("*.json")):
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
