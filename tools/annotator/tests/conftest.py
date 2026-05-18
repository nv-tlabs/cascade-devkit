# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the annotator server tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

CORPUS = Path("/home/horde/01_json_annotations")


@pytest.fixture(scope="session")
def corpus_path() -> Path:
    """The richest corpus file by size — used as the smoke fixture."""
    if not CORPUS.is_dir():
        pytest.skip(f"corpus directory not found: {CORPUS}")
    return max(CORPUS.glob("*.json"), key=lambda p: p.stat().st_size)


@pytest.fixture()
def corpus_copy(corpus_path: Path, tmp_path: Path) -> Path:
    """A writable copy of the smoke-fixture annotation JSON in `tmp_path`."""
    dest = tmp_path / corpus_path.name
    shutil.copy(corpus_path, dest)
    return dest
