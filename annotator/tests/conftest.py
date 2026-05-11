"""Shared fixtures for the annotator server tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from causal_ai_av.io import load_file

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


@pytest.fixture(scope="session")
def bbox_corpus_path() -> Path:
    """First corpus file that has at least one agent carrying bounding_boxes.

    Failing this fixture is a corpus-level invariant violation — every
    deployment exercise of the annotator should be able to round-trip a
    file with bbox frames. (At the time of writing, 246 of the 276 corpus
    files have at least one bbox-bearing agent.)
    """
    if not CORPUS.is_dir():
        pytest.skip(f"corpus directory not found: {CORPUS}")
    for p in sorted(CORPUS.glob("*.json")):
        try:
            bundle = load_file(p)
        except Exception:
            continue
        for agent in bundle.annotation.agents:
            if agent.bounding_boxes:
                return p
    pytest.fail("no corpus file has agents with bbox frames — fixture invariant broken")


@pytest.fixture()
def bbox_corpus_copy(bbox_corpus_path: Path, tmp_path: Path) -> Path:
    """A writable copy of the bbox-bearing corpus file in `tmp_path`."""
    dest = tmp_path / bbox_corpus_path.name
    shutil.copy(bbox_corpus_path, dest)
    return dest
