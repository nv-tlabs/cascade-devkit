"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal_ai_av.io import load_file
from causal_ai_av.spec import AnnotationBundle

CORPUS = Path("/home/horde/01_json_annotations")


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
