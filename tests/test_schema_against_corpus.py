# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate every JSON in /home/horde/01_json_annotations/ against the schema.

If a file fails, the schema is wrong — not the file. The on-disk corpus is the
ground truth for schema_version 2.0.0. Triage failures by reading the
diagnostic, then relax / extend the corresponding field in
`causal_ai_av.spec.schema`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal_ai_av.spec import AnnotationBundle

CORPUS = Path("/home/horde/01_json_annotations")
ALL_FILES = sorted(CORPUS.glob("*.json"))


def test_corpus_present() -> None:
    """Sanity: refuse to silently pass if the corpus directory is empty."""
    assert len(ALL_FILES) > 0, f"No JSONs in {CORPUS}"


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.stem[:13])
def test_corpus_file_validates(path: Path) -> None:
    data = json.loads(path.read_text())
    AnnotationBundle.model_validate(data)


def test_extras_passthrough_on_richest_file() -> None:
    """A rich corpus file must round-trip with no top-level key loss.

    Picks the largest file (richest schema usage) and asserts the set of
    annotation-level keys matches after load + dump.
    """
    rich = max(ALL_FILES, key=lambda p: p.stat().st_size)
    original = json.loads(rich.read_text())
    bundle = AnnotationBundle.model_validate(original)
    redumped = bundle.model_dump(mode="json", by_alias=True, exclude_unset=True)
    assert set(original.keys()) == set(redumped.keys()), (
        f"top-level key loss in {rich.name}: "
        f"original={set(original.keys())} dumped={set(redumped.keys())}"
    )
    assert set(original["annotation"].keys()) == set(redumped["annotation"].keys()), (
        f"annotation key loss in {rich.name}"
    )
