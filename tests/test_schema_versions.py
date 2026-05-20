# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke checks on the schema-version registry.

The schema history was rebooted on 2026-05-20 (PR #113); the registry now
carries a single 2.0.0 row aligned with the upstream
``sil-dense-annotation-tool 0.4.5`` shape. These tests assert structural
invariants of the registry that hold regardless of how many rows it has.
"""

from __future__ import annotations

from cascade_av.spec import (
    CURRENT_SCHEMA_VERSION,
    SCHEMA_HISTORY,
    SUPPORTED_SCHEMA_VERSIONS,
    SchemaVersion,
    changelog_for,
    is_known,
)


def test_current_schema_version_is_the_latest_history_entry() -> None:
    assert SCHEMA_HISTORY, "SCHEMA_HISTORY must not be empty"
    assert CURRENT_SCHEMA_VERSION == SCHEMA_HISTORY[-1].version


def test_schema_history_is_chronological() -> None:
    """Release dates must not regress as we walk the history."""
    dates = [entry.released for entry in SCHEMA_HISTORY]
    assert dates == sorted(dates), f"SCHEMA_HISTORY not chronological: {dates}"


def test_schema_history_versions_are_unique() -> None:
    versions = [entry.version for entry in SCHEMA_HISTORY]
    assert len(versions) == len(set(versions)), versions


def test_supported_versions_matches_history() -> None:
    assert SUPPORTED_SCHEMA_VERSIONS == frozenset(e.version for e in SCHEMA_HISTORY)


def test_is_known_handles_known_and_unknown() -> None:
    assert is_known(CURRENT_SCHEMA_VERSION)
    assert not is_known("9.9.9")
    assert not is_known("")


def test_changelog_for_returns_matching_entry() -> None:
    entry = changelog_for(CURRENT_SCHEMA_VERSION)
    assert isinstance(entry, SchemaVersion)
    assert entry.version == CURRENT_SCHEMA_VERSION
    assert changelog_for("9.9.9") is None


def test_current_version_row_is_non_breaking_baseline() -> None:
    """The 2.0.0 reboot row is the baseline — explicitly not flagged as
    a breaking change relative to itself. Future entries may set
    ``breaking=True``; until then we pin this invariant."""
    entry = changelog_for(CURRENT_SCHEMA_VERSION)
    assert entry is not None
    assert entry.breaking is False
