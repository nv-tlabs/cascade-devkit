# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke checks on the schema-version registry."""

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
    assert is_known("2.0.0")
    assert is_known("2.1.0")
    assert not is_known("9.9.9")
    assert not is_known("")


def test_changelog_for_returns_matching_entry() -> None:
    entry = changelog_for("2.1.0")
    assert isinstance(entry, SchemaVersion)
    assert entry.version == "2.1.0"
    assert entry.breaking is True
    assert "bounding_boxes" in entry.summary
    assert changelog_for("9.9.9") is None


def test_2_1_0_marked_breaking_and_2_0_0_not_breaking() -> None:
    assert changelog_for("2.0.0").breaking is False
    assert changelog_for("2.1.0").breaking is True
