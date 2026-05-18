# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for the soft-warn loader contract on old schema_version values.

The library accepts older `schema_version` strings (no hard rejection) but
emits a ``DeprecationWarning`` once per process per non-current version
pointing the user at ``cascade-migrate``.
"""

from __future__ import annotations

import warnings

import pytest

from cascade_av.spec import AnnotationBundle, CURRENT_SCHEMA_VERSION
from cascade_av.spec.schema import _reset_version_warn_cache_for_tests


@pytest.fixture(autouse=True)
def _reset_warn_cache() -> None:
    _reset_version_warn_cache_for_tests()
    yield
    _reset_version_warn_cache_for_tests()


def _make_bundle(version: str) -> dict[str, object]:
    return {
        "schema_version": version,
        "video": {"clip_id": "warn-test", "fps": 30.0, "duration_s": 1.0},
    }


def test_old_schema_fires_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match=r"2\.0\.0"):
        AnnotationBundle.model_validate(_make_bundle("2.0.0"))


def test_warning_message_mentions_cascade_migrate() -> None:
    with pytest.warns(DeprecationWarning, match=r"cascade-migrate"):
        AnnotationBundle.model_validate(_make_bundle("2.0.0"))


def test_current_version_does_not_warn() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AnnotationBundle.model_validate(_make_bundle(CURRENT_SCHEMA_VERSION))
    relevant = [w for w in caught if "schema_version" in str(w.message)]
    assert relevant == []


def test_warn_fires_once_per_version_per_process() -> None:
    """Loading 100 bundles with the same old version fires exactly one warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(100):
            AnnotationBundle.model_validate(_make_bundle("2.0.0"))
    deprecations = [
        w for w in caught
        if issubclass(w.category, DeprecationWarning) and "2.0.0" in str(w.message)
    ]
    assert len(deprecations) == 1, (
        f"expected exactly one DeprecationWarning, got {len(deprecations)}"
    )


def test_distinct_old_versions_each_warn_once() -> None:
    """Two different non-current versions each get their own one-shot warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AnnotationBundle.model_validate(_make_bundle("2.0.0"))
        AnnotationBundle.model_validate(_make_bundle("1.0.0"))
        # Second loads of the same version do not fire again.
        AnnotationBundle.model_validate(_make_bundle("2.0.0"))
        AnnotationBundle.model_validate(_make_bundle("1.0.0"))
    deprecations = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    versions_warned = {
        m.group(1)
        for w in deprecations
        for m in [__import__("re").search(r"schema_version=('[^']+')", str(w.message))]
        if m
    }
    assert versions_warned == {"'2.0.0'", "'1.0.0'"}
