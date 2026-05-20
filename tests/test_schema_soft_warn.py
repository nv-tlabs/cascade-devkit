# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for the soft-warn loader contract on non-current ``schema_version``.

The library accepts arbitrary ``schema_version`` strings (no hard rejection)
but emits a ``DeprecationWarning`` once per process per non-current version,
asking the producer to re-export against the current schema. Pre-reboot
versions like ``"1.99.0"`` exercise the same code path as any future
non-current version since the 2.0.0 reboot collapsed the registry to a
single row.
"""

from __future__ import annotations

import warnings

import pytest

from cascade_av.spec import AnnotationBundle, CURRENT_SCHEMA_VERSION
from cascade_av.spec.schema import _reset_version_warn_cache_for_tests

# A synthetic non-current schema_version. The loader is open by design;
# any string other than CURRENT_SCHEMA_VERSION exercises the warn path.
_OLD_VERSION = "1.99.0"
_OTHER_OLD_VERSION = "1.0.0"


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


def test_non_current_schema_fires_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match=r"1\.99\.0"):
        AnnotationBundle.model_validate(_make_bundle(_OLD_VERSION))


def test_warning_message_mentions_re_export() -> None:
    """The advisory points the producer at the fix: re-export against the
    current schema. There is no in-tree migration path post-reboot."""
    with pytest.warns(DeprecationWarning, match=r"Re-export"):
        AnnotationBundle.model_validate(_make_bundle(_OLD_VERSION))


def test_warning_message_mentions_current_version() -> None:
    """The warning text quotes ``CURRENT_SCHEMA_VERSION`` so the producer
    knows the target version to write."""
    with pytest.warns(DeprecationWarning, match=CURRENT_SCHEMA_VERSION):
        AnnotationBundle.model_validate(_make_bundle(_OLD_VERSION))


def test_current_version_does_not_warn() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AnnotationBundle.model_validate(_make_bundle(CURRENT_SCHEMA_VERSION))
    relevant = [w for w in caught if "schema_version" in str(w.message)]
    assert relevant == []


def test_warn_fires_once_per_version_per_process() -> None:
    """Loading 100 bundles with the same non-current version fires exactly one warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(100):
            AnnotationBundle.model_validate(_make_bundle(_OLD_VERSION))
    deprecations = [
        w for w in caught
        if issubclass(w.category, DeprecationWarning) and _OLD_VERSION in str(w.message)
    ]
    assert len(deprecations) == 1, (
        f"expected exactly one DeprecationWarning, got {len(deprecations)}"
    )


def test_distinct_old_versions_each_warn_once() -> None:
    """Two different non-current versions each get their own one-shot warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AnnotationBundle.model_validate(_make_bundle(_OLD_VERSION))
        AnnotationBundle.model_validate(_make_bundle(_OTHER_OLD_VERSION))
        # Second loads of the same version do not fire again.
        AnnotationBundle.model_validate(_make_bundle(_OLD_VERSION))
        AnnotationBundle.model_validate(_make_bundle(_OTHER_OLD_VERSION))
    deprecations = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    versions_warned = {
        m.group(1)
        for w in deprecations
        for m in [__import__("re").search(r"schema_version=('[^']+')", str(w.message))]
        if m
    }
    assert versions_warned == {f"'{_OLD_VERSION}'", f"'{_OTHER_OLD_VERSION}'"}
