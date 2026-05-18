# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema-version registry for the CASCADE annotation format.

Every released schema version gets one :class:`SchemaVersion` row in
:data:`SCHEMA_HISTORY` (chronological, oldest first). The current
version is :data:`CURRENT_SCHEMA_VERSION` and is the default value of
:attr:`cascade_av.spec.AnnotationBundle.schema_version`.

The companion human-readable changelog lives at
``docs/dev/schema-history.md`` and must be updated in the same PR
that changes :data:`SCHEMA_HISTORY`.

The loader does not (yet) reject unknown versions; an old version
triggers a one-shot ``DeprecationWarning`` per process that points
the user at the ``cascade-migrate`` CLI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchemaVersion:
    """One released version of the CASCADE annotation schema."""

    version: str
    released: str  # ISO date
    summary: str  # one-line human description
    breaking: bool  # field removal or semantic change


SCHEMA_HISTORY: tuple[SchemaVersion, ...] = (
    SchemaVersion(
        version="2.0.0",
        released="2026-05-12",
        summary=(
            "Initial CASCADE schema; bounding_boxes typed on Agent / "
            "TrafficObject / TrafficLight."
        ),
        breaking=False,
    ),
    SchemaVersion(
        version="2.1.0",
        released="2026-05-18",
        summary=(
            "Removed bounding_boxes from the base schema. Deprecated data "
            "moves to a sibling <stem>.extra.json sidecar under "
            "extensions['bbox/1.0']. Run cascade-migrate to convert."
        ),
        breaking=True,
    ),
    SchemaVersion(
        version="2.2.0",
        released="2026-05-18",
        summary=(
            "Removed _*_track_index UI-layout fields from every typed model. "
            "Indices move to a sibling <stem>.extra.json sidecar under "
            "extensions['ui/1.0'], keyed by entity id. Run cascade-migrate "
            "to convert; chains 2.0.0 → 2.1.0 → 2.2.0 in one pass."
        ),
        breaking=True,
    ),
)

#: Default value for ``AnnotationBundle.schema_version`` on new bundles
#: and the only version cascade_av writes when serialising.
CURRENT_SCHEMA_VERSION: str = SCHEMA_HISTORY[-1].version

#: Set of all version strings cascade_av recognises. Older versions still
#: load (with a deprecation warning); strings outside this set load too
#: but are flagged as unknown.
SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset(v.version for v in SCHEMA_HISTORY)


def is_known(version: str) -> bool:
    """Return True if ``version`` appears in :data:`SCHEMA_HISTORY`."""
    return version in SUPPORTED_SCHEMA_VERSIONS


def changelog_for(version: str) -> SchemaVersion | None:
    """Look up the :class:`SchemaVersion` row for ``version``, or None."""
    for entry in SCHEMA_HISTORY:
        if entry.version == version:
            return entry
    return None
