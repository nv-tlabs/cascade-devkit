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

The history was rebooted on 2026-05-20 against the upstream
``sil-dense-annotation-tool 0.4.5`` shape. There is no in-tree migration
path from pre-reboot bundles; downstream producers re-export against this
schema. The loader still tolerates unknown ``schema_version`` strings —
they trigger a one-shot ``DeprecationWarning`` per process.
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
        released="2026-05-20",
        summary="Initial CASCADE schema aligned with sil-dense-annotation-tool 0.4.5.",
        breaking=False,
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
