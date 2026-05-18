# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-wide registry of CASCADE schema extensions.

Two ways to populate the registry:

* Call :func:`register` directly with an :class:`Extension` instance.
* Ship an entry point in the ``cascade_av.extensions`` group; it will
  be discovered lazily the first time the registry is queried.

Discovery is wrapped in a broad ``try/except`` so a crashy third-party
extension cannot take down the loader.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from cascade_av.extensions.base import Extension

_log = logging.getLogger(__name__)

_ENTRY_POINT_GROUP = "cascade_av.extensions"

_registry: dict[str, Extension] = {}
_discovered: bool = False


def register(ext: Extension) -> None:
    """Register ``ext`` under its :attr:`Extension.key`.

    Re-registering the same key replaces the previous entry. Useful for
    tests; production extensions should each have a unique key.
    """
    _registry[ext.key] = ext


def registered() -> dict[str, Extension]:
    """Return the live registry, running entry-point discovery on first call."""
    _ensure_discovered()
    return _registry


def _ensure_discovered() -> None:
    global _discovered
    if _discovered:
        return
    _discovered = True
    try:
        for ep in entry_points(group=_ENTRY_POINT_GROUP):
            try:
                obj = ep.load()
                ext = obj() if isinstance(obj, type) else obj
                if isinstance(ext, Extension):
                    register(ext)
                else:
                    _log.warning(
                        "entry point %s did not yield an Extension instance",
                        ep.name,
                    )
            except Exception:
                _log.exception(
                    "failed to load cascade_av extension entry point %s", ep.name
                )
    except Exception:
        _log.exception("cascade_av extension discovery failed")


def _reset_for_tests() -> None:
    """Wipe the registry and re-register the first-party defaults.

    Intended for use by test fixtures that want a known starting state. After
    this call, the registry contains exactly the in-tree first-party
    extensions (e.g. :class:`~cascade_av.extensions.ui.UiExtension`) — same
    as immediately after ``import cascade_av.extensions`` at startup. Tests
    that need a *completely* empty registry can clear it again themselves.
    """
    global _discovered
    _registry.clear()
    _discovered = False
    # Re-import (cheap; module is already loaded) so the side-effect of
    # `_register_default_extensions()` runs again. We do this lazily to avoid
    # a circular import at module load time.
    from cascade_av.extensions import _register_default_extensions
    _register_default_extensions()
