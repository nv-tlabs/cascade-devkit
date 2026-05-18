# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema-extension surface for CASCADE.

Extensions live alongside a main annotation file as a sibling JSON
sidecar (``<stem>.extra.json``). Each extension declares a ``key``
(``<id>/<version>``, e.g. ``"bbox/1.0"``) and round-trips its own
data through :class:`Extension.load` and :class:`Extension.dump`.

One first-party extension ships in-tree: :class:`UiExtension`
(``"ui/1.0"``), which carries the annotator's timeline-layout indices
that lived as typed fields on the main schema through 2.1.0. Third
parties can register additional extensions via the
``cascade_av.extensions`` entry-point group, or by calling
:func:`register` explicitly.
"""

from cascade_av.extensions.base import Extension
from cascade_av.extensions.registry import register, registered
from cascade_av.extensions.ui import UI_INDEX_KEYS, UiExtension, UiIndexes


def _register_default_extensions() -> None:
    """Register the first-party extensions that ship in-tree.

    Called once at module-import time, and again from
    :func:`cascade_av.extensions.registry._reset_for_tests` so a test that
    wipes the registry to install a stub does not permanently un-register
    the defaults.
    """
    register(UiExtension())


_register_default_extensions()


__all__ = [
    "Extension",
    "UI_INDEX_KEYS",
    "UiExtension",
    "UiIndexes",
    "register",
    "registered",
]
