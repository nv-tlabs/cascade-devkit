# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema-extension surface for CASCADE.

Extensions live alongside a main annotation file as a sibling JSON
sidecar (``<stem>.extra.json``). Each extension declares a ``key``
(``<id>/<version>``, e.g. ``"bbox/1.0"``) and round-trips its own
data through :class:`Extension.load` and :class:`Extension.dump`.

No first-party extensions ship in-tree as of the 2.0.0 reboot — the
``ui/1.0`` extension that carried timeline-layout indices is retired now
that those indices live inline on the typed schema again. Sidecar payload
written under any unknown key is preserved verbatim in
:attr:`AnnotationBundle._sidecar_raw` so third-party producers (such as
``bbox/1.0``) round-trip safely. Register first-party extensions in this
module via :func:`register`, or third-party ones through the
``cascade_av.extensions`` entry-point group.
"""

from cascade_av.extensions.base import Extension
from cascade_av.extensions.registry import register, registered


def _register_default_extensions() -> None:
    """Register the first-party extensions that ship in-tree.

    Currently a no-op — the 2.0.0 reboot retired the only first-party
    extension (``ui/1.0``). The function is kept for parity with the
    registry's ``_reset_for_tests`` contract, which calls back into it.
    """


_register_default_extensions()


__all__ = [
    "Extension",
    "register",
    "registered",
]
