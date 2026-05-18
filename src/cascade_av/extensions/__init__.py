# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema-extension surface for CASCADE.

Extensions live alongside a main annotation file as a sibling JSON
sidecar (``<stem>.extra.json``). Each extension declares a ``key``
(``<id>/<version>``, e.g. ``"bbox/1.0"``) and round-trips its own
data through :class:`Extension.load` and :class:`Extension.dump`.

This package is the scaffolding only. No extensions are registered
or shipped in-tree; the registry is populated by third-party
packages via the ``cascade_av.extensions`` entry-point group, or by
calling :func:`register` explicitly.
"""

from cascade_av.extensions.base import Extension
from cascade_av.extensions.registry import register, registered

__all__ = ["Extension", "register", "registered"]
