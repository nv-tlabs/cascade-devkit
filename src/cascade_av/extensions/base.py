# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for CASCADE schema extensions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cascade_av.spec import AnnotationBundle


class Extension(ABC):
    """One side-car-resident addition to the CASCADE schema.

    Subclasses are stateless encoders. The actual data lives on the
    :class:`~cascade_av.spec.AnnotationBundle` (accessed via
    :meth:`AnnotationBundle.ext`); the extension only knows how to
    read its payload off disk and write it back.

    A registered extension claims its ``key`` in the sidecar's
    ``extensions`` object. Anything not claimed by a registered
    extension is preserved verbatim on round-trip (see
    :mod:`cascade_av.io.local`).
    """

    #: Identifier of the form ``"<id>/<version>"`` (e.g. ``"bbox/1.0"``).
    #: Must be unique within the registry.
    key: str

    #: Main-bundle ``schema_version`` strings this extension understands.
    #: The loader does not enforce this today; it is advisory metadata
    #: future versions of the loader may gate on.
    schema_versions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, bundle: AnnotationBundle, ext_data: dict[str, Any]) -> None:
        """Attach ``ext_data`` to ``bundle`` (typically via ``bundle._extensions``)."""

    @abstractmethod
    def dump(self, bundle: AnnotationBundle) -> dict[str, Any] | None:
        """Extract this extension's data from ``bundle`` for sidecar serialisation.

        Return ``None`` (or an empty dict) to signal there is nothing to
        write; the IO layer will then omit this key from the sidecar.
        """
