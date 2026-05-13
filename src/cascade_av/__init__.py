# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cascade_av — DevKit for CASCADE annotations on the Physical AI AV Dataset."""

from cascade_av.spec import AnnotationBundle

# `dataset` depends on `physical_ai_av`, which is gated behind the `[hf]`
# extra. Import lazily so this top-level package stays usable when the
# extra is not installed.
try:
    from cascade_av.dataset import CascadeDataset, Sequence
except ImportError:  # pragma: no cover — exercised only without `[hf]`
    CascadeDataset = None  # type: ignore[assignment]
    Sequence = None  # type: ignore[assignment]

__all__ = ["AnnotationBundle", "CascadeDataset", "Sequence"]
