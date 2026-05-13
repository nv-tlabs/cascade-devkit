# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless frame renderer — the Pillow-backed v0 of `render_frame`.

In v0, this just decodes the underlying video frame at `t` via the parent
`SeekVideoReader`. On-frame overlays (bounding boxes, keypoints, ego badges)
are deferred to v1 per `meta/10_visualization_api_plan.md` section 0a. The
`overlays` kwarg is accepted but ignored, so the call signature is stable
across the v0 → v1 transition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

# Re-export `render_timeline` so `from cascade_av.viz import render_timeline`
# (and the equivalent `from cascade_av.viz.render import render_timeline`)
# both work. The actual figure-building logic lives in `viz.timeline`; this
# module stays focused on raster-frame rendering.
from cascade_av.viz.timeline import render_timeline

if TYPE_CHECKING:  # pragma: no cover — typing only
    from cascade_av.dataset import Sequence


def render_frame(
    seq: "Sequence",
    t: float,
    overlays: dict[str, Any] | None = None,
) -> Image.Image:
    """Decode and return a single video frame as a `PIL.Image`.

    Args:
        seq: a `Sequence` whose `.video` resolves to a `SeekVideoReader`.
        t: timestamp in seconds.
        overlays: reserved for v1. Accepted but ignored in v0 — boxes /
            keypoints / ego-badge rendering land in a later PR. Kept on the
            signature so callers don't need to change when overlays arrive.

    Returns:
        A `PIL.Image.Image` in RGB mode containing the decoded frame.

    Raises:
        RuntimeError: if `seq.video` is unavailable (the clip has no
            video-rooted reader — annotation-only or stripped-down datasets
            cannot be rendered).
    """
    del overlays  # v0 no-op; signature placeholder for v1.

    try:
        video = seq.video
    except Exception as exc:
        raise RuntimeError(
            "render_frame requires a video-rooted Sequence; "
            "seq.video raised. Construct the Sequence via "
            "CascadeDataset.get_sequence on a parent dataset that owns "
            "the clip's MP4."
        ) from exc

    if video is None:
        raise RuntimeError(
            "render_frame requires seq.video; got None. The dataset does "
            "not have a video reader for this clip."
        )

    t_us = np.array([int(round(t * 1_000_000))], dtype=np.int64)
    images, _ = video.decode_images_from_timestamps(t_us)
    if images is None or len(images) == 0:
        raise RuntimeError(f"video.decode_images_from_timestamps returned no frames for t={t}")

    frame = np.asarray(images[0])
    return Image.fromarray(frame).convert("RGB")


__all__ = ["render_frame", "render_timeline"]
