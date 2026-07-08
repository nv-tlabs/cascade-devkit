# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static, publication-oriented composition of video frames and tracks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from cascade_av.viz.timeline import (
    TrackVisibility,
    _paint_timeline_onto,
    _timeline_px_for,
)
from cascade_av.viz.widget import _encode_frame_jpeg

if TYPE_CHECKING:  # pragma: no cover - typing only
    import plotly.graph_objects as go

    from cascade_av.dataset import Sequence as CascadeSequence


_MAX_FRAMES = 3
_DEFAULT_WIDTH = 1400
_MAX_FRAME_ROW_PX = 480
_PAPER_FRAME_QUALITY = 90
_PAPER_FRAME_MAX_DIM = 1920
_FRAME_GAP_FRACTION = 0.015
_CAPTION_GAP_PX = 32
_MIN_PLOTLY_DIMENSION = 10


def _validate_timestamps(timestamps: Sequence[float]) -> list[float]:
    """Return finite timestamp seconds, preserving caller order."""
    if isinstance(timestamps, (str, bytes)):
        raise TypeError("timestamps must be a sequence of numeric seconds")

    values = list(timestamps)
    if len(values) > _MAX_FRAMES:
        raise ValueError(
            f"paper_figure accepts at most {_MAX_FRAMES} timestamps; "
            f"got {len(values)}"
        )

    resolved: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"timestamps[{index}] must be a numeric number of seconds; "
                f"got {value!r}"
            )
        timestamp = float(value)
        if not math.isfinite(timestamp):
            raise ValueError(f"timestamps[{index}] must be finite; got {value!r}")
        resolved.append(timestamp)
    return resolved


def _validate_dimension(
    name: str, value: int | None, *, optional: bool = False
) -> int | None:
    """Enforce Plotly's integral, non-boolean 10px dimension floor."""
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    resolved = int(value)
    if resolved < _MIN_PLOTLY_DIMENSION:
        raise ValueError(
            f"{name} must be at least {_MIN_PLOTLY_DIMENSION}px, got {value!r}"
        )
    return resolved


def _decode_frames(
    seq: "CascadeSequence", timestamps: list[float]
) -> list[Image.Image]:
    """Decode explicit timestamps in one request, without silent clamping."""
    if not timestamps:
        return []

    try:
        video = seq.video
    except Exception as exc:
        raise RuntimeError(
            "paper_figure frames require a video-rooted Sequence; "
            "seq.video is unavailable"
        ) from exc
    if video is None:
        raise RuntimeError("paper_figure frames require seq.video; got None")

    try:
        available_us = np.asarray(video.timestamps, dtype=np.int64)
    except Exception as exc:
        raise RuntimeError(
            "paper_figure requires the video reader to expose frame timestamps"
        ) from exc
    if available_us.size == 0:
        raise RuntimeError(
            "paper_figure cannot decode frames from a video with no timestamps"
        )

    video_min = float(available_us.min()) / 1_000_000
    video_max = float(available_us.max()) / 1_000_000
    for index, timestamp in enumerate(timestamps):
        if timestamp < video_min or timestamp > video_max:
            raise ValueError(
                f"timestamps[{index}]={timestamp} is outside the video's "
                f"timestamp coverage [{video_min}, {video_max}]"
            )

    requested_us = np.asarray(
        [int(round(timestamp * 1_000_000)) for timestamp in timestamps],
        dtype=np.int64,
    )
    try:
        images, _decoded_timestamps = video.decode_images_from_timestamps(
            requested_us
        )
    except Exception as exc:
        raise RuntimeError(
            "video.decode_images_from_timestamps failed for paper_figure "
            f"timestamps {timestamps!r}"
        ) from exc
    if images is None or len(images) != len(timestamps):
        decoded_count = 0 if images is None else len(images)
        raise RuntimeError(
            "video.decode_images_from_timestamps returned "
            f"{decoded_count} frame(s) for {len(timestamps)} timestamp(s)"
        )

    return [Image.fromarray(np.asarray(frame)).convert("RGB") for frame in images]


def render_paper_figure(
    seq: "CascadeSequence",
    timestamps: Sequence[float] = (),
    *,
    highlight: tuple[float, float] | None = None,
    arrows: dict[str, bool] | None = None,
    entity_kinds: list[str] | None = None,
    agent_ids: list[str] | None = None,
    track_groups: list[str] | None = None,
    families: list[str] | None = None,
    track_visibility: TrackVisibility | None = None,
    width: int = _DEFAULT_WIDTH,
    height: int | None = None,
    show_inline_labels: bool = True,
) -> "go.Figure":
    """Return a static Plotly figure with up to three frames above tracks.

    ``timestamps`` are seconds in the source video. Their input order is
    preserved left-to-right, duplicates are allowed, and explicit values are
    never silently clamped. An empty sequence is valid and produces a
    timeline-only paper figure without accessing ``seq.video``. Requested
    frames are embedded as quality-90 JPEGs, capped at 1920 pixels on the
    longer edge, so notebook and HTML artifacts stay portable.

    ``track_visibility`` switches whole kinds or individual top-level entity
    IDs. For example, ``{"agent": {"agent_4": False}}`` hides that Agent's
    parent and child rows; causal arrows touching hidden rows disappear too.
    Omitted switches remain visible. The other filters match
    :func:`render_timeline` and compose by intersection.

    Args:
        seq: source sequence.
        timestamps: zero to three finite timestamps in seconds.
        highlight: optional timeline highlight interval.
        arrows: optional per-arrow-family switches.
        entity_kinds: optional entity-kind whitelist.
        agent_ids: backward-compatible Agent ID whitelist.
        track_groups: optional category whitelist.
        families: optional family-leaf whitelist.
        track_visibility: grouped kind/entity visibility switches.
        width: figure width in pixels.
        height: optional total figure height in pixels. The default adapts to
            both the frame aspect ratio and timeline lane depth.
        show_inline_labels: whether timeline bars carry inline text.

    Returns:
        A plain ``plotly.graph_objects.Figure`` suitable for HTML or static
        image export.

    Raises:
        TypeError: if timestamps, dimensions, or visibility switches have
            invalid types.
        ValueError: for more than three timestamps, an out-of-video timestamp,
            an unknown visibility selector, or dimensions below 10 pixels.
        RuntimeError: if requested frames cannot be decoded.
    """
    import plotly.graph_objects as go

    resolved_timestamps = _validate_timestamps(timestamps)
    resolved_width = _validate_dimension("width", width)
    resolved_height = _validate_dimension("height", height, optional=True)
    assert resolved_width is not None

    fig = go.Figure()
    paint = _paint_timeline_onto(
        fig,
        seq,
        highlight=highlight,
        arrows=arrows,
        entity_kinds=entity_kinds,
        agent_ids=agent_ids,
        track_groups=track_groups,
        families=families,
        track_visibility=track_visibility,
        show_inline_labels=show_inline_labels,
    )
    timeline_px = _timeline_px_for(paint)
    frames = _decode_frames(seq, resolved_timestamps)

    if not frames:
        fig.update_layout(
            template="plotly_white",
            width=resolved_width,
            height=(
                resolved_height
                if resolved_height is not None
                else timeline_px
            ),
            margin={"l": 80, "r": 20, "t": 20, "b": 40},
            showlegend=False,
        )
        return fig

    frame_count = len(frames)
    usable_width = max(1, resolved_width - 100)
    frame_gap_px = _FRAME_GAP_FRACTION * usable_width * (frame_count - 1)
    slot_width_px = max(1.0, (usable_width - frame_gap_px) / frame_count)
    natural_heights = [
        slot_width_px / (frame.width / frame.height)
        for frame in frames
        if frame.width > 0 and frame.height > 0
    ]
    frame_row_px = min(
        _MAX_FRAME_ROW_PX,
        max(natural_heights, default=float(_MAX_FRAME_ROW_PX)),
    )
    content_px = timeline_px + _CAPTION_GAP_PX + frame_row_px
    timeline_domain_top = timeline_px / content_px
    frame_domain_bottom = (timeline_px + _CAPTION_GAP_PX) / content_px

    gap = _FRAME_GAP_FRACTION
    slot_width = (1.0 - gap * (frame_count - 1)) / frame_count
    images: list[dict[str, object]] = []
    for index, frame in enumerate(frames):
        x0 = index * (slot_width + gap)
        images.append(
            {
                "source": _encode_frame_jpeg(
                    np.asarray(frame),
                    quality=_PAPER_FRAME_QUALITY,
                    max_dim=_PAPER_FRAME_MAX_DIM,
                ),
                "xref": "paper",
                "yref": "paper",
                "x": x0,
                "y": 1.0,
                "sizex": slot_width,
                "sizey": 1.0 - frame_domain_bottom,
                "xanchor": "left",
                "yanchor": "top",
                "sizing": "contain",
                "layer": "above",
                "name": f"frame-{index}",
            }
        )

    fig.update_layout(
        template="plotly_white",
        width=resolved_width,
        height=(
            resolved_height
            if resolved_height is not None
            else int(round(content_px))
        ),
        margin={"l": 80, "r": 20, "t": 20, "b": 40},
        showlegend=False,
        images=images,
        xaxis={"domain": [0.0, 1.0]},
        yaxis={"domain": [0.0, timeline_domain_top]},
    )

    caption_y = timeline_domain_top + (
        frame_domain_bottom - timeline_domain_top
    ) / 2.0
    for index, timestamp in enumerate(resolved_timestamps):
        x0 = index * (slot_width + gap)
        fig.add_annotation(
            x=x0 + slot_width / 2.0,
            y=caption_y,
            xref="paper",
            yref="paper",
            text=f"t = {timestamp:g} s",
            showarrow=False,
            font={"size": 11, "color": "#334155"},
            xanchor="center",
            yanchor="middle",
        )

    return fig


__all__ = ["render_paper_figure"]
