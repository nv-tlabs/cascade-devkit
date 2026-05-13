# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`ClipPlayer` — interactive Plotly + ipywidgets scrubber.

The PR-4 v0 of the interactive viz surface, per
`meta/10_visualization_api_plan.md`. Composes a `plotly.graph_objects.
FigureWidget` (top subplot: video frame `go.Image` trace; bottom
subplot: the static `render_timeline` figure, painted via
`_paint_timeline_onto`) with an `ipywidgets.FloatSlider` and an
`ipywidgets.Play` button stacked vertically.

Architecture is **pure server-side**: Python decodes one frame per
slider tick through `Sequence.video.decode_images_from_timestamps`,
the resulting numpy RGB array is dropped onto the FigureWidget's
Image trace inside a `batch_update()` so the playhead line and the
frame change atomically. No JavaScript extraction from the
annotator, no per-frame bbox / keypoint overlays in v0 — both are
deferred to v1 per the spec.

No prefetch ring, no worker thread, no background decode in v0. The
slider observer blocks the kernel for one decode per tick. At the
plan's target of ~8 fps scrub the latency is acceptable on the
parent dataset's 1080p MP4s; if a future PR wants smoother playback
the place to add a prefetch buffer is here.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from causal_ai_av.viz.timeline import (
    _PX_PER_LANE,
    _paint_timeline_onto,
    _timeline_px_for,
)

if TYPE_CHECKING:  # pragma: no cover — typing only
    import plotly.graph_objects as go

    from causal_ai_av.dataset import Sequence


# Subplot vertical-gap fraction. Top subplot is the video frame
# (taller); bottom is the timeline. The split is computed adaptively
# from the timeline's lane count in `__init__` — a busy clip with
# stacked sub-lanes gets a taller timeline so each lane stays
# readable, while a sparse clip stays compact. The annotator stacks
# video on top of timeline inside the same browser tab; we follow
# that reading order here.
#
# Tightened from 0.05 → 0.015 so the video and timeline read as one
# clustered viewer rather than two separated panes — matches the
# annotator's tight stack.
_VERTICAL_SPACING = 0.015

# Default scrub rate. The spec calls 8 fps "scrub-not-playback" and
# uses it as the smoke-test bar in `meta/10_visualization_api_plan.md`.
_DEFAULT_FPS = 8.0

# Pixel budget for the video subplot in the adaptive-height path.
# 480px keeps a 1080p frame legible at the default Jupyter cell width;
# users who want a giant video pane can override via the `height`
# kwarg (which pins the *total* height; the timeline's share scales
# down to compensate).
_VIDEO_PX = 480

# Chrome (margins + ipywidgets row) that the figure layout needs to
# leave room for. Adding it to the video + timeline pixels gives the
# default `height` value when callers don't override.
_CHROME_PX = 60

# Default JPEG quality for frame transport. Plotly's `go.Image` ships
# the trace's `z=` array to the browser as a JSON list-of-lists; for a
# 1920x1080 RGB frame that's ~32MB of JSON per tick, which dominates
# Play latency over the Jupyter Comm channel. Switching to `source=` —
# a base64 data URI of a JPEG-encoded frame — cuts the payload by ~50x
# and the Python-side serialization cost by ~25x. Quality 80 is a
# common "good enough at scrub-time" knee on the rate / distortion
# curve; power users can dial via the `frame_quality` kwarg.
_DEFAULT_FRAME_QUALITY = 80


def _encode_frame_jpeg(
    frame: np.ndarray,
    quality: int = _DEFAULT_FRAME_QUALITY,
    max_dim: int | None = None,
) -> str:
    """JPEG-encode a uint8 RGB frame into a ``data:image/jpeg;base64,...`` URI.

    Plotly's ``go.Image`` trace accepts either ``z=`` (raw pixel array,
    serialized as JSON list-of-lists over Comm transport) or ``source=``
    (a data URI consumed by the browser as a regular image). The data-
    URI path is ~50x smaller and ~25x faster on the Python side for
    1080p frames — see the ``_DEFAULT_FRAME_QUALITY`` comment above.

    Args:
        frame: ``(H, W, 3)`` uint8 RGB numpy array. Other dtypes are
            cast to uint8 first; mirrors ``render_frame``'s behaviour.
        quality: JPEG quality 1-95. Default 80 is a sensible scrub-time
            choice; pushing toward 90+ doubles the payload for marginal
            visual gain. Pillow clamps internally past 95.
        max_dim: optional pixel cap on the frame's longer dimension. The
            frame is downscaled with ``Image.LANCZOS`` so the longer
            side equals ``max_dim`` while preserving aspect ratio. ``None``
            (default) skips the downscale entirely.

    Returns:
        ``"data:image/jpeg;base64,<...>"``, ready to drop onto
        ``go.Image(source=...)`` or assign to ``trace.source``.
    """
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    img = Image.fromarray(frame)
    if max_dim is not None and max(img.size) > max_dim:
        # `img.size` is `(W, H)`. Scale so the longer side equals
        # `max_dim`; the other side keeps the aspect ratio.
        w, h = img.size
        scale = max_dim / float(max(w, h))
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


class ClipPlayer:
    """Interactive Plotly + ipywidgets player for a clip's video + timeline.

    Pure server-side — Python decodes frames on demand, Plotly's
    `FigureWidget` composites them with the static timeline. No
    JavaScript extraction from the annotator; no per-frame bbox /
    keypoint overlays (deferred to v1).

    Args:
        sequence: a `Sequence`. `sequence.video.decode_images_from_timestamps`
            is called per slider tick — for tests, mock it.
        t_start: playback window start, seconds. Defaults to `0.0`.
        t_end: playback window end, seconds. Defaults to
            `sequence.duration_s`.
        fps: scrub rate (frames per second). Defaults to `8.0`,
            matching the plan's "scrub-not-playback" target.
        highlight: forwarded to `_paint_timeline_onto`; translucent
            yellow band on the bottom subplot.
        arrows: forwarded to `_paint_timeline_onto`; per-family
            arrow-on/off toggles.
        entity_kinds: forwarded to `_paint_timeline_onto`; kind
            whitelist. See `render_timeline` for valid values.
        agent_ids: forwarded to `_paint_timeline_onto`; `Agent.id`
            whitelist.
        track_groups: forwarded to `_paint_timeline_onto`; group-row
            whitelist.
        height: optional explicit pixel height for the whole player.
            `None` (default) means adaptive — the timeline subplot
            scales to the deepest sub-lane stack on the bottom row
            and the video subplot keeps a fixed 480px budget. Pass
            an int to pin a specific total height; the video / timeline
            split adjusts so the timeline still gets at least the
            adaptive lane budget when possible.
        frame_quality: JPEG quality (1-95) used to encode each frame
            before shipping it over the Jupyter Comm channel. Default
            80 is a good scrub-time choice; lower values cut the
            payload further at the cost of visible compression
            artifacts. See ``_encode_frame_jpeg``.
        frame_max_dim: optional pixel cap on the frame's longer
            dimension. When set, frames are downscaled (LANCZOS) so
            their longer side equals ``frame_max_dim`` before JPEG
            encoding — a power knob for very large frames or slow
            transports. ``None`` (default) keeps the source resolution.
        show_inline_labels: forwarded to ``_paint_timeline_onto``; when
            False, suppress every inline label annotation on the
            timeline subplot. Hover tooltips still fire. Defaults to
            True (historical behaviour).

    Display protocol:
        - In Jupyter / JupyterLab the widget renders inline through
          `_ipython_display_` (the standard ipywidgets path).
        - For environments that don't speak the widget protocol
          (nbviewer, GitHub blob renders, etc.), `_repr_mimebundle_`
          falls back to an HTML snapshot of the current figure so the
          user still sees the timeline + the initial frame.

    Performance:
        v0 has **no prefetch and no worker thread**. Each slider tick
        blocks the kernel for one
        `decode_images_from_timestamps([t_us])` call plus a JPEG
        encode of the result. Frames cross the Jupyter Comm channel
        as base64 data URIs (`source=` on the `go.Image` trace)
        rather than raw `z=` arrays — see `_encode_frame_jpeg`
        for the ~50x payload / ~25x encode-time win on 1080p frames.
        On 1080p MP4 clips this comfortably hits the spec's 8 fps
        scrub bar; if a future PR wants smoother playback the place
        to add a prefetch buffer is here.
    """

    def __init__(
        self,
        sequence: "Sequence",
        *,
        t_start: float | None = None,
        t_end: float | None = None,
        fps: float = _DEFAULT_FPS,
        highlight: tuple[float, float] | None = None,
        arrows: dict[str, bool] | None = None,
        entity_kinds: list[str] | None = None,
        agent_ids: list[str] | None = None,
        track_groups: list[str] | None = None,
        height: int | None = None,
        frame_quality: int = _DEFAULT_FRAME_QUALITY,
        frame_max_dim: int | None = None,
        show_inline_labels: bool = True,
    ) -> None:
        # Defer the optional-extra imports so importing this module
        # without `[viz]` doesn't blow up at module load — only at
        # construction time.
        import ipywidgets
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps!r}")

        # Resolve playback window. Both endpoints are seconds.
        duration = float(sequence.duration_s) if sequence.duration_s else 0.0
        resolved_start = 0.0 if t_start is None else float(t_start)
        resolved_end = duration if t_end is None else float(t_end)
        if resolved_end <= resolved_start:
            raise ValueError(
                f"t_end ({resolved_end}) must be greater than t_start "
                f"({resolved_start}); resolved from sequence.duration_s="
                f"{duration!r}."
            )

        self._sequence = sequence
        self._t_start = resolved_start
        self._t_end = resolved_end
        self._fps = float(fps)
        self._t = resolved_start
        # Frame-transport knobs — stored so `_apply_t` re-uses the same
        # quality / downscale on every tick that the construction-time
        # trace used. See `_encode_frame_jpeg` for the rationale.
        self._frame_quality = int(frame_quality)
        self._frame_max_dim = (
            int(frame_max_dim) if frame_max_dim is not None else None
        )

        # ------------------------------------------------------------
        # 1. Build the underlying FigureWidget — two stacked subplots:
        #    row 1 video frame, row 2 timeline.
        #
        #    The `row_heights` split is computed adaptively after the
        #    timeline is painted, since the timeline's pixel budget
        #    scales with the deepest sub-lane stack. We start with a
        #    placeholder 50/50 split, then override the subplot
        #    domains and total height once `_paint_timeline_onto`
        #    has reported the lane count.
        # ------------------------------------------------------------
        base = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.5, 0.5],
            vertical_spacing=_VERTICAL_SPACING,
            shared_xaxes=False,
        )

        # Initial frame at t_start. We decode through the sequence's
        # video reader; tests substitute a fake reader that returns a
        # known fake array. The frame is JPEG-encoded into a data URI
        # rather than passed as a raw `z=` array — see
        # `_encode_frame_jpeg` for the rationale (avoids ~32MB of JSON
        # per tick over the Jupyter Comm channel).
        initial_frame = self._decode_frame(self._t)
        base.add_trace(
            go.Image(
                source=_encode_frame_jpeg(
                    initial_frame,
                    quality=self._frame_quality,
                    max_dim=self._frame_max_dim,
                ),
                name="frame",
            ),
            row=1,
            col=1,
        )

        # Convert to a FigureWidget *before* painting the timeline so
        # the playhead-line shape can be added through the same
        # `update_layout` path on a live widget.
        fig = go.FigureWidget(base)

        # Hide the image axes ticks — the video frame doesn't have
        # meaningful axis units; we only want the picture. Total
        # `height` is overridden below once the adaptive sizing has
        # been computed from the timeline's lane counts.
        fig.update_layout(
            template="plotly_white",
            # Tight top + bottom margins so the video + timeline read
            # as one clustered viewer (matches the annotator's stack).
            margin={"l": 80, "r": 20, "t": 6, "b": 30},
            showlegend=False,
            xaxis={"showticklabels": False, "showgrid": False, "zeroline": False},
            yaxis={"showticklabels": False, "showgrid": False, "zeroline": False},
        )

        # Paint the timeline onto the second subplot (x2 / y2). The
        # painter returns a `PaintResult` describing the band stack —
        # band keys in y-order, per-band lane counts, the y-coord
        # range, and the total lane count — so we can size the bottom
        # subplot proportionally to the deepest stack across bands.
        paint = _paint_timeline_onto(
            fig,
            sequence,
            xref="x2",
            yref="y2",
            xaxis_key="xaxis2",
            yaxis_key="yaxis2",
            highlight=highlight,
            arrows=arrows,
            entity_kinds=entity_kinds,
            agent_ids=agent_ids,
            track_groups=track_groups,
            show_inline_labels=show_inline_labels,
        )

        # Adaptive height: frame_px on top, lane-count × _PX_PER_LANE
        # on the bottom (with a floor so a sparse clip still gets
        # breathing room), plus chrome for margins / slider. Callers
        # can pass `height=N` to pin a total value — we then keep the
        # timeline at its computed pixel share and let the video pane
        # absorb the rest, preserving the lane-band readability the
        # whole adaptive path is about.
        timeline_px = _timeline_px_for(paint)
        if height is not None:
            total_height = int(height)
            video_px = max(
                _PX_PER_LANE,  # avoid collapsing the video pane to zero
                total_height - timeline_px - _CHROME_PX,
            )
        else:
            video_px = _VIDEO_PX
            total_height = video_px + timeline_px + _CHROME_PX

        # `make_subplots` writes y-domain fractions onto the two
        # yaxes; override them so the split tracks pixel counts
        # rather than the placeholder 50/50 we passed in above.
        # `_VERTICAL_SPACING` is a domain fraction (matching the
        # `make_subplots` argument we passed earlier); reserving it
        # at the bottom of the top subplot keeps the gap consistent
        # with the placeholder layout.
        subplot_total = video_px + timeline_px
        usable = 1.0 - _VERTICAL_SPACING
        timeline_domain_top = (timeline_px / subplot_total) * usable
        fig.update_layout(
            height=total_height,
            yaxis={
                "showticklabels": False,
                "showgrid": False,
                "zeroline": False,
                "domain": [timeline_domain_top + _VERTICAL_SPACING, 1.0],
            },
            yaxis2={"domain": [0.0, timeline_domain_top]},
        )

        # ------------------------------------------------------------
        # 2. Playhead — a vertical line on the bottom subplot's axes.
        #    Appended after the timeline shapes so it sits on top.
        #    The y-range comes from the painter's `PaintResult` (rather
        #    than the legacy 5-row integer grid) so the playhead spans
        #    the full populated band stack, however many bands that is.
        # ------------------------------------------------------------
        y_min, y_max = paint.y_range
        playhead = {
            "type": "line",
            "x0": self._t,
            "x1": self._t,
            "y0": y_min,
            "y1": y_max,
            "xref": "x2",
            "yref": "y2",
            "line": {"color": "#fbbf24", "width": 2, "dash": "dash"},
            "layer": "above",
            "name": "playhead",
        }
        fig.update_layout(shapes=list(fig.layout.shapes or ()) + [playhead])
        # Track the playhead's index in `layout.shapes` so we can
        # mutate it cheaply in the slider observer.
        self._playhead_index = len(fig.layout.shapes) - 1

        self._fig: go.FigureWidget = fig

        # ------------------------------------------------------------
        # 3. ipywidgets — FloatSlider visible to the user, Play button
        #    in integer-ms units linked through a Python observer.
        #
        #    DEVIATION FROM SPEC: ipywidgets.Play stores its `value` as
        #    `CInt`; a `jslink((play, "value"), (slider, "value"))` to
        #    a `FloatSlider` truncates seconds → 0/1/.../N silently.
        #    The plan flagged this as a known fallback option. We keep
        #    the user-visible slider in seconds (per the spec's
        #    `description="t (s)"`) and link the two widgets through
        #    Python `.observe` translators that round-trip via
        #    milliseconds.
        # ------------------------------------------------------------
        slider_step = 1.0 / self._fps
        self._slider = ipywidgets.FloatSlider(
            min=self._t_start,
            max=self._t_end,
            step=slider_step,
            value=self._t,
            description="t (s)",
            continuous_update=True,
            readout=True,
            readout_format=".3f",
        )
        play_step_ms = max(1, int(round(1000.0 / self._fps)))
        self._play = ipywidgets.Play(
            value=int(round(self._t * 1000)),
            min=int(round(self._t_start * 1000)),
            max=int(round(self._t_end * 1000)),
            step=play_step_ms,
            interval=play_step_ms,
            description="play",
        )

        # Re-entrancy guard for the bidirectional Play ↔ Slider link.
        self._suspend_link = False
        self._play.observe(self._on_play_change, names="value")
        self._slider.observe(self._on_slider_change, names="value")

        # Container with the slider above the Play button, both above
        # the figure. ipywidgets boxes flow top → bottom by default.
        self._controls = ipywidgets.HBox([self._play, self._slider])
        self._container = ipywidgets.VBox([self._fig, self._controls])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def t(self) -> float:
        """Current playhead position, seconds."""
        return self._t

    @property
    def widget(self) -> Any:
        """The underlying ipywidgets composite (`VBox` of figure + controls).

        Returned for embedding in other ipywidgets containers — e.g. the
        `MatchSet.visualize()` carousel which stacks one labeled
        `ClipPlayer` per match inside a `VBox` / `GridBox`. The
        ipywidgets display protocol (`_ipython_display_` /
        `_repr_mimebundle_`) is still the preferred path for rendering
        a player on its own; this property exists so other widgets can
        wrap one without reaching into a private attribute.
        """
        return self._container

    def seek(self, t: float) -> None:
        """Move the playhead to `t` (seconds), updating the figure.

        Raises:
            ValueError: if `t` is outside `[t_start, t_end]`.
        """
        t = float(t)
        if t < self._t_start or t > self._t_end:
            raise ValueError(
                f"seek({t}) is outside the playback window "
                f"[{self._t_start}, {self._t_end}]."
            )
        # Drive the seek through the slider so observers fire exactly
        # once (the slider observer is the central handler). Suspend
        # the bidirectional link so we don't ping-pong with Play.
        self._suspend_link = True
        try:
            self._slider.value = t
        finally:
            self._suspend_link = False
        # The slider observer would normally run, but ipywidgets may
        # short-circuit if the value didn't actually change (e.g., a
        # second seek to the same `t`). Apply the figure update
        # unconditionally so callers don't have to think about it.
        self._apply_t(t)

    # ------------------------------------------------------------------
    # ipywidgets display protocol
    # ------------------------------------------------------------------

    def _ipython_display_(self) -> None:
        """Render the widget inline in a Jupyter / JupyterLab cell."""
        from IPython.display import display

        display(self._container)

    def _repr_mimebundle_(
        self,
        include: Any | None = None,
        exclude: Any | None = None,
    ) -> dict[str, Any]:
        """Return a multi-format mimebundle.

        - `text/html`: a static snapshot of the current figure so the
          widget still renders something readable in environments that
          don't speak the ipywidgets protocol (nbviewer, GitHub blob
          renders, archived `.ipynb` files).
        - `application/vnd.jupyter.widget-view+json`: the live
          ipywidgets view, used by Jupyter / JupyterLab front-ends.
        """
        # The widget-view payload lets Jupyter look up the live model.
        widget_view = {
            "version_major": 2,
            "version_minor": 0,
            "model_id": getattr(self._container, "model_id", None),
        }
        # Static HTML snapshot. Cheap to build; mirrors the static
        # `render_timeline` figure plus the initial frame.
        html_snapshot = self._fig.to_html(
            full_html=False, include_plotlyjs="cdn"
        )
        return {
            "text/html": html_snapshot,
            "application/vnd.jupyter.widget-view+json": widget_view,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_slider_change(self, change: dict[str, Any]) -> None:
        """Slider moved — update the figure, and mirror to Play."""
        new_t = float(change["new"])
        self._apply_t(new_t)
        if not self._suspend_link:
            self._suspend_link = True
            try:
                self._play.value = int(round(new_t * 1000))
            finally:
                self._suspend_link = False

    def _on_play_change(self, change: dict[str, Any]) -> None:
        """Play tick — mirror to the slider, which drives the figure.

        The slider's `value` assignment fires `_on_slider_change`,
        which itself calls `_apply_t`. Do **not** call `_apply_t`
        again here — doing so caused every Play tick to decode the
        same frame twice (once from the slider observer, once from
        this method), doubling the per-tick latency.
        """
        if self._suspend_link:
            return
        new_t = float(change["new"]) / 1000.0
        # Mirror to the slider; the slider observer then applies the
        # figure update.
        self._suspend_link = True
        try:
            self._slider.value = new_t
        finally:
            self._suspend_link = False

    def _apply_t(self, t: float) -> None:
        """Decode the frame at `t`, swap the Image trace + playhead atomically.

        The playhead shape is mutated **in place** via direct
        attribute writes on `fig.layout.shapes[playhead_index]`
        inside a `batch_update()`. Plotly 6.x propagates those
        writes to the FigureWidget without rebuilding the full
        shapes list — the old `shapes = list(...) → mutate →
        reassign` pattern triggered a full layout recompute
        (O(num_shapes) per tick), which dominated Play latency
        once the duplicate-decode bug was fixed. Surgical writes
        measured ~7x faster on a 40-shape figure (403ms → 58ms
        over 100 iterations).
        """
        frame = self._decode_frame(t)
        self._t = t
        with self._fig.batch_update():
            # JPEG data URI rather than raw `z=` array: keeps the
            # per-tick payload to ~1.8MB instead of ~32MB on a 1080p
            # frame, which is the dominant Play-latency cost over the
            # Jupyter Comm channel.
            self._fig.data[0].source = _encode_frame_jpeg(
                frame,
                quality=self._frame_quality,
                max_dim=self._frame_max_dim,
            )
            playhead = self._fig.layout.shapes[self._playhead_index]
            playhead.x0 = t
            playhead.x1 = t

    def _decode_frame(self, t: float) -> np.ndarray:
        """Decode a single video frame at `t` (seconds) → uint8 RGB array.

        Delegates straight to `seq.video.decode_images_from_timestamps`.
        Tests substitute a fake reader so this never touches real
        ffmpeg / mp4 decode.
        """
        t_us = np.array([int(round(t * 1_000_000))], dtype=np.int64)
        images, _ = self._sequence.video.decode_images_from_timestamps(t_us)
        if images is None or len(images) == 0:
            raise RuntimeError(
                f"video.decode_images_from_timestamps returned no frames for t={t}"
            )
        frame = np.asarray(images[0])
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        return frame


__all__ = ["ClipPlayer"]
