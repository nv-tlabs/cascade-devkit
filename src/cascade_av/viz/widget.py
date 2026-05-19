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

from cascade_av.viz.timeline import (
    _PX_PER_LANE,
    _paint_timeline_onto,
    _timeline_px_for,
)

if TYPE_CHECKING:  # pragma: no cover — typing only
    import plotly.graph_objects as go

    from cascade_av.dataset import Sequence


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

# Fallback pixel budget for the video pane when the source frame's
# aspect ratio can't be measured (zero-byte frame, decode failure,
# pre-PR-44 unit tests stubbing _decode_frame to return something
# pathological). Real frames drive video_px via
# `_aspect_fit_video_px` so the figure height tracks the source
# resolution and no letterboxing creeps in.
_VIDEO_PX = 480

# Layout-image paper-x extents for the video pane. Negative left
# edge so the box reaches into the y-tick-label margin (matching
# the entity-block bands in the timeline at `paper-x=[-0.18, 1.0]`).
# Right edge sits at the plot right edge. `sizing="contain"` on
# the layout image preserves the source aspect inside the box.
_VIDEO_PAPER_X_LEFT: float = -0.18
_VIDEO_PAPER_X_RIGHT: float = 1.0

# Assumed Jupyter cell width in pixels, used to size the video
# pane's height when aspect-fitting AND to pin `fig.layout.width`
# as a default. `autosize=True` + `ipywidgets.Layout(width="100%")`
# alone don't reliably grow a FigureWidget past Plotly's ~700px
# first-render default; an explicit `width` is required. 1400 is
# a useful default for modern wide-screen Jupyter sessions while
# still fitting a 1080p source's aspect comfortably. Users on
# narrower cells can override via the `width=` kwarg on
# `ClipPlayer.__init__`.
_DEFAULT_WIDTH: int = 1400
_ASSUMED_CELL_WIDTH_PX: int = _DEFAULT_WIDTH

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
        families: forwarded to `_paint_timeline_onto`; family-leaf
            whitelist. See `render_timeline` for the recognized leaf
            names. Parent entity headers auto-render for entities
            whose sub-rows survive — orphan headers are dropped.
            `None` = all families.
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
        families: list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
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
        # known fake array. Use the initial frame's resolution to
        # compute the source aspect ratio — drives the layout image's
        # paper-y extent so we avoid the letterboxing-inside-subplot
        # that the old `go.Image` trace approach produced.
        initial_frame = self._decode_frame(self._t)
        src_h, src_w = initial_frame.shape[:2]
        self._video_aspect = float(src_w) / float(src_h) if src_h > 0 else 16.0 / 9.0

        # Convert to a FigureWidget *before* painting the timeline so
        # the playhead-line shape can be added through the same
        # `update_layout` path on a live widget. The top subplot's
        # axes are hidden — the video lives at the figure level as a
        # `layout.images[0]` entry positioned in paper coords (so it
        # can reach into the y-tick-label margin AND fill the figure
        # width without being clipped to the subplot's `xaxis.domain`,
        # which Plotly hard-clamps to [0, 1]).
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
            # Autosize horizontally so the widget fills the Jupyter
            # cell rather than defaulting to Plotly's fixed ~700px.
            # Total height stays pinned (set below) so the lane-band
            # readability survives.
            autosize=True,
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
            families=families,
            show_inline_labels=show_inline_labels,
        )

        # Adaptive height: video_px is derived from the source aspect
        # at a target box width (the layout-image's paper-x extent
        # times an assumed plot pixel width). Eliminates the
        # letterboxing-inside-subplot that the old fixed _VIDEO_PX
        # path produced — the video pane is now sized exactly to fit
        # the source aspect, so there's no internal padding between
        # the video bottom edge and the timeline.
        #
        # Callers can pass `height=N` to pin a total value — we then
        # keep the timeline at its computed pixel share and let the
        # video pane absorb the rest, preserving the lane-band
        # readability the whole adaptive path is about.
        timeline_px = _timeline_px_for(paint)
        if height is not None:
            total_height = int(height)
            video_px = max(
                _PX_PER_LANE,  # avoid collapsing the video pane to zero
                total_height - timeline_px - _CHROME_PX,
            )
        else:
            video_px = self._aspect_fit_video_px()
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
        video_domain_bottom = timeline_domain_top + _VERTICAL_SPACING
        # Pin a default width — `autosize=True` + `Layout(width="100%")`
        # alone don't reliably expand a `FigureWidget` in ipywidgets
        # (the widget's first-render pixel width tends to stick at
        # Plotly's ~700px default regardless of container width).
        # Setting `fig.layout.width` explicitly forces a useful render
        # size. Users with smaller cells can override via `width=`.
        resolved_width = int(width) if width is not None else _DEFAULT_WIDTH
        fig.update_layout(
            height=total_height,
            width=resolved_width,
            yaxis={
                "showticklabels": False,
                "showgrid": False,
                "zeroline": False,
                "visible": False,  # hide axes — video lives at layout level
                "domain": [video_domain_bottom, 1.0],
            },
            yaxis2={"domain": [0.0, timeline_domain_top]},
        )

        # Attach the video as a layout-level image. Paper xref means
        # we can put the box's left edge at `x=-0.18` (reaching into
        # the y-tick-label margin) and the right edge at `x=1.0` so
        # the box spans labels + bars + chrome.
        #
        # `sizing="stretch"` fills the box exactly — no internal
        # letterboxing. The PR-44 `sizing="contain"` attempt
        # preserved aspect by adding letterboxes inside the box,
        # which recreated the very gap between video bottom and
        # timeline top that the layout-image refactor was meant to
        # kill (at any cell width != the assumed 1000px, the box
        # aspect didn't match the source 16:9, so vertical letterbox
        # appeared). Stretching distorts the video ~18% horizontally
        # for a 16:9 source in a 1.18-paper-wide box — acceptable
        # trade-off for a tight visual stack with no gap.
        fig.update_layout(
            images=[
                {
                    "source": _encode_frame_jpeg(
                        initial_frame,
                        quality=self._frame_quality,
                        max_dim=self._frame_max_dim,
                    ),
                    "xref": "paper",
                    "yref": "paper",
                    "x": _VIDEO_PAPER_X_LEFT,
                    "y": 1.0,
                    "sizex": _VIDEO_PAPER_X_RIGHT - _VIDEO_PAPER_X_LEFT,
                    "sizey": 1.0 - video_domain_bottom,
                    "xanchor": "left",
                    "yanchor": "top",
                    "sizing": "stretch",
                    "layer": "above",
                    "name": "frame",
                }
            ]
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
        # Fill the Jupyter cell horizontally. Without an explicit
        # `width="100%"` on the VBox / HBox, ipywidgets sizes each
        # child to its preferred fit (Plotly's default ~700px),
        # leaving the cell mostly empty. Paired with the
        # `fig.layout.autosize=True` setting above, the figure now
        # tracks the container width.
        full_width = ipywidgets.Layout(width="100%")
        self._controls = ipywidgets.HBox(
            [self._play, self._slider], layout=full_width
        )
        self._container = ipywidgets.VBox(
            [self._fig, self._controls], layout=full_width
        )

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

    def _aspect_fit_video_px(self) -> int:
        """Pixel height of the video pane derived from the source
        aspect ratio and the layout image's paper-x extent.

        The layout image occupies paper-x =
        `[_VIDEO_PAPER_X_LEFT, _VIDEO_PAPER_X_RIGHT]` which equals
        ~1.18 plot widths. At an assumed Jupyter cell width of
        `_ASSUMED_CELL_WIDTH_PX`, that translates to a target box
        width in pixels; dividing by the source aspect gives the
        height in pixels needed to fit the source without
        letterboxing inside the box.

        Real cell widths vary — the assumed width is a target for the
        pixel-height math, so the result is "approximately right" at
        the typical cell width but degrades smoothly at narrower or
        wider widths (small letterboxes appear, but never the giant
        gap the old fixed-`_VIDEO_PX = 480` path produced).
        """
        plot_w_px = _ASSUMED_CELL_WIDTH_PX - 80 - 20  # margins
        box_w_paper = _VIDEO_PAPER_X_RIGHT - _VIDEO_PAPER_X_LEFT
        box_w_px = plot_w_px * box_w_paper
        aspect = self._video_aspect or (16.0 / 9.0)
        return max(_PX_PER_LANE, int(round(box_w_px / aspect)))

    def _apply_t(self, t: float) -> None:
        """Decode the frame at `t`, swap the layout image + playhead atomically.

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

        Since PR-44 the video frame lives at
        `layout.images[0].source` (was `data[0].source`). The
        per-tick update target moved, but PR-33's JPEG transport
        is otherwise untouched — encoding still happens once per
        tick into a base64 data URI.
        """
        frame = self._decode_frame(t)
        self._t = t
        with self._fig.batch_update():
            # JPEG data URI rather than raw `z=` array: keeps the
            # per-tick payload to ~1.8MB instead of ~32MB on a 1080p
            # frame, which is the dominant Play-latency cost over the
            # Jupyter Comm channel.
            self._fig.layout.images[0].source = _encode_frame_jpeg(
                frame,
                quality=self._frame_quality,
                max_dim=self._frame_max_dim,
            )
            playhead = self._fig.layout.shapes[self._playhead_index]
            playhead.x0 = t
            playhead.x1 = t

    def _decode_frame(self, t: float) -> np.ndarray:
        """Decode a single video frame at `t` (seconds) → uint8 RGB array.

        Delegates to `seq.video.decode_images_from_timestamps`. Tests
        substitute a fake reader so this never touches real ffmpeg /
        mp4 decode.

        The requested timestamp is clamped to the reader's actual frame
        timestamp range. The first frame is rarely at exactly `0us` —
        HEVC GOP boundaries and variable-frame-rate captures leave the
        first frame ~16–33ms in for typical 30/60fps footage — and the
        last frame can sit just before the dataset's reported duration.
        `decode_images_from_timestamps` raises if the request falls
        outside that range, which crashed `matches.visualize(...)` for
        any whole-clip match (`interval=None`, computed as
        `t_start=0.0`). Clamping shows the nearest available frame
        instead.
        """
        timestamps = self._sequence.video.timestamps
        t_us_int = int(round(t * 1_000_000))
        t_us_int = max(t_us_int, int(timestamps[0]))
        t_us_int = min(t_us_int, int(timestamps[-1]))
        t_us = np.array([t_us_int], dtype=np.int64)
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
