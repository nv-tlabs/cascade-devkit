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

from typing import TYPE_CHECKING, Any

import numpy as np

from causal_ai_av.viz.timeline import _paint_timeline_onto

if TYPE_CHECKING:  # pragma: no cover — typing only
    import plotly.graph_objects as go

    from causal_ai_av.dataset import Sequence


# Layout constants. Top subplot is the video frame (taller); bottom is
# the timeline (shorter). The split mirrors the annotator's two-pane
# layout but flipped — the annotator stacks video on top of timeline
# inside the same browser tab, and we follow that reading order here.
_ROW_HEIGHTS: tuple[float, float] = (0.7, 0.3)
_VERTICAL_SPACING = 0.05

# Default scrub rate. The spec calls 8 fps "scrub-not-playback" and
# uses it as the smoke-test bar in `meta/10_visualization_api_plan.md`.
_DEFAULT_FPS = 8.0


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
        `decode_images_from_timestamps([t_us])` call. On 1080p MP4
        clips this comfortably hits the spec's 8 fps scrub bar; if a
        future PR wants smoother playback the place to add a
        prefetch buffer is here.
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

        # ------------------------------------------------------------
        # 1. Build the underlying FigureWidget — two stacked subplots:
        #    row 1 video frame, row 2 timeline.
        # ------------------------------------------------------------
        base = make_subplots(
            rows=2,
            cols=1,
            row_heights=list(_ROW_HEIGHTS),
            vertical_spacing=_VERTICAL_SPACING,
            shared_xaxes=False,
        )

        # Initial frame at t_start. We decode through the sequence's
        # video reader; tests substitute a fake reader that returns a
        # known fake array.
        initial_frame = self._decode_frame(self._t)
        base.add_trace(go.Image(z=initial_frame, name="frame"), row=1, col=1)

        # Convert to a FigureWidget *before* painting the timeline so
        # the playhead-line shape can be added through the same
        # `update_layout` path on a live widget.
        fig = go.FigureWidget(base)

        # Hide the image axes ticks — the video frame doesn't have
        # meaningful axis units; we only want the picture.
        fig.update_layout(
            template="plotly_dark",
            height=600,
            margin={"l": 80, "r": 20, "t": 20, "b": 40},
            showlegend=False,
            xaxis={"showticklabels": False, "showgrid": False, "zeroline": False},
            yaxis={"showticklabels": False, "showgrid": False, "zeroline": False},
        )

        # Paint the timeline onto the second subplot (x2 / y2).
        _paint_timeline_onto(
            fig,
            sequence,
            xref="x2",
            yref="y2",
            xaxis_key="xaxis2",
            yaxis_key="yaxis2",
            highlight=highlight,
            arrows=arrows,
        )

        # ------------------------------------------------------------
        # 2. Playhead — a vertical line on the bottom subplot's axes.
        #    Appended after the timeline shapes so it sits on top.
        # ------------------------------------------------------------
        from causal_ai_av.viz.timeline import _TRACK_GROUPS

        playhead = {
            "type": "line",
            "x0": self._t,
            "x1": self._t,
            "y0": -0.5,
            "y1": len(_TRACK_GROUPS) - 0.5,
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
        """Play tick — mirror to the slider, which drives the figure."""
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
        self._apply_t(new_t)

    def _apply_t(self, t: float) -> None:
        """Decode the frame at `t`, swap the Image trace + playhead atomically."""
        frame = self._decode_frame(t)
        self._t = t
        with self._fig.batch_update():
            self._fig.data[0].z = frame
            # Move the playhead by replacing the shapes list. We
            # rebuild the list each tick because Plotly's
            # `layout.shapes` is a tuple of validators that doesn't
            # mutate well in place; the cost is O(num_shapes) per
            # tick, which is fine at scrub rates.
            shapes = list(self._fig.layout.shapes)
            ph = dict(shapes[self._playhead_index].to_plotly_json())
            ph["x0"] = t
            ph["x1"] = t
            shapes[self._playhead_index] = ph
            self._fig.layout.shapes = shapes

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
