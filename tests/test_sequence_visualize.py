# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatch-matrix coverage for ``Sequence.visualize``.

The placeholder dict-shape return was replaced in PR-6 by a polymorphic
dispatcher into ``cascade_av.viz``. The dispatch matrix is:

    seq.visualize()                 -> ClipPlayer over the whole clip
    seq.visualize(t=2.5)            -> ClipPlayer ±1s around 2.5
    seq.visualize(t=2.5, static=True) -> PIL.Image
    seq.visualize(t=(2, 5))         -> ClipPlayer over [2, 5]
    seq.visualize(match=m)          -> ClipPlayer over m.interval ± pad
    seq.visualize(context=cw)       -> ClipPlayer over cw.interval

Plus the error paths (mutually-exclusive args, static without a scalar
`t`, mismatched clip_id, …). No real video decode — we lean on the
shared ``FakeVideoReader`` fixture from ``tests/conftest.py``.
"""

from __future__ import annotations

import sys

import numpy as np
import plotly.graph_objects as go
import pytest
from PIL import Image

from cascade_av.dataset import Sequence
from cascade_av.query.context import context_at
from cascade_av.query.engine import Match
from cascade_av.query.time import Interval
from cascade_av.spec import AnnotationBundle, SilAvAnnotation, VideoMeta
from cascade_av.viz import ClipPlayer
from tests.conftest import FakeVideoReader


class _FakeVideoReaderWithTimestamps(FakeVideoReader):
    """`FakeVideoReader` extended with a public `timestamps` (np.int64 μs).

    The real `physical_ai_av.video.SeekVideoReader` exposes a numpy
    int64 microsecond array as `.timestamps`. The base fake omits it
    because the original widget tests don't need clamping, but the
    Bug-A fix (`dataset.py`) reads this attribute. Subclass-here keeps
    the original tests unchanged while letting the new tests cover the
    non-zero-start path.
    """

    def __init__(
        self,
        timestamps: np.ndarray,
        height: int = 8,
        width: int = 12,
    ) -> None:
        super().__init__(height=height, width=width)
        self.timestamps = np.asarray(timestamps, dtype=np.int64)


# ---------------------------------------------------------------------------
# Fixtures — minimal Sequence with a fake video reader injected.
# ---------------------------------------------------------------------------


def _make_bundle(duration: float = 10.0, clip_id: str = "vis_clip") -> AnnotationBundle:
    """Smallest valid bundle — empty annotation, fixed duration."""
    return AnnotationBundle(
        video=VideoMeta(clip_id=clip_id, duration_s=duration),
        annotation=SilAvAnnotation(),
    )


def _seq_with_fake_video(
    duration: float = 10.0, clip_id: str = "vis_clip"
) -> Sequence:
    """Sequence whose `.video` returns the shared `FakeVideoReader`.

    Mirrors the `_seq_with_fake_video` helper in `test_viz_widget.py`:
    `Sequence.video` is a property that hits the parent dataset, so we
    inject the fake reader through the `_cameras` slot instead of
    assigning to the read-only property.
    """
    bundle = _make_bundle(duration=duration, clip_id=clip_id)
    seq = Sequence.from_annotation(bundle)
    # ``Sequence.visualize`` clamps the requested ``[t_start, t_end]``
    # window to the video's actual timestamp coverage. Match the fake
    # reader's timestamps to the bundle duration so the clamp is a
    # no-op and assertions like ``_t_end == 10.0`` still hold.
    fake_ts = np.arange(0, int(duration * 1_000_000) + 1, 1_000, dtype=np.int64)
    seq._cameras = {  # type: ignore[assignment]
        seq.annotation_camera: FakeVideoReader(timestamps=fake_ts),
    }
    return seq


# ---------------------------------------------------------------------------
# 1. No-args → ClipPlayer over the whole clip.
# ---------------------------------------------------------------------------


def test_no_args_returns_clipplayer_over_full_clip() -> None:
    seq = _seq_with_fake_video(duration=10.0)
    player = seq.visualize()
    assert isinstance(player, ClipPlayer)
    assert player._t_start == 0.0
    assert player._t_end == 10.0


# ---------------------------------------------------------------------------
# 2. Scalar `t` → ClipPlayer with a ±1s window.
# ---------------------------------------------------------------------------


def test_scalar_t_centers_clipplayer_with_one_second_padding() -> None:
    seq = _seq_with_fake_video(duration=10.0)
    player = seq.visualize(t=2.5)
    assert isinstance(player, ClipPlayer)
    assert player._t_start == pytest.approx(1.5)
    assert player._t_end == pytest.approx(3.5)


def test_scalar_t_clamps_window_to_clip_bounds() -> None:
    seq = _seq_with_fake_video(duration=10.0)
    # Near the start: window's lower edge clamps to 0.
    player = seq.visualize(t=0.4)
    assert player._t_start == 0.0
    assert player._t_end == pytest.approx(1.4)
    # Near the end: window's upper edge clamps to duration.
    player = seq.visualize(t=9.7)
    assert player._t_start == pytest.approx(8.7)
    assert player._t_end == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 3. Scalar `t` + static=True → PIL.Image.
# ---------------------------------------------------------------------------


def test_scalar_t_with_static_returns_pil_image() -> None:
    seq = _seq_with_fake_video(duration=10.0)
    out = seq.visualize(t=2.5, static=True)
    assert isinstance(out, Image.Image)


# ---------------------------------------------------------------------------
# 4. Tuple `t` → ClipPlayer over the explicit window.
# ---------------------------------------------------------------------------


def test_tuple_t_uses_explicit_window() -> None:
    seq = _seq_with_fake_video(duration=10.0)
    player = seq.visualize(t=(2.0, 5.0))
    assert isinstance(player, ClipPlayer)
    assert player._t_start == 2.0
    assert player._t_end == 5.0


# ---------------------------------------------------------------------------
# 5. match=m → ClipPlayer over m.interval ± pad, highlight = raw interval.
# ---------------------------------------------------------------------------


def test_match_widens_to_pad_and_paints_raw_highlight() -> None:
    seq = _seq_with_fake_video(duration=10.0)
    iv = Interval(3.0, 5.0)
    m = Match(clip_id=seq.clip_id, entity=object(), interval=iv)
    player = seq.visualize(match=m, pad=1.5)
    assert isinstance(player, ClipPlayer)
    # Padded window: [3 - 1.5, 5 + 1.5] = [1.5, 6.5].
    assert player._t_start == pytest.approx(1.5)
    assert player._t_end == pytest.approx(6.5)
    # The yellow highlight band tracks the *raw* interval (not the
    # padded window) — matches the carousel's behavior in PR-5.
    bands = [
        dict(s.to_plotly_json())
        for s in player._fig.layout.shapes
        if dict(s.to_plotly_json()).get("name") == "highlight"
    ]
    assert len(bands) == 1
    assert float(bands[0]["x0"]) == pytest.approx(3.0)
    assert float(bands[0]["x1"]) == pytest.approx(5.0)


def test_match_pad_clamps_to_clip_bounds() -> None:
    seq = _seq_with_fake_video(duration=10.0)
    # Match abuts t=0; the lower clamp should kick in.
    m = Match(clip_id=seq.clip_id, entity=object(), interval=Interval(0.5, 1.0))
    player = seq.visualize(match=m, pad=2.0)
    assert player._t_start == 0.0
    assert player._t_end == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 6. context=cw → ClipPlayer over cw.interval.
# ---------------------------------------------------------------------------


def test_context_window_drives_playback_window() -> None:
    seq = _seq_with_fake_video(duration=10.0)
    cw = context_at(seq.annotation, Interval(2.0, 5.0))
    player = seq.visualize(context=cw)
    assert isinstance(player, ClipPlayer)
    assert player._t_start == 2.0
    assert player._t_end == 5.0


# ---------------------------------------------------------------------------
# 7. Dispatcher errors — mutual exclusion, static gating, clip_id mismatch.
# ---------------------------------------------------------------------------


def test_mixing_t_and_match_raises_value_error() -> None:
    seq = _seq_with_fake_video()
    m = Match(clip_id=seq.clip_id, entity=object(), interval=Interval(2.0, 4.0))
    with pytest.raises(ValueError, match="at most one"):
        seq.visualize(t=2.5, match=m)


def test_mixing_match_and_context_raises_value_error() -> None:
    seq = _seq_with_fake_video()
    m = Match(clip_id=seq.clip_id, entity=object(), interval=Interval(2.0, 4.0))
    cw = context_at(seq.annotation, Interval(2.0, 5.0))
    with pytest.raises(ValueError, match="at most one"):
        seq.visualize(match=m, context=cw)


def test_static_without_scalar_t_raises_value_error() -> None:
    seq = _seq_with_fake_video()
    with pytest.raises(ValueError, match="scalar"):
        seq.visualize(static=True)
    with pytest.raises(ValueError, match="scalar"):
        seq.visualize(t=(2.0, 5.0), static=True)


def test_match_with_mismatched_clip_id_raises_value_error() -> None:
    seq = _seq_with_fake_video(clip_id="vis_clip")
    m = Match(clip_id="some_other_clip", entity=object(), interval=Interval(2.0, 4.0))
    with pytest.raises(ValueError, match="different clip"):
        seq.visualize(match=m)


def test_context_with_mismatched_clip_id_raises_value_error() -> None:
    seq = _seq_with_fake_video(clip_id="vis_clip")
    # Build a context window against a different clip's annotation.
    other_bundle = _make_bundle(clip_id="some_other_clip")
    cw = context_at(other_bundle, Interval(2.0, 5.0))
    with pytest.raises(ValueError, match="different clip"):
        seq.visualize(context=cw)


# ---------------------------------------------------------------------------
# 8. Publication-figure mode.
# ---------------------------------------------------------------------------


def test_paper_figure_mode_returns_plain_figure_and_batches_timestamps() -> None:
    seq = _seq_with_fake_video(duration=10.0)
    reader = seq.video

    fig = seq.visualize(
        mode="paper_figure",
        timestamps=[2.0, 1.0, 3.0],
    )

    assert type(fig) is go.Figure
    assert len(reader.calls) == 1
    np.testing.assert_array_equal(
        reader.calls[0], np.array([2_000_000, 1_000_000, 3_000_000])
    )
    assert len(fig.layout.images) == 3


def test_paper_figure_mode_rejects_incompatible_arguments_before_decode() -> None:
    seq = _seq_with_fake_video(duration=10.0)
    reader = seq.video
    match = Match(
        clip_id=seq.clip_id,
        entity=object(),
        interval=Interval(2.0, 4.0),
    )
    context = context_at(seq.annotation, Interval(2.0, 4.0))

    incompatible = [
        {"mode": "paper_figure", "timestamps": [2.0], "t": 2.0},
        {"mode": "paper_figure", "timestamps": [2.0], "match": match},
        {"mode": "paper_figure", "timestamps": [2.0], "context": context},
        {"mode": "paper_figure", "timestamps": [2.0], "static": True},
        {"timestamps": [2.0]},
        {"mode": "not-a-mode"},
    ]
    for kwargs in incompatible:
        with pytest.raises(ValueError):
            seq.visualize(**kwargs)  # type: ignore[arg-type]

    assert reader.calls == []


# ---------------------------------------------------------------------------
# 9. Video timestamp clamping — Bug A. Real clips' video manifests rarely
# start at t=0us; `seq.visualize()` must clamp `[t_start, t_end]` to
# `seq.video.timestamps.min()/max()` so the eager first-frame decode in
# `ClipPlayer.__init__` doesn't trip `decode_images_from_timestamps([0])`.
# ---------------------------------------------------------------------------


def test_visualize_clamps_to_video_timestamp_range() -> None:
    """No-args path: clamp `(t_start, t_end)` to the video's covered span.

    Mirrors the real-corpus reproducer in `meta/viz-e2e/findings.md`:
    clip `004c2001-...` has `timestamps=[3464, 20103730]μs`. Without
    clamping the ClipPlayer constructor eagerly decoded `t=0.0` and
    raised `ValueError` from the parent dataset's video reader.
    """
    bundle = _make_bundle(duration=20.13)
    seq = Sequence.from_annotation(bundle)
    seq._cameras = {  # type: ignore[assignment]
        seq.annotation_camera: _FakeVideoReaderWithTimestamps(
            timestamps=np.array([3464, 1_000_000, 20_103_730], dtype=np.int64),
        )
    }

    player = seq.visualize()
    assert isinstance(player, ClipPlayer)
    # `t_start` must move forward from 0.0 to the first available μs.
    assert player._t_start == pytest.approx(0.003464)
    # `t_end` must move back from `duration_s` (20.13) to the last μs.
    assert player._t_end == pytest.approx(20.103730)


def test_visualize_raises_when_window_outside_video_coverage() -> None:
    """Tuple `t` entirely before the video starts → clear ValueError.

    A silent snap to a 1-frame sliver would hide the misconfiguration;
    surface it as a `ValueError` instead.
    """
    bundle = _make_bundle(duration=30.0)
    seq = Sequence.from_annotation(bundle)
    seq._cameras = {  # type: ignore[assignment]
        seq.annotation_camera: _FakeVideoReaderWithTimestamps(
            timestamps=np.array([10_000_000, 20_000_000], dtype=np.int64),
        )
    }

    with pytest.raises(ValueError, match="outside the video's timestamp coverage"):
        seq.visualize(t=(0.0, 5.0))


# ---------------------------------------------------------------------------
# 10. Import hygiene — `from cascade_av.dataset import Sequence` must NOT
# pull in Pillow / Plotly / ipywidgets. The viz extras are optional and the
# dispatcher lazy-imports them inside `.visualize()`.
# ---------------------------------------------------------------------------


def test_importing_sequence_does_not_eager_load_viz_extras() -> None:
    """Run a fresh `python -c` so prior tests' imports don't pollute
    `sys.modules`. We import only `Sequence`, then assert that none of
    the viz-extra packages were transitively pulled in.
    """
    import subprocess

    script = (
        "import sys\n"
        "from cascade_av.dataset import Sequence  # noqa: F401\n"
        "banned = ['PIL', 'plotly', 'ipywidgets']\n"
        "leaked = [m for m in banned if m in sys.modules]\n"
        "print('LEAKED=' + ','.join(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    # The subprocess prints `LEAKED=` with a comma-separated list. We
    # assert the list is empty — `Sequence` must be importable without
    # the viz extras.
    last = result.stdout.strip().splitlines()[-1]
    assert last == "LEAKED=", (
        f"importing Sequence eagerly loaded a viz-extra module: "
        f"stdout={result.stdout!r}, stderr={result.stderr!r}"
    )
