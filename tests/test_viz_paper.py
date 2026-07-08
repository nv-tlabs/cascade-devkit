# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused contract tests for the publication-oriented figure renderer."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import plotly.graph_objects as go
import pytest
from PIL import Image

from cascade_av.dataset import Sequence
from cascade_av.spec import (
    AnnotationBundle,
    EgoAction,
    EgoVehicle,
    SilAvAnnotation,
    VideoMeta,
)
from cascade_av.viz import render_paper_figure
from tests.conftest import FakeVideoReader


class _SolidFrameVideoReader(FakeVideoReader):
    """Fake whose full-frame color survives lossy JPEG transport."""

    def decode_images_from_timestamps(
        self, t_us: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(np.asarray(t_us).copy())
        frames = np.zeros(
            (len(t_us), self.height, self.width, 3), dtype=np.uint8
        )
        for index, timestamp in enumerate(t_us):
            frames[index, :, :, 0] = (int(timestamp) // 1_000_000) * 60
        return frames, np.asarray(t_us, dtype=np.int64)


def _sequence_with_video() -> tuple[Sequence, FakeVideoReader]:
    bundle = AnnotationBundle(
        video=VideoMeta(clip_id="paper", duration_s=3.0),
        annotation=SilAvAnnotation(
            ego_vehicle=EgoVehicle(
                actions=[
                    EgoAction(
                        id="ego_action",
                        type="oxd:Continue",
                        start_timestamp="0:0.25",
                        end_timestamp="0:2.75",
                    )
                ]
            )
        ),
    )
    seq = Sequence.from_annotation(bundle)
    reader = FakeVideoReader(
        height=8,
        width=12,
        timestamps=np.array([0, 1_000_000, 2_000_000, 3_000_000]),
    )
    seq._cameras = {seq.annotation_camera: reader}  # type: ignore[assignment]
    return seq, reader


def _decode_image_source(source: str) -> Image.Image:
    _header, payload = source.split(",", 1)
    with Image.open(BytesIO(base64.b64decode(payload))) as image:
        return image.convert("RGB").copy()


def test_zero_timestamps_returns_timeline_without_touching_video() -> None:
    """The timeline-only form works for an annotation-only Sequence."""
    bundle = AnnotationBundle(
        video=VideoMeta(clip_id="paper-no-video", duration_s=3.0),
        annotation=SilAvAnnotation(
            ego_vehicle=EgoVehicle(
                actions=[
                    EgoAction(
                        id="ego_action",
                        type="oxd:Continue",
                        start_timestamp="0:0.25",
                        end_timestamp="0:2.75",
                    )
                ]
            )
        ),
    )

    fig = render_paper_figure(Sequence.from_annotation(bundle), timestamps=())

    assert type(fig) is go.Figure
    assert tuple(fig.layout.images) == ()
    assert any((shape.name or "").startswith("segment:") for shape in fig.layout.shapes)


@pytest.mark.parametrize(
    "timestamps",
    [
        pytest.param((1.0,), id="one"),
        pytest.param((1.0, 2.0), id="two"),
        pytest.param((1.0, 2.0, 3.0), id="three"),
    ],
)
def test_one_to_three_timestamps_are_decoded_in_one_batch(
    timestamps: tuple[float, ...],
) -> None:
    seq, reader = _sequence_with_video()

    fig = render_paper_figure(seq, timestamps=timestamps)

    assert type(fig) is go.Figure
    assert len(reader.calls) == 1
    np.testing.assert_array_equal(
        reader.calls[0],
        np.asarray(timestamps, dtype=float) * 1_000_000,
    )
    assert len(fig.layout.images) == len(timestamps)


def test_timestamp_order_is_preserved_left_to_right() -> None:
    seq, _reader = _sequence_with_video()
    reader = _SolidFrameVideoReader(
        height=8,
        width=12,
        timestamps=np.array([0, 1_000_000, 2_000_000, 3_000_000]),
    )
    seq._cameras = {seq.annotation_camera: reader}  # type: ignore[assignment]
    requested = (2.0, 1.0, 3.0)

    fig = render_paper_figure(seq, timestamps=requested)

    assert len(reader.calls) == 1
    np.testing.assert_array_equal(reader.calls[0], np.array([2_000_000, 1_000_000, 3_000_000]))
    images = list(fig.layout.images)
    assert [image.name for image in images] == ["frame-0", "frame-1", "frame-2"]
    assert [float(image.x) for image in images] == sorted(float(image.x) for image in images)
    assert all(
        str(image.source).startswith("data:image/jpeg;base64,")
        for image in images
    )
    transported_red = [
        _decode_image_source(str(image.source)).getpixel((0, 0))[0]
        for image in images
    ]
    assert transported_red == pytest.approx([120, 60, 180], abs=3)
    captions = [
        annotation.text
        for annotation in fig.layout.annotations
        if annotation.xref == "paper" and annotation.yref == "paper"
    ]
    assert captions == ["t = 2 s", "t = 1 s", "t = 3 s"]


def test_embedded_frame_long_edge_is_capped_at_1920_pixels() -> None:
    seq, _reader = _sequence_with_video()
    reader = _SolidFrameVideoReader(
        height=2000,
        width=3000,
        timestamps=np.array([0, 1_000_000, 2_000_000, 3_000_000]),
    )
    seq._cameras = {seq.annotation_camera: reader}  # type: ignore[assignment]

    fig = render_paper_figure(seq, timestamps=(1.0,))

    transported = _decode_image_source(str(fig.layout.images[0].source))
    assert transported.size == (1920, 1280)


def test_frames_are_positioned_above_the_timeline() -> None:
    seq, _reader = _sequence_with_video()

    fig = render_paper_figure(seq, timestamps=(0.5, 1.5, 2.5))

    timeline_top = float(fig.layout.yaxis.domain[1])
    images = list(fig.layout.images)
    assert images
    for image in images:
        assert image.xref == "paper"
        assert image.yref == "paper"
        image_bottom = float(image.y) - float(image.sizey)
        assert image_bottom > timeline_top


@pytest.mark.parametrize(
    ("timestamps", "exception"),
    [
        pytest.param((0.0, 1.0, 2.0, 3.0), ValueError, id="four"),
        pytest.param((float("nan"),), ValueError, id="nan"),
        pytest.param((float("inf"),), ValueError, id="positive-infinity"),
        pytest.param((float("-inf"),), ValueError, id="negative-infinity"),
        pytest.param((True,), TypeError, id="boolean"),
        pytest.param(("1.0",), TypeError, id="string"),
        pytest.param((-0.001,), ValueError, id="before-video"),
        pytest.param((3.001,), ValueError, id="after-video"),
    ],
)
def test_invalid_timestamp_requests_fail_before_decode(
    timestamps: tuple[object, ...],
    exception: type[Exception],
) -> None:
    seq, reader = _sequence_with_video()

    with pytest.raises(exception):
        render_paper_figure(seq, timestamps=timestamps)  # type: ignore[arg-type]

    assert reader.calls == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param(
            {"families": ["not-a-family"]},
            "unknown family leaves",
            id="invalid-family",
        ),
        pytest.param(
            {"track_visibility": {"spaceship": False}},
            "unknown track_visibility kinds",
            id="unknown-kind",
        ),
        pytest.param(
            {"track_visibility": {"agent": {"missing-agent": False}}},
            "unknown 'agent' entity IDs",
            id="unknown-entity-id",
        ),
    ],
)
def test_invalid_timeline_filters_fail_before_video_decode(
    kwargs: dict[str, object],
    message: str,
) -> None:
    seq, reader = _sequence_with_video()

    with pytest.raises(ValueError, match=message):
        render_paper_figure(
            seq,
            timestamps=(1.0,),
            **kwargs,  # type: ignore[arg-type]
        )

    assert reader.calls == []


@pytest.mark.parametrize(
    ("dimension", "value", "exception"),
    [
        pytest.param("width", True, TypeError, id="width-boolean"),
        pytest.param("height", False, TypeError, id="height-boolean"),
        pytest.param("width", 10.5, TypeError, id="width-non-integral"),
        pytest.param("height", 10.5, TypeError, id="height-non-integral"),
        pytest.param("width", float("nan"), TypeError, id="width-nan"),
        pytest.param("height", float("nan"), TypeError, id="height-nan"),
        pytest.param("width", 9, ValueError, id="width-below-plotly-floor"),
        pytest.param("height", 9, ValueError, id="height-below-plotly-floor"),
    ],
)
def test_invalid_dimensions_fail_before_video_decode(
    dimension: str,
    value: object,
    exception: type[Exception],
) -> None:
    seq, reader = _sequence_with_video()

    with pytest.raises(exception, match=dimension):
        render_paper_figure(
            seq,
            timestamps=(1.0,),
            **{dimension: value},  # type: ignore[arg-type]
        )

    assert reader.calls == []
