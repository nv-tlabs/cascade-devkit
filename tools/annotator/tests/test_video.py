# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Phase-3 video route + transcode resolver.

Real `ffmpeg` / `ffprobe` are never invoked here — every subprocess call is
monkeypatched so the test suite stays fast and environment-independent.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from annotator.server.app import create_app
from annotator.server.io_adapter import build_clip_index
from annotator.server.video import (
    VideoNotFound,
    VideoResolver,
    VideoToolsMissing,
    _parse_frame_rate,
    _transcode_to_h264,
    probe_video_meta,
)


# -----------------------------------------------------------------------------
# Helpers — fake subprocess.run for ffmpeg/ffprobe
# -----------------------------------------------------------------------------

@dataclass
class _Recorded:
    cmd: list[str]


def _fake_h264_probe(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
    """ffprobe stub: report codec=h264 so no transcode is needed."""
    return subprocess.CompletedProcess(
        args=_args[0] if _args else [],
        returncode=0,
        stdout=json.dumps({"streams": [{"codec_name": "h264"}]}),
        stderr="",
    )


def _fake_hevc_probe(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=_args[0] if _args else [],
        returncode=0,
        stdout=json.dumps({"streams": [{"codec_name": "hevc"}]}),
        stderr="",
    )


def _fake_ffmpeg_run(recorded: list[_Recorded]):
    """Factory: returns a stub for subprocess.run that 'transcodes' by
    writing a fixed byte to the output and recording the invocation."""

    def _run(cmd, **kwargs):
        recorded.append(_Recorded(cmd=list(cmd)))
        # Detect ffprobe — read codec from disk if asked, default h264.
        if cmd[0] == "ffprobe":
            return _fake_h264_probe(cmd)
        # ffmpeg: write the tmp file so os.replace succeeds.
        # Find "-i <src>" and the final dst arg.
        dst = Path(cmd[-1])
        dst.write_bytes(b"transcoded")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return _run


# -----------------------------------------------------------------------------
# VideoResolver — cache hit on second call (no re-transcode)
# -----------------------------------------------------------------------------

def test_video_resolver_hits_cache_on_second_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First resolve transcodes; second resolve serves from cache."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    src = video_dir / "clipA.mp4"
    src.write_bytes(b"fake-hevc-bytes")

    cache_dir = tmp_path / "cache"

    # ffprobe → "hevc" so we hit the transcode branch.
    probe_calls = {"n": 0}
    ffmpeg_calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffprobe":
            probe_calls["n"] += 1
            return _fake_hevc_probe(cmd)
        # ffmpeg
        ffmpeg_calls["n"] += 1
        dst = Path(cmd[-1])
        dst.write_bytes(b"transcoded")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("annotator.server.video.subprocess.run", fake_run)
    monkeypatch.setattr(
        "annotator.server.video.shutil.which", lambda name: f"/usr/bin/{name}",
    )

    resolver = VideoResolver(mode="local", video_dir=video_dir, cache_dir=cache_dir)

    out1 = resolver.resolve("clipA")
    assert out1 == cache_dir / "clipA.mp4"
    assert out1.read_bytes() == b"transcoded"
    assert ffmpeg_calls["n"] == 1, "first call should run ffmpeg once"

    out2 = resolver.resolve("clipA")
    assert out2 == out1
    assert ffmpeg_calls["n"] == 1, "second call must hit cache (no re-transcode)"
    # ffprobe is allowed to run again (cheap), but ffmpeg must not.
    assert probe_calls["n"] >= 1


# -----------------------------------------------------------------------------
# VideoResolver — already-h264 source is returned verbatim
# -----------------------------------------------------------------------------

def test_video_resolver_passthrough_when_already_h264(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    src = video_dir / "h264clip.mp4"
    src.write_bytes(b"already-h264")

    def fake_run(cmd, **kwargs):
        assert cmd[0] == "ffprobe", "ffmpeg must not be invoked for h264 source"
        return _fake_h264_probe(cmd)

    monkeypatch.setattr("annotator.server.video.subprocess.run", fake_run)
    monkeypatch.setattr(
        "annotator.server.video.shutil.which", lambda name: f"/usr/bin/{name}",
    )

    resolver = VideoResolver(
        mode="local", video_dir=video_dir, cache_dir=tmp_path / "cache",
    )
    out = resolver.resolve("h264clip")
    assert out == src


# -----------------------------------------------------------------------------
# /api/clips/{clip_id}/video — 404 path
# -----------------------------------------------------------------------------

def test_video_404_on_unknown_clip(tmp_path: Path) -> None:
    video = tmp_path / "knownclip.mp4"
    video.touch()
    index = build_clip_index([tmp_path])

    resolver = VideoResolver(mode="local", video_dir=tmp_path)
    app = create_app(
        index, read_only=False, destination_dir=tmp_path, video_resolver=resolver,
    )
    client = TestClient(app)
    r = client.get("/api/clips/nonexistent/video")
    assert r.status_code == 404
    assert "nonexistent" in r.json()["detail"]


# -----------------------------------------------------------------------------
# /api/clips/{clip_id}/video — 503 when ffmpeg missing
# -----------------------------------------------------------------------------

def test_video_503_when_ffmpeg_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clipB.mp4"
    video.write_bytes(b"x")
    index = build_clip_index([tmp_path])

    # Pretend ffmpeg/ffprobe are not on PATH.
    monkeypatch.setattr("annotator.server.video.shutil.which", lambda _name: None)

    resolver = VideoResolver(mode="local", video_dir=tmp_path)
    app = create_app(
        index, read_only=False, destination_dir=tmp_path, video_resolver=resolver,
    )
    client = TestClient(app)
    r = client.get("/api/clips/clipB/video")
    assert r.status_code == 503
    body = r.json()
    assert "ffmpeg" in body["detail"] or "ffprobe" in body["detail"]


# -----------------------------------------------------------------------------
# /api/clips/{clip_id}/video — 404 when video-dir doesn't have the file
# -----------------------------------------------------------------------------

def test_video_404_when_local_file_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Build the index from a directory with one (.mp4) — but point video_dir
    # elsewhere so the resolver can't find the file.
    video = tmp_path / "indexed.mp4"
    video.touch()
    index = build_clip_index([tmp_path])

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    monkeypatch.setattr(
        "annotator.server.video.shutil.which", lambda name: f"/usr/bin/{name}",
    )

    resolver = VideoResolver(mode="local", video_dir=empty_dir)
    app = create_app(
        index, read_only=False, destination_dir=tmp_path, video_resolver=resolver,
    )
    client = TestClient(app)
    r = client.get("/api/clips/indexed/video")
    assert r.status_code == 404


# -----------------------------------------------------------------------------
# VideoResolver — local mode without --video-dir is rejected at construction
# -----------------------------------------------------------------------------

def test_video_resolver_local_without_dir_raises() -> None:
    with pytest.raises(ValueError, match="video-dir"):
        VideoResolver(mode="local", video_dir=None)


# -----------------------------------------------------------------------------
# Transcode error path → TranscodeError surfaces stderr tail
# -----------------------------------------------------------------------------

def test_transcode_error_surfaces_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from annotator.server.video import TranscodeError

    src = tmp_path / "src.mp4"
    src.touch()
    dst = tmp_path / "dst.mp4"

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            1, cmd, output="", stderr="line1\nline2\nffmpeg: panicked",
        )

    monkeypatch.setattr("annotator.server.video.subprocess.run", fake_run)
    with pytest.raises(TranscodeError) as excinfo:
        _transcode_to_h264(src, dst)
    assert "panicked" in str(excinfo.value)


# -----------------------------------------------------------------------------
# Local-mode resolver explores all supported extensions
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("ext", [".mp4", ".mkv", ".mov", ".avi"])
def test_video_resolver_local_finds_by_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ext: str,
) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    src = video_dir / f"clipX{ext}"
    src.write_bytes(b"v")

    monkeypatch.setattr(
        "annotator.server.video.shutil.which", lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.setattr(
        "annotator.server.video.subprocess.run",
        lambda cmd, **kwargs: _fake_h264_probe(cmd),
    )
    resolver = VideoResolver(mode="local", video_dir=video_dir, cache_dir=tmp_path / "c")
    out = resolver.resolve("clipX")
    assert out == src


# -----------------------------------------------------------------------------
# VideoToolsMissing / VideoNotFound propagate as the right HTTP codes
# -----------------------------------------------------------------------------

def test_video_resolver_raises_videonotfound_in_local_mode_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    monkeypatch.setattr(
        "annotator.server.video.shutil.which", lambda name: f"/usr/bin/{name}",
    )
    resolver = VideoResolver(mode="local", video_dir=video_dir)
    with pytest.raises(VideoNotFound):
        resolver.resolve("ghost-clip")


def test_video_resolver_raises_tools_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "c.mp4").write_bytes(b"x")
    monkeypatch.setattr("annotator.server.video.shutil.which", lambda _name: None)
    resolver = VideoResolver(mode="local", video_dir=video_dir)
    with pytest.raises(VideoToolsMissing):
        resolver.resolve("c")


# -----------------------------------------------------------------------------
# probe_video_meta — fps / duration extraction
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("30/1", 30.0),
        ("30000/1001", 30000.0 / 1001.0),  # NTSC 29.97
        ("60", 60.0),
        ("0/0", None),
        ("/", None),
        ("", None),
        (None, None),
        ("not a number", None),
        ("30/abc", None),
        ("0/1", None),  # zero rate is not a usable fps
    ],
)
def test_parse_frame_rate_handles_ffprobe_strings(
    raw: str | None, expected: float | None
) -> None:
    """`avg_frame_rate` comes back as a rational string from ffprobe.
    Confirm the parser handles every realistic input (clean integer
    rates, NTSC fractions) and degrades gracefully on garbage."""
    result = _parse_frame_rate(raw)
    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert abs(result - expected) < 1e-9


def _fake_meta_probe(
    stdout: str,
) -> Any:
    """Build a fake `subprocess.run` that returns `stdout` from a single
    ffprobe call. Used by the probe_video_meta tests below."""

    def _run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert cmd[0] == "ffprobe"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    return _run


def test_probe_video_meta_returns_fps_and_duration_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Happy path: ffprobe returns a v:0 stream with `avg_frame_rate`
    and a container `format.duration`. Both end up in the tuple."""
    payload = json.dumps({
        "streams": [{"avg_frame_rate": "30/1"}],
        "format": {"duration": "12.345"},
    })
    monkeypatch.setattr(
        "annotator.server.video.subprocess.run", _fake_meta_probe(payload)
    )
    out = probe_video_meta(tmp_path / "clip.mp4")
    assert out is not None
    fps, duration = out
    assert fps == 30.0
    assert duration == pytest.approx(12.345)


def test_probe_video_meta_handles_ntsc_fractional_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NTSC clips report `30000/1001` rather than `29.97`. Confirm the
    rational parser feeds the right value into the tuple."""
    payload = json.dumps({
        "streams": [{"avg_frame_rate": "30000/1001"}],
        "format": {"duration": "9.9"},
    })
    monkeypatch.setattr(
        "annotator.server.video.subprocess.run", _fake_meta_probe(payload)
    )
    out = probe_video_meta(tmp_path / "ntsc.mp4")
    assert out is not None
    fps, _ = out
    assert fps == pytest.approx(30000.0 / 1001.0)


def test_probe_video_meta_returns_none_on_no_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A container ffprobe can read but where v:0 produces no streams
    must collapse to `None` so the caller falls back to schema defaults."""
    payload = json.dumps({"streams": [], "format": {"duration": "5.0"}})
    monkeypatch.setattr(
        "annotator.server.video.subprocess.run", _fake_meta_probe(payload)
    )
    assert probe_video_meta(tmp_path / "no_video.mp4") is None


def test_probe_video_meta_returns_none_on_unparseable_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`avg_frame_rate` of `0/0` (ffprobe's "unknown" sentinel) must
    collapse to `None` rather than emitting a zero-fps bundle."""
    payload = json.dumps({
        "streams": [{"avg_frame_rate": "0/0"}],
        "format": {"duration": "5.0"},
    })
    monkeypatch.setattr(
        "annotator.server.video.subprocess.run", _fake_meta_probe(payload)
    )
    assert probe_video_meta(tmp_path / "unknown_fps.mp4") is None


def test_probe_video_meta_returns_none_on_missing_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Some containers (matroska without seek index) report no
    `format.duration`. The probe must collapse to `None` rather than
    fabricating a bogus duration."""
    payload = json.dumps({
        "streams": [{"avg_frame_rate": "30/1"}],
        "format": {},
    })
    monkeypatch.setattr(
        "annotator.server.video.subprocess.run", _fake_meta_probe(payload)
    )
    assert probe_video_meta(tmp_path / "no_duration.mp4") is None


def test_probe_video_meta_returns_none_when_ffprobe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ffprobe missing / file unreadable / non-zero exit — every
    `subprocess` failure mode collapses to `None`. The caller cannot
    raise on probe failure (fresh-bundle creation must succeed even
    for unreadable videos)."""

    def _failing(cmd: list[str], **_kwargs: Any) -> Any:
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr("annotator.server.video.subprocess.run", _failing)
    assert probe_video_meta(tmp_path / "unreadable.mp4") is None


def test_probe_video_meta_returns_none_on_malformed_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ffprobe returning non-JSON (e.g. stderr leak, partial output)
    must not crash — collapse to `None` and log a warning."""

    def _bad_json(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="not json {", stderr="")

    monkeypatch.setattr("annotator.server.video.subprocess.run", _bad_json)
    assert probe_video_meta(tmp_path / "garbled.mp4") is None
