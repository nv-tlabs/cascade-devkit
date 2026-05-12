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
    _transcode_to_h264,
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
