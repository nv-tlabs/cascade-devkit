# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video resolution + HEVC→H.264 transcode for the annotator server.

The browser's HTML5 `<video>` element does not play HEVC/H.265, but the
corpus is HEVC. This module resolves a clip_id to a *playable* H.264 MP4
path on disk, transcoding when needed and caching the result.

Three source modes (set by the CLI via `--video-source`):

  - ``local``: a directory of `<clip_id>.<ext>` files supplied by the user.
  - ``hf``: the HuggingFace Physical AI AV dataset (requires the `[hf]`
    optional dependency: `pip install cascade_av[hf]`).
  - ``auto``: try ``hf`` first, fall back to ``local`` if the `[hf]` extra
    is not installed.

The resolver is intentionally lazy — the dataset is only touched on the
first request, not at startup, so CLI launches stay fast when video is
never browsed. The transcode cache lives in
``$XDG_CACHE_HOME/cascade-annotator/transcoded/`` (or
``~/.cache/cascade-annotator/transcoded/`` when XDG is unset).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LOG = logging.getLogger(__name__)

VideoSourceMode = Literal["auto", "hf", "local"]
LOCAL_VIDEO_EXTS: tuple[str, ...] = (".mp4", ".mkv", ".mov", ".avi")

# Coarse stages a clip walks through during :meth:`VideoResolver.resolve`.
# Surfaced to the browser via /api/clips/{id}/video/status so the UI can show
# *what* is happening during a long HF resolve instead of an inscrutable spinner.
# Order is not enforced — a clip can skip stages (e.g. a local H.264 file goes
# locating → probing → ready and never enters downloading/extracting/transcoding).
VideoStage = Literal[
    "idle",         # never requested, or status cleared after a successful serve
    "locating",     # picking a source file (local lookup or HF chunk resolution)
    "downloading",  # HF chunk-zip download from the hub
    "extracting",   # unzip of the per-clip MP4 out of the chunk archive
    "probing",      # ffprobe codec detection
    "transcoding",  # ffmpeg HEVC → H.264
    "ready",        # resolve() returned a path; the FileResponse is being served
    "error",        # last resolve() raised; ``message`` carries the human reason
]


class VideoToolsMissing(RuntimeError):
    """Raised when `ffmpeg` and/or `ffprobe` are not on PATH."""


class VideoNotFound(LookupError):
    """Raised when a clip_id cannot be resolved to a source file."""


@dataclass
class TranscodeError(RuntimeError):
    """ffmpeg failed; the stderr tail is surfaced in the API 503 body."""

    stderr: str

    def __str__(self) -> str:  # pragma: no cover — trivial
        return self.stderr or "ffmpeg transcode failed (no stderr captured)"


def _default_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "cascade-annotator" / "transcoded"


def _require_tools() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise VideoToolsMissing(
            f"required executable(s) not on PATH: {', '.join(missing)}. "
            "Install via `apt install ffmpeg` (or the equivalent for your OS)."
        )


def _probe_codec(path: Path) -> str | None:
    """Return the video codec_name (e.g. ``'h264'``, ``'hevc'``) or None."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name", "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        LOG.warning("ffprobe failed on %s: %s", path, exc)
        return None
    try:
        streams = json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError:
        return None
    if not streams:
        return None
    codec = streams[0].get("codec_name")
    return str(codec).lower() if codec else None


def _parse_frame_rate(raw: str | None) -> float | None:
    """Parse ffprobe's ``avg_frame_rate`` / ``r_frame_rate`` field.

    ffprobe reports rates as rational strings: ``"30/1"`` for 30 fps,
    ``"30000/1001"`` for NTSC 29.97, ``"0/0"`` for unknown. Returns
    ``None`` for any value that doesn't parse to a positive float.
    """
    if not raw:
        return None
    if "/" in raw:
        num_s, _, den_s = raw.partition("/")
        try:
            num, den = float(num_s), float(den_s)
        except ValueError:
            return None
        if den <= 0:
            return None
        rate = num / den
        return rate if rate > 0 else None
    try:
        val = float(raw)
    except ValueError:
        return None
    return val if val > 0 else None


def probe_video_meta(path: Path) -> tuple[float, float] | None:
    """Return ``(fps, duration_s)`` for ``path``, or ``None`` on failure.

    Uses a single ``ffprobe`` invocation that asks for both
    ``stream.avg_frame_rate`` (the canonical fps field) and
    ``format.duration`` (more reliable than ``stream.duration`` across
    container types). All failure modes — missing ffprobe, malformed
    JSON output, no video stream, unparseable rate — collapse to
    ``None``; callers fall back to schema defaults.

    No exception is raised on probe failure: ``make_empty_bundle`` must
    succeed for any fresh clip even if its video is unprobeable.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate:format=duration",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        LOG.warning("ffprobe failed on %s: %s", path, exc)
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        LOG.warning("ffprobe returned non-JSON for %s", path)
        return None

    streams = payload.get("streams") or []
    if not streams:
        return None
    fps = _parse_frame_rate(streams[0].get("avg_frame_rate"))
    if fps is None:
        return None

    fmt = payload.get("format") or {}
    raw_dur = fmt.get("duration")
    try:
        duration_s = float(raw_dur) if raw_dur is not None else None
    except (TypeError, ValueError):
        duration_s = None
    if duration_s is None or duration_s < 0:
        return None

    return fps, duration_s


def _transcode_to_h264(src: Path, dst: Path) -> None:
    """Run ffmpeg ``src`` → unique tmp file → rename to ``dst`` on success.

    The tmp filename embeds a fresh UUID so two transcodes that race on the
    same destination cannot stomp on each other's output. The per-clip lock
    in :class:`VideoResolver` is the primary serializer; this is the
    belt-and-suspenders backstop — worst case is wasted ffmpeg CPU, never
    a corrupt cache file.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(f".tmp.{uuid.uuid4().hex}.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=True)
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        tail = (exc.stderr or "").strip().splitlines()[-20:]
        raise TranscodeError(stderr="\n".join(tail)) from exc
    except subprocess.TimeoutExpired as exc:
        tmp.unlink(missing_ok=True)
        raise TranscodeError(stderr=f"ffmpeg timed out after {exc.timeout}s") from exc
    os.replace(tmp, dst)


class VideoResolver:
    """Resolve a clip_id to a browser-playable H.264 MP4 on disk.

    Stateless w.r.t. mutations — calls are idempotent and the transcode
    cache is the only "memory" between requests. Construct one instance
    per server process.
    """

    def __init__(
        self,
        mode: VideoSourceMode,
        *,
        video_dir: Path | None = None,
        hf_source: str | Path | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        if mode == "local" and video_dir is None:
            raise ValueError("--video-source=local requires --video-dir")
        self.mode = mode
        self.video_dir = video_dir
        self.hf_source = hf_source
        self.cache_dir = cache_dir if cache_dir is not None else _default_cache_dir()
        self._hf_dataset = None  # lazy
        # Whether HF mode is usable in this env. Decided eagerly so `auto`
        # collapses to `local` without paying the import cost per request.
        self._hf_available = self._check_hf_available(mode)
        # Per-clip transcode/extract serialization. FastAPI dispatches sync
        # routes through a thread pool, so two concurrent GET /video requests
        # for the same clip would otherwise both transcode (or both extract
        # from the HF zip) into the same cache path. Followers re-check the
        # cache after acquiring the lock and skip the heavy work when the
        # leader has already populated it.
        self._clip_locks: dict[str, threading.Lock] = {}
        self._clip_locks_guard = threading.Lock()
        # Per-clip progress tracker — the browser polls
        # ``GET /api/clips/{id}/video/status`` while the synchronous
        # ``GET /video`` request is in flight, so it can show a stage label
        # ("Downloading clip from HuggingFace…") instead of a blind spinner.
        # Reads and writes are protected by ``_clip_state_guard`` so a poller
        # never observes a half-written dict.
        self._clip_state: dict[str, tuple[str, str]] = {}
        self._clip_state_guard = threading.Lock()

    def _set_state(self, clip_id: str, stage: str, message: str = "") -> None:
        """Record the current progress stage for ``clip_id``.

        Called from inside the per-clip lock during resolve(). The state dict
        has its own guard (cheaper than holding the per-clip lock across the
        status read path), so status pollers do not contend with each other.
        """
        with self._clip_state_guard:
            self._clip_state[clip_id] = (stage, message)

    def get_state(self, clip_id: str) -> dict[str, str]:
        """Return the most recent stage + message for ``clip_id``.

        Returns ``{"stage": "idle", "message": ""}`` for clips that have not
        been requested in this process — there is no historical state to
        report and "idle" is the only honest answer.
        """
        with self._clip_state_guard:
            stage, message = self._clip_state.get(clip_id, ("idle", ""))
        return {"stage": stage, "message": message}

    def _clip_lock(self, clip_id: str) -> threading.Lock:
        """Return the lock for ``clip_id``, lazily creating it on first use."""
        with self._clip_locks_guard:
            lock = self._clip_locks.get(clip_id)
            if lock is None:
                lock = threading.Lock()
                self._clip_locks[clip_id] = lock
            return lock

    @staticmethod
    def _check_hf_available(mode: VideoSourceMode) -> bool:
        if mode == "local":
            return False
        try:
            import cascade_av.dataset  # noqa: F401
            return True
        except ImportError:
            if mode == "hf":
                # Explicit hf mode + missing extra → fail loudly at startup.
                raise ImportError(
                    "--video-source=hf requires the `[hf]` extra: "
                    "`pip install cascade_av[hf]` or "
                    "`uv sync --extra hf`."
                ) from None
            return False

    def describe(self) -> str:
        """One-line human-readable description for the startup log."""
        if self.mode == "local":
            return f"local ({self.video_dir})"
        if self.mode == "hf":
            return f"hf ({self.hf_source or 'corpus dataset'})"
        if self._hf_available:
            return f"auto → hf ({self.hf_source or 'corpus dataset'})"
        if self.video_dir is not None:
            return f"auto → local ({self.video_dir})"
        return "auto (no source resolved)"

    # ----- mode dispatchers --------------------------------------------------

    def _resolve_local(self, clip_id: str) -> Path:
        if self.video_dir is None:
            raise VideoNotFound(
                f"clip {clip_id!r}: --video-source=local but no --video-dir set"
            )
        for ext in LOCAL_VIDEO_EXTS:
            candidate = self.video_dir / f"{clip_id}{ext}"
            if candidate.is_file():
                return candidate
        raise VideoNotFound(
            f"clip {clip_id!r}: no file matching <video-dir>/{clip_id}.{{mp4,mkv,mov,avi}}"
        )

    def _resolve_hf(self, clip_id: str) -> Path:
        try:
            import zipfile

            from cascade_av.dataset import (
                DEFAULT_ANNOTATION_CAMERA,
                CascadeDataset,
            )
        except ImportError as exc:
            raise VideoNotFound(
                f"clip {clip_id!r}: HF backend unavailable ({exc})"
            ) from exc

        # Lazy: only construct on first need, not at startup.
        if self._hf_dataset is None:
            if self.hf_source is None:
                # The dataset constructor refuses `None` (no remote default
                # yet). For Phase 3, falling back to the parent interface lets
                # us reach the HF repo for video bytes even without a local
                # annotation directory.
                from physical_ai_av import PhysicalAIAVDatasetInterface
                self._hf_dataset = PhysicalAIAVDatasetInterface()
            else:
                self._hf_dataset = CascadeDataset(self.hf_source)

        ds = self._hf_dataset
        try:
            chunk_id = ds.get_clip_chunk(clip_id)
        except (KeyError, IndexError) as exc:
            raise VideoNotFound(
                f"clip {clip_id!r} not present in HF clip index"
            ) from exc

        zip_filename = ds.features.get_chunk_feature_filename(
            chunk_id, DEFAULT_ANNOTATION_CAMERA
        )
        # The HF hub client streams the chunk archive on first access and
        # serves a cached path afterwards — we cannot tell the two apart
        # without poking at the cache, so the "downloading" stage covers
        # both fast (cached) and slow (cold) returns. The browser only
        # sees this label long enough to matter on the cold path.
        self._set_state(clip_id, "downloading", "fetching chunk from HuggingFace")
        try:
            zip_path = Path(ds.download_file(zip_filename))
        except Exception as exc:  # noqa: BLE001 — surface anything as VideoNotFound
            raise VideoNotFound(
                f"clip {clip_id!r}: HF download failed ({exc})"
            ) from exc

        # Extract the MP4 for this clip to our cache dir.
        extracted_dir = self.cache_dir / "hf_extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        extracted_mp4 = extracted_dir / f"{clip_id}.mp4"
        if not extracted_mp4.is_file() or extracted_mp4.stat().st_mtime < zip_path.stat().st_mtime:
            self._set_state(clip_id, "extracting", f"unzipping {clip_id}.mp4")
            with zipfile.ZipFile(zip_path, "r") as zf:
                candidates = [
                    n for n in zf.namelist()
                    if clip_id in n and n.lower().endswith(".mp4")
                ]
                if not candidates:
                    raise VideoNotFound(
                        f"clip {clip_id!r}: no MP4 named like clip_id in chunk zip"
                    )
                match = next(
                    (c for c in candidates if Path(c).stem == clip_id),
                    candidates[0],
                )
                # Unique tmp suffix per call so two concurrent extractions
                # cannot stomp on each other's output; the per-clip lock in
                # ``resolve()`` is the primary serializer.
                tmp = extracted_mp4.with_suffix(f".tmp.{uuid.uuid4().hex}.mp4")
                try:
                    with zf.open(match) as src, tmp.open("wb") as out:
                        shutil.copyfileobj(src, out)
                    os.replace(tmp, extracted_mp4)
                except BaseException:
                    tmp.unlink(missing_ok=True)
                    raise
        return extracted_mp4

    # ----- public entry point ------------------------------------------------

    def resolve(self, clip_id: str) -> Path:
        """Return a playable H.264 MP4 path for `clip_id`.

        Concurrent calls for the same ``clip_id`` are serialised by a
        per-clip lock so two requests do not both transcode (HEVC source)
        or both extract from the HF zip into the same destination. Calls
        for *different* clip_ids run in parallel.

        Raises:
            VideoNotFound: clip_id cannot be located in any configured source.
            VideoToolsMissing: ffmpeg/ffprobe are absent from PATH.
            TranscodeError: ffmpeg ran but returned non-zero.
        """
        # ffprobe is needed on every path (codec detection), so check tools
        # up front — otherwise the HF mode would burn a chunk-zip download
        # before discovering ffmpeg/ffprobe are missing.
        _require_tools()

        with self._clip_lock(clip_id):
            try:
                path = self._resolve_locked(clip_id)
            except Exception as exc:
                # Best-effort progress reporting — surface the failure to
                # status pollers so the UI can switch from "in progress" to
                # an error affordance instead of leaving its last stage
                # label hanging until the GET /video request itself fails.
                self._set_state(clip_id, "error", str(exc))
                raise
            self._set_state(clip_id, "ready", "")
            return path

    def _resolve_locked(self, clip_id: str) -> Path:
        """Same contract as :meth:`resolve`, executed under the per-clip lock."""
        # 1. Locate the source file. HF mode performs zip extraction with
        #    its own cache check; re-running under the lock is idempotent.
        self._set_state(clip_id, "locating", "")
        if self.mode == "local":
            src = self._resolve_local(clip_id)
        elif self.mode == "hf":
            src = self._resolve_hf(clip_id)
        else:  # auto
            if self._hf_available:
                try:
                    src = self._resolve_hf(clip_id)
                except VideoNotFound:
                    if self.video_dir is None:
                        raise
                    src = self._resolve_local(clip_id)
            elif self.video_dir is not None:
                src = self._resolve_local(clip_id)
            else:
                raise VideoNotFound(
                    f"clip {clip_id!r}: no video source configured "
                    "(install the [hf] extra or pass --video-dir)"
                )

        # 2. If it's already H.264, serve as-is. Otherwise transcode.
        self._set_state(clip_id, "probing", "")
        codec = _probe_codec(src)
        if codec is None:
            # ffprobe could read the file but found no stream — surface a clean
            # 404-equivalent rather than masquerading as a transcode error.
            raise VideoNotFound(
                f"clip {clip_id!r}: ffprobe could not detect a video stream in {src}"
            )
        if codec == "h264":
            return src

        cached = self.cache_dir / f"{clip_id}.mp4"
        cached.parent.mkdir(parents=True, exist_ok=True)
        if cached.is_file() and cached.stat().st_mtime >= src.stat().st_mtime:
            return cached

        LOG.info("transcoding %s (%s → h264) → %s", clip_id, codec, cached)
        self._set_state(clip_id, "transcoding", f"{codec.upper()} → H.264")
        _transcode_to_h264(src, cached)
        return cached
