# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter between source paths on disk and `AnnotationBundle` objects.

This module is the only place in the server that talks to `cascade_av.io`.
It owns clip-index construction, fresh-bundle creation for unlabelled clips
(video-only sources), and the save side of the round-trip.
"""

from __future__ import annotations

import logging
import os
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from annotator.server.video import probe_video_meta
from cascade_av.io import load_file, save_file
from cascade_av.spec import AnnotationBundle, SilAvAnnotation, VideoMeta

LOG = logging.getLogger(__name__)

ClipKind = Literal["annotated", "unlabelled"]

# Recognized video extensions (must match cli.VIDEO_EXTS).
VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".mkv", ".mov", ".avi"})


@dataclass
class ClipEntry:
    """One clip's on-disk presence.

    `path` is `None` for fresh (unlabelled) clips that have a video but no
    annotation JSON yet — until the first save mutates the entry in-place.
    """

    clip_id: str
    path: Path | None = None
    video_path: Path | None = None
    kind: ClipKind = "unlabelled"
    # Internal: list of duplicate JSON paths dropped during indexing.
    _dropped_paths: list[Path] = field(default_factory=list)
    # Internal: True after the server has taken a `.bak` snapshot of `path`
    # this session. The snapshot is taken once, on the first save, so an
    # accidental overwrite or corrupted bundle can be recovered.
    _bak_written: bool = False


def _classify(path: Path) -> Literal["annotation", "video"] | None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "annotation"
    if suffix in VIDEO_EXTS:
        return "video"
    return None


def _iter_paths(sources: list[Path]) -> list[Path]:
    """Expand `sources` into a flat, deterministic file list.

    Directories are recursed for `*.json` and known video extensions. Files
    are accepted as-is. Order is `sorted(rglob)` per source, sources kept in
    user-supplied order. Non-existent paths raise `FileNotFoundError` (so the
    CLI can surface a clean error before uvicorn boots).
    """
    out: list[Path] = []
    for src in sources:
        if src.is_dir():
            for child in sorted(src.rglob("*.json")):
                out.append(child)
            for ext in sorted(VIDEO_EXTS):
                for child in sorted(src.rglob(f"*{ext}")):
                    out.append(child)
        elif src.is_file():
            out.append(src)
        else:
            raise FileNotFoundError(f"source does not exist: {src}")
    return out


def build_clip_index(sources: list[Path]) -> dict[str, ClipEntry]:
    """Walk `sources`, classify files, and key the result by clip_id.

    Annotation JSONs have their clip_id read from `bundle.video.clip_id`.
    Videos derive clip_id from the filename stem. When the same clip_id has
    both a JSON and a video, both paths populate the same `ClipEntry` and
    `kind = "annotated"`. Duplicate JSONs for one clip_id emit a single
    summary warning (mirroring `cascade_av.dataset.CascadeDataset._scan`).
    """
    paths = _iter_paths(sources)

    # First pass: collect annotations and videos by clip_id.
    annotations: dict[str, list[Path]] = {}
    videos: dict[str, Path] = {}
    for path in paths:
        kind = _classify(path)
        if kind == "annotation":
            try:
                bundle = load_file(path)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("failed to parse %s: %s", path, exc)
                continue
            annotations.setdefault(bundle.video.clip_id, []).append(path)
        elif kind == "video":
            clip_id = path.stem
            # Only the first video wins (videos with the same stem in
            # different dirs are unusual — keep deterministic).
            videos.setdefault(clip_id, path)

    # Second pass: build the index.
    index: dict[str, ClipEntry] = {}
    duplicate_summary: list[tuple[str, Path, list[Path]]] = []
    for clip_id, ann_paths in annotations.items():
        kept = ann_paths[0]
        dropped = ann_paths[1:]
        if dropped:
            duplicate_summary.append((clip_id, kept, dropped))
        index[clip_id] = ClipEntry(
            clip_id=clip_id,
            path=kept,
            video_path=videos.get(clip_id),
            kind="annotated",
            _dropped_paths=dropped,
        )
    for clip_id, video_path in videos.items():
        if clip_id in index:
            continue
        index[clip_id] = ClipEntry(
            clip_id=clip_id,
            path=None,
            video_path=video_path,
            kind="unlabelled",
        )

    if duplicate_summary:
        n = len(duplicate_summary)
        warnings.warn(
            f"{n} clip_id(s) had multiple annotation files; kept first by filename. "
            "(Set CASCADE_AV_VERBOSE=1 for per-file detail.)",
            stacklevel=2,
        )
        if os.environ.get("CASCADE_AV_VERBOSE"):
            for clip_id, kept, dropped in duplicate_summary:
                dropped_names = ", ".join(p.name for p in dropped)
                LOG.warning("clip %s: kept %s; dropped %s", clip_id, kept.name, dropped_names)

    return index


def make_empty_bundle(
    clip_id: str, video_path: Path | None = None
) -> AnnotationBundle:
    """Construct a minimum-valid `AnnotationBundle` for a brand-new clip.

    Explicitly sets `schema_version` and `status` so fresh-clip JSONs are
    self-describing on disk — `exclude_unset=True` in save_file would
    otherwise drop them along with any other field still at its default.

    When `video_path` is supplied, ffprobe is invoked to populate
    `VideoMeta.fps` / `VideoMeta.duration_s` with the real values from
    the file header. Probe failures (missing ffprobe, no video stream,
    unparseable rate, container with no duration) silently fall back to
    the schema defaults (`30.0` / `0.0`) — the bundle remains valid
    either way. Pass `video_path=None` (the default) to skip probing
    entirely, which keeps unit tests and synthetic fixtures fast.
    """
    probe_kwargs: dict[str, float] = {}
    if video_path is not None:
        probed = probe_video_meta(video_path)
        if probed is not None:
            fps_val, duration_val = probed
            probe_kwargs = {"fps": fps_val, "duration_s": duration_val}
    return AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id=clip_id, **probe_kwargs),
        annotation=SilAvAnnotation(),
        status="annotating",
    )


def load_bundle(entry: ClipEntry) -> AnnotationBundle:
    """Load (or synthesize) the bundle for an entry."""
    if entry.kind == "annotated":
        assert entry.path is not None  # invariant of `kind == "annotated"`
        return load_file(entry.path)
    return make_empty_bundle(entry.clip_id, video_path=entry.video_path)


def save_bundle(
    entry: ClipEntry,
    bundle: AnnotationBundle,
    destination_dir: Path | None,
) -> Path:
    """Save `bundle` for `entry`, returning the resulting path on disk.

    For already-annotated entries, writes back to `entry.path` (atomic
    rename inside `cascade_av.io.save_file`). For unlabelled entries,
    creates `<destination_dir>/<clip_id>.json` and mutates the entry to
    `kind="annotated"` with its new path.

    On the *first* save for a clip in this server session, the current
    on-disk file (if any) is copied to a sibling `<path>.bak`. This gives
    the user a single-shot recovery option in case the new save introduces
    a regression; subsequent saves in the same session do not re-stamp the
    backup, so the user's original input is preserved.

    Raises `ValueError` if `entry.path` is unset and `destination_dir` is
    `None`.
    """
    if entry.kind == "annotated":
        assert entry.path is not None
        _maybe_write_bak(entry)
        save_file(bundle, entry.path)
        return entry.path

    # Fresh save for an unlabelled clip. Prefer destination_dir; fall back to
    # the video's parent directory if available.
    target_dir = destination_dir
    if target_dir is None and entry.video_path is not None:
        target_dir = entry.video_path.parent
    if target_dir is None:
        raise ValueError(
            f"cannot determine destination for fresh clip {entry.clip_id!r}: "
            "pass destination_dir, or attach a video_path to the entry."
        )

    target_path = target_dir / f"{entry.clip_id}.json"
    save_file(bundle, target_path)
    entry.path = target_path
    entry.kind = "annotated"
    # Fresh clips have no prior content to back up — mark _bak_written so a
    # later save in this same session also skips the .bak step.
    entry._bak_written = True
    return target_path


def _maybe_write_bak(entry: ClipEntry) -> None:
    """Take a one-shot `.bak` snapshot of the existing file."""
    if entry._bak_written:
        return
    if entry.path is None or not entry.path.is_file():
        entry._bak_written = True
        return
    bak_path = entry.path.with_suffix(entry.path.suffix + ".bak")
    try:
        shutil.copy2(entry.path, bak_path)
    except OSError as exc:  # noqa: BLE001
        # Don't block a save if the snapshot fails — log and move on.
        LOG.warning("failed to write .bak for %s: %s", entry.path, exc)
    entry._bak_written = True
