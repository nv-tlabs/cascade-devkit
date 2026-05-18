# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter between source paths on disk and `AnnotationBundle` objects.

This module is the only place in the server that talks to `cascade_av.io`.
It owns clip-index construction, fresh-bundle creation for unlabelled clips
(video-only sources), and the save side of the round-trip.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from annotator.server.video import probe_video_meta
from cascade_av.io import load_file, save_file
from cascade_av.spec import (
    CURRENT_SCHEMA_VERSION,
    AnnotationBundle,
    SilAvAnnotation,
    VideoMeta,
)

LOG = logging.getLogger(__name__)

ClipKind = Literal["annotated", "unlabelled"]

# Recognized video extensions (must match cli.VIDEO_EXTS).
VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".mkv", ".mov", ".avi"})


@dataclass
class ClipEntry:
    """One clip's on-disk presence.

    `path` is `None` for fresh (unlabelled) clips that have a video but no
    annotation JSON yet — until the first save mutates the entry in-place.

    `status` mirrors ``AnnotationBundle.status`` and is read lazily from disk
    via ``json.load`` during ``build_clip_index`` so the sidebar can render a
    per-clip badge ("in progress" / "complete") without instantiating the
    full Pydantic tree for every entry. ``None`` means "no annotation file"
    (unlabelled) or "annotation file has no `status` key" (legacy bundles).
    """

    clip_id: str
    path: Path | None = None
    video_path: Path | None = None
    kind: ClipKind = "unlabelled"
    status: str | None = None
    # Internal: list of duplicate JSON paths dropped during indexing.
    _dropped_paths: list[Path] = field(default_factory=list)
    # Internal: True after the server has taken a `.bak` snapshot of `path`
    # this session. The snapshot is taken once, on the first save, so an
    # accidental overwrite or corrupted bundle can be recovered.
    _bak_written: bool = False


def _classify(path: Path) -> Literal["annotation", "video"] | None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        # Sidecar files (`<stem>.extra.json`) end in `.json` too — Path.suffix
        # only looks at the final dot-segment. Skip them: they carry extension
        # payload (the ``ui/1.0`` track indices, etc.) and aren't standalone
        # annotation bundles. Without this, `load_file` would try to parse one
        # as an AnnotationBundle and log a noisy `WARNING ... validation
        # error: 'video' Field required` per sidecar on every launch.
        if path.name.lower().endswith(".extra.json"):
            return None
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

    # First pass: collect annotations and videos by clip_id. The status field
    # is captured alongside clip_id so the second pass can populate
    # `ClipEntry.status` without re-reading the file.
    annotations: dict[str, list[tuple[Path, str | None]]] = {}
    videos: dict[str, Path] = {}
    for path in paths:
        kind = _classify(path)
        if kind == "annotation":
            try:
                bundle = load_file(path)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("failed to parse %s: %s", path, exc)
                continue
            # Pluck status straight from the on-disk JSON without going
            # through Pydantic again — keeps the sidebar load O(small) for
            # large corpora. Errors silently degrade to `status=None`, which
            # the sidebar treats as "in progress" for annotated clips.
            status: str | None = None
            try:
                raw = json.loads(path.read_text())
                if isinstance(raw, dict):
                    val = raw.get("status")
                    if isinstance(val, str):
                        status = val
            except Exception:  # noqa: BLE001
                pass
            annotations.setdefault(bundle.video.clip_id, []).append((path, status))
        elif kind == "video":
            clip_id = path.stem
            # Only the first video wins (videos with the same stem in
            # different dirs are unusual — keep deterministic).
            videos.setdefault(clip_id, path)

    # Second pass: build the index.
    index: dict[str, ClipEntry] = {}
    duplicate_summary: list[tuple[str, Path, list[Path]]] = []
    for clip_id, ann_paths in annotations.items():
        kept_path, kept_status = ann_paths[0]
        dropped = [p for p, _ in ann_paths[1:]]
        if dropped:
            duplicate_summary.append((clip_id, kept_path, dropped))
        index[clip_id] = ClipEntry(
            clip_id=clip_id,
            path=kept_path,
            video_path=videos.get(clip_id),
            kind="annotated",
            status=kept_status,
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
        schema_version=CURRENT_SCHEMA_VERSION,
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


def bundle_to_wire(bundle: AnnotationBundle) -> dict[str, Any]:
    """Serialise a bundle for the wire (annotator GET response).

    The on-disk format keeps schema-extension payloads in the sibling
    ``<stem>.extra.json`` sidecar; the wire format folds them into a single
    JSON object under ``_extensions`` so the frontend can hydrate everything
    from one request. The shape is::

        {
          ...main bundle fields (schema_version, video, annotation, …)…,
          "_extensions": {
            "ui/1.0":   {...},
            "bbox/1.0": {...},   # if applicable
            ...
          }
        }

    ``_extensions`` is absent (not empty) when no extension produced data,
    matching the on-disk convention.
    """
    from cascade_av.extensions.registry import registered

    data = bundle.model_dump(by_alias=True, exclude_unset=True, mode="json")
    extensions: dict[str, Any] = dict(bundle._sidecar_raw)
    for key, ext in registered().items():
        payload = ext.dump(bundle)
        if payload:
            extensions[key] = payload
    if extensions:
        data["_extensions"] = extensions
    return data


def bundle_from_wire(payload: dict[str, Any]) -> AnnotationBundle:
    """Inverse of :func:`bundle_to_wire`.

    Pops the ``_extensions`` envelope before validating the main bundle, then
    re-attaches each entry to ``bundle._extensions`` (via the registered
    extension's :meth:`load`) or to ``bundle._sidecar_raw`` if no extension
    is registered for that key. Raises whatever ``AnnotationBundle.model_validate``
    raises on the trimmed payload — callers handle ``ValidationError``.
    """
    from cascade_av.extensions.registry import registered

    extensions_in = payload.pop("_extensions", None) if isinstance(payload, dict) else None
    bundle = AnnotationBundle.model_validate(payload)
    if not isinstance(extensions_in, dict):
        return bundle
    handlers = registered()
    for key, ext_data in extensions_in.items():
        ext = handlers.get(key)
        if ext is None:
            bundle._sidecar_raw[key] = ext_data
            continue
        try:
            ext.load(bundle, ext_data)
        except Exception:
            # Defensive: preserve the raw payload so the next save round-trips
            # it even if a handler crashed. Mirrors the io.local fallback.
            bundle._sidecar_raw[key] = ext_data
    return bundle


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
        # Refresh cached status so the sidebar badge reflects this save without
        # re-scanning every JSON on disk.
        entry.status = bundle.status
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
    entry.status = bundle.status
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
