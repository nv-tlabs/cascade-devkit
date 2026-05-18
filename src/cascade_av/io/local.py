# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local-filesystem I/O for CASCADE annotation JSON files.

Filename convention in the reference corpus: `<annotation_uuid>__<clip_id>.json`.

A main bundle file at ``<stem>.json`` may be accompanied by a sidecar at
``<stem>.extra.json`` carrying schema-extension payloads (see
:mod:`cascade_av.extensions`). The pair is written sidecar-first then main,
so a crash mid-save leaves the deprecated/extension data on disk and the
main file at its previous state.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from cascade_av.extensions.registry import registered
from cascade_av.spec import AnnotationBundle

_log = logging.getLogger(__name__)

# One-shot per process: keys we have already warned about as "sidecar key
# with no registered extension". Reset by `_reset_warn_cache_for_tests`.
_warned_unknown_keys: set[str] = set()


def _sidecar_path_for(path: Path) -> Path:
    """Sibling sidecar path: ``foo.json`` → ``foo.extra.json``.

    Uses :attr:`Path.stem` which strips the last suffix only, so paths
    with embedded dots (``foo.bar.json``) become ``foo.bar.extra.json``.
    """
    return path.with_name(f"{path.stem}.extra.json")


def load_file(path: str | Path) -> AnnotationBundle:
    """Parse one annotation JSON file into an :class:`AnnotationBundle`.

    If a sibling ``<stem>.extra.json`` sidecar exists, its ``extensions``
    block is dispatched to registered extensions; entries with no
    registered handler are stashed on the bundle's ``_sidecar_raw`` so
    they survive a subsequent :func:`save_file`.
    """
    p = Path(path)
    bundle = AnnotationBundle.model_validate_json(p.read_bytes())
    _load_sidecar_into(bundle, _sidecar_path_for(p))
    return bundle


def _load_sidecar_into(bundle: AnnotationBundle, sidecar_path: Path) -> None:
    if not sidecar_path.exists():
        return
    try:
        sidecar = json.loads(sidecar_path.read_bytes())
    except (OSError, json.JSONDecodeError):
        _log.exception("could not parse sidecar %s; ignoring", sidecar_path)
        return
    extensions = sidecar.get("extensions") if isinstance(sidecar, dict) else None
    if not isinstance(extensions, dict):
        return
    handlers = registered()
    for key, payload in extensions.items():
        ext = handlers.get(key)
        if ext is None:
            bundle._sidecar_raw[key] = payload
            if key not in _warned_unknown_keys:
                _warned_unknown_keys.add(key)
                warnings.warn(
                    f"sidecar key {key!r} has no registered extension; "
                    f"payload will round-trip verbatim. Install the "
                    f"matching extension to consume it.",
                    stacklevel=3,
                )
            continue
        try:
            ext.load(bundle, payload)
        except Exception:
            _log.exception(
                "extension %s failed to load sidecar payload from %s",
                key, sidecar_path,
            )
            # Preserve the raw payload so save still round-trips it.
            bundle._sidecar_raw[key] = payload


def save_file(bundle: AnnotationBundle, path: str | Path, *, indent: int = 2) -> None:
    """Serialize a bundle to a JSON file atomically.

    Uses ``by_alias=True`` so leading-underscore fields (``_track_index``, etc.)
    serialize with their on-disk names, and ``exclude_unset=True`` so absent
    optional fields stay absent (preserving the input's field set).

    Creates parent directories if missing.

    If any extension produces sidecar data (or the bundle was loaded with
    unclaimed sidecar payload), a sibling ``<stem>.extra.json`` is written
    *before* the main file. Each individual file write is atomic via temp
    file + rename. Pair-write across the two files is best-effort: a crash
    between the sidecar write and the main write leaves the sidecar
    refreshed and the main file at its previous bytes, which re-running
    the save corrects.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    sidecar_dest = _sidecar_path_for(dest)
    sidecar_payload = _build_sidecar_payload(bundle, dest.name)
    if sidecar_payload is not None:
        _atomic_write_text(
            sidecar_dest,
            json.dumps(sidecar_payload, indent=indent, sort_keys=True),
        )

    main_payload = bundle.model_dump_json(
        indent=indent, by_alias=True, exclude_unset=True,
    )
    _atomic_write_text(dest, main_payload)


def _build_sidecar_payload(
    bundle: AnnotationBundle, main_filename: str,
) -> dict[str, Any] | None:
    """Merge unclaimed sidecar payload + registered-extension dumps.

    Returns ``None`` when the merged ``extensions`` block is empty, so
    callers can skip writing the sidecar entirely.
    """
    # Start from any unknown payload the bundle carried in; registered
    # extensions overwrite their own key but cannot remove unknown ones.
    merged: dict[str, Any] = dict(bundle._sidecar_raw)
    for key, ext in registered().items():
        payload = ext.dump(bundle)
        if payload:
            merged[key] = payload
    if not merged:
        return None
    return {
        "schema_version": bundle.schema_version,
        "main_file": main_filename,
        "extensions": merged,
    }


def _atomic_write_text(dest: Path, payload: str) -> None:
    """Write ``payload`` to ``dest`` atomically via temp-file + rename."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=dest.parent,
        prefix=f".{dest.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
    finally:
        tmp.close()
    try:
        Path(tmp.name).replace(dest)
    except Exception:
        try:
            Path(tmp.name).unlink()
        except FileNotFoundError:
            pass
        raise


def iter_dir(
    path: str | Path,
    *,
    glob: str = "*.json",
    on_error: Literal["raise", "skip"] = "raise",
) -> Iterator[tuple[Path, AnnotationBundle]]:
    """Yield `(path, bundle)` for every JSON in a directory.

    With `on_error="skip"`, malformed or schema-violating files are silently
    dropped. With `on_error="raise"` (default), the first failure aborts.

    Sidecar files (``*.extra.json``) are skipped — they are paired with
    their main file and loaded automatically by :func:`load_file`.
    """
    for p in sorted(Path(path).glob(glob)):
        if p.name.endswith(".extra.json"):
            continue
        try:
            yield p, load_file(p)
        except Exception:
            if on_error == "raise":
                raise


def load_dir(
    path: str | Path,
    *,
    glob: str = "*.json",
    on_error: Literal["raise", "skip"] = "raise",
) -> list[AnnotationBundle]:
    """Load every annotation JSON in a directory into a flat list."""
    return [b for _, b in iter_dir(path, glob=glob, on_error=on_error)]


def parse_filename(path: str | Path) -> tuple[str, str]:
    """Split a `<annotation_uuid>__<clip_id>.json` filename into its two UUIDs.

    Returns `(annotation_id, clip_id)`. Raises `ValueError` on mismatch.
    """
    stem = Path(path).stem
    if "__" not in stem:
        raise ValueError(f"filename does not match `<a>__<b>` convention: {path}")
    annotation_id, _, clip_id = stem.rpartition("__")
    return annotation_id, clip_id


def group_by_clip_id(
    bundles: Iterable[AnnotationBundle],
) -> dict[str, list[AnnotationBundle]]:
    """Group bundles by `video.clip_id`. Useful when a clip has >1 annotation."""
    out: dict[str, list[AnnotationBundle]] = {}
    for b in bundles:
        out.setdefault(b.video.clip_id, []).append(b)
    return out


def _reset_warn_cache_for_tests() -> None:
    """Wipe the warn-once cache. Test fixtures only."""
    _warned_unknown_keys.clear()
