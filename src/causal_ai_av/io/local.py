# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local-filesystem I/O for AV Causal annotation JSON files.

Filename convention in the reference corpus: `<annotation_uuid>__<clip_id>.json`.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from causal_ai_av.spec import AnnotationBundle


def load_file(path: str | Path) -> AnnotationBundle:
    """Parse one annotation JSON file into an `AnnotationBundle`."""
    return AnnotationBundle.model_validate_json(Path(path).read_bytes())


def save_file(bundle: AnnotationBundle, path: str | Path, *, indent: int = 2) -> None:
    """Serialize a bundle to a JSON file atomically.

    Uses `by_alias=True` so leading-underscore fields (`_track_index`, etc.)
    serialize with their on-disk names, and `exclude_unset=True` so absent
    optional fields stay absent (preserving the input's field set).

    Creates parent directories if missing.

    The write is atomic: the payload is written to a temp file in the same
    directory, fsynced, and renamed over the target. Concurrent readers will
    see either the old bytes or the new bytes — never a truncated file.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = bundle.model_dump_json(indent=indent, by_alias=True, exclude_unset=True)
    # NamedTemporaryFile with delete=False so we can rename on close.
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
        # Best-effort cleanup of the temp file if rename failed.
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
    """
    for p in sorted(Path(path).glob(glob)):
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
