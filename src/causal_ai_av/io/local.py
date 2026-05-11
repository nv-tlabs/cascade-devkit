"""Local-filesystem I/O for AV Causal annotation JSON files.

Filename convention in the reference corpus: `<annotation_uuid>__<clip_id>.json`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from causal_ai_av.spec import AnnotationBundle


def load_file(path: str | Path) -> AnnotationBundle:
    """Parse one annotation JSON file into an `AnnotationBundle`."""
    return AnnotationBundle.model_validate_json(Path(path).read_bytes())


def save_file(bundle: AnnotationBundle, path: str | Path, *, indent: int = 2) -> None:
    """Serialize a bundle to a JSON file.

    Uses `by_alias=True` so leading-underscore fields (`_track_index`, etc.)
    serialize with their on-disk names, and `exclude_unset=True` so absent
    optional fields stay absent (preserving the input's field set).
    """
    Path(path).write_text(
        bundle.model_dump_json(indent=indent, by_alias=True, exclude_unset=True),
        encoding="utf-8",
    )


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
