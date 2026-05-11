"""Command-line entry point for `causal-av-annotate`.

Phase 1 scope: argument parsing and source resolution only. Server startup
(uvicorn + FastAPI app) is wired in Step 3 (this same phase, later step).
Video probing for unlabelled clips lands in Phase 3.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

# Recognized video extensions (lower-case, with leading dot).
VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".mkv", ".mov", ".avi"})

SourceKind = Literal["annotation", "video"]


def _expand_source(source: Path) -> Iterable[Path]:
    """Yield concrete files from a single source argument.

    - A directory yields every `*.json` (recursive) plus every recognized
      video file (recursive).
    - A file is yielded as-is. Callers classify it by extension.
    """
    if source.is_dir():
        for child in sorted(source.rglob("*.json")):
            yield child
        for ext in sorted(VIDEO_EXTS):
            for child in sorted(source.rglob(f"*{ext}")):
                yield child
        return
    if source.is_file():
        yield source
        return
    raise FileNotFoundError(f"source does not exist: {source}")


def _classify(path: Path) -> SourceKind | None:
    """Return `"annotation"` for `*.json`, `"video"` for a known video ext,
    or `None` for anything else (skipped with a warning by the caller)."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "annotation"
    if suffix in VIDEO_EXTS:
        return "video"
    return None


def _resolve_clip_id(path: Path, kind: SourceKind) -> str | None:
    """Best-effort clip-id extraction.

    For annotations: parse the JSON and read `bundle.video.clip_id`.
    For videos: use the filename stem.

    Returns `None` on parse failure (caller logs + skips).
    """
    if kind == "video":
        return path.stem
    # Import lazily — keeps `--help` fast and avoids importing pydantic etc.
    # before argparse has had a chance to short-circuit on `-h`/`--help`.
    from causal_ai_av.io import load_file

    try:
        bundle = load_file(path)
    except Exception as exc:  # noqa: BLE001 — surface parse errors as warnings
        logging.warning("failed to parse %s: %s", path, exc)
        return None
    return bundle.video.clip_id


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causal-av-annotate",
        description=(
            "Local annotation tool for the AV Causal Dataset. Accepts one or "
            "more directories or files (annotation JSONs and/or video files) "
            "and serves a browser UI for editing them."
        ),
    )
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help=(
            "One or more paths. Directories are recursed for *.json and known "
            "video files (.mp4, .mkv, .mov, .avi). Files are accepted directly."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Bind port (default: 8765).",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Refuse PUT to annotation routes.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open a browser window on startup.",
    )
    parser.add_argument(
        "--video-source",
        choices=("auto", "hf", "local"),
        default="auto",
        help="Where to source video frames (Phase 3 — currently a placeholder).",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="When --video-source=local, the directory containing video files.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override log level explicitly (e.g. info, warning, error).",
    )
    return parser


def _resolve_log_level(args: argparse.Namespace) -> str:
    if args.log_level is not None:
        return args.log_level.lower()
    return "debug" if args.verbose else "info"


def _resolve_sources(
    sources: list[Path],
) -> list[tuple[str, SourceKind, Path]]:
    """Expand each source, classify, and return `(clip_id, kind, path)` rows.

    Mirrors `causal_ai_av.dataset.CausalAVDataset._resolve_annotation_paths`
    in spirit (dir vs file mode) but additionally accepts video files. Order
    is `sorted(paths)` per source, sources processed in the order the user
    supplied them. Duplicate clip_ids are emitted in the returned list — the
    caller (Step 3 io_adapter) is responsible for merging.
    """
    rows: list[tuple[str, SourceKind, Path]] = []
    for source in sources:
        for path in _expand_source(source):
            kind = _classify(path)
            if kind is None:
                logging.debug("skipping unrecognized file: %s", path)
                continue
            clip_id = _resolve_clip_id(path, kind)
            if clip_id is None:
                continue
            rows.append((clip_id, kind, path))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=_resolve_log_level(args).upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        rows = _resolve_sources(args.sources)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Phase 1: print and exit. Server wiring lands in Step 3.
    for clip_id, kind, path in rows:
        print(f"{clip_id}\t{kind}\t{path}")
    print(f"# resolved {len(rows)} source row(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
