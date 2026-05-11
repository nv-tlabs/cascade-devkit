"""Command-line entry point for `causal-av-annotate`.

Resolves source paths into a clip index, builds the FastAPI app, and runs
uvicorn. Video probing (real fps/duration) lands in Phase 3.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import webbrowser
from pathlib import Path

# Recognized video extensions (kept in sync with `io_adapter.VIDEO_EXTS`).
VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".mkv", ".mov", ".avi"})


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


def _resolve_destination_dir(sources: list[Path]) -> Path:
    """The directory where fresh (unlabelled) clips get their first JSON.

    Convention: first source if it's a directory; otherwise the parent of
    the first source file.
    """
    first = sources[0]
    return first if first.is_dir() else first.parent


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    log_level = _resolve_log_level(args)
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Defer heavy imports until past argparse so `--help` stays snappy.
    from annotator.server.app import create_app
    from annotator.server.io_adapter import build_clip_index

    try:
        clip_index = build_clip_index(args.sources)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    destination_dir = _resolve_destination_dir(args.sources)
    logging.info(
        "indexed %d clip(s); destination dir for fresh JSONs: %s",
        len(clip_index),
        destination_dir,
    )

    app = create_app(
        clip_index,
        read_only=args.read_only,
        destination_dir=destination_dir,
    )

    if not args.no_browser:
        url = f"http://{args.host}:{args.port}/"
        # Fire the browser open ~1s after uvicorn.run starts so the server is
        # ready by the time the browser sends its first GET. Daemonized so a
        # quick Ctrl-C before the timer fires doesn't block shutdown.
        timer = threading.Timer(1.0, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=log_level,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
