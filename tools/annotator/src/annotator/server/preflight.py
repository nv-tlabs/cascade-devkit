# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Startup environment checks for `cascade-annotate`.

The preflight surface answers a question that today's startup banner does
not: *can this annotator process actually do its job?* Before the
overlay-rendered "video resolve" pipeline gets a chance to surface
problems by 401-ing or crashing at request time, this module reports —
at server start — which of {ffmpeg, ffprobe, HF auth, .env, video
source} are present, missing, or misconfigured.

Every check is a pure function. The aggregator
:func:`run_preflight` returns a flat ``list[CheckResult]`` so tests can
assert against the structured output and the CLI can decide how to log
each line. Failures don't abort — the user might still want to launch
in a partially-broken state (e.g. local-only video without HF auth).
The contract is: print enough that the user knows what's wrong before
they click anything, then start uvicorn.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckResult:
    """One row in the preflight report.

    ``ok=False`` rows render with a ✗ glyph at WARNING level so they
    stand out in the launcher's log stream.
    """

    name: str
    ok: bool
    detail: str


def check_ffmpeg(*, which: Callable[[str], str | None] = shutil.which) -> CheckResult:
    """`ffmpeg` is required for transcoding non-H.264 video into a
    browser-playable stream. Without it, the video pipeline falls back to
    raw streaming, which fails the moment a clip is HEVC."""
    path = which("ffmpeg")
    if path is None:
        return CheckResult(
            "ffmpeg",
            False,
            "not found on PATH — install via apt/brew/conda or video transcoding will fail",
        )
    try:
        proc = subprocess.run(  # noqa: S603 — fixed binary, no user input
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        first_line = (proc.stdout or "").split("\n", 1)[0].strip() or path
        return CheckResult("ffmpeg", True, first_line)
    except (subprocess.TimeoutExpired, OSError):
        return CheckResult("ffmpeg", True, f"present at {path} (version probe failed)")


def check_ffprobe(*, which: Callable[[str], str | None] = shutil.which) -> CheckResult:
    """`ffprobe` is required for clip metadata (duration, fps). It ships
    in the same package as `ffmpeg` so the only realistic failure mode
    is a stripped install."""
    path = which("ffprobe")
    if path is None:
        return CheckResult(
            "ffprobe",
            False,
            "not found on PATH (usually bundled with ffmpeg) — clip duration probing will fail",
        )
    return CheckResult("ffprobe", True, path)


def check_hf_auth(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> CheckResult:
    """Either ``HF_TOKEN`` in the env or a ``huggingface-cli`` token on
    disk lets us reach gated repos. The annotator's parent dataset
    (``nvidia/PhysicalAI-Autonomous-Vehicles``) is gated, so a missing
    token surfaces as a generic 401 on the very first video request —
    well after the user has lost their place."""
    env = env if env is not None else os.environ
    if env.get("HF_TOKEN"):
        return CheckResult("hf_auth", True, "HF_TOKEN present in environment")
    home = home if home is not None else Path.home()
    token_file = home / ".cache" / "huggingface" / "token"
    if token_file.is_file():
        return CheckResult("hf_auth", True, f"huggingface-cli token at {token_file}")
    return CheckResult(
        "hf_auth",
        False,
        "no HF_TOKEN and no huggingface-cli login — gated repos (e.g. "
        "nvidia/PhysicalAI-Autonomous-Vehicles) will 401",
    )


def check_dotenv(dotenv_path: Path | None) -> CheckResult:
    """Whether `_load_dotenv` found and loaded a file. Not a failure
    when absent — many environments (CI, containers, hgx) inject
    secrets directly into the process env."""
    if dotenv_path is None:
        return CheckResult(
            "dotenv",
            True,
            "no .env in working tree (using shell environment only)",
        )
    return CheckResult("dotenv", True, f"loaded {dotenv_path}")


def check_video_source(
    mode: str,
    video_dir: Path | None,
    *,
    hf_auth_ok: bool,
) -> CheckResult:
    """Cross-reference ``--video-source`` against what's actually
    available. Catches the three common shapes of misconfiguration:

    * ``--video-source=local`` without ``--video-dir`` (or pointing at
      a non-directory).
    * ``--video-source=hf`` without HF auth (every request will 401).
    * ``--video-source=auto`` with neither HF auth nor ``--video-dir``
      (no source can succeed).
    """
    if mode == "local":
        if video_dir is None:
            return CheckResult(
                "video_source",
                False,
                "--video-source=local requires --video-dir",
            )
        if not video_dir.is_dir():
            return CheckResult(
                "video_source",
                False,
                f"--video-source=local but {video_dir} is not a directory",
            )
        return CheckResult("video_source", True, f"local from {video_dir}")
    if mode == "hf":
        if not hf_auth_ok:
            return CheckResult(
                "video_source",
                False,
                "--video-source=hf but no HF auth — every video request will 401",
            )
        return CheckResult("video_source", True, "hf (Physical AI AV dataset)")
    # mode == "auto" (argparse `choices=` guarantees one of the three).
    detail = "auto (hf preferred, falls back to local)"
    if not hf_auth_ok and video_dir is None:
        return CheckResult(
            "video_source",
            False,
            f"{detail} — neither HF auth nor --video-dir; no source will succeed",
        )
    return CheckResult("video_source", True, detail)


def run_preflight(
    *,
    video_source: str,
    video_dir: Path | None,
    dotenv_path: Path | None,
) -> list[CheckResult]:
    """Build the full preflight report in display order.

    Order: dotenv (closest to user intent) → ffmpeg/ffprobe (binaries the
    pipeline depends on) → hf_auth (credentials) → video_source (the
    cross-reference). The CLI logs them top-to-bottom so the user sees a
    natural narrative of "what did the launcher see, in the order the
    pipeline will touch it."
    """
    results: list[CheckResult] = [
        check_dotenv(dotenv_path),
        check_ffmpeg(),
        check_ffprobe(),
    ]
    hf = check_hf_auth()
    results.append(hf)
    results.append(
        check_video_source(video_source, video_dir, hf_auth_ok=hf.ok)
    )
    return results


def log_preflight(results: list[CheckResult]) -> None:
    """Emit one log line per check; failures route through WARNING so
    they show up in any default-level filter. Ships its own banner line
    so the report is visually grouped under a single header."""
    logger.info("preflight:")
    for r in results:
        glyph = "✓" if r.ok else "✗"
        line = f"  {glyph} {r.name}: {r.detail}"
        if r.ok:
            logger.info(line)
        else:
            logger.warning(line)
