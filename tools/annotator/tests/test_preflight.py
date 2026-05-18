# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `annotator.server.preflight`.

Each check function takes its dependencies (PATH resolver, env mapping,
home directory) as keyword arguments, so the tests can drive the state
explicitly without monkeypatching globals. The integration test at the
bottom locks the `cli.main` → `run_preflight` → `log_preflight` wiring.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from annotator.server.preflight import (
    CheckResult,
    check_dotenv,
    check_ffmpeg,
    check_ffprobe,
    check_hf_auth,
    check_video_source,
    log_preflight,
    run_preflight,
)


# -----------------------------------------------------------------------------
# check_ffmpeg / check_ffprobe
# -----------------------------------------------------------------------------

def test_check_ffmpeg_missing_returns_failure() -> None:
    r = check_ffmpeg(which=lambda _name: None)
    assert r.name == "ffmpeg"
    assert r.ok is False
    assert "not found" in r.detail.lower()


def test_check_ffmpeg_present_runs_version_probe() -> None:
    """When `ffmpeg -version` succeeds, the first stdout line becomes the detail.

    The check shells out to the binary the resolver returned; we mock
    subprocess.run so the test is hermetic and doesn't depend on the host's
    actual ffmpeg version string.
    """
    fake_stdout = "ffmpeg version 6.1.1 Copyright (c) 2000-2023 the FFmpeg developers\nbuilt with gcc\n"
    with patch(
        "annotator.server.preflight.subprocess.run",
        return_value=type("R", (), {"stdout": fake_stdout, "returncode": 0})(),
    ):
        r = check_ffmpeg(which=lambda _name: "/usr/bin/ffmpeg")
    assert r.ok is True
    assert "ffmpeg version 6.1.1" in r.detail


def test_check_ffmpeg_present_but_version_probe_crashes_still_passes() -> None:
    """A weird/stripped ffmpeg build that hangs or errors on `-version` shouldn't
    cause the preflight to claim it's missing — the binary IS on PATH; we just
    can't get a version string. Report ok but with a softer detail."""
    with patch(
        "annotator.server.preflight.subprocess.run",
        side_effect=OSError("nope"),
    ):
        r = check_ffmpeg(which=lambda _name: "/opt/weird/ffmpeg")
    assert r.ok is True
    assert "/opt/weird/ffmpeg" in r.detail


def test_check_ffprobe_missing_returns_failure() -> None:
    r = check_ffprobe(which=lambda _name: None)
    assert r.ok is False
    assert "ffprobe" in r.detail.lower() or "not found" in r.detail.lower()


def test_check_ffprobe_present_reports_path() -> None:
    r = check_ffprobe(which=lambda _name: "/usr/bin/ffprobe")
    assert r.ok is True
    assert "/usr/bin/ffprobe" in r.detail


# -----------------------------------------------------------------------------
# check_hf_auth
# -----------------------------------------------------------------------------

def test_check_hf_auth_uses_env_var_when_set(tmp_path: Path) -> None:
    r = check_hf_auth(env={"HF_TOKEN": "hf_secret123"}, home=tmp_path)
    assert r.ok is True
    assert "HF_TOKEN" in r.detail


def test_check_hf_auth_falls_back_to_cli_token_file(tmp_path: Path) -> None:
    """`huggingface-cli login` writes ~/.cache/huggingface/token; we accept
    either that or the env var. Set up the token file at the synthetic home."""
    token = tmp_path / ".cache" / "huggingface" / "token"
    token.parent.mkdir(parents=True)
    token.write_text("hf_xxxxxxxx")
    r = check_hf_auth(env={}, home=tmp_path)
    assert r.ok is True
    assert "huggingface-cli token" in r.detail


def test_check_hf_auth_no_token_anywhere_fails(tmp_path: Path) -> None:
    r = check_hf_auth(env={}, home=tmp_path)
    assert r.ok is False
    assert "401" in r.detail  # the user-facing hint
    assert "PhysicalAI" in r.detail  # name the gated repo so it's recognisable


def test_check_hf_auth_empty_string_token_does_not_count(tmp_path: Path) -> None:
    """`HF_TOKEN=""` in `.env.example` would otherwise read as truthy via the
    raw dict. Our check uses `env.get(...)` which returns "" → falsy. Lock it."""
    r = check_hf_auth(env={"HF_TOKEN": ""}, home=tmp_path)
    assert r.ok is False


# -----------------------------------------------------------------------------
# check_dotenv
# -----------------------------------------------------------------------------

def test_check_dotenv_absent_is_ok_with_explanation() -> None:
    r = check_dotenv(None)
    assert r.ok is True
    assert "no .env" in r.detail


def test_check_dotenv_present_records_path(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("X=1")
    r = check_dotenv(p)
    assert r.ok is True
    assert str(p) in r.detail


# -----------------------------------------------------------------------------
# check_video_source
# -----------------------------------------------------------------------------

def test_check_video_source_local_without_dir_fails() -> None:
    r = check_video_source("local", None, hf_auth_ok=True)
    assert r.ok is False
    assert "--video-dir" in r.detail


def test_check_video_source_local_with_nonexistent_dir_fails(tmp_path: Path) -> None:
    r = check_video_source("local", tmp_path / "missing", hf_auth_ok=True)
    assert r.ok is False
    assert "not a directory" in r.detail


def test_check_video_source_local_happy_path(tmp_path: Path) -> None:
    r = check_video_source("local", tmp_path, hf_auth_ok=False)
    assert r.ok is True
    assert str(tmp_path) in r.detail


def test_check_video_source_hf_without_auth_fails() -> None:
    r = check_video_source("hf", None, hf_auth_ok=False)
    assert r.ok is False
    assert "401" in r.detail


def test_check_video_source_hf_with_auth_passes() -> None:
    r = check_video_source("hf", None, hf_auth_ok=True)
    assert r.ok is True


def test_check_video_source_auto_with_neither_source_fails(tmp_path: Path) -> None:
    """`--video-source=auto` is the default; if a user doesn't set HF_TOKEN
    AND doesn't pass --video-dir, neither leg of the fallback can work.
    This is the silent footgun the preflight is built to surface."""
    r = check_video_source("auto", None, hf_auth_ok=False)
    assert r.ok is False
    assert "no source will succeed" in r.detail


def test_check_video_source_auto_with_hf_only_passes() -> None:
    r = check_video_source("auto", None, hf_auth_ok=True)
    assert r.ok is True


def test_check_video_source_auto_with_local_only_passes(tmp_path: Path) -> None:
    r = check_video_source("auto", tmp_path, hf_auth_ok=False)
    assert r.ok is True


# -----------------------------------------------------------------------------
# run_preflight (aggregator)
# -----------------------------------------------------------------------------

def test_run_preflight_emits_one_row_per_check(tmp_path: Path) -> None:
    """The aggregator's row count is the panel's contract — the CLI logs
    one line per result, so adding a check is an intentional UX change.
    Lock the surface so it can't drift accidentally."""
    results = run_preflight(
        video_source="auto",
        video_dir=None,
        dotenv_path=None,
    )
    names = [r.name for r in results]
    # dotenv, ffmpeg, ffprobe, hf_auth, video_source — 5 rows.
    assert names == ["dotenv", "ffmpeg", "ffprobe", "hf_auth", "video_source"]


def test_run_preflight_passes_hf_status_into_video_source_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When HF auth is absent, the video_source check sees that and the
    auto-mode-no-fallback row fails. This is the integration that makes the
    "silent 401" problem visible — verify the wiring."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no token file under here
    with patch("annotator.server.preflight.Path.home", return_value=tmp_path):
        results = run_preflight(
            video_source="auto",
            video_dir=None,
            dotenv_path=None,
        )
    hf = next(r for r in results if r.name == "hf_auth")
    vs = next(r for r in results if r.name == "video_source")
    assert hf.ok is False
    assert vs.ok is False  # cascade from hf failure
    assert "no source will succeed" in vs.detail


# -----------------------------------------------------------------------------
# log_preflight
# -----------------------------------------------------------------------------

def test_log_preflight_emits_warning_for_failures(caplog: pytest.LogCaptureFixture) -> None:
    """Failures route through WARNING so they survive default-INFO log filters
    AND so they're visually loud in the launcher output."""
    results = [
        CheckResult("ok_check", True, "fine"),
        CheckResult("bad_check", False, "broken"),
    ]
    with caplog.at_level(logging.DEBUG, logger="annotator.server.preflight"):
        log_preflight(results)
    levels_by_name = {r.message: r.levelname for r in caplog.records}
    # The header line is INFO; the ok row is INFO; the bad row is WARNING.
    assert any(
        msg.endswith("ok_check: fine") and lvl == "INFO"
        for msg, lvl in levels_by_name.items()
    )
    assert any(
        msg.endswith("bad_check: broken") and lvl == "WARNING"
        for msg, lvl in levels_by_name.items()
    )


def test_log_preflight_renders_glyphs_for_visual_scan(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A ✓ / ✗ prefix per row so users can scan the report at a glance."""
    with caplog.at_level(logging.DEBUG, logger="annotator.server.preflight"):
        log_preflight([
            CheckResult("a", True, "x"),
            CheckResult("b", False, "y"),
        ])
    messages = [r.message for r in caplog.records]
    assert any("✓ a:" in m for m in messages)
    assert any("✗ b:" in m for m in messages)
