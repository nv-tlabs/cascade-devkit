# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `annotator.server.cli._load_dotenv`.

The helper bridges the gap between the documented "fill in `.env`" step and
the running server: `make annotator-dev` does not source dotenv itself, so
without this autoload an `HF_TOKEN` in the file never reaches the
HuggingFace client and clip resolves 401 against gated repos.

Tests use the explicit ``start=`` parameter (no cwd manipulation) so the
suite stays parallel-safe.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from annotator.server.cli import _load_dotenv


def test_load_dotenv_returns_none_when_file_absent(tmp_path: Path) -> None:
    """A missing `.env` is normal in CI / production; not an error."""
    assert _load_dotenv(start=tmp_path) is None


def test_load_dotenv_populates_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `.env` with one assignment lands in `os.environ`."""
    (tmp_path / ".env").write_text("CASCADE_AV_TEST_TOKEN=from-dotenv\n")
    monkeypatch.delenv("CASCADE_AV_TEST_TOKEN", raising=False)

    out = _load_dotenv(start=tmp_path)
    assert out == tmp_path / ".env"
    assert os.environ["CASCADE_AV_TEST_TOKEN"] == "from-dotenv"


def test_load_dotenv_does_not_override_shell_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The contract: explicit shell exports win over file values.

    Rationale: a CI runner / `hgx` secret / per-command `VAR=… make …`
    should not be silently shadowed by a stale `.env` entry. If a user
    truly wants the file value, they delete the shell var first.
    """
    monkeypatch.setenv("CASCADE_AV_TEST_TOKEN", "from-shell")
    (tmp_path / ".env").write_text("CASCADE_AV_TEST_TOKEN=from-dotenv\n")

    out = _load_dotenv(start=tmp_path)
    assert out == tmp_path / ".env"
    assert os.environ["CASCADE_AV_TEST_TOKEN"] == "from-shell"


def test_load_dotenv_fills_only_missing_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`.env` should top up the environment, not replace it.

    Shell sets ``A=shell``; `.env` lists both A and B; after load,
    ``A`` is still ``shell`` (precedence) and ``B`` is from the file.
    """
    monkeypatch.setenv("CASCADE_AV_TEST_A", "shell")
    monkeypatch.delenv("CASCADE_AV_TEST_B", raising=False)
    (tmp_path / ".env").write_text(
        "CASCADE_AV_TEST_A=dotenv\nCASCADE_AV_TEST_B=from-dotenv\n"
    )

    _load_dotenv(start=tmp_path)
    assert os.environ["CASCADE_AV_TEST_A"] == "shell"
    assert os.environ["CASCADE_AV_TEST_B"] == "from-dotenv"


def test_load_dotenv_ignores_commented_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Commented `#HF_TOKEN=` lines (the `.env.example` default) must not
    leak the empty string into the process — pollutes downstream checks."""
    monkeypatch.delenv("CASCADE_AV_TEST_COMMENTED", raising=False)
    (tmp_path / ".env").write_text("#CASCADE_AV_TEST_COMMENTED=value\n")

    _load_dotenv(start=tmp_path)
    assert "CASCADE_AV_TEST_COMMENTED" not in os.environ


def test_load_dotenv_with_directory_lacking_env_returns_none(
    tmp_path: Path,
) -> None:
    """`start=<dir>` looks at exactly that directory, not its ancestors —
    so an explicit start of a clean tmp dir yields ``None`` even if the
    parent has a `.env`. (The cwd-walk-up path is exercised by main()'s
    integration in real use.)"""
    (tmp_path / "child").mkdir()
    # No .env in tmp_path / "child"
    assert _load_dotenv(start=tmp_path / "child") is None
