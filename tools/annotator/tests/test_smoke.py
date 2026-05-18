# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test: spawn the CLI in a subprocess, hit a few
routes, do an edit round-trip via the HTTP API.

This test is intentionally heavier than the rest of the suite — it
spawns uvicorn for real — but it catches regressions that unit-level
TestClient stubs miss (wiring between cli.py → create_app → uvicorn,
startup logs, port binding).
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from cascade_av.spec import AnnotationBundle, VideoMeta


def _free_port() -> int:
    """Bind to port 0 to discover a free TCP port, close, and return it."""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_ready(url: str, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.25)
    return False


@pytest.fixture()
def smoke_corpus(tmp_path: Path) -> Path:
    """One bundle saved as JSON under tmp_path."""
    clip_id = "smoke-clip-001"
    bundle = AnnotationBundle(
        schema_version="2.1.0",
        video=VideoMeta(clip_id=clip_id),
        status="annotating",
    )
    path = tmp_path / f"{clip_id}.json"
    path.write_text(bundle.model_dump_json(by_alias=True))
    return tmp_path


def test_smoke_round_trip(smoke_corpus: Path, tmp_path: Path) -> None:
    """Spawn the server in a subprocess; hit /api/health, /api/clips,
    GET annotations, then PUT a mutation and reload from disk."""
    port = _free_port()
    base = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    # Keep tests deterministic — disable browser auto-open.
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "annotator.server.cli",
            str(smoke_corpus),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--no-browser",
            "--log-level", "warning",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    try:
        ready = _wait_for_ready(f"{base}/api/health", timeout=15.0)
        if not ready:
            # Give the user a useful error if the boot failed.
            proc.terminate()
            try:
                _, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                _, stderr = proc.communicate()
            pytest.fail(
                f"server failed to come up within timeout. "
                f"stderr:\n{stderr.decode(errors='replace')}"
            )

        # /api/health
        with urllib.request.urlopen(f"{base}/api/health") as r:
            health = json.loads(r.read())
        assert health["ok"] is True
        assert health["clip_count"] == 1

        # /api/clips
        with urllib.request.urlopen(f"{base}/api/clips") as r:
            clips = json.loads(r.read())
        assert len(clips) == 1
        clip_id = clips[0]["clip_id"]

        # GET annotations
        with urllib.request.urlopen(
            f"{base}/api/clips/{clip_id}/annotations"
        ) as r:
            bundle = json.loads(r.read())
        assert bundle["video"]["clip_id"] == clip_id

        # PUT a mutation
        bundle["status"] = "needs_revision"
        put = urllib.request.Request(
            f"{base}/api/clips/{clip_id}/annotations",
            data=json.dumps(bundle).encode(),
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        with urllib.request.urlopen(put) as r:
            saved = json.loads(r.read())
        assert saved["bundle"]["status"] == "needs_revision"

        # Reload from disk → mutation persisted.
        on_disk = json.loads((smoke_corpus / f"{clip_id}.json").read_text())
        assert on_disk["status"] == "needs_revision"

    finally:
        proc.terminate()
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
