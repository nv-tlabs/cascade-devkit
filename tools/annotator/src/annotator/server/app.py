# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI app for the cascade-annotator local server.

Routes:
  - GET  /api/health
  - GET  /api/clips
  - GET  /api/clips/{clip_id}/annotations
  - PUT  /api/clips/{clip_id}/annotations
  - GET  /api/clips/{clip_id}/video           (Phase 3)

The frontend dist is mounted at `/` (SPA fallback). If the dist directory
does not exist, the server logs a warning and only the `/api/*` routes
respond.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from cascade_av.spec import AnnotationBundle

from annotator.server.io_adapter import ClipEntry, load_bundle, save_bundle
from annotator.server.video import (
    TranscodeError,
    VideoNotFound,
    VideoResolver,
    VideoToolsMissing,
)

LOG = logging.getLogger(__name__)

# `tools/annotator/web/dist` resolved relative to this file:
#   tools/annotator/src/annotator/server/app.py  →  tools/annotator/web/dist
_DEFAULT_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


def create_app(
    clip_index: dict[str, ClipEntry],
    *,
    read_only: bool,
    destination_dir: Path,
    dist_dir: Path | None = None,
    video_resolver: VideoResolver | None = None,
) -> FastAPI:
    """Construct the FastAPI app bound to this in-memory clip index."""
    app = FastAPI(title="cascade-annotator", version="0.1.0")

    # ----- API routes -----

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "clip_count": len(clip_index),
            "read_only": read_only,
        }

    @app.get("/api/clips")
    def list_clips() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for clip_id in sorted(clip_index.keys()):
            entry = clip_index[clip_id]
            out.append({
                "clip_id": clip_id,
                "kind": entry.kind,
                "has_video": entry.video_path is not None,
            })
        return out

    @app.get("/api/clips/{clip_id}/annotations")
    def get_annotations(clip_id: str) -> dict[str, Any]:
        if clip_id not in clip_index:
            raise HTTPException(status_code=404, detail=f"unknown clip_id: {clip_id}")
        bundle = load_bundle(clip_index[clip_id])
        return bundle.model_dump(by_alias=True, exclude_unset=True, mode="json")

    @app.put("/api/clips/{clip_id}/annotations")
    async def put_annotations(clip_id: str, request: Request) -> JSONResponse:
        if clip_id not in clip_index:
            raise HTTPException(status_code=404, detail=f"unknown clip_id: {clip_id}")
        if read_only:
            return JSONResponse(
                status_code=409,
                content={"error": "server is read-only"},
            )
        payload = await request.json()
        try:
            bundle = AnnotationBundle.model_validate(payload)
        except ValidationError as exc:
            return JSONResponse(status_code=422, content={"errors": exc.errors()})
        # Force clip_id alignment — the URL is authoritative.
        if bundle.video.clip_id != clip_id:
            return JSONResponse(
                status_code=422,
                content={
                    "error": (
                        f"video.clip_id {bundle.video.clip_id!r} does not match "
                        f"URL clip_id {clip_id!r}"
                    ),
                },
            )
        try:
            saved_path = save_bundle(clip_index[clip_id], bundle, destination_dir)
        except OSError as exc:
            LOG.exception("save failed for %s", clip_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return JSONResponse(
            status_code=200,
            content={
                "saved_to": str(saved_path),
                "bundle": bundle.model_dump(by_alias=True, exclude_unset=True, mode="json"),
            },
        )

    # ----- Video route (Phase 3) -----

    @app.get("/api/clips/{clip_id}/video")
    def get_video(clip_id: str) -> FileResponse:
        if clip_id not in clip_index:
            raise HTTPException(status_code=404, detail=f"unknown clip_id: {clip_id}")
        if video_resolver is None:
            raise HTTPException(
                status_code=503,
                detail="video source not configured on this server",
            )
        try:
            path = video_resolver.resolve(clip_id)
        except VideoToolsMissing as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except VideoNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except TranscodeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        # FileResponse populates Content-Length and handles HEAD; HTML5 <video>
        # in Chrome/Firefox accepts a single full response over the lifetime of
        # a local playback session, and seeks work via decoded-buffer scrubbing.
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    # ----- Static frontend mount -----

    dist = dist_dir if dist_dir is not None else _DEFAULT_DIST
    if dist.is_dir():
        index_path = dist / "index.html"

        # Serve index.html with no-cache headers so browsers don't pin a stale
        # HTML pointing at hashed asset names that have been replaced by a
        # fresh build. The hashed assets themselves under /assets/* are
        # content-addressed and safe to cache aggressively (StaticFiles default).
        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(
                index_path,
                media_type="text/html",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

        app.mount(
            "/",
            StaticFiles(directory=str(dist), html=True),
            name="frontend",
        )
    else:
        LOG.warning(
            "Frontend dist not found at %s — run `cd tools/annotator/web && npm run build`. "
            "Server will still serve /api/* routes.",
            dist,
        )

    return app
