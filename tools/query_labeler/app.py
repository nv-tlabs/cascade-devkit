# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local web tool for labeling DSL query matches as true / false positives.

Usage:

    uv run python tools/query_labeler/app.py \\
        --queries /home/horde/cascade_scenarios.jsonl \\
        --corpus /home/horde/ego_centric_annotations \\
        --labels /home/horde/cascade_scenarios.labels.jsonl

`--corpus` may be either a directory of annotation JSONs or a directory of
batch subdirectories (the file list is collected recursively).

Labels are appended to the labels JSONL append-only — every click is one
row `{query, clip_id, label, labeler, ts, notes?}`. The latest row per
`(query, clip_id)` wins.

Video playback reuses the annotator's `VideoResolver`, which transcodes
HEVC source clips from HuggingFace to browser-playable H.264 MP4 via the
ffmpeg cache at `~/.cache/cascade-annotator/`. First view of a clip
triggers an HF fetch (~few hundred MB) and a few seconds of transcode;
subsequent views are instant.
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from annotator.server.video import VideoResolver  # noqa: E402
from cascade_av.io import iter_dir  # noqa: E402
from cascade_av.query import find_on_bundle  # noqa: E402
from cascade_av.spec import AnnotationBundle  # noqa: E402

LOG = logging.getLogger("query_labeler")
HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"


# ---------------------------------------------------------------------------
# Bundle index — flat clip_id → AnnotationBundle map across the corpus.
# ---------------------------------------------------------------------------


def _iter_bundles(corpus_root: Path) -> Iterator[tuple[str, AnnotationBundle, Path]]:
    """Yield every (clip_id, bundle, path) under ``corpus_root``.

    Walks subdirectories so ``--corpus /home/horde/ego_centric_annotations``
    picks up batch_1/, batch_2/, batch_3/ in one shot.

    Some clips have an in-batch duplicate annotation (e.g.
    ``<clip>.json`` plus ``<clip>__<8hex>.json``). The audit showed the
    primary file is sometimes nearly empty while the suffixed file holds
    the substantive bundle. We collect everything first, then keep the
    *richest* bundle per clip_id, measured by annotation-content count
    (agents + ego actions + environments + conditions + lights + objs).
    """
    def richness(b: AnnotationBundle) -> int:
        ann = b.annotation
        return (
            len(ann.agents)
            + len(ann.ego_vehicle.actions)
            + len(ann.environments)
            + len(ann.conditions)
            + len(ann.traffic_lights)
            + len(ann.traffic_objects)
        )

    best: dict[str, tuple[int, AnnotationBundle, Path]] = {}
    for path, bundle in iter_dir(corpus_root, glob="**/*.json"):
        cid = bundle.video.clip_id
        score = richness(bundle)
        prev = best.get(cid)
        if prev is None or score > prev[0]:
            best[cid] = (score, bundle, path)
    for cid, (_, bundle, path) in best.items():
        yield cid, bundle, path


# ---------------------------------------------------------------------------
# Query evaluation
# ---------------------------------------------------------------------------


def _evaluate_query(query: str, bundles: list[AnnotationBundle]) -> list[str]:
    """Return the clip_ids whose bundle matches ``query``.

    Signature is `find_on_bundle(bundle, dsl_text)` (bundle first); MatchSet
    is non-empty iff at least one entity satisfied the predicate.
    """
    hits: list[str] = []
    for bundle in bundles:
        try:
            ms = find_on_bundle(bundle, query)
        except Exception as e:  # noqa: BLE001 — surface DSL errors to the caller
            raise HTTPException(status_code=400, detail=f"query error: {e}") from e
        if len(ms) > 0:
            hits.append(bundle.video.clip_id)
    return hits


# ---------------------------------------------------------------------------
# Labels JSONL — append-only persistence
# ---------------------------------------------------------------------------


def _read_labels(labels_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Return latest label per (query, clip_id). Empty dict if file is absent."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    if not labels_path.exists():
        return out
    for line in labels_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            LOG.warning("skipping malformed labels row: %r", line[:80])
            continue
        key = (row.get("query", ""), row.get("clip_id", ""))
        out[key] = row
    return out


def _append_label(labels_path: Path, row: dict[str, Any]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("a") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def create_app(
    *,
    queries_path: Path,
    corpus_root: Path,
    labels_path: Path,
    video_resolver: VideoResolver,
) -> FastAPI:
    app = FastAPI(title="cascade query labeler", version="0.1.0")

    # Eager load: queries + bundles. Both fit easily in memory.
    queries: list[dict[str, Any]] = [
        json.loads(line)
        for line in queries_path.read_text().splitlines()
        if line.strip()
    ]
    LOG.info("loaded %d queries from %s", len(queries), queries_path)

    bundles_by_clip: dict[str, AnnotationBundle] = {}
    for clip_id, bundle, _ in _iter_bundles(corpus_root):
        bundles_by_clip[clip_id] = bundle
    bundle_list = list(bundles_by_clip.values())
    LOG.info("loaded %d bundles from %s", len(bundle_list), corpus_root)

    # Per-query match cache. Eager-computed at startup so the UI snaps.
    matches_cache: dict[int, list[str]] = {}
    for i, q in enumerate(queries):
        try:
            matches_cache[i] = _evaluate_query(q["query"], bundle_list)
        except HTTPException as e:
            LOG.warning("query %d failed: %s — %s", i, q.get("query", "")[:60], e.detail)
            matches_cache[i] = []
    LOG.info(
        "match cache built: %d queries → %d total matches",
        len(matches_cache),
        sum(len(v) for v in matches_cache.values()),
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "queries": len(queries),
            "bundles": len(bundle_list),
            "labels_path": str(labels_path),
        }

    @app.get("/api/queries")
    def list_queries() -> list[dict[str, Any]]:
        labels = _read_labels(labels_path)
        out: list[dict[str, Any]] = []
        for i, q in enumerate(queries):
            cids = matches_cache.get(i, [])
            labeled = sum(1 for c in cids if (q["query"], c) in labels)
            out.append(
                {
                    "idx": i,
                    "query": q["query"],
                    "description": q.get("description", ""),
                    "scenario_name": q.get("scenario_name"),
                    "theme": q.get("theme", ""),
                    "total": q.get("total", len(cids)),
                    "matches": len(cids),
                    "labeled": labeled,
                }
            )
        return out

    @app.get("/api/query/{idx}/matches")
    def query_matches(idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(queries):
            raise HTTPException(status_code=404, detail=f"query index out of range: {idx}")
        q = queries[idx]
        clip_ids = matches_cache.get(idx, [])
        labels = _read_labels(labels_path)
        out_clips: list[dict[str, Any]] = []
        for cid in clip_ids:
            bundle = bundles_by_clip.get(cid)
            if bundle is None:
                continue
            ann = bundle.annotation
            ego_actions = [a.type for a in ann.ego_vehicle.actions if a.type]
            agent_types = sorted({a.type for a in ann.agents if a.type})
            light_colors = sorted(
                {
                    s.color
                    for tl in ann.traffic_lights
                    for h in tl.signal_heads
                    for s in h.state_sequence
                    if s.color
                }
            )
            obj_types = sorted({o.type for o in ann.traffic_objects if o.type})
            cond_types = sorted({t for c in ann.conditions for t in c.type if t})
            key = (q["query"], cid)
            label_row = labels.get(key)
            out_clips.append(
                {
                    "clip_id": cid,
                    "brief_description": ann.brief_description,
                    "duration_s": bundle.video.duration_s,
                    "fps": bundle.video.fps,
                    "ego_actions": ego_actions,
                    "agent_types": agent_types,
                    "light_colors": light_colors,
                    "obj_types": obj_types,
                    "cond_types": cond_types,
                    "label": label_row.get("label") if label_row else None,
                    "label_ts": label_row.get("ts") if label_row else None,
                    "labeler": label_row.get("labeler") if label_row else None,
                    "notes": label_row.get("notes") if label_row else None,
                }
            )
        return {
            "idx": idx,
            "query": q["query"],
            "description": q.get("description", ""),
            "scenario_name": q.get("scenario_name"),
            "theme": q.get("theme", ""),
            "clips": out_clips,
        }

    @app.get("/api/clip/{clip_id}/video")
    def get_video(clip_id: str) -> FileResponse:
        if clip_id not in bundles_by_clip:
            raise HTTPException(status_code=404, detail=f"unknown clip: {clip_id}")
        try:
            mp4_path = video_resolver.resolve(clip_id)
        except Exception as e:  # noqa: BLE001 — surface as HTTP 503
            raise HTTPException(status_code=503, detail=str(e)) from e
        return FileResponse(
            mp4_path,
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes"},
        )

    @app.get("/api/clip/{clip_id}/video/status")
    def get_video_status(clip_id: str) -> dict[str, str]:
        return video_resolver.get_state(clip_id)

    @app.post("/api/labels")
    async def post_label(request: Request) -> JSONResponse:
        body = await request.json()
        query = body.get("query")
        clip_id = body.get("clip_id")
        label = body.get("label")
        notes = body.get("notes")
        if not query or not clip_id or label not in ("tp", "fp", "skip"):
            raise HTTPException(
                status_code=400,
                detail="body must include {query, clip_id, label in (tp|fp|skip)}",
            )
        row = {
            "id": str(uuid.uuid4()),
            "query": query,
            "clip_id": clip_id,
            "label": label,
            "labeler": getpass.getuser(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if notes:
            row["notes"] = notes
        _append_label(labels_path, row)
        return JSONResponse({"ok": True, "row": row})

    @app.get("/api/labels")
    def list_labels() -> list[dict[str, Any]]:
        return list(_read_labels(labels_path).values())

    # --- static UI ---
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cascade-label",
        description="Local web tool for labeling DSL query matches as true/false positives.",
    )
    p.add_argument(
        "--queries", type=Path, required=True,
        help="path to a queries JSONL (e.g. /home/horde/cascade_scenarios.jsonl)",
    )
    p.add_argument(
        "--corpus", type=Path, required=True,
        help="path to a directory of annotation JSONs (recursed)",
    )
    p.add_argument(
        "--labels", type=Path, default=None,
        help="path to the labels JSONL (default: <queries>.labels.jsonl)",
    )
    p.add_argument(
        "--video-source", choices=("auto", "local", "hf"), default="auto",
        help="how to resolve clip videos (default: auto)",
    )
    p.add_argument(
        "--video-dir", type=Path, default=None,
        help="when --video-source=local, directory of MP4 files",
    )
    p.add_argument(
        "--host", default="127.0.0.1",
        help="bind host (default: 127.0.0.1)",
    )
    p.add_argument(
        "--port", type=int, default=8766,
        help="bind port (default: 8766 — annotator uses 8765)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)
    if not args.queries.exists():
        print(f"error: queries file not found: {args.queries}", file=sys.stderr)
        return 2
    if not args.corpus.exists():
        print(f"error: corpus dir not found: {args.corpus}", file=sys.stderr)
        return 2
    labels_path = args.labels or args.queries.with_suffix(args.queries.suffix + ".labels.jsonl")

    resolver = VideoResolver(
        mode=args.video_source,
        video_dir=args.video_dir,
    )

    app = create_app(
        queries_path=args.queries,
        corpus_root=args.corpus,
        labels_path=labels_path,
        video_resolver=resolver,
    )

    LOG.info("queries:   %s", args.queries)
    LOG.info("corpus:    %s", args.corpus)
    LOG.info("labels:    %s", labels_path)
    LOG.info("video src: %s", args.video_source)
    LOG.info("listening on http://%s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
