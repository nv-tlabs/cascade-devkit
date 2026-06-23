# query_labeler

Local web tool for labeling DSL query matches as true / false positives.

Given a queries JSONL file (one query per line, e.g. `cascade_scenarios.jsonl`)
and a corpus of annotation JSONs, the tool runs each query, lists its matching
clips, plays the dashcam video alongside an annotation summary, and lets you
click **True positive** / **False positive** / **Skip** per clip. Labels are
appended to a sibling JSONL keyed by `(query, clip_id)`.

## Quick start

```bash
cd /path/to/cascade-devkit
set -a && . ./.env && set +a   # provides HF_TOKEN

uv run python tools/query_labeler/app.py \
    --queries /home/horde/cascade_scenarios.jsonl \
    --corpus  /home/horde/ego_centric_annotations
```

The server binds to `http://127.0.0.1:8766/` (port `8766` so it doesn't clash
with the annotator on `8765`). Open the URL in a browser.

## Inputs

- `--queries <path>`: a JSONL file. Each row must include at least a `query`
  field (the DSL query string). Other fields surfaced in the UI:
  `description`, `scenario_name`, `theme`, `total`, `b1`/`b2`/`b3`.
- `--corpus <path>`: a directory of annotation JSONs (recursed). For the
  Ego-centric work, this is `/home/horde/ego_centric_annotations` containing
  `batch_1/`, `batch_2/`, `batch_3/`.

## Outputs

- `--labels <path>` (default: `<queries-path>.labels.jsonl`): one row per
  labeling event, `{id, query, clip_id, label, labeler, ts, notes?}`. The
  newest row per `(query, clip_id)` wins. Append-only, so the file doubles
  as an audit log.

## Video resolution

Videos are fetched + transcoded by the existing `annotator.server.video.VideoResolver`:

- `--video-source=auto` (default): HuggingFace if available, falls back to local.
- `--video-source=hf`: always pull from HuggingFace (needs `HF_TOKEN` for the
  gated `nvidia/PhysicalAI-Autonomous-Vehicles` repo).
- `--video-source=local --video-dir <dir>`: read MP4s from a local directory.

First view of a clip triggers an HF fetch (~few hundred MB) and a few seconds
of HEVC → H.264 transcode. Subsequent views are instant. The transcode cache
lives at `~/.cache/cascade-annotator/`.

## Keyboard shortcuts

- `T` — true positive
- `F` — false positive
- `S` — skip
- `←` / `J` — previous clip
- `→` / `K` / `Space` — next clip

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Server status + counts |
| GET | `/api/queries` | List queries + per-query labeled counts |
| GET | `/api/query/{idx}/matches` | Matches for one query + per-clip metadata |
| GET | `/api/clip/{clip_id}/video` | Stream the canonical-camera MP4 |
| GET | `/api/clip/{clip_id}/video/status` | Resolve-pipeline stage probe |
| POST | `/api/labels` | Append a label `{query, clip_id, label}` |
| GET | `/api/labels` | List all labels (latest per pair) |

## Implementation notes

- Queries + bundles are loaded eagerly at startup. Matches are pre-computed
  in memory so the UI snaps when switching queries. For ~2000 queries × 300
  bundles this takes a few seconds; the labeler is a long-running tool,
  amortized that's fine.
- The frontend is a single `static/index.html` file (vanilla JS, no build
  step). Two panes: query list on the left, current match on the right.
- The labels file is JSONL append-only — every click is one row. Reloading
  the tool reads back the latest row per `(query, clip_id)` and renders the
  per-query progress.
