---
name: annotator-dev
description: Launch the annotator backend, rebuild the React frontend, and diagnose common dev-loop issues (blank page, port collision, video 503, locked edits).
---

# Develop on the annotator

The annotator is a slim FastAPI server hosting a vendored Vite + React
frontend. Source under `tools/annotator/`; conventions in
`tools/annotator/AGENTS.md`.

## Procedure

1. From the repo root, ensure dependencies are current:
   ```bash
   make install
   ```
2. Launch the backend pointed at a data directory:
   ```bash
   make annotator-dev DATA=~/01_json_annotations
   ```
   `DATA` is optional when `CASCADE_AV_DATASET_ROOT` is set in `.env`
   (Makefile sources `.env` and falls back to that var). The target
   exits 2 only if neither is supplied. `annotator-dev` always
   rebuilds the frontend bundle (~2-3s) before launching uvicorn, so
   the UI you load is always in sync with the source on disk — no
   "did my fix not land?" moments from a stale `dist/`.
3. Open <http://127.0.0.1:8765/>. The page loads without flicker
   because theme hydration runs synchronously in `main.tsx` before
   React mounts.
4. After editing UI source files, just relaunch `annotator-dev` —
   it rebuilds every time. Reach for `make annotator-build` only
   when you want to compile the bundle without booting the backend
   (e.g. to confirm `tsc -b` passes after a refactor).

## Troubleshooting

- **Page is blank / "Frontend dist not found" in logs** — the React
  bundle isn't built. Should not happen on `make annotator-dev` (it
  rebuilds every launch); only fires if the user invoked
  `uv run cascade-annotate` directly. Run `make annotator-build`.
- **Port 8765 already in use** — another process owns the default
  port. Override via the target's optional `PORT=` variable:
  `make annotator-dev DATA=~/01_json_annotations PORT=20000`
  (equivalent to
  `uv run cascade-annotate $DATA --port 20000`).
- **Video plays as a black square / 503 on
  `/api/clips/<id>/video`** — `ffmpeg` or `ffprobe` isn't on PATH.
  Install with `sudo apt install -y ffmpeg` (Ubuntu / Horde DGXC).
  The annotator transcodes HEVC → H.264 on first request.
- **Edits do nothing in the UI** — the bundle is locked
  (default-locked policy, per clip session). Click the unlock toggle
  in the lock bar; confirm in the dialog.
- **"Cannot save while locked" on a clip switch** — by design.
  Unlock first, save explicitly (`Cmd/Ctrl+S` or the Save button),
  then switch.
- **Edits land in the wrong file** — the first save in a server
  session stamps `<file>.json.bak` (atomic via `os.replace`).
  Restore from the `.bak` if needed; do not delete `.bak` files
  from inside the running app.

## Hard "don't"s

- Do **not** re-introduce a global `* { padding: 0 }` reset in
  `tools/annotator/web/src/index.css`. See commit `9ef41f5`.
- Do **not** add autosave or remove the explicit-save model.
- Do **not** bypass `os.replace` for disk writes — atomicity is a
  load-bearing invariant.
- Do **not** move dirty-tracking out of the Zustand store. The
  store is the single source of truth for
  `dirty` / `locked` / `serverReadOnly` / `saveError`.
