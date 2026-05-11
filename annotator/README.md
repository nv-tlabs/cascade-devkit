# causal-av-annotator

Local annotation tool for the AV Causal Dataset. A slim FastAPI server hosts a
vendored React/Vite frontend; you point it at a directory of annotation JSONs
(and/or videos) and edit them in the browser.

## Install

```bash
# From the repo root:
uv sync --extra annotator
cd annotator/web && npm install && npm run build
```

## Launch

```bash
uv run causal-av-annotate /path/to/annotations
```

Then open `http://127.0.0.1:8765/`.

Options:

- `--read-only` — refuse PUT to annotation routes.
- `--no-browser` — don't auto-open a browser window.
- `--host`, `--port` — bind address (default `127.0.0.1:8765`).
- `--video-source {auto,hf,local}` — where to source video frames (wired
  in Phase 3).
- `--video-dir DIR` — when `--video-source local`, where to find videos.

`sources` can be one or more directories or files. Directories are recursed
for `*.json` and known video extensions (`.mp4`, `.mkv`, `.mov`, `.avi`).
Video-only clips (no JSON yet) appear as "unlabelled" — the first save
creates a fresh JSON in the source's parent directory.

## Lock & save

Editing is **default-locked**. The right-panel lock must be explicitly
toggled off before any field accepts input; toggling it back on (or
clicking Save) writes to disk via atomic rename (no autosave). The UI
prompts for confirmation before unlocking. This is deliberate: the
dataset is ground truth, so we make accidental edits hard.
