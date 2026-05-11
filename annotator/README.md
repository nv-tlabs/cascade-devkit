# causal-av-annotator

Local annotation tool for the AV Causal Dataset. A slim FastAPI server
hosts a vendored React/Vite frontend; you point it at a directory of
annotation JSONs and/or videos and edit them in the browser. No auth,
no database, no admin layer.

## Install

```bash
# From the repo root:
uv sync --extra annotator
cd annotator/web && npm install && npm run build
```

For HEVC video playback you also need `ffmpeg` + `ffprobe` on PATH:

```bash
sudo apt install ffmpeg     # Ubuntu/Debian
brew install ffmpeg         # macOS
```

Without ffmpeg, the JSON editing flow still works in full; only video
playback degrades (the `/api/clips/{id}/video` route 503s and the
`<video>` element stays blank).

## Launch

```bash
uv run causal-av-annotate /path/to/annotations
```

Then open `http://127.0.0.1:8765/`.

Common variants:

```bash
# Start from a directory of videos (no JSONs yet) — unlabelled clips
# appear in the sidebar. The first save creates <dir>/<clip_id>.json.
uv run causal-av-annotate /path/to/videos

# Local video source (HF dataset not installed):
uv run causal-av-annotate \
    --video-source local --video-dir /path/to/videos \
    /path/to/annotations

# HF dataset (requires the [hf] extra):
uv sync --extra hf
uv run causal-av-annotate --video-source hf /path/to/annotations

# Browse without write access:
uv run causal-av-annotate --read-only /path/to/annotations
```

Options:

- `--host`, `--port` — bind address (default `127.0.0.1:8765`).
- `--read-only` — refuse PUT to annotation routes.
- `--no-browser` — don't auto-open a browser window.
- `--video-source {auto,hf,local}` — where to source video bytes.
  `auto` (default) tries `hf` first, falls back to `local`.
- `--video-dir DIR` — when `--video-source local`, the directory
  containing video files (`.mp4`, `.mkv`, `.mov`, `.avi`).
- `--hf-source PATH` — optional annotations dir/repo passed to
  `CausalAVDataset` when `--video-source hf`.

`sources` can be one or more directories or files. Directories are
recursed for `*.json` and known video extensions. Video-only clips
(no JSON yet) appear as "unlabelled" in the sidebar; the first save
creates a fresh JSON in the source directory.

## Lock model

Editing is **default-locked** to make accidental edits hard:

- Every clip switch re-locks. You explicitly toggle the unlock button
  (with a confirm dialog) before any field accepts input.
- `--read-only` permanently hides the unlock toggle.
- The lock state is per-clip-session; locking is non-destructive.

## Save model

Saves are **explicit** — there is no autosave.

- Click **Save** in the right-panel header, or hit `Cmd/Ctrl+S`.
- A successful save writes to disk via `os.replace` (atomic rename),
  so an interrupted process can never produce a half-written JSON.
- On the *first* save for a clip in this server session, the prior
  on-disk file is copied to `<path>.json.bak`. Subsequent saves in the
  same session do not re-stamp the backup, so your original input
  survives a chain of regressions.
- Switching clips with unsaved changes opens a confirm dialog:
  *Save / Discard / Cancel*. Save is disabled while the bundle is
  locked or the server is read-only.
- The browser's native "leave site?" prompt fires on close while
  dirty.
- Undo restores the previous annotation snapshot. If undo lands you
  back at the on-disk state, `dirty` clears automatically.

## Fresh-annotation flow

Point the CLI at a directory of videos with no JSONs:

```bash
uv run causal-av-annotate /path/to/videos
```

- Each video becomes an "unlabelled" clip in the sidebar.
- Selecting it loads a minimum-valid bundle (schema 2.0.0,
  `status="annotating"`).
- The first save creates `<dir>/<clip_id>.json`; the clip flips to
  "annotated" in the sidebar.

## Architecture

```
annotator/
  src/annotator/server/
    cli.py        # argparse + uvicorn entry point
    app.py        # FastAPI app, /api/* routes
    io_adapter.py # disk ↔ AnnotationBundle (with .bak snapshot)
    video.py      # HEVC→H.264 transcode + HF/local resolver
  web/
    src/          # React/Vite frontend (zustand store, Radix dialogs)
    dist/         # vite build output, mounted at /
  tests/          # pytest — server + video routes + smoke test
```

## Tests

```bash
# From the repo root:
uv run pytest annotator/
```

The video tests mock `subprocess.run` / `shutil.which` so real
`ffmpeg` is never invoked in CI.
