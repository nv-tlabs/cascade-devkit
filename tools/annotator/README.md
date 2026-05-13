# cascade-annotator

Local annotation tool for the CASCADE dataset. A slim FastAPI server
hosts a vendored React/Vite frontend; you point it at a directory of
annotation JSONs and/or videos and edit them in the browser. No auth,
no database, no admin layer.

## Install

Three steps. All three are required for the full experience; only the
first two are needed if you don't care about video playback.

**1. System packages — `ffmpeg` + `ffprobe`.** The corpus is HEVC and
browsers can't play HEVC natively, so the annotator transcodes to H.264
on first request. `ffprobe` is also used to detect codec on every video
request, so both binaries must be on PATH.

```bash
sudo apt install -y ffmpeg     # Ubuntu / Debian (Horde DGXC VMs)
brew install ffmpeg            # macOS
```

If ffmpeg is missing, JSON editing still works; only `/api/clips/{id}/video`
degrades and the player shows a "Video unavailable" overlay with the
install hint.

**2. Python dependencies.** From the repo root:

```bash
uv sync --extra annotator
```

For HuggingFace-sourced video (the default), also install the `[hf]`
extra so the parent Physical AI dataset is reachable:

```bash
uv sync --extra annotator --extra hf
```

**3. Frontend build.** One-time; rebuild after pulling UI changes:

```bash
cd tools/annotator/web && npm install && npm run build
```

## Launch

```bash
uv run cascade-annotate /path/to/annotations
```

Then open `http://127.0.0.1:8765/`.

Common variants:

```bash
# Start from a directory of videos (no JSONs yet) — unlabelled clips
# appear in the sidebar. The first save creates <dir>/<clip_id>.json.
uv run cascade-annotate /path/to/videos

# Local video source (HF dataset not installed):
uv run cascade-annotate \
    --video-source local --video-dir /path/to/videos \
    /path/to/annotations

# HF dataset (requires the [hf] extra):
uv sync --extra hf
uv run cascade-annotate --video-source hf /path/to/annotations

# Browse without write access:
uv run cascade-annotate --read-only /path/to/annotations
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
  `CascadeDataset` when `--video-source hf`.

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
uv run cascade-annotate /path/to/videos
```

- Each video becomes an "unlabelled" clip in the sidebar.
- Selecting it loads a minimum-valid bundle (schema 2.0.0,
  `status="annotating"`).
- The first save creates `<dir>/<clip_id>.json`; the clip flips to
  "annotated" in the sidebar.

## Architecture

```
tools/annotator/
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
uv run pytest tools/annotator/
```

The video tests mock `subprocess.run` / `shutil.which` so real
`ffmpeg` is never invoked in CI.
