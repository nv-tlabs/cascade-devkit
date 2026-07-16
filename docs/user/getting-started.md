# Getting started

A 5-minute walkthrough from a fresh clone to your first saved
annotation. If you already have the repo installed and just want the
annotator reference, jump to [`docs/user/annotator.md`](annotator.md).

This guide assumes Ubuntu 22.04+ / Debian Bookworm or the bundled
devcontainer. macOS works the same after the prerequisite step;
Windows is supported via WSL2.

---

## 1. Install

One line installs system prerequisites (apt + Node LTS + `uv`), clones
the repo, runs `make install`, and smoke-tests the Python side:

```bash
curl -LsSf https://raw.githubusercontent.com/nv-tlabs/cascade-devkit/main/scripts/install.sh | bash
```

Re-running is safe (idempotent). If you already have `uv`, Node 20+,
and `ffmpeg`/`ffprobe` on PATH, you can skip the bootstrap and just:

```bash
git clone https://github.com/nv-tlabs/cascade-devkit.git
cd cascade-devkit
make install
```

See [`README.md` § Install](../../README.md#install) for the manual
recipe.

## 2. Get a HuggingFace token

The parent dataset (`nvidia/PhysicalAI-Autonomous-Vehicles`) is gated.
Without a HuggingFace token, the annotator can't fetch video for the
clips you'll be editing — every `GET /video` request hits a 401.

1. Generate a token at <https://huggingface.co/settings/tokens>
   (the "Read" scope is sufficient).
2. Visit the dataset page and click **Request access** — approval is
   automatic for the public copy.
3. Provide the token to the annotator one of two ways:

   **Option A — `.env` file.** Copy the template and fill it in:

   ```bash
   cp .env.example .env
   $EDITOR .env   # uncomment HF_TOKEN=... and paste your token
   ```

   `cascade-annotate` walks up from the working directory at startup
   and auto-loads `.env`. Shell exports still win, so you can override
   per-session without touching the file.

   **Option B — `huggingface-cli login`.** Stores the token in
   `~/.cache/huggingface/token`; the annotator picks it up
   automatically:

   ```bash
   uv run huggingface-cli login
   ```

Either is enough. Don't do both with different tokens — pick one.

## 3. Launch the annotator

You need a directory of annotation JSONs to edit. Published bundles are
available in the [CASCADE dataset on Hugging Face](https://huggingface.co/datasets/nvidia/cascade);
the [`README.md` data guide](../../README.md#getting-the-data) covers the
streaming and local-directory workflows. If you don't have annotations yet,
the annotator can also start against a directory of videos (it'll create empty
annotation bundles for each clip on first open).

```bash
make annotator-dev DATA=/path/to/your/json_annotations
```

If you set `CASCADE_AV_DATASET_ROOT` in your `.env` during step 2 (or
already had it set), you can drop the `DATA=` argument — `make
annotator-dev` will pick the path up from there.

The server defaults to port `8765`. Override with `PORT=9000` if it
clashes with something else. The launcher tries to open a browser tab
at `http://127.0.0.1:8765/` automatically; pass `--no-browser` to the
underlying CLI if you'd rather not.

## 4. Read the preflight log

The first thing `cascade-annotate` prints is a 5-row ✓/✗ checklist
of what it found in the environment:

```
INFO  annotator.server.preflight: preflight:
INFO  annotator.server.preflight:   ✓ dotenv: loaded /home/you/work/cascade-devkit/.env
INFO  annotator.server.preflight:   ✓ ffmpeg: ffmpeg version 6.1.1 ...
INFO  annotator.server.preflight:   ✓ ffprobe: /usr/bin/ffprobe
INFO  annotator.server.preflight:   ✓ hf_auth: HF_TOKEN present in environment
INFO  annotator.server.preflight:   ✓ video_source: auto (hf preferred, falls back to local)
```

**If any row is `✗`, stop and fix it before clicking a clip** — the
failure modes (especially HF auth) surface much later as a generic
"video failed to load" overlay, which is hard to debug after the
fact. Common fixes:

| Failure | Fix |
|---------|-----|
| `✗ ffmpeg: not found on PATH` | `sudo apt-get install ffmpeg`, or use Homebrew/conda on other platforms. |
| `✗ ffprobe: not found on PATH` | Usually shipped with `ffmpeg`. Same install fix. |
| `✗ hf_auth: no HF_TOKEN and no huggingface-cli login` | Re-do step 2 above. |
| `✗ video_source: --video-source=hf but no HF auth` | Same as above — HF auth missing. |
| `✗ video_source: --video-source=local but --video-dir not set` | Pass `--video-dir=/path/to/videos` to the CLI, or switch to `--video-source=auto`. |

## 5. Click your first clip

The left sidebar lists every clip the annotator found in the data
directory you passed. Each row shows a status badge:

| Badge | Meaning |
|-------|---------|
| Gray dot **UNLABELLED** | No annotation JSON yet (fresh clip). |
| Amber dot **IN PROGRESS** | Annotation exists, status is `annotating`. |
| Green check **COMPLETE** | Annotation exists, status is `complete` — validation passed. |

Click a clip to open it. The center pane shows the video; the bottom
pane shows the timeline; the right pane shows the entity editor for
whatever segment you've selected.

## 6. Make an edit

Two common workflows on a freshly-loaded clip:

- **Add an action to the ego vehicle.** Right-click an empty area of
  the ego's action track at the time where the action starts. A new
  segment appears; the right pane opens its editor. Fill in the
  `action_type`, drag the right edge to set the end time, and that's
  it — the timestamp window populates from the segment positions.
- **Fix a label.** Click an existing segment (any track). The right
  pane shows its fields; edit them in place. Changes flow through the
  store and mark the bundle dirty.

The video stays in sync with the timeline. Use space to play/pause;
arrow keys step frame-by-frame.

## 7. Save your work

Three ways to save:

- **`Cmd/Ctrl-S`** — quick save. Same as the **Save** button in the
  bottom-right of the right panel.
- **The Save button** itself. Disabled until something is dirty.
- **Autosave** — the annotator persists on every blur of an editing
  field. Save is mostly a checkpoint.

The first save for a clip in a server session also stamps a
`<clipname>.json.bak` so you have an undo-of-last-resort. Subsequent
saves overwrite the main JSON atomically (write-temp + `os.replace`).

## 8. Mark complete

When you've worked through every action, agent, and condition the
clip needs, click the green **Mark complete** button next to Save.
The annotator:

1. Runs the validation contract against your bundle (see [`docs/user/annotator.md` § Validation](annotator.md)).
2. If clean, stamps `status="complete"` and saves. The sidebar badge
   flips to green.
3. If there are errors, shows them in the right-panel **Validation**
   section, each row clickable to jump to the offending entity. Save
   is still available — you can checkpoint partial work and come back
   later. Warnings (orange) don't block completion; errors (red) do.

A transport-layer failure (network down, server crash) shows a banner
above the validation list but doesn't wipe out the prior findings —
you keep your in-progress fix list visible.

---

## Where next

- [`docs/user/annotator.md`](annotator.md) — full reference: keyboard
  shortcuts, every panel's behavior, the validation rule catalog.
- [`docs/user/query_language.md`](query_language.md) — searching the
  annotated corpus from Python / a notebook.
- [`docs/user/visualization.md`](visualization.md) — programmatic
  rendering, the carousel, timeline plots.
- [`README.md` § Quickstart](../../README.md#quickstart) — load a clip
  in Python and explore its tree.

## Troubleshooting

| Symptom | First place to look |
|---------|---------------------|
| Annotator opens but the clip list is empty | Did you pass `DATA=` to `make annotator-dev`? Are the JSONs actually in that directory? |
| Clip loads but video stays as a loading overlay forever | The preflight log. The overlay reports the failure stage; the preflight names the root cause. |
| `make install` fails on `npm install` | Make sure you have Node 20+; older versions silently misparse Vite's config. |
| Mark-complete blocks on a rule that doesn't seem to apply | Open the issue row in the panel; it's clickable and jumps to the entity. The rule names are listed in [`docs/user/annotator.md`](annotator.md). |
