# CASCADE DevKit

Python DevKit for the **CASCADE dataset** — *Causal Spatio-Temporal
Analysis of Driving Environments*. Causal and spatio-temporal action
annotations on top of NVIDIA's *Physical AI AV Dataset*. The DevKit
parses the annotation JSON into a typed Pydantic tree and exposes a
small query language for searching the corpus by entity, attribute,
time, and cause.

```python
from cascade_av.dataset import CascadeDataset

ds = CascadeDataset("/path/to/json_annotations")
ds.count("ego.action = decel because_of agent.type = ped")
# → number of clips where the ego decelerates *because of* a pedestrian
```

## What's annotated

Each clip is a short front-facing driving video paired with a JSON
annotation bundle (schema `2.0.0`). Every clip is annotated with:

| entity | what it captures |
|---|---|
| **Agents** | non-ego actors — vehicles, pedestrians, cyclists, animals, officers — with type, ego-relative position, visibility intervals, and per-interval actions |
| **Ego vehicle** | actions taken by the recording car (drive, decel, stop, turn, change-lane, …) and the entities that motivated them |
| **Environments** | road, intersection, crosswalk, sidewalk, roundabout, cycle-lane, tunnel, … with lane counts and one-way flags |
| **Conditions** | weather, lighting, construction, occlusion |
| **Traffic lights** | signal-head colors and state transitions, plus annotator-tagged flags such as `could_have_cleared` and `ego_in_on_yellow` |
| **Traffic objects** | stop signs, yield signs, cones, debris, barriers |

The schema includes a `because_of` edge on every action: each action
can name the entities that caused it. The query DSL surfaces this
through a `because_of` operator.

## Install

This project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management. Clone the repo first; the package is not on
PyPI.

### Quick install (Ubuntu / Debian)

After cloning, one command does everything below:

```bash
./scripts/install.sh
```

The script installs the system prerequisites (apt + Node LTS + uv),
runs `make install`, and smoke-tests the Python side. It is
idempotent — re-running is safe. Read on for what the script does
step-by-step, or skip to **Install the project** if you already have
the prerequisites.

### Prerequisites (Ubuntu / Debian)

You need four things on PATH before `make install` works:

| Tool | Why |
|------|-----|
| **Python ≥ 3.11** | DevKit baseline. Ships with Ubuntu 22.04+ and Debian Bookworm; older releases need [deadsnakes](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) or `pyenv`. |
| **`make`** | All documented install / test / run targets are make recipes. |
| **Node ≥ 18 (LTS) + npm** | Only used by the annotator frontend (Vite bundle). |
| **`ffmpeg` + `ffprobe`** | Only used by the annotator (HEVC → H.264 transcode + codec detection). See [`tools/annotator/README.md`](tools/annotator/README.md) for which features degrade if missing. |

Plus `uv` itself — installed separately because it manages your Python
environments and is not a Python package.

Manual recipe (what `./scripts/install.sh` does on a fresh
Ubuntu 22.04+ / Debian Bookworm):

```bash
sudo apt-get update
sudo apt-get install -y make build-essential ffmpeg

# Node 20 LTS via NodeSource
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# uv via the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Container alternative.** A [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json)
ships with the repo and installs all of the above automatically — open
the folder in VS Code Dev Containers, GitHub Codespaces, or any
compatible host. No host-side setup needed beyond the IDE.

### Install the project

```bash
# One-shot — Python deps (all extras) + the annotator's npm deps.
make install

# Or run the steps explicitly:
uv sync                                    # core install
uv sync --extra hf                         # + parent-dataset integration (video loading)
uv sync --all-extras --group notebooks     # + notebook tooling (JupyterLab, matplotlib, pandas)
```

## Getting the data

The DevKit reads two kinds of artifacts:

1. **Annotation JSON bundles** — the causal/spatio-temporal labels
   this repo adds. You point `CascadeDataset` at a local directory of
   `*.json` files.
2. **Sensor data** (camera videos, LiDAR, radar, egomotion) — the
   underlying *Physical AI AV Dataset* on HuggingFace. Pulled
   on-demand by `ds.download_clips(...)` when you need pixels or
   sensors. See [Working with the sensor data](#working-with-the-sensor-data)
   below.

Until the annotation bundles ship publicly, obtain them from the
project owners and put the directory of `*.json` files anywhere on
local disk.

Examples and notebooks read the path from an environment variable:

```bash
export CASCADE_AV_DATASET_ROOT=/path/to/json_annotations
```

A [`.env.example`](.env.example) ships at the repo root — copy it to
`.env` if your tooling auto-loads dotenv (IDE test runners, Docker
Compose, `dotenv-cli`; plain `uv run` does not).

## Quickstart

```python
from cascade_av.dataset import CascadeDataset

ds = CascadeDataset("/path/to/json_annotations")
print(f"{len(ds)} clips")

# Count clips matching a query
ds.count("agent.type = ped and env.type = crosswalk")

# Get the full MatchSet — a tuple of (clip_id, entity, interval) `Match`es
matches = ds.find("light.color = red and ego.action = stop")
clips = sorted(set(matches.clips()))

# Group matches by an attribute
ds.group_by("agent.type = vehicle or agent.type = vru", key="agent.type")
```

The DSL composes over entities and their attributes. Full grammar in
[`docs/user/query_language.md`](docs/user/query_language.md). A taste:

```text
agent(type = ped, action(jaywalk = true)) and ego.action in (stop, yield, decel)
light.color = yellow then(3) ego.action = stop
within light.color = red: not ego.action = stop
ego.action = decel because_of agent.type = ped
```

## Visualization

`cascade_av.viz` renders any subset of a clip — a single instant, a
time range, or a `MatchSet` — as a decoded camera frame paired with
the clip's annotation timeline. The timeline carries one bar per
agent action, ego action, environment, condition, and traffic-light
state, plus five families of causal arrow — `because_of`, `link_to`,
`containment`, `influence`, `action_target` — and a yellow highlight
band over any match interval.

The viz API requires the optional `[viz]` extra and runs in
JupyterLab, VS Code / Cursor notebooks, and similar Jupyter-protocol
environments. Colab support is best-effort.

```bash
uv sync --extra viz
```

The three common entry points hang off the objects you already use.
`ds.get_sequence(clip_id)` returns a `Sequence` — the per-clip handle
that bundles the parsed annotation with camera / sensor accessors and
the `.visualize()` method:

```python
seq = ds.get_sequence(clip_id)

# Whole-clip scrubbable widget. Pass t=2.5 to open at a single instant,
# or t=(2, 5) for an explicit window.
seq.visualize()

# Single match in context — pads the match interval by `pad` seconds
# and paints `m.interval` as the yellow highlight band.
m = ds.find("agent.type = ped while ego.action = decel").matches[0]
seq.visualize(match=m, pad=1.0)

# Fan out a whole MatchSet into a carousel of mini-players (one per
# match, capped at `limit`).
ds.find("ego.action = decel because_of agent.type = ped").visualize()
```

For reports, doc figures, or headless pipelines, two functions return
plain values you can pickle, save, or post-process:

```python
from cascade_av import viz

frame = viz.render_frame(seq, t=3.0)          # -> PIL.Image (RGB)
fig   = viz.render_timeline(seq,              # -> plotly.graph_objects.Figure
                            highlight=(2, 5))
frame.save("/tmp/clip_t3.png")
fig.write_image("/tmp/timeline.png")          # needs the `kaleido` extra
```

`seq.visualize(t=2.5, static=True)` is the shorthand for
`viz.render_frame(seq, 2.5)` — handy when you start interactive and
want a single still without switching modules.

### Filtering the timeline

Five filter kwargs narrow what shows up — `arrows`, `entity_kinds`,
`agent_ids`, `track_groups`, `families`. They work on every entry point
and AND together:

```python
# Just agent actions, hide the Ego row, drop because_of arrows.
seq.visualize(
    track_groups=["Agents"],
    families=["action"],
    arrows={"because_of": False},
)
```

Full reference (every value each filter accepts, parent-retention
semantics for `families`, composition rules) in
[`docs/user/visualization.md`](docs/user/visualization.md).

## Examples and notebooks

Runnable Python scripts under `examples/`:

| file | what it shows |
|---|---|
| `01_quickstart.py` | smallest possible end-to-end usage |
| `02_query_operators.py` | tour of every DSL operator |
| `03_statistics.py` | count / group-by aggregations |
| `04_scenarios.py` | 20 driving scenarios encoded as DSL queries |
| `05_context.py` | inspect what else was happening during each match |
| `06_sensor_data.py` | catalog of available sensors + recipes for fetching extras |
| `07_visualize.py` | headless render — single frame and timeline figure |

Run any of them with:

```bash
CASCADE_AV_DATASET_ROOT=/path/to/json_annotations \
    uv run python examples/01_quickstart.py
```

Jupyter notebooks under `notebooks/` cover the same ground with
richer narrative and charts, plus `05_video_inspection.ipynb` which
pulls the original camera video from HuggingFace and renders frames
across a match's interval, and `06_visualize.ipynb` which walks
through the interactive viz API (clip player + timeline + match
carousel). Launch JupyterLab with:

```bash
CASCADE_AV_DATASET_ROOT=/path/to/json_annotations \
    uv run --all-extras --group notebooks jupyter lab notebooks/
```

## Working with the sensor data

The annotation bundles are paired with the original Physical AI AV
Dataset on HuggingFace — every clip ships with a full sensor stack:

| group | members | feature names |
|---|---|---|
| Cameras (7) | front-wide 120°, front-tele 30°, 2× cross 120°, 2× rear-side 70°, rear-tele 30° | `camera_front_wide_120fov`, `camera_front_tele_30fov`, `camera_cross_{left,right}_120fov`, `camera_rear_{left,right}_70fov`, `camera_rear_tele_30fov` |
| LiDAR (1) | roof-mounted 360° | `lidar_top_360fov` |
| Radars (19) | front-center (SRR/MRR/imaging-LRR), 4 corner radars, 2 side radars × 2 modes, 2 rear-side radars × 2 ranges | `radar_*` |
| Egomotion (2) | raw + offline-smoothed | `egomotion`, `egomotion.offline` |
| Calibration (6) | sensor extrinsics, camera/LiDAR intrinsics, vehicle dimensions | `sensor_extrinsics{,.offline}`, `camera_intrinsics{,.offline}`, `lidar_intrinsics.offline`, `vehicle_dimensions` |
| Derived labels (1) | preprocessed obstacle tracks | `obstacle.offline` |

Prefetch a batch of clips so they're cached locally before you start
iterating. `download_clips` accepts an iterable of clip ids and an
optional `features=` list; defaults are the canonical front-wide
camera plus egomotion — the minimum for `get_sequence`:

```python
ds.download_clips([clip_id])                                  # canonical camera + egomotion
ds.download_clips()                                            # every clip in the corpus
ds.download_clips([clip_id], features=ds.features.CAMERA.ALL)  # full 7-camera rig
ds.download_clips([clip_id], features=ds.features.LIDAR.ALL)   # LiDAR sweeps
ds.download_clips([clip_id], features=ds.features.RADAR.ALL)   # all 19 radars
ds.download_clips([clip_id], features=ds.features.ALL)         # everything
```

Then read sensors off the `Sequence`:

```python
seq = ds.get_sequence(clip_id)
seq.video                          # SeekVideoReader for the canonical camera
seq.cameras["camera_rear_tele_30fov"]   # any other camera
ds.get_clip_feature(clip_id, "lidar_top_360fov")
```

`SeekVideoReader.decode_images_from_timestamps(np.array([t_us], dtype=np.int64))`
decodes frames at microsecond timestamps.

> **Heads-up — chunk-granularity downloads.** The parent dataset
> stores features in chunks containing many clips, so opting into
> one extra sensor for one clip can pull several GB. The dataset
> constructor's `confirm_download_threshold_gb` (default 10) prompts
> for confirmation before crossing that threshold; raise it to run
> unattended.

`examples/06_sensor_data.py` prints the full sensor catalog and the
download recipes; set `SENSOR_DEMO_DOWNLOAD=1` to also fetch a sister
camera and decode a frame from it.

## Project layout

```
src/cascade_av/
  spec/        # Pydantic schema for the annotation JSON
  io.py        # parse JSON → AnnotationBundle
  dataset.py   # CascadeDataset / Sequence — corpus and per-clip API
  query/       # DSL lexer + parser + evaluator + query helpers
  viz/         # render_frame / render_timeline / ClipPlayer (optional [viz] extra)

tools/
  annotator/   # local FastAPI + React annotation tool (see below)

docs/
  user/
    query_language.md  # DSL specification (grammar + semantics)
    visualization.md   # viz filter reference
    annotator.md       # annotator UI guide

examples/      # runnable Python scripts
notebooks/     # Jupyter notebooks (built from scripts/build_notebooks.py)
tests/         # pytest suite
```

## Annotator

`tools/annotator/` ships a local annotation tool — a slim FastAPI server
hosting a vendored React frontend — for editing the JSON bundles
in-browser. No auth, no database, no admin layer; you point it at a
directory of annotations (or fresh videos) and it serves an editor over
`localhost`.

```bash
make install                                              # Python + npm deps (once)
make annotator-dev DATA=/path/to/json_annotations         # launches on :8765
```

If you want the explicit steps without `make`:

```bash
uv sync --extra annotator
cd tools/annotator/web && npm install && npm run build
uv run cascade-annotate /path/to/json_annotations
```

Lock-by-default, explicit Save (with `.bak` on first save), HEVC→H.264
transcode pipeline for browser playback. Two docs cover the rest:

- [`docs/user/annotator.md`](docs/user/annotator.md) — UI walkthrough
  end-to-end: mouse, keyboard, lock model, arrows, troubleshooting.
- [`tools/annotator/README.md`](tools/annotator/README.md) — install,
  CLI flags, transcode pipeline, architecture.

## Development

```bash
# Run the test suite
make test                  # or: uv run pytest

# Lint + format
make lint
make fmt

# Regenerate notebooks from the source-of-truth builder
uv run --group notebooks python scripts/build_notebooks.py
```

`make help` lists every target.

Notebooks are committed without embedded outputs — the cells are
short, regenerable, and ship the narrative rather than the data.

## License

See [`LICENSE`](LICENSE).
