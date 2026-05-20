# CASCADE DevKit

Python DevKit for the **CASCADE dataset**: Causal Spatio-Temporal
Analysis of Driving Environments. It parses the annotation JSON
into a typed Pydantic tree, exposes a small query language for
searching by entity, attribute, time, and cause, and renders any
clip or query result as a video frame paired with its annotation
timeline.

```python
from cascade_av.dataset import CascadeDataset

ds = CascadeDataset("/path/to/json_annotations")
ds.count("ego.action = decel because_of agent.type = ped")
# → number of clips where the ego decelerates *because of* a pedestrian
```

## What's annotated

Each clip is a short front-facing driving video paired with a JSON
annotation bundle, including:

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
dependency management. Pick one of the two paths below — they produce
the same result.

### Option A: one-line install (Ubuntu / Debian, recommended)

`scripts/install.sh` does everything end-to-end: clones the repo (if
you ran it via curl-pipe), installs apt prerequisites (`make`,
`build-essential`, `ffmpeg`), installs Node 20 LTS, installs `uv`,
runs `make install` (Python deps + annotator npm), and smoke-tests
the Python side. Idempotent — re-running is safe.

Fresh host (no checkout yet):

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA-dev/av-causal-dataset-tools/main/scripts/install.sh | bash
```

Already cloned, run from the repo root:

```bash
./scripts/install.sh
```

> **First time using the annotator?** The 5-minute walkthrough at
> [`docs/user/getting-started.md`](docs/user/getting-started.md) takes
> you from a fresh clone to your first saved annotation, including
> HuggingFace token setup and reading the startup preflight log.

### Option B: manual install

For non-Debian hosts, hardened environments, devcontainers, or
finer-grained control over Python extras: see
[`docs/user/install.md`](docs/user/install.md) for the prerequisites
table, the apt + Node + uv recipe, and the `uv sync` variants.

## Getting the data

The DevKit reads two kinds of artifacts: **annotation JSON bundles**
(this repo's causal/spatio-temporal labels) and **sensor data**
(camera videos, LiDAR, radar, egomotion — the underlying *Physical AI
AV Dataset* on HuggingFace). Sensor data is always pulled on-demand by
`ds.download_clips(...)`; see
[Working with the sensor data](#working-with-the-sensor-data). For the
annotation JSONs, pick one of the two paths below.

### Option A: stream from HuggingFace

The JSONs live in a HuggingFace dataset repo. The `[hf]` extra ships
with `make install` (i.e. both install paths above); pull either the
whole corpus or a named, versioned split:

```python
from cascade_av.io.hf import CausalAnnotationsHfRepo

repo = CausalAnnotationsHfRepo("nvidia/cascade")   # path_in_repo auto-detected

# Whole corpus
for bundle in repo.iter_annotations():
    ...

# Named, versioned splits — declared by `data/dataset_split.yaml`
# in the repo and never rewritten once published, so a (name, split)
# pair is a stable, citeable handle.
repo.available_splits()                            # {"cascade-v0.1": ["train", "validation"]}
train = repo.load_split("cascade-v0.1", "train")   # list[AnnotationBundle]
```

Downloads land in the standard `huggingface_hub` cache; re-running
hits the cache, not the network. Replace `"nvidia/cascade"` with your
own fork/mirror if needed — both `CASCADE_REPO_URL` (for `install.sh`)
and `repo_id=` (for the Python API) are fully overridable.

### Option B: point at a local directory

If you already have JSONs on disk — a previous HF download, a fork, a
private mirror, or a freshly-recorded session — point `CascadeDataset`
at the directory directly. Examples and notebooks read the path from
an environment variable:

```bash
export CASCADE_AV_DATASET_ROOT=/path/to/json_annotations
```

A [`.env.example`](.env.example) ships at the repo root. Copy it to
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
agent(type = ped, action.type = "oxd:Walk (jaywalk)") and ego.action in (stop, yield, decel)
light.color = yellow then(3) ego.action = stop
within light.color = red: not ego.action = stop
ego.action = decel because_of agent.type = ped
```

## Visualization

`cascade_av.viz` renders any subset of a clip — a single instant, a
time range, or a `MatchSet` — as a decoded camera frame paired with
the clip's annotation timeline (one bar per agent action, ego action,
environment, condition, traffic-light state, plus five families of
causal arrow). Requires the optional `[viz]` extra:

```bash
uv sync --extra viz
```

```python
seq = ds.get_sequence(clip_id)

seq.visualize()                                  # scrubbable widget over the whole clip
seq.visualize(match=m, pad=1.0)                  # single match in context
ds.find("ego.action = decel because_of agent.type = ped").visualize()  # MatchSet carousel
```

For reports, doc figures, or headless pipelines, `viz.render_frame()`
and `viz.render_timeline()` return plain `PIL.Image` and
`plotly.graph_objects.Figure` values you can save or post-process.

Full reference — every entry point, filter kwarg, headless variant,
the `static=True` shorthand, and the carousel's `unique_clips` knob —
in [`docs/user/visualization.md`](docs/user/visualization.md).

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
make notebooks
```

`make notebooks` sources `.env` and exports `CASCADE_AV_DATASET_ROOT`
into the kernel, so notebooks that iterate the corpus just work.
Override the path per-invocation with `make notebooks DATA=/some/other/dir`.
The underlying command (`uv run --all-extras --group notebooks
jupyter lab notebooks/`) still works if you'd rather invoke it
directly.

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

`DATA` is optional if `CASCADE_AV_DATASET_ROOT` is set in `.env` (see
[`.env.example`](.env.example)) — `make annotator-dev` (no args) will
pick it up. Override per-invocation by passing `DATA=...` explicitly.

If you want the explicit steps without `make`:

```bash
uv sync --extra annotator
cd tools/annotator/web && npm install && npm run build
uv run cascade-annotate /path/to/json_annotations
```

Lock-by-default, explicit Save (with `.bak` on first save), HEVC→H.264
transcode pipeline for browser playback. Annotators mark a clip complete
via the right panel; the server validates against the rule set in
`cascade_av.validate` before accepting `status="complete"`. Two docs cover
the rest:

- [`docs/user/annotator.md`](docs/user/annotator.md) — UI walkthrough
  end-to-end: mouse, keyboard, lock model, arrows, troubleshooting.
- [`tools/annotator/README.md`](tools/annotator/README.md) — install,
  CLI flags, transcode pipeline, architecture.

**Opening older bundles.** Current schema is **2.0.0** (rebooted
2026-05-20 against the upstream `sil-dense-annotation-tool 0.4.5` shape).
Files written against an older version still load with a one-shot
`DeprecationWarning`; there is no in-tree migration path — the corpus is
being reconverted upstream. See
[`docs/dev/schema-history.md`](docs/dev/schema-history.md) for the
version history and sidecar layout.

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
