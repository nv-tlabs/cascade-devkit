# causal_ai_av

A Python DevKit for the **AV Causal Dataset** — causal and
spatio-temporal action annotations on top of NVIDIA's *Physical AI
AV Dataset*. The DevKit parses the annotation JSON into a typed
Pydantic tree and exposes a small query language for searching the
corpus by entity, attribute, time, and cause.

```python
from causal_ai_av.dataset import CausalAVDataset

ds = CausalAVDataset("/path/to/json_annotations")
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
dependency management.

```bash
# Core install
uv sync

# With the Physical AI AV Dataset integration (video loading)
uv sync --extra hf

# Plus the notebook tooling (JupyterLab, matplotlib, pandas)
uv sync --all-extras --group notebooks
```

Requires Python ≥ 3.11.

## Pointing at the corpus

Examples and notebooks read the dataset root from an environment
variable:

```bash
export CAUSAL_AV_DATASET_ROOT=/path/to/json_annotations
```

## Quickstart

```python
from causal_ai_av.dataset import CausalAVDataset

ds = CausalAVDataset("/path/to/json_annotations")
print(f"{len(ds)} clips")

# Count clips matching a query
ds.count("agent.type = ped and env.type = crosswalk")

# Get the full MatchSet — (clip_id, entity, interval) tuples
matches = ds.find("light.color = red and ego.action = stop")
clips = sorted(set(matches.clips()))

# Group matches by an attribute
ds.group_by("agent.type = vehicle or agent.type = vru", key="agent.type")
```

The DSL composes over entities and their attributes. Full grammar in
[`docs/query_language.md`](docs/query_language.md). A taste:

```text
agent(type = ped, action(jaywalk = true)) and ego.action in (stop, yield, decel)
light.color = yellow then(3) ego.action = stop
within light.color = red: not ego.action = stop
ego.action = decel because_of agent.type = ped
```

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

Run any of them with:

```bash
CAUSAL_AV_DATASET_ROOT=/path/to/json_annotations \
    uv run python examples/01_quickstart.py
```

Jupyter notebooks under `notebooks/` cover the same ground with
richer narrative and charts, plus `05_video_inspection.ipynb` which
pulls the original camera video from HuggingFace and renders frames
across a match's interval. Launch JupyterLab with:

```bash
CAUSAL_AV_DATASET_ROOT=/path/to/json_annotations \
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
src/causal_ai_av/
  spec/        # Pydantic schema for the annotation JSON
  io.py        # parse JSON → AnnotationBundle
  dataset.py   # CausalAVDataset / Sequence — corpus and per-clip API
  query/       # DSL lexer + parser + evaluator + query helpers

docs/
  query_language.md   # DSL specification (grammar + semantics)

examples/      # runnable Python scripts
notebooks/     # Jupyter notebooks (built from scripts/build_notebooks.py)
tests/         # pytest suite
```

## Development

```bash
# Run the test suite
uv run pytest

# Regenerate notebooks from the source-of-truth builder
uv run --group notebooks python scripts/build_notebooks.py
```

Notebooks are committed without embedded outputs — the cells are
short, regenerable, and ship the narrative rather than the data.

## License

See [`LICENSE`](LICENSE).
