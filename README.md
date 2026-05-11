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

Run any of them with:

```bash
CAUSAL_AV_DATASET_ROOT=/path/to/json_annotations \
    uv run python examples/01_quickstart.py
```

Matching Jupyter notebooks under `notebooks/` cover the same ground
with richer narrative and charts. Launch JupyterLab with:

```bash
CAUSAL_AV_DATASET_ROOT=/path/to/json_annotations \
    uv run --all-extras --group notebooks jupyter lab notebooks/
```

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
