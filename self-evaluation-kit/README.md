# Self-Evaluation Kit

Use this kit to score a CASCADE submission **artifact** locally on a
CASCADE-shaped retrieval split and produce an evaluation JSON file.

It mirrors the official evaluator: it reconstructs your submission's delta layers
on the base image and runs your entrypoint **in place with the network disabled**
(`docker run --network none`), so you catch accidental inference-time downloads
before submitting.

The runner converts CASCADE `data/` and `tasks/retrieval/` files into the same
input contract used by the official evaluator:

- `/input/queries.jsonl`
- `/input/videos.jsonl`
- `/input/videos/<clip_id>.mp4`

Ground-truth qrels are kept outside `/input` and are used only after the run
finishes.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r self-evaluation-kit/requirements.txt
```

Docker must also be available on the machine.

## Dataset Requirements

`--dataset-root` must point at a checkout of the
[CASCADE dataset](https://huggingface.co/datasets/nvidia/cascade) with this layout:

```text
dataset-root/
├── data/
│   └── batch_00001/
│       └── <annotation>.json
└── tasks/
    └── retrieval/
        ├── retrieval_split.yaml
        └── batch_00001/
            ├── train_queries_0817.json
            └── val_queries_0817.json
```

The base corpus under `data/` is not split. Each task owns its split manifest;
for retrieval, `--split-name` must match an entry in
`tasks/retrieval/retrieval_split.yaml`, such as `cascade-v1.0`. Query filenames
are resolved from that manifest rather than inferred from the split name.

`--video-root` must contain real video files. By default the runner expects
`<video-root>/<clip_id>.mp4` for every clip in `data/batch_*/*.json`. If your
videos use a different layout, pass `--video-manifest`, either as JSON:

```json
{
  "clip-id-1": "nested/path/clip-id-1.mp4"
}
```

or JSONL:

```jsonl
{"clip_id": "clip-id-1", "path": "nested/path/clip-id-1.mp4"}
```

The tool fails before running Docker if any required video file is missing.

## Run

First build your submission artifact with the docker-prep-kit, then evaluate it:

```bash
# Produce the artifact (manifest.json + layers/) without uploading:
python docker-prep-kit/build_submission.py \
  --context my-submission \
  --base-image python:3.12@sha256:2575347025c314e37d89d4b353904edbe1824a6117b8eeffe52254879e4f6146 \
  --no-push

# Score it on a CASCADE split, reconstructed on the base with NO network:
python self-evaluation-kit/self_eval.py \
  --artifact my-submission/.cascade-build \
  --base-image python:3.12@sha256:2575347025c314e37d89d4b353904edbe1824a6117b8eeffe52254879e4f6146 \
  --dataset-root /path/to/cascade \
  --video-root /path/to/videos \
  --split-name cascade-v1.0 \
  --split val \
  --out evaluation.json
# add --gpus all to use the GPU (requires the NVIDIA Container Toolkit)
```

The output JSON includes the split, image, corpus size, query count, primary
score, and detailed retrieval metrics: Average R-Precision, MAP, Precision@k,
Recall@k, and Hit@k.

## Metrics

The local kit uses the same retrieval-metric formulas as the official scorer.
For each query, let `GT` be its set of relevant videos, let `R = |GT|`, and let
the submitted video IDs be ranked from strongest to weakest. Repeated video IDs
are removed while preserving their first position.

- **Average R-Precision (primary metric):** for each query, count the relevant
  videos among the first `R` ranked results and divide by `R`. The official
  `score` is the mean R-Precision across scored queries.
- **Mean Average Precision (MAP):** Average Precision for one query is
  `(1 / R) * sum(P@i)` over ranks `i` containing a relevant video. MAP is the
  mean AP across scored queries.
- **Precision@k:** relevant videos among the first `k` results, divided by
  exactly `k`, even when fewer than `k` results were returned.
- **Recall@k:** relevant videos among the first `k` results, divided by `R`.
- **Hit@k:** `1` when at least one relevant video appears among the first `k`
  results, otherwise `0`. Its aggregate is the fraction of scored queries with
  a hit.

Precision, Recall, and Hit are reported at `k = 1, 3, 5, 10`. Every aggregate
is an unweighted macro-average, so each scored query contributes equally even
when queries have different numbers of relevant videos. Queries with no
ground-truth relevant videos are excluded from metric averages and reported in
`num_queries_without_relevance`.

Missing prediction rows and empty rankings score zero for that query. Ranking
order alone determines the metrics; confidence values are neither required nor
used. The evaluation JSON also reports `num_queries`, `num_scored_queries`, and
`num_predicted` (the number of query IDs with submitted predictions).

The public leaderboard keeps the best successful submission for each stable
Hugging Face account. Results are ordered by Average R-Precision, then MAP,
Recall@10, Precision@10, and finally earliest completion time.

## Score Existing Predictions

To debug scoring without running the submission, pass an existing predictions
file:

```bash
python self-evaluation-kit/self_eval.py \
  --dataset-root /path/to/cascade \
  --video-root /path/to/videos \
  --split-name cascade-v1.0 \
  --split val \
  --predictions /tmp/predictions.jsonl \
  --out evaluation.json
```

Predictions must contain one JSON object per query:

```json
{"query_id": "cascade-v1.0_val_batch_00001_00001", "video_ids": ["clip-id"]}
```

`video_ids` should contain only videos your system predicts match the query,
ranked strongest match first. The list may be shorter than `TOP_K`; if no video
is likely to match, use an empty list. Do not pad with arbitrary corpus videos.
