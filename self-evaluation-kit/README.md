# Self-Evaluation Kit

Use this kit to run a challenge Docker image locally on a CASCADE-shaped
retrieval split and produce an evaluation JSON file.

The runner converts CASCADE `data/` and `tasks/retrieval/` files into the same
container input contract used by the official evaluator:

- `/input/queries.jsonl`
- `/input/videos.jsonl`
- `/input/videos/<clip_id>.mp4`

Ground-truth qrels are kept outside `/input` and are used only after the
container exits.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r self-evaluation-kit/requirements.txt
```

Docker must also be available on the machine.

## Dataset Requirements

`--dataset-root` must point at a CASCADE-shaped checkout:

```text
dataset-root/
├── data/
│   └── batch_00001/
│       └── <annotation>.json
└── tasks/
    └── retrieval/
        ├── retrieval_split.yaml
        └── batch_00001/
            ├── train_queries.json
            └── val_queries.json
```

`--split-name` must match an entry in
`tasks/retrieval/retrieval_split.yaml`, for example `cascade-v0.1`.

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

Build your submission image, then evaluate it:

```bash
docker build -t my-crc-submission docker-prep-kit/template

python self-evaluation-kit/self_eval.py \
  --image my-crc-submission \
  --dataset-root /path/to/cascade \
  --video-root /path/to/videos \
  --split-name cascade-v0.1 \
  --split val \
  --out evaluation.json
```

The output JSON includes the split, image, corpus size, query count, primary
score, and detailed retrieval metrics: Average R-Precision, MAP, Precision@k,
Recall@k, and Hit@k.

## Score Existing Predictions

To debug scoring without running Docker, pass an existing predictions file:

```bash
python self-evaluation-kit/self_eval.py \
  --image my-crc-submission \
  --dataset-root /path/to/cascade \
  --video-root /path/to/videos \
  --split-name cascade-v0.1 \
  --split val \
  --predictions /tmp/predictions.jsonl \
  --out evaluation.json
```

Predictions must contain one JSON object per query:

```json
{"query_id": "cascade-v0.1_val_batch_00001_00001", "video_ids": ["clip-id"]}
```
