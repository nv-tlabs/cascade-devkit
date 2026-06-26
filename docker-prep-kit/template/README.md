# Submission template

Copy this directory and turn it into your submission. The build tool
(`build_submission.py`) builds this `FROM` the CASCADE common base, extracts the
`/opt/submission` prefix, and publishes it as a portable artifact.

## Files

- `Dockerfile` — builds `FROM` the common base and installs everything under
  `/opt/submission`. Replace the dependency and weight steps with your own.
- `requirements.txt` — your Python dependencies (installed into the prefix env).
- `run.py` — baseline entrypoint. Replace `predict_matching_videos` with your
  retrieval system.
- `submission.yaml` — declares how the evaluator launches your system.
- `.dockerignore` — keeps the build context small.

## Rules that make or break a submission

1. **Everything under `/opt/submission`.** Anything installed elsewhere is not
   shipped. Use the venv (or a conda env) inside the prefix.
2. **Download weights at build time.** The evaluator runs your system with the
   network **disabled**. Bake weights into the prefix; do not fetch at run time.
3. **CUDA toolkit ≤ host driver.** Match the base image's CUDA line.
4. **Single GPU.** Multi-GPU/NCCL may be blocked by the no-network sandbox.

## Runtime contract

Your entrypoint runs on GPU with the network disabled and must:

- Read `/input/queries.jsonl`: `{"query_id": "...", "text": "..."}`.
- Read `/input/videos.jsonl`: `{"video_id": "...", "path": "videos/<id>.mp4"}`.
- Read video files from `/input/videos/<video_id>.mp4`.
- Write `/output/predictions.jsonl`, one object per query:
  `{"query_id": "...", "video_ids": ["...predicted matches, strongest first..."]}`.
- Exit `0` after writing predictions.

Every query appears once. `video_ids` are your predicted matches ranked by
confidence. `TOP_K` is a maximum, not a target — return an empty list when there
is no likely match; do not pad with non-matching videos.
