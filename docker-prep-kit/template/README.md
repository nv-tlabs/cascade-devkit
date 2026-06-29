# Submission template

Copy this directory and turn it into your submission. The build tool
(`build_submission.py`) builds this `FROM` the pinned CASCADE base (a stock
`python:3.12` image), ships the layers your build adds on top of it, and
publishes them as a portable artifact. The artifact is `manifest.json` plus
`layers/layer-*.tar`; the Docker image itself is never uploaded.

GPU/CUDA comes from your framework's pip wheels (e.g. `pip install torch` pulls
CUDA + cuDNN) plus the host driver — there is no CUDA base image.

## Files

- `Dockerfile` — builds `FROM` the pinned base and installs your stack. Install
  wherever is natural (the example uses a venv, but system/conda also work).
- `requirements.txt` — your Python dependencies.
- `run.py` — baseline entrypoint. Replace `predict_matching_videos` with your
  retrieval system.
- `submission.yaml` — declares how the evaluator launches your system.
- `.dockerignore` — keeps the build context small.

## Rules that make or break a submission

1. **Install at build time, anywhere.** Whatever your build adds on top of the
   base (any location — venv, conda, system, `~/.cache`) is captured. There is no
   required prefix.
2. **Download weights at build time.** The evaluator runs your system with the
   network **disabled**; do not fetch anything at run time.
3. **CUDA toolkit ≤ host driver.** Target CUDA 13.0 or older for the current
   evaluation host.
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

## Publishing

Publish with `build_submission.py --repo-id
your-hf-username/your-submission`. The default target is a private Hugging Face
**model repo**, created in the authenticated token owner's personal namespace.
Keep it private, then grant `grossanchez` read access in the repo's Hugging Face
access settings so the evaluator can fetch the manifest and layers. Submit the
repo ID through the challenge frontend while signed in as that same owner. A
Space, dataset repo, public repo, or Docker image is not a valid submission.
