# Submission template

Copy this directory and turn it into your submission. The build tool
(`build_submission.py`) builds this `FROM` the pinned CASCADE base (a stock
`python:3.12` image), ships the layers your build adds on top of it, and
publishes them as a portable artifact. The artifact is `manifest.json` plus
`layers/layer-*.tar`; the Docker image itself is never uploaded.

GPU/CUDA comes from your framework's pip wheels (e.g. `pip install torch` pulls
CUDA + cuDNN) plus the host driver — there is no CUDA base image. Official
evaluation uses one **NVIDIA A100 80 GB**, compute capability **8.0**
(`sm_80`). Your framework wheel and all compiled CUDA extensions must include
`sm_80` kernels; a CUDA version string alone does not establish compatibility.

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
3. **Include `sm_80` kernels.** For PyTorch CUDA extensions, build with
   `TORCH_CUDA_ARCH_LIST="8.0"`. A maximum CUDA version is not, by itself, a
   compatibility test.
4. **Single A100 80 GB.** Multi-GPU/NCCL may be blocked by the no-network
   sandbox.

## PyTorch GPU preflight

Build locally with `build_submission.py --no-push`, then test the exact image
before publishing. The default local image tag is `cascade-submission:local`.

```bash
docker run --rm -i --gpus all --network none \
  --entrypoint /opt/app/.venv/bin/python \
  cascade-submission:local - <<'PY'
import torch

arches = set(torch.cuda.get_arch_list())
if "sm_80" not in arches:
    raise SystemExit(f"PyTorch wheel is missing sm_80; compiled for {sorted(arches)}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the submission image")

capability = torch.cuda.get_device_capability(0)
x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
torch.testing.assert_close((x.square() + 1).cpu(), torch.tensor([2.0, 5.0, 10.0]))
torch.cuda.synchronize()
print({"device": torch.cuda.get_device_name(0), "capability": capability,
       "torch_cuda": torch.version.cuda, "compiled_arches": sorted(arches),
       "kernel": "ok"})
PY
```

On an A100, `capability` must be `(8, 0)`. On another local GPU, this still
confirms that PyTorch advertises `sm_80` and that a CUDA kernel runs locally.
It does not inspect third-party or custom extensions, so build those for
`sm_80` and exercise their inference path on an A100 too.

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

First use the challenge frontend to create your submission repository. It must be
a private Hugging Face **model repo** in your personal namespace. Authenticate
your development machine once with the project-managed CLI, then publish with:

```bash
uv run --project ../.. --extra hf hf auth login
uv run --project ../.. --extra hf python ../build_submission.py \
  --context . \
  --repo-id your-hf-username/frontend-generated-submission
```

The uploader uses the normal cached HF login, so you do not pass or export a
token. It does not create a repo or change visibility, and it rejects missing,
public, organization-owned, or non-model targets. The artifact is replaced in
one commit and the command reports its exact commit SHA. Return to the challenge
frontend to submit the uploaded revision; no access grant to an organizer's personal
account is required. A Space, dataset repo, public repo, or Docker image is not a
valid submission.

The build is fixed to the challenge's `linux/amd64` evaluation platform. On
Docker 23+, the tool also pulls the pinned base into Docker's image store when
BuildKit has cached it only internally; participants do not need a manual pull.

If model weights are private or gated, download them on the host with that
cached login and copy only the weight files into the Docker build. Never use a
token in Docker `ARG`/`ENV` or copy a token into the context; delta layers are
published verbatim.
