# Docker Preparation Kit

Prepare and publish your submission to the AV Causal Scenario Retrieval
Challenge.

## How evaluation works (and why this kit exists)

The evaluator does **not** run your Docker image as a container (the sandbox has
no nested containers, and a `chroot` cannot be given GPU access). Instead:

1. You build your image **`FROM` the pinned CASCADE base** — a **stock Python
   image** (`python:3.12`, pinned by digest), installing whatever you need
   **wherever is natural** (a venv, conda, system packages, weights in
   `~/.cache/huggingface`, ...). There is **no required prefix**.
2. This kit ships only the **layers your build added on top of the base** (the
   delta), plus a `manifest.json`, to a private Hugging Face **model repo** in
   your personal namespace. It does not publish or run your Docker image.
3. Hugging Face mounts the pinned artifact read-only into the evaluation Job.
   Trusted launcher code validates it, disables network egress, reapplies your
   delta layers on the **identical** base, and only then starts your declared
   entrypoint **in place** on GPU with the hidden inputs available.

Because both sides share one pinned base, your environment is reproduced exactly.

## GPU / CUDA

There is **no CUDA base image**. Your GPU stack comes from your framework's pip
wheels — e.g. `pip install torch` pulls the matching CUDA + cuDNN — plus the GPU
**driver** provided by the evaluation host (recent; supports CUDA 13.0). If you
compile custom CUDA, install the toolkit yourself in your Dockerfile (`apt` or the
`nvidia-*` pip packages); the base already includes `git`/`gcc`/`make`/`curl`.

## Two rules that make or break a submission

1. **Download everything at build time** (dependencies *and* model weights). The
   inference phase has **no network**.
2. Target a **single GPU** (multi-GPU/NCCL may be blocked by the no-network
   sandbox) and keep your framework's CUDA build compatible with the host driver.

## Quick start

```bash
cp -r template my-submission && cd my-submission
# edit requirements.txt, run.py, submission.yaml; bake weights in the Dockerfile

# Build FROM the pinned base and extract the delta layers (no upload yet).
# The tool defaults --base-image to the pinned challenge base; pass it
# explicitly to be safe:
python ../build_submission.py \
  --context . \
  --base-image python:3.12@sha256:2575347025c314e37d89d4b353904edbe1824a6117b8eeffe52254879e4f6146 \
  --no-push
# -> .cascade-build/manifest.json + .cascade-build/layers/layer-*.tar

# Publish the artifact to a private model repo in your personal HF namespace.
# The tool creates the repo if it does not exist; "model" and private are the
# defaults. The token must belong to the namespace named by --repo-id.
HF_TOKEN=*** python ../build_submission.py \
  --context . \
  --base-image python:3.12@sha256:2575347025c314e37d89d4b353904edbe1824a6117b8eeffe52254879e4f6146 \
  --repo-id your-hf-username/your-submission
```

The tool runs `docker build` locally, so use a machine with Docker. Use the exact
pinned base digest from the challenge instructions. Keep the model repo
**private**. In its Hugging Face access settings, grant the user `grossanchez`
read access so the private evaluator can pin and fetch your artifact. Register
that model repo ID in the challenge frontend using the same HF account that owns
the namespace. Do not submit a Space, a dataset repo, or a container image.

## Test it locally with no network (recommended)

Reconstruct the exact artifact on the base and run it with the network disabled —
this is what the evaluator does, so it catches accidental inference-time
downloads before you submit:

```bash
python reconstruct_submission.py \
  --artifact my-submission/.cascade-build \
  --base-image python:3.12@sha256:2575347025c314e37d89d4b353904edbe1824a6117b8eeffe52254879e4f6146 \
  --input /path/to/sample-input --output /tmp/out --network none
```

For a scored local run against a CASCADE split, use the `self-evaluation-kit`.

## What gets published to the model repo

- `manifest.json` — pins the base image identity (layer diff-ids), lists the
  ordered delta layers + sha256s, and declares the launch `entrypoint`/`env`.
- `layers/layer-*.tar` — only the layers your build added on top of the base.

The model repo is an artifact store for these files; it is not a model that the
Hub serves and it is not a runnable Docker Space.

The delta is valid **only** on the exact base it was built on, so always use the
base **pinned by digest** from the challenge instructions.

## Layout

- `template/` — copy this to start your submission.
- `build_submission.py` — build → extract delta layers → manifest → push.
- `reconstruct_submission.py` + `_apply_layers.py` — reapply the delta on the
  base and run it (used for local no-network checks and by the evaluator).
- `validate_submission.py` — check `predictions.jsonl` shape locally.
