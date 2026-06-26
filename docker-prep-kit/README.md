# Docker Preparation Kit

Prepare and publish your submission to the AV Causal Scenario Retrieval
Challenge.

## How evaluation works (and why this kit exists)

The evaluator does **not** run your Docker image as a container. The evaluation
sandbox cannot launch nested containers, and a `chroot` cannot be given GPU
access. So instead:

1. You build your environment **`FROM` the CASCADE common base image**, installing
   everything into the `/opt/submission` prefix.
2. This kit **extracts that prefix** and publishes it as a portable artifact
   (`submission.tar.gz` + `manifest.json`) to your private Hugging Face Space.
3. The evaluator restores your prefix on the **identical** base image and runs it
   **in place**, on GPU, in two phases:
   - **fetch phase** — network on, no test data present;
   - **inference phase** — network **disabled** (egress blocked), hidden test
     inputs mounted, your entrypoint runs.

Because both sides share one base image, your environment is ABI-compatible at
evaluation time — no relocation or CUDA/glibc surprises.

## Quick start

```bash
# 1. Copy the template and make it your own.
cp -r template my-submission && cd my-submission
# edit requirements.txt, run.py, submission.yaml; bake weights in the Dockerfile

# 2. Build, extract, and publish to your private Space.
python ../build_submission.py \
  --context . \
  --base-image <registry>/cascade-base:cuda13.0-py312 \
  --repo-id your-team/your-submission \
  --repo-type space

# Preview everything without Docker or the Hub:
python ../build_submission.py --context . --dry-run

# Build and inspect the artifact locally without publishing:
python ../build_submission.py --context . --no-push
```

The tool runs `docker build` locally, so run it on a machine with Docker. It
needs an `HF_TOKEN` (or `--token`) with write access to push.

## What gets published

- `submission.tar.gz` — the contents of `/opt/submission` from your built image.
- `manifest.json` — base image, prefix, launch `entrypoint`, `env`, the
  artifact's sha256, and CUDA target. The evaluator reads this to restore and run
  your submission.

## Rules

- Install **everything under `/opt/submission`** (use the venv/conda env in the
  prefix). Files outside the prefix are not shipped.
- **Download weights at build time** — the inference phase has no network.
- Keep the **CUDA toolkit compatible with the host driver**; target a single GPU.
- The entrypoint must read `/input` and write `/output/predictions.jsonl` (see
  `template/README.md`).

## Local prediction check

After a local run that produced `predictions.jsonl`, validate its shape:

```bash
python validate_submission.py \
  --predictions /tmp/out/predictions.jsonl \
  --queries /path/to/input/queries.jsonl \
  --videos /path/to/input/videos.jsonl
```

## Layout

- `base-image/` — the shared base image (the ABI contract). See its README.
- `template/` — copy this to start your submission.
- `build_submission.py` — build → extract prefix → manifest → push.
- `validate_submission.py` — check `predictions.jsonl` shape locally.
