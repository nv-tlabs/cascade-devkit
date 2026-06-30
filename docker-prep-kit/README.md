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

For private or gated weights, use the cached `hf auth login` on the host to
download them into the submission build context, then `COPY` those weight files
in the Dockerfile. Never put an HF credential in `ARG`, `ENV`, a Dockerfile
command, or the build context: every resulting delta layer is published.

## Quick start

Install [uv](https://docs.astral.sh/uv/), then run these commands from the
`docker-prep-kit` directory. `uv run --project ../.. --extra hf` creates/uses the
project environment with PyYAML and the Hugging Face CLI; no activated virtual
environment or system packages are assumed.

```bash
cp -r template my-submission && cd my-submission
# edit requirements.txt, run.py, submission.yaml; bake weights in the Dockerfile

# Build FROM the pinned base and extract the delta layers (no upload yet).
# The tool defaults --base-image to the pinned challenge base; pass it
# explicitly to be safe:
uv run --project ../.. --extra hf python ../build_submission.py \
  --context . \
  --base-image python:3.12@sha256:2575347025c314e37d89d4b353904edbe1824a6117b8eeffe52254879e4f6146 \
  --no-push
# -> .cascade-build/manifest.json + .cascade-build/layers/layer-*.tar

# In the challenge frontend, create a submission repository and copy its repo ID.
# It will be a private model repo in your personal HF namespace.

# Authenticate this machine once. The uploader uses the cached login; do not
# paste a token into the command or export one into your shell.
uv run --project ../.. --extra hf hf auth login
uv run --project ../.. --extra hf hf auth whoami

# Publish to the repository created by the challenge frontend.
uv run --project ../.. --extra hf python ../build_submission.py \
  --context . \
  --base-image python:3.12@sha256:2575347025c314e37d89d4b353904edbe1824a6117b8eeffe52254879e4f6146 \
  --repo-id your-hf-username/frontend-generated-submission
```

The tool runs `docker build` locally, so use a machine with Docker. It targets
`linux/amd64` and automatically pulls the digest-pinned base when Docker does not
have that reference in its image store. This explicit materialization is needed
on clean Docker 23+ installations, where BuildKit may otherwise keep the `FROM`
image only in its private build cache. Use the exact pinned base digest from the
challenge instructions.

The uploader deliberately does **not** create repositories or change their
visibility: it requires the challenge frontend to have already created a
**private model repo in the signed-in user's personal namespace**. It refuses
missing, public, organization-owned, and non-model targets. No participant token
is sent to the challenge and no access grant to an organizer's personal account
is required. Return to the challenge frontend after the upload and submit the
uploaded revision. Do not submit a Space, a dataset repo, a public repo, or a
container image.

The manifest and all replacement layers are uploaded in one Hub commit. On a
current `huggingface_hub` client, the final line reports the exact immutable
revision in copyable form:

```text
Artifact revision: your-hf-username/frontend-generated-submission@<commit-sha>
```

Keep that line with your submission record. If another process updates the repo
between validation and upload, the command fails instead of silently racing it.

## Test it locally with no network (recommended)

Reconstruct the exact artifact on the base and run it with the network disabled —
this is what the evaluator does, so it catches accidental inference-time
downloads before you submit:

```bash
uv run --project ../.. --extra hf python ../reconstruct_submission.py \
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

## Operator upload smoke check

Before opening submissions, validate the production flow with a non-admin HF test
account:

1. Have the challenge frontend create the account's private personal model repo.
2. Run `hf auth login`, then publish this template without `--token` or
   `HF_TOKEN`; confirm the uploader reports an exact commit SHA.
3. Confirm that revision's artifact paths contain one `manifest.json` and only
   its current `layers/layer-*.tar` files (no stale layers), then complete an
   evaluation through the frontend.
4. Confirm missing, public, and organization-owned repo IDs are rejected and that
   the participant is never asked to share a token or grant a person access.

## Layout

- `template/` — copy this to start your submission.
- `build_submission.py` — build → extract delta layers → manifest → push.
- `reconstruct_submission.py` + `_apply_layers.py` — reapply the delta on the
  base and run it (used for local no-network checks and by the evaluator).
- `validate_submission.py` — check `predictions.jsonl` shape locally.
