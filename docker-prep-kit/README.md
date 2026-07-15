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

Official evaluations run on one **NVIDIA A100 80 GB** GPU. The A100 has CUDA
compute capability **8.0**, so your framework wheels and every compiled CUDA
extension must include **`sm_80`** kernels.

There is **no CUDA base image**. Your GPU stack comes from your framework's pip
wheels — e.g. `pip install torch` pulls the matching CUDA + cuDNN — plus the GPU
**driver** provided by the evaluation host. A CUDA version label by itself does
not prove GPU compatibility: the framework runtime must be supported by the host
driver **and** its binaries must target `sm_80`. Being below a maximum CUDA
version is not, by itself, a compatibility test.

If you compile custom CUDA, install the toolkit yourself in your Dockerfile
(`apt` or the `nvidia-*` pip packages) and compile for `sm_80`. For PyTorch CUDA
extensions, set `TORCH_CUDA_ARCH_LIST="8.0"` during the build. The base already
includes `git`/`gcc`/`make`/`curl`.

## Two rules that make or break a submission

1. **Download everything at build time** (dependencies *and* model weights). The
   inference phase has **no network**.
2. Target the evaluation's **single A100 80 GB (`sm_80`)**. Multi-GPU/NCCL may
   be blocked by the no-network sandbox.

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

On constrained hosts with no working Docker bridge — rootless, or a daemon
started without `iptables`/`nft` (e.g. inside an unprivileged container) —
build-time `apt`/`pip`/weight downloads can fail DNS resolution. Pass
`--network host` to build in the host network namespace and sidestep that. It
affects only the build; evaluation always runs with the network disabled.

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

### PyTorch GPU preflight

After `build_submission.py --no-push`, run this against the exact local image
(the default tag is `cascade-submission:local`):

```bash
docker run --rm -i --gpus all --network none \
  --entrypoint /opt/app/.venv/bin/python \
  cascade-submission:local - <<'PY'
import torch

required_arch = "sm_80"
arches = set(torch.cuda.get_arch_list())
if required_arch not in arches:
    raise SystemExit(f"PyTorch wheel is missing {required_arch}; compiled for {sorted(arches)}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the submission image")

name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
torch.testing.assert_close((x.square() + 1).cpu(), torch.tensor([2.0, 5.0, 10.0]))
torch.cuda.synchronize()
print({"device": name, "capability": capability, "torch_cuda": torch.version.cuda,
       "compiled_arches": sorted(arches), "kernel": "ok"})
PY
```

On an A100, `capability` must print `(8, 0)`. On another local GPU, the kernel
checks that GPU while `get_arch_list()` separately confirms that the installed
PyTorch library contains `sm_80`. This does not inspect third-party or custom
CUDA extensions; build those for `sm_80` and exercise their real inference path
on an A100 before publishing.

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
