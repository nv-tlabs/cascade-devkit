# CASCADE common base image

This is the shared base that makes the submission contract work. Both the
participant submission and the official evaluator use the **same** image, so an
environment built on top of it is ABI-compatible at evaluation time.

## Why a shared base

Submissions are not run as containers by the evaluator (the evaluation sandbox
cannot launch nested containers and cannot give a `chroot` GPU access). Instead,
the participant ships only the image **layers they added on top of this base**,
and the evaluator reapplies those layers on this exact base and runs them **in
place**. For that to work, the delta must be built against the exact
OS / libc / CUDA-toolkit / driver ABI it will run on — hence a single published
base for both sides.

**Pin by digest.** A submission's delta layers are only valid on the precise base
they were built on (the manifest records the base's layer diff-ids and the
evaluator verifies them). Publish this image and pin it by digest for the
challenge run.

## Building and publishing (maintainers)

```bash
docker build -t cascade-base:cuda13.0-py312 base-image/
# Tag and push to a registry participants can pull FROM, then pin by digest:
docker tag cascade-base:cuda13.0-py312 <registry>/cascade-base:cuda13.0-py312
docker push <registry>/cascade-base:cuda13.0-py312
```

Publish to a registry participants can reach (e.g. a public GHCR/Docker Hub
repo). Pin the exact digest in the challenge instructions and pass it to the
build tool with `--base-image`.

## Contract guarantees

- Imposes no install prefix — submissions install wherever is natural (venv,
  conda, system packages, `~/.cache`, ...); every layer added on top of this base
  is captured and reapplied at evaluation time.
- Provides `python3`, `pip`, `venv`, `build-essential`, and `git` so submissions
  can create an isolated environment and compile native/CUDA extensions during
  their own build.
- CUDA userspace comes from the `nvidia/cuda` base; the GPU **driver** is
  injected by the evaluation host at run time. Keep the CUDA toolkit you install
  compatible with the host driver.
