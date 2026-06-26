# CASCADE common base image

This is the shared base that makes the submission contract work. Both the
participant submission and the official evaluator use the **same** image, so an
environment built on top of it is ABI-compatible at evaluation time.

## Why a shared base

Submissions are not run as containers by the evaluator (the evaluation sandbox
cannot launch nested containers and cannot give a `chroot` GPU access). Instead,
the participant's `/opt/submission` prefix is restored on top of this base and
run **in place**. For that to work reliably, the environment must be built
against the exact OS / libc / CUDA-toolkit / driver ABI it will run on — hence a
single published base for both sides.

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

- Provides a fixed prefix at `/opt/submission` (also exported as
  `$SUBMISSION_PREFIX`). Everything a submission needs must live there.
- Provides `python3`, `pip`, `venv`, `build-essential`, and `git` so submissions
  can create an isolated environment and compile native/CUDA extensions during
  their own build.
- CUDA userspace comes from the `nvidia/cuda` base; the GPU **driver** is
  injected by the evaluation host at run time. Keep the CUDA toolkit you install
  compatible with the host driver.
