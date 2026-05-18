# Manual install

Most users should run `scripts/install.sh` (see [`README.md` §
Install](../../README.md#install)) — it does the same end-to-end work
that the recipe below performs by hand. This doc is for cases where
the script can't be used:

- Non-Debian hosts (Fedora, Arch, macOS, etc.).
- Hardened environments where piping a network-fetched script into a
  shell is disallowed by policy.
- CI runners with custom apt mirrors or air-gapped package caches.
- Anyone who wants to read every command before running it.

## Prerequisites

These need to be on `PATH` before `make install` works:

| Tool | Why |
|------|-----|
| **Python ≥ 3.11** | DevKit baseline. Ships with Ubuntu 22.04+ and Debian Bookworm; older releases need [deadsnakes](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) or `pyenv`. |
| **`make`** | All documented install / test / run targets are make recipes. |
| **Node ≥ 18 (LTS) + npm** | Only used by the annotator frontend (Vite bundle). |
| **`ffmpeg` + `ffprobe`** | Only used by the annotator (HEVC → H.264 transcode + codec detection). See [`tools/annotator/README.md`](../../tools/annotator/README.md) for which features degrade if missing. |
| **`uv`** | Manages your Python environments. Not a Python package — installed separately. |

## Recipe (Ubuntu 22.04+ / Debian Bookworm)

This is what `scripts/install.sh` runs on a fresh Debian-family host.
Adapt the apt commands to your distro's package manager; the Node and
`uv` installers are distro-agnostic.

```bash
sudo apt-get update
sudo apt-get install -y make build-essential ffmpeg

# Node 20 LTS via NodeSource
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# uv via the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh

# Project deps — Python (all extras) + the annotator's npm deps.
make install
```

## Python deps without `make install`

`make install` calls `uv sync --all-extras` and then `npm install` in
`tools/annotator/web`. If you want finer-grained control over the
Python side (e.g. you don't need the notebook tooling, or you don't
want the HF integration), call `uv sync` directly:

```bash
uv sync                                    # core install
uv sync --extra hf                         # + parent-dataset integration (video loading)
uv sync --all-extras --group notebooks     # + notebook tooling (JupyterLab, matplotlib, pandas)
```

The annotator frontend still needs `npm install` for the dev server;
skip it if you only use the DevKit from Python and never launch the
annotator UI.

## Container alternative

A [`.devcontainer/devcontainer.json`](../../.devcontainer/devcontainer.json)
ships with the repo and installs all of the above automatically — open
the folder in VS Code Dev Containers, GitHub Codespaces, or any
compatible host. No host-side setup needed beyond the IDE.

## Verifying the install

```bash
make test
```

Unit tests don't need the corpus; they only verify the install
landed. Set `CASCADE_AV_DATASET_ROOT` before running tests that
iterate the real annotation directory (see [`README.md` § Getting
the data](../../README.md#getting-the-data)).
