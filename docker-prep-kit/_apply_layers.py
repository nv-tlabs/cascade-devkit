"""Apply a submission's delta layers onto the running base, then exec it.

Runs INSIDE the base container during reconstruction (local self-eval and the
official evaluator). Stdlib only, so it keeps working even if a layer overwrites
interpreter files on disk (this process is already loaded). It:

  1. reads /artifact/manifest.json,
  2. applies each delta layer tar over "/" in order, honoring OCI/AUFS
     whiteouts (`.wh.<name>` deletions and `.wh..wh..opq` opaque dirs),
  3. sets the manifest env, and
  4. execs the manifest entrypoint.

The network is expected to already be disabled by the caller (``--network none``
locally, or the seccomp egress sandbox in the official evaluator).
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

WHITEOUT_PREFIX = ".wh."
OPAQUE_MARKER = ".wh..wh..opq"


def _open_tar(path: Path) -> tarfile.TarFile:
    with path.open("rb") as probe:
        magic = probe.read(2)
    if magic == b"\x1f\x8b":  # gzip
        return tarfile.open(fileobj=gzip.open(path, "rb"), mode="r|")
    return tarfile.open(path, mode="r")


def _apply_layer(layer_path: Path, root: Path) -> None:
    with _open_tar(layer_path) as tar:
        for member in tar:
            name = member.name.lstrip("./")
            if not name:
                continue
            base = os.path.basename(name)
            target_dir = root / os.path.dirname(name)

            if base == OPAQUE_MARKER:
                # Clear existing contents of the directory marked opaque.
                if target_dir.is_dir():
                    for child in target_dir.iterdir():
                        _remove(child)
                continue

            if base.startswith(WHITEOUT_PREFIX):
                victim = target_dir / base[len(WHITEOUT_PREFIX):]
                _remove(victim)
                continue

            tar.extract(member, path=root, set_attrs=True, numeric_owner=True)


def _remove(path: Path) -> None:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass


def main() -> int:
    artifact = Path(sys.argv[1] if len(sys.argv) > 1 else "/artifact")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))

    root = Path("/")
    for layer in manifest["layers"]:
        _apply_layer(artifact / "layers" / layer["name"], root)

    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (manifest.get("env") or {}).items()})

    entrypoint = manifest["entrypoint"]
    if not entrypoint:
        print("ERROR: manifest has no entrypoint", file=sys.stderr)
        return 2
    os.execvpe(entrypoint[0], list(entrypoint), env)


if __name__ == "__main__":
    raise SystemExit(main())
