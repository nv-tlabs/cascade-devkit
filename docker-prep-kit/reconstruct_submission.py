"""Reconstruct a CASCADE submission artifact on the base image and run it.

Applies the submission's delta layers onto the base image *in place*, then runs
the declared entrypoint against a mounted ``/input``, writing ``/output``. This
is the core shared by:

  - local self-checks (this script / the self-evaluation kit), which disable the
    network with ``docker run --network none``; and
  - the official evaluator, which runs the same applier under the seccomp egress
    sandbox instead.

Running in place (no chroot) keeps GPU access working, because the container uses
the host's GPU device nodes and injected driver.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KIT_DIR = Path(__file__).resolve().parent
APPLIER = KIT_DIR / "_apply_layers.py"


class ReconstructError(Exception):
    pass


def load_manifest(artifact_dir: Path) -> dict[str, Any]:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ReconstructError(f"No manifest.json in {artifact_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def inspect_layers(image: str) -> list[str]:
    out = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", image],
        check=True, capture_output=True, text=True,
    ).stdout
    return list(json.loads(out))


def verify_base(base_image: str, manifest: dict[str, Any]) -> None:
    """Ensure the base image matches the one the delta was built on."""
    expected = list(manifest.get("base_image", {}).get("diff_ids", []))
    if not expected:
        return
    actual = inspect_layers(base_image)
    if actual != expected:
        raise ReconstructError(
            f"Base image {base_image} does not match the manifest's base "
            "(layer diff-ids differ). The delta layers are only valid on the "
            "exact base they were built on; pull/pin the correct base image."
        )


@dataclass
class RunSpec:
    artifact_dir: Path
    base_image: str
    input_dir: Path
    output_dir: Path
    network: str = "none"
    gpus: str | None = None
    extra_mounts: list[str] = field(default_factory=list)


def docker_run_command(spec: RunSpec) -> list[str]:
    cmd = [
        "docker", "run", "--rm",
        "--network", spec.network,
    ]
    if spec.gpus:
        cmd += ["--gpus", spec.gpus]
    cmd += [
        "-v", f"{spec.artifact_dir.resolve()}:/artifact:ro",
        "-v", f"{APPLIER}:/cascade/_apply_layers.py:ro",
        "-v", f"{spec.input_dir.resolve()}:/input:ro",
        "-v", f"{spec.output_dir.resolve()}:/output",
    ]
    for mount in spec.extra_mounts:
        cmd += ["-v", mount]
    cmd += [
        spec.base_image,
        "python3", "/cascade/_apply_layers.py", "/artifact",
    ]
    return cmd


def run(spec: RunSpec, *, verify: bool = True) -> None:
    manifest = load_manifest(spec.artifact_dir)
    if verify:
        verify_base(spec.base_image, manifest)
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(docker_run_command(spec), check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reconstruct and run a submission artifact.")
    parser.add_argument("--artifact", type=Path, required=True, help="Artifact dir (manifest.json + layers/).")
    parser.add_argument("--base-image", required=True, help="Base image to reconstruct on (must match manifest).")
    parser.add_argument("--input", type=Path, required=True, help="Directory mounted read-only at /input.")
    parser.add_argument("--output", type=Path, required=True, help="Directory mounted at /output.")
    parser.add_argument("--network", default="none", help="Docker network (default: none = no egress).")
    parser.add_argument("--gpus", default=None, help="Pass through to docker --gpus, e.g. 'all'.")
    parser.add_argument("--mount", action="append", default=[], help="Extra docker -v mount (repeatable).")
    parser.add_argument("--no-verify-base", action="store_true", help="Skip base-image identity check.")
    args = parser.parse_args(argv)

    spec = RunSpec(
        artifact_dir=args.artifact.resolve(),
        base_image=args.base_image,
        input_dir=args.input,
        output_dir=args.output,
        network=args.network,
        gpus=args.gpus,
        extra_mounts=list(args.mount),
    )
    try:
        run(spec, verify=not args.no_verify_base)
    except ReconstructError as exc:
        print(f"ERROR: {exc}")
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: reconstruction run failed: {exc}")
        return 1
    print(f"Wrote outputs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
