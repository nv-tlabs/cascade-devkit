"""Build a CASCADE submission and publish it as an image-layer delta.

The evaluator cannot run participant Docker images directly (the sandbox has no
nested containers, and a ``chroot`` cannot be given GPU access). Instead,
participants build their image ``FROM`` the shared CASCADE base image, and this
tool ships only the layers they added **on top of** the base.

Because a submission is built FROM the base, its filesystem layers are exactly
the base's layers plus the participant's new ones. We keep only the new (delta)
layers; at evaluation time they are reapplied on the identical base and run in
place, on GPU, with the network disabled. Dependencies can live anywhere
(system site-packages, conda, ``~/.cache/huggingface``, ...) — whatever the build
added is captured.

Pipeline:
    1. ``docker build`` the image FROM the base.
    2. Verify the base's layers are a prefix of the submission's layers.
    3. ``docker save`` and copy out only the added (delta) layers.
    4. Write ``manifest.json`` (base identity, ordered layers, entrypoint/env).
    5. Push the manifest + layers to a private Hugging Face repo (Space).

The delta is only valid on the exact base it was built on, so the manifest pins
the base image identity (its layer diff-ids). Use ``--dry-run`` to preview.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a declared dependency
    yaml = None


TOOL_VERSION = "2.0.0"
MANIFEST_SCHEMA_VERSION = 2
DEFAULT_BASE_IMAGE = "ghcr.io/nv-tlabs/cascade-base:cuda13.0-py312"
DEFAULT_CONFIG_NAME = "submission.yaml"
MANIFEST_NAME = "manifest.json"
LAYERS_DIRNAME = "layers"


class SubmissionError(Exception):
    """User-facing submission build/validation problem."""


@dataclass
class SubmissionConfig:
    """Declares how the evaluator launches a submission."""

    entrypoint: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cuda: str | None = None
    notes: str | None = None


# --------------------------------------------------------------------------- #
# Pure logic (unit-tested without Docker or the Hub)
# --------------------------------------------------------------------------- #
def parse_submission_config(text: str) -> SubmissionConfig:
    if yaml is None:  # pragma: no cover - defensive
        raise SubmissionError("PyYAML is required to read submission.yaml")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise SubmissionError("submission.yaml must be a mapping")

    entrypoint = data.get("entrypoint")
    if not isinstance(entrypoint, list) or not entrypoint:
        raise SubmissionError(
            "submission.yaml must set 'entrypoint' to a non-empty list, e.g.\n"
            '  entrypoint: ["/opt/app/.venv/bin/python", "/opt/app/run.py"]'
        )
    if not all(isinstance(part, str) for part in entrypoint):
        raise SubmissionError("submission.yaml 'entrypoint' entries must all be strings")

    env_raw = data.get("env", {}) or {}
    if not isinstance(env_raw, dict):
        raise SubmissionError("submission.yaml 'env' must be a mapping of string to string")
    env = {str(key): str(value) for key, value in env_raw.items()}

    return SubmissionConfig(
        entrypoint=[str(part) for part in entrypoint],
        env=env,
        cuda=(str(data["cuda"]) if data.get("cuda") is not None else None),
        notes=(str(data["notes"]) if data.get("notes") is not None else None),
    )


def load_submission_config(context: Path, config_name: str = DEFAULT_CONFIG_NAME) -> SubmissionConfig:
    config_path = context / config_name
    if not config_path.is_file():
        raise SubmissionError(
            f"Missing {config_name} in build context {context}. It must declare the "
            "evaluator entrypoint."
        )
    return parse_submission_config(config_path.read_text(encoding="utf-8"))


def compute_delta_layers(base_layers: list[str], submission_layers: list[str]) -> list[str]:
    """Return the submission layers added on top of the base.

    The submission must be built FROM the base, so the base's layer diff-ids are a
    prefix of the submission's. Returns the remaining (delta) diff-ids.
    """
    if len(submission_layers) < len(base_layers) or submission_layers[: len(base_layers)] != base_layers:
        raise SubmissionError(
            "The submission image was not built FROM the given base image "
            "(its layers do not extend the base's). Pass the correct --base-image "
            "and use `FROM <base>` in your Dockerfile."
        )
    delta = submission_layers[len(base_layers):]
    if not delta:
        raise SubmissionError(
            "The submission adds no layers on top of the base — nothing to ship. "
            "Install your dependencies/code/weights in the Dockerfile."
        )
    return delta


def read_save_layer_paths(save_dir: Path) -> list[str]:
    """Return the ordered layer blob paths from a `docker save` archive."""
    manifest_path = save_dir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not data or "Layers" not in data[0]:
        raise SubmissionError("Unexpected `docker save` archive: no Layers in manifest.json")
    return list(data[0]["Layers"])


def docker_build_command(*, context: Path, dockerfile: Path, base_image: str, image_tag: str) -> list[str]:
    return [
        "docker", "build",
        "--build-arg", f"BASE_IMAGE={base_image}",
        "--tag", image_tag,
        "--file", str(dockerfile),
        str(context),
    ]


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    base_image: str,
    base_image_id: str,
    base_layers: list[str],
    layer_files: list[dict[str, Any]],
    config: SubmissionConfig,
    created_at: float | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "base_image": {
            "ref": base_image,
            "id": base_image_id,
            "diff_ids": list(base_layers),
        },
        "layers": layer_files,
        "entrypoint": list(config.entrypoint),
        "env": dict(config.env),
        "cuda": config.cuda,
        "notes": config.notes,
        "created_at": _iso(created_at if created_at is not None else time.time()),
    }


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


# --------------------------------------------------------------------------- #
# Docker helpers (side-effecting)
# --------------------------------------------------------------------------- #
def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _capture(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def inspect_layers(image: str) -> list[str]:
    out = _capture(["docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", image])
    return list(json.loads(out))


def inspect_id(image: str) -> str:
    return _capture(["docker", "image", "inspect", "--format", "{{.Id}}", image]).strip()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class BuildPlan:
    context: Path
    dockerfile: Path
    base_image: str
    image_tag: str
    staging_dir: Path
    config: SubmissionConfig

    @property
    def layers_dir(self) -> Path:
        return self.staging_dir / LAYERS_DIRNAME

    @property
    def manifest_path(self) -> Path:
        return self.staging_dir / MANIFEST_NAME


def build_and_extract_delta(plan: BuildPlan, *, skip_build: bool = False) -> dict[str, Any]:
    """Build the image, extract the delta layers, and write the manifest."""
    if not skip_build:
        _run(docker_build_command(
            context=plan.context,
            dockerfile=plan.dockerfile,
            base_image=plan.base_image,
            image_tag=plan.image_tag,
        ))

    base_layers = inspect_layers(plan.base_image)
    submission_layers = inspect_layers(plan.image_tag)
    delta = compute_delta_layers(base_layers, submission_layers)
    base_count = len(base_layers)

    plan.layers_dir.mkdir(parents=True, exist_ok=True)
    for existing in plan.layers_dir.glob("layer-*.tar"):
        existing.unlink()

    layer_files: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        save_tar = Path(tmp) / "image.tar"
        save_dir = Path(tmp) / "save"
        save_dir.mkdir()
        _run(["docker", "save", plan.image_tag, "-o", str(save_tar)])
        _run(["tar", "xf", str(save_tar), "-C", str(save_dir)])

        all_layer_paths = read_save_layer_paths(save_dir)
        delta_layer_paths = all_layer_paths[base_count:]
        if len(delta_layer_paths) != len(delta):
            raise SubmissionError(
                "Mismatch between inspected delta layers and saved layers; "
                "rebuild and retry."
            )

        for index, rel_path in enumerate(delta_layer_paths):
            src = save_dir / rel_path
            dest = plan.layers_dir / f"layer-{index:02d}.tar"
            shutil.copyfile(src, dest)
            layer_files.append({
                "name": dest.name,
                "sha256": sha256_file(dest),
                "bytes": dest.stat().st_size,
            })

    manifest = build_manifest(
        base_image=plan.base_image,
        base_image_id=inspect_id(plan.base_image),
        base_layers=base_layers,
        layer_files=layer_files,
        config=plan.config,
    )
    plan.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def push_to_hub(plan: BuildPlan, *, repo_id: str, repo_type: str, private: bool, token: str | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(plan.manifest_path),
        path_in_repo=MANIFEST_NAME,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        commit_message="Add submission manifest",
    )
    api.upload_folder(
        folder_path=str(plan.layers_dir),
        path_in_repo=LAYERS_DIRNAME,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        commit_message="Add submission layers",
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context", type=Path, default=Path("."), help="Build context with the Dockerfile.")
    parser.add_argument("--dockerfile", type=Path, default=None, help="Dockerfile path (default: <context>/Dockerfile).")
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE, help="CASCADE base image to build FROM.")
    parser.add_argument("--image-tag", default="cascade-submission:local", help="Local tag for the built image.")
    parser.add_argument("--output-dir", type=Path, default=Path(".cascade-build"), help="Staging directory.")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME, help="Submission config file name in context.")
    parser.add_argument("--repo-id", default=None, help="Target HF repo id, e.g. your-team/your-submission.")
    parser.add_argument("--repo-type", default="space", choices=["space", "dataset", "model"], help="HF repo type.")
    parser.add_argument("--public", action="store_true", help="Create the repo public (default: private).")
    parser.add_argument("--token", default=None, help="HF token (default: HF_TOKEN env).")
    parser.add_argument("--skip-build", action="store_true", help="Reuse an existing local image tag.")
    parser.add_argument("--no-push", action="store_true", help="Build and extract only; do not push.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running Docker or pushing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    context = args.context.resolve()
    dockerfile = (args.dockerfile or (context / "Dockerfile")).resolve()
    staging = args.output_dir.resolve()

    if not context.is_dir():
        print(f"ERROR: build context not found: {context}", file=sys.stderr)
        return 2
    if not dockerfile.is_file() and not args.dry_run:
        print(f"ERROR: Dockerfile not found: {dockerfile}", file=sys.stderr)
        return 2

    try:
        config = load_submission_config(context, args.config_name)
    except SubmissionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    plan = BuildPlan(
        context=context,
        dockerfile=dockerfile,
        base_image=args.base_image,
        image_tag=args.image_tag,
        staging_dir=staging,
        config=config,
    )

    if args.dry_run:
        print("[dry-run] would build:", " ".join(docker_build_command(
            context=context, dockerfile=dockerfile, base_image=args.base_image, image_tag=args.image_tag,
        )))
        print("[dry-run] would docker save, keep only the layers added on top of the base,")
        print(f"[dry-run] and write {plan.manifest_path} + {plan.layers_dir}/layer-*.tar")
        print(f"[dry-run] entrypoint: {config.entrypoint}")
        return 0

    staging.mkdir(parents=True, exist_ok=True)
    try:
        manifest = build_and_extract_delta(plan, skip_build=args.skip_build)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"ERROR: docker step failed: {exc}", file=sys.stderr)
        return 1
    except SubmissionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    total = sum(layer["bytes"] for layer in manifest["layers"])
    print(f"Extracted {len(manifest['layers'])} delta layer(s), {total} bytes total")
    print(f"Wrote manifest: {plan.manifest_path}")
    print(f"Layers: {plan.layers_dir}")

    if args.no_push:
        print("Skipping push (--no-push).")
        return 0

    if not args.repo_id:
        print("ERROR: --repo-id is required to push (or pass --no-push).", file=sys.stderr)
        return 2

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: no HF token (pass --token or set HF_TOKEN).", file=sys.stderr)
        return 2

    try:
        push_to_hub(plan, repo_id=args.repo_id, repo_type=args.repo_type, private=not args.public, token=token)
    except Exception as exc:  # noqa: BLE001 - surface hub errors to the user
        print(f"ERROR: push failed: {exc}", file=sys.stderr)
        return 1

    print(f"Pushed submission to {args.repo_type}: {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
