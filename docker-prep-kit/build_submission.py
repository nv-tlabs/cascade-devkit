"""Build a CASCADE submission and publish it as a portable environment artifact.

The challenge evaluator does not run participant Docker images directly (the
evaluation sandbox cannot launch nested containers, and a ``chroot`` cannot be
given GPU access). Instead, participants build their environment ``FROM`` the
shared CASCADE base image, and this tool extracts the ``/opt/submission`` prefix
they installed into and publishes it as a tarball plus a ``manifest.json``.

At evaluation time the prefix is restored on the identical base image and run in
place, on GPU, with network access disabled.

Pipeline:
    1. ``docker build`` the submission image ``FROM`` the common base.
    2. Extract the ``/opt/submission`` prefix from the built image.
    3. Pack it into ``submission.tar.gz`` and write ``manifest.json``.
    4. Push the artifact + manifest to a private Hugging Face repo (Space).

Use ``--dry-run`` to see every step without invoking Docker or the Hub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a declared dependency
    yaml = None


TOOL_VERSION = "1.0.0"
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_BASE_IMAGE = "ghcr.io/nv-tlabs/cascade-base:cuda13.0-py312"
DEFAULT_PREFIX = "/opt/submission"
DEFAULT_CONFIG_NAME = "submission.yaml"
ARTIFACT_NAME = "submission.tar.gz"
MANIFEST_NAME = "manifest.json"


class SubmissionError(Exception):
    """Raised for user-facing submission build/validation problems."""


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
    """Parse and validate a ``submission.yaml`` document."""
    if yaml is None:  # pragma: no cover - defensive
        raise SubmissionError("PyYAML is required to read submission.yaml")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise SubmissionError("submission.yaml must be a mapping")

    entrypoint = data.get("entrypoint")
    if not isinstance(entrypoint, list) or not entrypoint:
        raise SubmissionError(
            "submission.yaml must set 'entrypoint' to a non-empty list, e.g.\n"
            '  entrypoint: ["/opt/submission/env/bin/python", "/opt/submission/run.py"]'
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


def docker_build_command(
    *,
    context: Path,
    dockerfile: Path,
    base_image: str,
    image_tag: str,
) -> list[str]:
    return [
        "docker",
        "build",
        "--build-arg",
        f"BASE_IMAGE={base_image}",
        "--tag",
        image_tag,
        "--file",
        str(dockerfile),
        str(context),
    ]


def docker_create_command(image_tag: str) -> list[str]:
    return ["docker", "create", image_tag]


def docker_copy_prefix_command(container_id: str, prefix: str, destination: Path) -> list[str]:
    # Trailing '/.' copies the directory contents into destination.
    return ["docker", "cp", f"{container_id}:{prefix}/.", str(destination)]


def docker_remove_command(container_id: str) -> list[str]:
    return ["docker", "rm", "-f", container_id]


def tar_command(*, prefix_dir: Path, artifact_path: Path) -> list[str]:
    return [
        "tar",
        "-C",
        str(prefix_dir),
        "-czf",
        str(artifact_path),
        ".",
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
    prefix: str,
    config: SubmissionConfig,
    artifact_sha256: str,
    artifact_bytes: int,
    created_at: float | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "base_image": base_image,
        "prefix": prefix,
        "entrypoint": list(config.entrypoint),
        "env": dict(config.env),
        "cuda": config.cuda,
        "notes": config.notes,
        "artifact": {
            "path": ARTIFACT_NAME,
            "sha256": artifact_sha256,
            "bytes": artifact_bytes,
        },
        "created_at": _iso(created_at if created_at is not None else time.time()),
    }


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


# --------------------------------------------------------------------------- #
# Side-effecting orchestration
# --------------------------------------------------------------------------- #
CommandRunner = Callable[[Sequence[str]], None]


def _default_runner(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


@dataclass
class BuildPlan:
    context: Path
    dockerfile: Path
    base_image: str
    prefix: str
    image_tag: str
    staging_dir: Path
    config: SubmissionConfig

    @property
    def prefix_dir(self) -> Path:
        return self.staging_dir / "submission"

    @property
    def artifact_path(self) -> Path:
        return self.staging_dir / ARTIFACT_NAME

    @property
    def manifest_path(self) -> Path:
        return self.staging_dir / MANIFEST_NAME


def build_and_extract(
    plan: BuildPlan,
    *,
    skip_build: bool = False,
    dry_run: bool = False,
    runner: CommandRunner | None = None,
    capture: Callable[[Sequence[str]], str] | None = None,
) -> None:
    """Build the image and extract the submission prefix into staging."""
    runner = runner or _default_runner
    capture = capture or _capture_output

    plan.prefix_dir.mkdir(parents=True, exist_ok=True)

    build_cmd = docker_build_command(
        context=plan.context,
        dockerfile=plan.dockerfile,
        base_image=plan.base_image,
        image_tag=plan.image_tag,
    )
    if dry_run:
        print("[dry-run] would build:", " ".join(build_cmd))
    elif not skip_build:
        runner(build_cmd)

    if dry_run:
        print("[dry-run] would create container, copy", plan.prefix, "and pack the tarball")
        return

    container_id = capture(docker_create_command(plan.image_tag)).strip()
    try:
        runner(docker_copy_prefix_command(container_id, plan.prefix, plan.prefix_dir))
    finally:
        try:
            runner(docker_remove_command(container_id))
        except Exception:  # pragma: no cover - best-effort cleanup
            pass

    runner(tar_command(prefix_dir=plan.prefix_dir, artifact_path=plan.artifact_path))


def _capture_output(command: Sequence[str]) -> str:
    result = subprocess.run(list(command), check=True, capture_output=True, text=True)
    return result.stdout


def write_manifest(plan: BuildPlan) -> dict[str, Any]:
    artifact_bytes = plan.artifact_path.stat().st_size
    manifest = build_manifest(
        base_image=plan.base_image,
        prefix=plan.prefix,
        config=plan.config,
        artifact_sha256=sha256_file(plan.artifact_path),
        artifact_bytes=artifact_bytes,
    )
    plan.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def push_to_hub(
    plan: BuildPlan,
    *,
    repo_id: str,
    repo_type: str,
    private: bool,
    token: str | None,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    for path in (plan.artifact_path, plan.manifest_path):
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message=f"Add {path.name}",
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context", type=Path, default=Path("."), help="Build context with the Dockerfile.")
    parser.add_argument("--dockerfile", type=Path, default=None, help="Dockerfile path (default: <context>/Dockerfile).")
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE, help="CASCADE common base image to build FROM.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Submission prefix to extract.")
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

    staging.mkdir(parents=True, exist_ok=True)
    plan = BuildPlan(
        context=context,
        dockerfile=dockerfile,
        base_image=args.base_image,
        prefix=args.prefix,
        image_tag=args.image_tag,
        staging_dir=staging,
        config=config,
    )

    try:
        build_and_extract(plan, skip_build=args.skip_build, dry_run=args.dry_run)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"ERROR: docker step failed: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        manifest_preview = build_manifest(
            base_image=plan.base_image,
            prefix=plan.prefix,
            config=plan.config,
            artifact_sha256="<computed-after-build>",
            artifact_bytes=0,
        )
        print("[dry-run] manifest preview:")
        print(json.dumps(manifest_preview, indent=2, sort_keys=True))
        return 0

    manifest = write_manifest(plan)
    print(f"Built artifact: {plan.artifact_path} ({manifest['artifact']['bytes']} bytes)")
    print(f"Wrote manifest: {plan.manifest_path}")

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
        push_to_hub(
            plan,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            private=not args.public,
            token=token,
        )
    except Exception as exc:  # noqa: BLE001 - surface hub errors to the user
        print(f"ERROR: push failed: {exc}", file=sys.stderr)
        return 1

    print(f"Pushed submission to {args.repo_type}: {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
