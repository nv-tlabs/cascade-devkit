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
    1. Ensure the pinned base is present in Docker's image store.
    2. ``docker build`` the image FROM the base for the evaluation platform.
    3. Verify the base's layers are a prefix of the submission's layers.
    4. ``docker save`` and copy out only the added (delta) layers.
    5. Write ``manifest.json`` (base identity, ordered layers, entrypoint/env).
    6. Push the manifest + layers to a private Hugging Face model repo.

The delta is only valid on the exact base it was built on, so the manifest pins
the base image identity (its layer diff-ids). Use ``--dry-run`` to preview.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a declared dependency
    yaml = None


TOOL_VERSION = "2.0.1"
MANIFEST_SCHEMA_VERSION = 2
# The shared base is a pinned stock Python image. GPU/CUDA comes from the
# participant's framework pip wheels (e.g. torch bundles CUDA + cuDNN) plus the
# evaluation host driver — no CUDA base image is required. Pin by digest so the
# participant build and the evaluator use the byte-identical base.
DEFAULT_BASE_IMAGE = (
    "python:3.12@sha256:2575347025c314e37d89d4b353904edbe1824a6117b8eeffe52254879e4f6146"
)
DEFAULT_PLATFORM = "linux/amd64"
DEFAULT_CONFIG_NAME = "submission.yaml"
MANIFEST_NAME = "manifest.json"
LAYERS_DIRNAME = "layers"
DEFAULT_REPO_TYPE = "model"
_SHA256_HEX_LENGTH = 64
_CONFIG_KEYS = frozenset({"entrypoint", "env", "cuda", "notes"})
_RUNTIME_ENV_KEYS = frozenset({
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "INPUT_DIR",
    "OUTPUT_DIR",
    "TOP_K",
    "CRC_SUBMISSION_ID",
    "CRC_RUN_TOKEN",
})


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

    unknown_keys = sorted(set(data) - _CONFIG_KEYS, key=str)
    if unknown_keys:
        raise SubmissionError(
            "submission.yaml contains unsupported field(s): "
            + ", ".join(str(key) for key in unknown_keys)
        )

    entrypoint = data.get("entrypoint")
    if not isinstance(entrypoint, list) or not entrypoint:
        raise SubmissionError(
            "submission.yaml must set 'entrypoint' to a non-empty list, e.g.\n"
            '  entrypoint: ["/opt/app/.venv/bin/python", "/opt/app/run.py"]'
        )
    if not all(isinstance(part, str) and part.strip() and "\0" not in part for part in entrypoint):
        raise SubmissionError(
            "submission.yaml 'entrypoint' entries must be non-empty strings without NUL bytes"
        )

    env_raw = data.get("env", {}) or {}
    if not isinstance(env_raw, dict):
        raise SubmissionError("submission.yaml 'env' must be a mapping of string to string")
    env: dict[str, str] = {}
    for key, value in env_raw.items():
        if not isinstance(key, str) or not key or "=" in key or "\0" in key:
            raise SubmissionError(
                "submission.yaml 'env' keys must be non-empty strings without '=' or NUL bytes"
            )
        if key in _RUNTIME_ENV_KEYS:
            raise SubmissionError(
                f"submission.yaml 'env' may not override evaluator-owned variable {key!r}"
            )
        if value is None or isinstance(value, (dict, list)):
            raise SubmissionError(
                f"submission.yaml 'env' value for {key!r} must be a scalar"
            )
        rendered = str(value)
        if "\0" in rendered:
            raise SubmissionError(
                f"submission.yaml 'env' value for {key!r} must not contain a NUL byte"
            )
        env[key] = rendered

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


def _saved_archive_file(save_dir: Path, relative_path: object, *, label: str) -> Path:
    """Resolve a file named by docker-save metadata without escaping the archive."""
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise SubmissionError(f"Unexpected `docker save` archive: invalid {label} path")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise SubmissionError(f"Unexpected `docker save` archive: unsafe {label} path")

    root = save_dir.resolve()
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SubmissionError(
            f"Unexpected `docker save` archive: missing or unsafe {label} path"
        ) from exc
    if not resolved.is_file():
        raise SubmissionError(f"Unexpected `docker save` archive: {label} is not a file")
    return resolved


def _saved_config_digest(relative_path: object) -> str:
    """Read the config digest encoded by classic or OCI-style save paths."""
    if not isinstance(relative_path, str):
        raise SubmissionError("Unexpected `docker save` archive: invalid image config path")
    name = PurePosixPath(relative_path).name
    digest = name[:-5] if name.endswith(".json") else name
    if not _is_sha256(digest, prefix=False):
        raise SubmissionError(
            "Unexpected `docker save` archive: image config path is not content-addressed"
        )
    return digest


def read_save_layer_paths(
    save_dir: Path,
    *,
    image_tag: str,
    expected_repo_tags: list[str],
    expected_diff_ids: list[str],
) -> list[str]:
    """Return validated, ordered layer blob paths from a `docker save` archive.

    Modern Docker image stores may add OCI-layout files to the archive. The
    compatibility manifest remains authoritative, but its first entry is not
    assumed to be the requested image. Instead, select the entry for the saved
    tag and confirm that its config describes the exact inspected RootFS layer
    sequence before using its layer paths. Selection cannot rely on image ID:
    containerd-backed Docker can report an OCI index ID while the archive names
    the underlying image-config digest.
    """
    manifest_path = save_dir / "manifest.json"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubmissionError(
            "Unexpected `docker save` archive: unreadable manifest.json"
        ) from exc
    if not isinstance(data, list) or not data:
        raise SubmissionError(
            "Unexpected `docker save` archive: manifest.json must be a non-empty list"
        )

    requested_tags = {image_tag, *expected_repo_tags}
    matching_entries: list[dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise SubmissionError(
                "Unexpected `docker save` archive: invalid manifest entry"
            )
        repo_tags = entry.get("RepoTags") or []
        if not isinstance(repo_tags, list) or not all(
            isinstance(tag, str) for tag in repo_tags
        ):
            raise SubmissionError(
                "Unexpected `docker save` archive: invalid RepoTags"
            )
        if requested_tags.intersection(repo_tags):
            matching_entries.append(entry)

    if len(matching_entries) > 1:
        exact_entries = [
            entry for entry in matching_entries if image_tag in entry.get("RepoTags", [])
        ]
        if len(exact_entries) == 1:
            matching_entries = exact_entries
    if not matching_entries:
        raise SubmissionError(
            "Unexpected `docker save` archive: the requested submission tag is missing"
        )
    if len(matching_entries) != 1:
        raise SubmissionError(
            "Unexpected `docker save` archive: multiple entries match the "
            "submission image"
        )

    entry = matching_entries[0]
    config_relative_path = entry.get("Config")
    config_path = _saved_archive_file(
        save_dir, config_relative_path, label="image config"
    )
    if sha256_file(config_path) != _saved_config_digest(config_relative_path):
        raise SubmissionError(
            "Unexpected `docker save` archive: image config digest does not match"
        )
    try:
        image_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubmissionError(
            "Unexpected `docker save` archive: unreadable image config"
        ) from exc
    rootfs = image_config.get("rootfs") if isinstance(image_config, dict) else None
    saved_diff_ids = rootfs.get("diff_ids") if isinstance(rootfs, dict) else None
    if saved_diff_ids != expected_diff_ids:
        raise SubmissionError(
            "Mismatch between the inspected submission layers and the saved "
            "image config; rebuild and retry."
        )

    layer_paths = entry.get("Layers")
    if (
        not isinstance(layer_paths, list)
        or len(layer_paths) != len(expected_diff_ids)
        or not all(isinstance(path, str) for path in layer_paths)
    ):
        raise SubmissionError(
            "Mismatch between the saved image config and layer blobs; rebuild and retry."
        )
    for relative_path in layer_paths:
        _saved_archive_file(save_dir, relative_path, label="layer")
    return list(layer_paths)


def docker_pull_command(*, image: str, platform: str = DEFAULT_PLATFORM) -> list[str]:
    return ["docker", "image", "pull", "--platform", platform, image]


def docker_build_command(
    *,
    context: Path,
    dockerfile: Path,
    base_image: str,
    image_tag: str,
    platform: str = DEFAULT_PLATFORM,
) -> list[str]:
    return [
        "docker", "build",
        "--platform", platform,
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
    manifest = {
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
    validate_manifest(manifest)
    return manifest


def _is_sha256(value: object, *, prefix: bool) -> bool:
    if not isinstance(value, str):
        return False
    expected_prefix = "sha256:" if prefix else ""
    if not value.startswith(expected_prefix):
        return False
    digest = value[len(expected_prefix):]
    return len(digest) == _SHA256_HEX_LENGTH and all(char in "0123456789abcdef" for char in digest)


def validate_manifest(manifest: object) -> None:
    """Validate the schema-2 artifact manifest produced by this tool."""
    if not isinstance(manifest, dict):
        raise SubmissionError("manifest must be a mapping")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise SubmissionError(
            f"manifest schema_version must be {MANIFEST_SCHEMA_VERSION}"
        )

    base = manifest.get("base_image")
    if not isinstance(base, dict):
        raise SubmissionError("manifest base_image must be a mapping")
    base_ref = base.get("ref")
    if not isinstance(base_ref, str) or "@sha256:" not in base_ref:
        raise SubmissionError("manifest base_image.ref must be pinned by sha256 digest")
    ref_digest = base_ref.rsplit("@sha256:", 1)[-1]
    if not _is_sha256(ref_digest, prefix=False):
        raise SubmissionError("manifest base_image.ref has an invalid sha256 digest")
    if not _is_sha256(base.get("id"), prefix=True):
        raise SubmissionError("manifest base_image.id must be a sha256 digest")
    diff_ids = base.get("diff_ids")
    if (
        not isinstance(diff_ids, list)
        or not diff_ids
        or not all(_is_sha256(item, prefix=True) for item in diff_ids)
    ):
        raise SubmissionError(
            "manifest base_image.diff_ids must be a non-empty list of sha256 digests"
        )

    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise SubmissionError("manifest layers must be a non-empty list")
    for index, layer in enumerate(layers):
        expected_name = f"layer-{index:02d}.tar"
        if not isinstance(layer, dict) or layer.get("name") != expected_name:
            raise SubmissionError(
                f"manifest layer {index} must be named {expected_name!r}"
            )
        if not _is_sha256(layer.get("sha256"), prefix=False):
            raise SubmissionError(
                f"manifest layer {expected_name!r} has an invalid sha256"
            )
        size = layer.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise SubmissionError(
                f"manifest layer {expected_name!r} must have a positive byte size"
            )

    entrypoint = manifest.get("entrypoint")
    if (
        not isinstance(entrypoint, list)
        or not entrypoint
        or not all(isinstance(part, str) and part.strip() and "\0" not in part for part in entrypoint)
    ):
        raise SubmissionError("manifest entrypoint must be a non-empty list of strings")
    env = manifest.get("env")
    if not isinstance(env, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in env.items()
    ):
        raise SubmissionError("manifest env must be a mapping of strings to strings")


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


# --------------------------------------------------------------------------- #
# Docker helpers (side-effecting)
# --------------------------------------------------------------------------- #
def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _capture(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


@dataclass(frozen=True)
class DockerImageMetadata:
    image_id: str
    layers: list[str]
    os: str
    architecture: str
    repo_tags: tuple[str, ...] = ()

    @property
    def platform(self) -> str:
        return f"{self.os}/{self.architecture}"


def inspect_image(image: str) -> DockerImageMetadata:
    out = _capture(["docker", "image", "inspect", image])
    try:
        result = json.loads(out)
    except json.JSONDecodeError as exc:
        raise SubmissionError(
            f"Docker returned invalid inspection data for image {image!r}"
        ) from exc
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
        raise SubmissionError(
            f"Docker returned unexpected inspection data for image {image!r}"
        )

    details = result[0]
    rootfs = details.get("RootFS")
    layers = rootfs.get("Layers") if isinstance(rootfs, dict) else None
    image_id = details.get("Id")
    os_name = details.get("Os")
    architecture = details.get("Architecture")
    repo_tags = details.get("RepoTags") or []
    if (
        not _is_sha256(image_id, prefix=True)
        or not isinstance(layers, list)
        or not layers
        or not all(_is_sha256(layer, prefix=True) for layer in layers)
        or not isinstance(os_name, str)
        or not os_name
        or not isinstance(architecture, str)
        or not architecture
        or not isinstance(repo_tags, list)
        or not all(isinstance(tag, str) for tag in repo_tags)
    ):
        raise SubmissionError(
            f"Docker returned incomplete inspection data for image {image!r}"
        )
    return DockerImageMetadata(
        image_id=image_id,
        layers=list(layers),
        os=os_name,
        architecture=architecture,
        repo_tags=tuple(repo_tags),
    )


def _require_platform(
    metadata: DockerImageMetadata,
    *,
    image: str,
    expected_platform: str,
) -> None:
    if metadata.platform != expected_platform:
        raise SubmissionError(
            f"Image {image!r} is for {metadata.platform}, but challenge evaluation "
            f"requires {expected_platform}."
        )


def ensure_base_image(
    image: str,
    *,
    platform: str = DEFAULT_PLATFORM,
) -> DockerImageMetadata:
    """Ensure BuildKit's base is also addressable in Docker's image store.

    Docker 23 made BuildKit the default. On a clean daemon, a base fetched by
    BuildKit can remain only in its private cache, so a successful build does
    not guarantee that ``docker image inspect <base>`` works. Pull the pinned
    base explicitly when it is missing (or cached for another platform).
    """
    try:
        metadata = inspect_image(image)
    except subprocess.CalledProcessError:
        metadata = None

    if metadata is not None and metadata.platform == platform:
        return metadata

    if metadata is None:
        reason = "is not registered in Docker's image store"
    else:
        reason = f"is cached for {metadata.platform}, not {platform}"
    print(f"Base image {image!r} {reason}; pulling it for {platform}.", file=sys.stderr)
    try:
        _run(docker_pull_command(image=image, platform=platform))
        metadata = inspect_image(image)
    except subprocess.CalledProcessError as exc:
        raise SubmissionError(
            f"Could not materialize pinned base image {image!r} for {platform}. "
            "Check Docker access and retry."
        ) from exc
    _require_platform(metadata, image=image, expected_platform=platform)
    return metadata


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
    platform: str = DEFAULT_PLATFORM

    @property
    def layers_dir(self) -> Path:
        return self.staging_dir / LAYERS_DIRNAME

    @property
    def manifest_path(self) -> Path:
        return self.staging_dir / MANIFEST_NAME


def build_and_extract_delta(plan: BuildPlan, *, skip_build: bool = False) -> dict[str, Any]:
    """Build the image, extract the delta layers, and write the manifest."""
    base_metadata = ensure_base_image(plan.base_image, platform=plan.platform)
    if not skip_build:
        _run(docker_build_command(
            context=plan.context,
            dockerfile=plan.dockerfile,
            base_image=plan.base_image,
            image_tag=plan.image_tag,
            platform=plan.platform,
        ))

    submission_metadata = inspect_image(plan.image_tag)
    _require_platform(
        submission_metadata,
        image=plan.image_tag,
        expected_platform=plan.platform,
    )
    base_layers = base_metadata.layers
    submission_layers = submission_metadata.layers
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

        all_layer_paths = read_save_layer_paths(
            save_dir,
            image_tag=plan.image_tag,
            expected_repo_tags=list(submission_metadata.repo_tags),
            expected_diff_ids=submission_layers,
        )
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
        base_image_id=base_metadata.image_id,
        base_layers=base_layers,
        layer_files=layer_files,
        config=plan.config,
    )
    plan.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def push_to_hub(
    plan: BuildPlan,
    *,
    repo_id: str,
    repo_type: str = DEFAULT_REPO_TYPE,
    private: bool = True,
) -> str | None:
    if repo_type != DEFAULT_REPO_TYPE or not private:
        raise SubmissionError(
            "Challenge submissions must use a private Hugging Face model repo"
        )

    from huggingface_hub import HfApi
    from huggingface_hub.errors import LocalTokenNotFoundError, RepositoryNotFoundError

    # Use only credentials cached by `hf auth login`. There is deliberately no
    # CLI token argument that could leak through shell history or process lists.
    api = HfApi()
    try:
        identity = api.whoami()
    except LocalTokenNotFoundError as exc:
        raise SubmissionError(
            "No cached Hugging Face login found. Run `hf auth login`, then retry."
        ) from exc
    username = identity.get("name") if isinstance(identity, dict) else None
    owner, separator, name = repo_id.partition("/")
    if not separator or not owner or not name or "/" in name:
        raise SubmissionError(
            "--repo-id must be <your-hf-username>/<submission-name>"
        )
    if not isinstance(username, str) or owner.casefold() != username.casefold():
        raise SubmissionError(
            "Submission repositories must be in the namespace of the authenticated "
            f"Hugging Face user ({username or 'unknown'}), not {owner!r}."
        )

    validate_manifest(json.loads(plan.manifest_path.read_text(encoding="utf-8")))
    try:
        repo_info = api.repo_info(
            repo_id=repo_id,
            repo_type=repo_type,
        )
    except RepositoryNotFoundError as exc:
        raise SubmissionError(
            f"Submission repository {repo_id!r} does not exist or is not accessible. "
            "Create it from the challenge frontend before uploading."
        ) from exc

    repo_is_private = (
        repo_info.get("private")
        if isinstance(repo_info, dict)
        else getattr(repo_info, "private", None)
    )
    if repo_is_private is not True:
        raise SubmissionError(
            f"Submission repository {repo_id!r} must already be private. "
            "Create a new submission repository from the challenge frontend."
        )

    # Publish one self-consistent revision. Reusing a repository is supported:
    # old manifest/layer files are removed in the same commit, so a build with
    # fewer layers cannot leave stale participant-controlled blobs behind.
    upload_kwargs: dict[str, Any] = {
        "folder_path": str(plan.staging_dir),
        "path_in_repo": "",
        "repo_id": repo_id,
        "repo_type": repo_type,
        "allow_patterns": [MANIFEST_NAME, f"{LAYERS_DIRNAME}/*"],
        "delete_patterns": [MANIFEST_NAME, f"{LAYERS_DIRNAME}/*"],
        "commit_message": "Publish submission artifact",
    }
    repo_head = (
        repo_info.get("sha")
        if isinstance(repo_info, dict)
        else getattr(repo_info, "sha", None)
    )
    if isinstance(repo_head, str) and repo_head:
        # Fail instead of silently racing another upload to the same repository.
        upload_kwargs["parent_commit"] = repo_head

    commit_info = api.upload_folder(
        **upload_kwargs,
    )
    commit_sha = getattr(commit_info, "oid", None)
    return commit_sha if isinstance(commit_sha, str) and commit_sha else None


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
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Target HF repo id, e.g. your-hf-username/your-submission.",
    )
    parser.add_argument(
        "--repo-type",
        default=DEFAULT_REPO_TYPE,
        choices=[DEFAULT_REPO_TYPE],
        help="HF repo type (default and required for challenge uploads: model).",
    )
    parser.add_argument("--public", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-build", action="store_true", help="Reuse an existing local image tag.")
    parser.add_argument("--no-push", action="store_true", help="Build and extract only; do not push.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running Docker or pushing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.no_push and (args.repo_type != DEFAULT_REPO_TYPE or args.public):
        print(
            "ERROR: challenge submissions must use a private Hugging Face model repo.",
            file=sys.stderr,
        )
        return 2

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
        print(
            "[dry-run] would ensure the base is locally available:",
            " ".join(docker_pull_command(image=args.base_image)),
            "(only if missing or for another platform)",
        )
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

    try:
        commit_sha = push_to_hub(
            plan,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            private=not args.public,
        )
    except Exception as exc:  # noqa: BLE001 - surface hub errors to the user
        print(f"ERROR: push failed: {exc}", file=sys.stderr)
        return 1

    print(f"Pushed submission to {args.repo_type}: {args.repo_id}")
    if commit_sha:
        print(f"Artifact revision: {args.repo_id}@{commit_sha}")
    else:
        print("Upload completed; this Hub client did not report the commit SHA.")
    print("Next: return to the challenge frontend and submit this uploaded revision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
