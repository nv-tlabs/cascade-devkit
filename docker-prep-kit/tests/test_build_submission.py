from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import build_submission as bs  # noqa: E402


SHA256_A = "sha256:" + "a" * 64
SHA256_B = "sha256:" + "b" * 64
SHA256_C = "sha256:" + "c" * 64


def write_saved_image(
    save_dir: Path,
    *,
    diff_ids: list[str],
    tag: str = "submission:test",
    suffix: str = "target",
    legacy_config_path: bool = False,
) -> tuple[str, dict[str, object]]:
    config_content = json.dumps(
        {"rootfs": {"type": "layers", "diff_ids": diff_ids}},
        separators=(",", ":"),
    ).encode()
    config_digest = hashlib.sha256(config_content).hexdigest()
    config_rel = (
        f"{config_digest}.json"
        if legacy_config_path
        else f"blobs/sha256/{config_digest}"
    )
    config_path = save_dir / config_rel
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(config_content)

    layer_paths = [f"layers/{suffix}-{index}.tar" for index in range(len(diff_ids))]
    for index, relative_path in enumerate(layer_paths):
        path = save_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"layer-{suffix}-{index}".encode())

    entry: dict[str, object] = {
        "Config": config_rel,
        "RepoTags": [tag],
        "Layers": layer_paths,
    }
    return f"sha256:{config_digest}", entry


class ParseConfigTest(unittest.TestCase):
    def test_parses_valid_config(self) -> None:
        cfg = bs.parse_submission_config(
            """
            entrypoint: ["/opt/app/.venv/bin/python", "/opt/app/run.py"]
            env:
              MODE: fast
            cuda: "13.0"
            notes: hello
            """
        )
        self.assertEqual(cfg.entrypoint, ["/opt/app/.venv/bin/python", "/opt/app/run.py"])
        self.assertEqual(cfg.env, {"MODE": "fast"})
        self.assertEqual(cfg.cuda, "13.0")
        self.assertEqual(cfg.notes, "hello")

    def test_missing_entrypoint_raises(self) -> None:
        with self.assertRaises(bs.SubmissionError):
            bs.parse_submission_config("env:\n  TOP_K: 1\n")

    def test_entrypoint_must_be_list(self) -> None:
        with self.assertRaises(bs.SubmissionError):
            bs.parse_submission_config('entrypoint: "python run.py"\n')

    def test_entrypoint_entries_must_not_be_blank(self) -> None:
        with self.assertRaisesRegex(bs.SubmissionError, "non-empty strings"):
            bs.parse_submission_config('entrypoint: ["python", "  "]\n')

    def test_unknown_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(bs.SubmissionError, "unsupported field"):
            bs.parse_submission_config('entrypoint: ["python"]\nentrypont: ["typo"]\n')

    def test_nested_environment_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(bs.SubmissionError, "must be a scalar"):
            bs.parse_submission_config('entrypoint: ["python"]\nenv:\n  OPTIONS: [a, b]\n')

    def test_runtime_environment_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(bs.SubmissionError, "evaluator-owned"):
            bs.parse_submission_config('entrypoint: ["python"]\nenv:\n  TOP_K: "5"\n')


class ComputeDeltaTest(unittest.TestCase):
    def test_returns_layers_added_on_top_of_base(self) -> None:
        base = ["a", "b", "c"]
        sub = ["a", "b", "c", "d", "e"]
        self.assertEqual(bs.compute_delta_layers(base, sub), ["d", "e"])

    def test_rejects_non_prefix_base(self) -> None:
        with self.assertRaises(bs.SubmissionError):
            bs.compute_delta_layers(["a", "b", "x"], ["a", "b", "c", "d"])

    def test_rejects_shorter_submission(self) -> None:
        with self.assertRaises(bs.SubmissionError):
            bs.compute_delta_layers(["a", "b", "c"], ["a", "b"])

    def test_rejects_empty_delta(self) -> None:
        with self.assertRaises(bs.SubmissionError):
            bs.compute_delta_layers(["a", "b"], ["a", "b"])


class ReadSaveLayerPathsTest(unittest.TestCase):
    def test_reads_layers_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save = Path(tmp)
            _, entry = write_saved_image(
                save,
                diff_ids=[SHA256_A, SHA256_B],
            )
            (save / "manifest.json").write_text(
                json.dumps([entry]),
                encoding="utf-8",
            )
            self.assertEqual(
                bs.read_save_layer_paths(
                    save,
                    image_tag="submission:test",
                    expected_repo_tags=["submission:test"],
                    expected_diff_ids=[SHA256_A, SHA256_B],
                ),
                ["layers/target-0.tar", "layers/target-1.tar"],
            )

    def test_selects_entry_by_inspected_repo_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save = Path(tmp)
            _, other_entry = write_saved_image(
                save,
                diff_ids=[SHA256_C],
                tag="other:test",
                suffix="other",
            )
            _, target_entry = write_saved_image(
                save,
                diff_ids=[SHA256_A, SHA256_B],
            )
            (save / "manifest.json").write_text(
                json.dumps([other_entry, target_entry]),
                encoding="utf-8",
            )

            paths = bs.read_save_layer_paths(
                save,
                image_tag="submission:test",
                expected_repo_tags=["submission:test"],
                expected_diff_ids=[SHA256_A, SHA256_B],
            )

            self.assertEqual(paths, ["layers/target-0.tar", "layers/target-1.tar"])

    def test_accepts_classic_docker_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save = Path(tmp)
            _, entry = write_saved_image(
                save,
                diff_ids=[SHA256_A],
                legacy_config_path=True,
            )
            (save / "manifest.json").write_text(json.dumps([entry]), encoding="utf-8")

            paths = bs.read_save_layer_paths(
                save,
                image_tag="submission:test",
                expected_repo_tags=["submission:test"],
                expected_diff_ids=[SHA256_A],
            )

            self.assertEqual(paths, ["layers/target-0.tar"])

    def test_uses_normalized_inspected_repo_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save = Path(tmp)
            _, entry = write_saved_image(
                save,
                diff_ids=[SHA256_A],
                tag="submission:latest",
            )
            (save / "manifest.json").write_text(json.dumps([entry]), encoding="utf-8")

            paths = bs.read_save_layer_paths(
                save,
                image_tag="submission",
                expected_repo_tags=["submission:latest"],
                expected_diff_ids=[SHA256_A],
            )

            self.assertEqual(paths, ["layers/target-0.tar"])

    def test_rejects_config_content_that_does_not_match_path_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save = Path(tmp)
            _, entry = write_saved_image(save, diff_ids=[SHA256_A])
            (save / str(entry["Config"])).write_text(
                json.dumps({"rootfs": {"diff_ids": [SHA256_A]}}),
                encoding="utf-8",
            )
            (save / "manifest.json").write_text(json.dumps([entry]), encoding="utf-8")

            with self.assertRaisesRegex(bs.SubmissionError, "config digest"):
                bs.read_save_layer_paths(
                    save,
                    image_tag="submission:test",
                    expected_repo_tags=["submission:test"],
                    expected_diff_ids=[SHA256_A],
                )

    def test_rejects_saved_config_with_different_diff_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save = Path(tmp)
            _, entry = write_saved_image(save, diff_ids=[SHA256_A])
            (save / "manifest.json").write_text(json.dumps([entry]), encoding="utf-8")

            with self.assertRaisesRegex(bs.SubmissionError, "saved image config"):
                bs.read_save_layer_paths(
                    save,
                    image_tag="submission:test",
                    expected_repo_tags=["submission:test"],
                    expected_diff_ids=[SHA256_B],
                )

    def test_rejects_layer_count_that_differs_from_saved_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save = Path(tmp)
            _, entry = write_saved_image(save, diff_ids=[SHA256_A])
            entry["Layers"] = []
            (save / "manifest.json").write_text(json.dumps([entry]), encoding="utf-8")

            with self.assertRaisesRegex(bs.SubmissionError, "config and layer blobs"):
                bs.read_save_layer_paths(
                    save,
                    image_tag="submission:test",
                    expected_repo_tags=["submission:test"],
                    expected_diff_ids=[SHA256_A],
                )

    def test_rejects_duplicate_entries_for_submission_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save = Path(tmp)
            _, entry = write_saved_image(save, diff_ids=[SHA256_A])
            (save / "manifest.json").write_text(
                json.dumps([entry, dict(entry)]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(bs.SubmissionError, "multiple entries"):
                bs.read_save_layer_paths(
                    save,
                    image_tag="submission:test",
                    expected_repo_tags=["submission:test"],
                    expected_diff_ids=[SHA256_A],
                )

    def test_rejects_unsafe_layer_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save = Path(tmp)
            _, entry = write_saved_image(save, diff_ids=[SHA256_A])
            entry["Layers"] = ["../outside.tar"]
            (save / "manifest.json").write_text(json.dumps([entry]), encoding="utf-8")

            with self.assertRaisesRegex(bs.SubmissionError, "unsafe layer path"):
                bs.read_save_layer_paths(
                    save,
                    image_tag="submission:test",
                    expected_repo_tags=["submission:test"],
                    expected_diff_ids=[SHA256_A],
                )


class ManifestTest(unittest.TestCase):
    def test_manifest_shape(self) -> None:
        cfg = bs.SubmissionConfig(entrypoint=["python", "run.py"], env={"TOP_K": "100"}, cuda="13.0")
        manifest = bs.build_manifest(
            base_image=bs.DEFAULT_BASE_IMAGE,
            base_image_id=SHA256_A,
            base_layers=[SHA256_A, SHA256_B],
            layer_files=[{"name": "layer-00.tar", "sha256": "c" * 64, "bytes": 10}],
            config=cfg,
            created_at=0,
        )
        self.assertEqual(manifest["schema_version"], bs.MANIFEST_SCHEMA_VERSION)
        self.assertEqual(manifest["base_image"]["ref"], bs.DEFAULT_BASE_IMAGE)
        self.assertEqual(manifest["base_image"]["id"], SHA256_A)
        self.assertEqual(manifest["base_image"]["diff_ids"], [SHA256_A, SHA256_B])
        self.assertEqual(manifest["layers"][0]["name"], "layer-00.tar")
        self.assertEqual(manifest["entrypoint"], ["python", "run.py"])
        self.assertEqual(manifest["env"], {"TOP_K": "100"})
        self.assertEqual(manifest["created_at"], "1970-01-01T00:00:00Z")
        json.dumps(manifest)

    def test_rejects_unpinned_base(self) -> None:
        with self.assertRaisesRegex(bs.SubmissionError, "pinned by sha256"):
            bs.build_manifest(
                base_image="python:3.12",
                base_image_id=SHA256_A,
                base_layers=[SHA256_A],
                layer_files=[{"name": "layer-00.tar", "sha256": "b" * 64, "bytes": 1}],
                config=bs.SubmissionConfig(entrypoint=["python"]),
            )

    def test_rejects_nonsequential_layer_names(self) -> None:
        with self.assertRaisesRegex(bs.SubmissionError, "layer-00.tar"):
            bs.build_manifest(
                base_image=bs.DEFAULT_BASE_IMAGE,
                base_image_id=SHA256_A,
                base_layers=[SHA256_A],
                layer_files=[{"name": "other.tar", "sha256": "b" * 64, "bytes": 1}],
                config=bs.SubmissionConfig(entrypoint=["python"]),
            )


class DockerBuildCommandTest(unittest.TestCase):
    def test_passes_base_arg_and_paths(self) -> None:
        cmd = bs.docker_build_command(
            context=Path("/ctx"),
            dockerfile=Path("/ctx/Dockerfile"),
            base_image="reg/base:tag",
            image_tag="sub:local",
        )
        self.assertIn("--build-arg", cmd)
        self.assertIn("BASE_IMAGE=reg/base:tag", cmd)
        self.assertIn("sub:local", cmd)
        self.assertEqual(cmd[2:4], ["--platform", "linux/amd64"])
        self.assertEqual(cmd[-1], "/ctx")

    def test_omits_network_by_default(self) -> None:
        cmd = bs.docker_build_command(
            context=Path("/ctx"),
            dockerfile=Path("/ctx/Dockerfile"),
            base_image="reg/base:tag",
            image_tag="sub:local",
        )
        self.assertNotIn("--network", cmd)

    def test_passes_network_when_set(self) -> None:
        cmd = bs.docker_build_command(
            context=Path("/ctx"),
            dockerfile=Path("/ctx/Dockerfile"),
            base_image="reg/base:tag",
            image_tag="sub:local",
            network="host",
        )
        self.assertEqual(cmd[-3:], ["--network", "host", "/ctx"])

    def test_pull_materializes_base_for_evaluation_platform(self) -> None:
        self.assertEqual(
            bs.docker_pull_command(image="reg/base@sha256:digest"),
            [
                "docker",
                "image",
                "pull",
                "--platform",
                "linux/amd64",
                "reg/base@sha256:digest",
            ],
        )


class EnsureBaseImageTest(unittest.TestCase):
    def _metadata(self, *, architecture: str = "amd64") -> bs.DockerImageMetadata:
        return bs.DockerImageMetadata(
            image_id=SHA256_A,
            layers=[SHA256_B],
            os="linux",
            architecture=architecture,
        )

    @patch.object(bs, "_run")
    @patch.object(bs, "inspect_image")
    def test_uses_already_materialized_base(self, inspect_image, run) -> None:
        metadata = self._metadata()
        inspect_image.return_value = metadata

        self.assertIs(bs.ensure_base_image("base:tag"), metadata)

        inspect_image.assert_called_once_with("base:tag")
        run.assert_not_called()

    @patch.object(bs, "_run")
    @patch.object(bs, "inspect_image")
    def test_pulls_base_missing_from_docker_image_store(self, inspect_image, run) -> None:
        metadata = self._metadata()
        inspect_image.side_effect = [
            subprocess.CalledProcessError(1, ["docker", "image", "inspect"]),
            metadata,
        ]

        self.assertIs(bs.ensure_base_image("base@sha256:digest"), metadata)

        run.assert_called_once_with(
            [
                "docker",
                "image",
                "pull",
                "--platform",
                "linux/amd64",
                "base@sha256:digest",
            ]
        )
        self.assertEqual(inspect_image.call_count, 2)

    @patch.object(bs, "_run")
    @patch.object(bs, "inspect_image")
    def test_repulls_base_cached_for_wrong_platform(self, inspect_image, run) -> None:
        arm = self._metadata(architecture="arm64")
        amd = self._metadata()
        inspect_image.side_effect = [arm, amd]

        self.assertIs(bs.ensure_base_image("base@sha256:digest"), amd)

        run.assert_called_once_with(bs.docker_pull_command(image="base@sha256:digest"))

    @patch.object(bs, "_run")
    @patch.object(bs, "inspect_image")
    def test_rejects_wrong_platform_after_pull(self, inspect_image, run) -> None:
        arm = self._metadata(architecture="arm64")
        inspect_image.side_effect = [arm, arm]

        with self.assertRaisesRegex(bs.SubmissionError, "requires linux/amd64"):
            bs.ensure_base_image("base@sha256:digest")

        run.assert_called_once()


class BuildAndExtractDeltaTest(unittest.TestCase):
    def _plan(self, root: Path) -> bs.BuildPlan:
        return bs.BuildPlan(
            context=root,
            dockerfile=root / "Dockerfile",
            base_image=bs.DEFAULT_BASE_IMAGE,
            image_tag="submission:test",
            staging_dir=root / "artifact",
            config=bs.SubmissionConfig(entrypoint=["python", "/opt/app/run.py"]),
        )

    def _build(self, *, skip_build: bool) -> list[object]:
        events: list[object] = []
        base = bs.DockerImageMetadata(
            image_id=SHA256_A,
            layers=[SHA256_A],
            os="linux",
            architecture="amd64",
        )
        submission = bs.DockerImageMetadata(
            image_id=SHA256_B,
            layers=[SHA256_A, SHA256_B],
            os="linux",
            architecture="amd64",
        )

        def ensure(*_args, **_kwargs):
            events.append("ensure-base")
            return base

        def inspect(*_args, **_kwargs):
            events.append("inspect-submission")
            return submission

        def run(command):
            events.append(command)
            if command[0] == "tar":
                save_dir = Path(command[-1])
                (save_dir / "base.tar").write_bytes(b"base")
                (save_dir / "delta.tar").write_bytes(b"delta")

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            bs, "ensure_base_image", side_effect=ensure
        ), patch.object(bs, "inspect_image", side_effect=inspect), patch.object(
            bs, "_run", side_effect=run
        ), patch.object(
            bs,
            "read_save_layer_paths",
            return_value=["base.tar", "delta.tar"],
        ):
            bs.build_and_extract_delta(self._plan(Path(tmp)), skip_build=skip_build)
        return events

    def test_materializes_base_before_build(self) -> None:
        events = self._build(skip_build=False)

        self.assertEqual(events[0], "ensure-base")
        self.assertEqual(events[1][0:2], ["docker", "build"])
        self.assertEqual(events[2], "inspect-submission")

    def test_materializes_base_when_reusing_existing_build(self) -> None:
        events = self._build(skip_build=True)

        self.assertEqual(events[0:2], ["ensure-base", "inspect-submission"])
        self.assertFalse(
            any(isinstance(event, list) and event[0:2] == ["docker", "build"] for event in events)
        )


class Sha256Test(unittest.TestCase):
    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blob"
            path.write_bytes(b"cascade")
            self.assertEqual(bs.sha256_file(path), hashlib.sha256(b"cascade").hexdigest())


class HubPushTest(unittest.TestCase):
    def _plan(self, root: Path) -> bs.BuildPlan:
        plan = bs.BuildPlan(
            context=root,
            dockerfile=root / "Dockerfile",
            base_image=bs.DEFAULT_BASE_IMAGE,
            image_tag="submission:test",
            staging_dir=root / "artifact",
            config=bs.SubmissionConfig(entrypoint=["python", "/opt/app/run.py"]),
        )
        plan.layers_dir.mkdir(parents=True)
        (plan.layers_dir / "layer-00.tar").write_bytes(b"layer")
        manifest = bs.build_manifest(
            base_image=bs.DEFAULT_BASE_IMAGE,
            base_image_id=SHA256_A,
            base_layers=[SHA256_A],
            layer_files=[{"name": "layer-00.tar", "sha256": "b" * 64, "bytes": 5}],
            config=plan.config,
            created_at=0,
        )
        plan.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return plan

    def test_cli_defaults_to_private_model_repo(self) -> None:
        args = bs.build_arg_parser().parse_args([])
        self.assertEqual(args.repo_type, "model")
        self.assertFalse(args.public)
        self.assertFalse(hasattr(args, "token"))

    @patch("huggingface_hub.HfApi")
    def test_push_requires_precreated_private_repo_and_uses_cached_login(self, api_type) -> None:
        api = api_type.return_value
        api.whoami.return_value = {"name": "alice"}
        api.repo_info.return_value = SimpleNamespace(private=True, sha="c" * 40)
        api.upload_folder.return_value = SimpleNamespace(oid="d" * 40)
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._plan(Path(tmp))
            commit_sha = bs.push_to_hub(
                plan,
                repo_id="alice/cascade-submission",
            )

        self.assertEqual(commit_sha, "d" * 40)
        api_type.assert_called_once_with()
        api.whoami.assert_called_once_with()
        api.repo_info.assert_called_once_with(
            repo_id="alice/cascade-submission",
            repo_type="model",
        )
        api.create_repo.assert_not_called()
        api.update_repo_settings.assert_not_called()
        api.upload_file.assert_not_called()
        upload = api.upload_folder.call_args.kwargs
        self.assertEqual(upload["repo_type"], "model")
        self.assertEqual(upload["path_in_repo"], "")
        self.assertNotIn("token", upload)
        self.assertEqual(upload["parent_commit"], "c" * 40)
        self.assertEqual(
            upload["allow_patterns"], ["manifest.json", "layers/*"]
        )
        self.assertEqual(
            upload["delete_patterns"], ["manifest.json", "layers/*"]
        )

    @patch("huggingface_hub.HfApi")
    def test_push_rejects_missing_precreated_repository(self, api_type) -> None:
        import httpx

        from huggingface_hub.errors import RepositoryNotFoundError

        api = api_type.return_value
        api.whoami.return_value = {"name": "alice"}
        api.repo_info.side_effect = RepositoryNotFoundError(
            "not found",
            response=httpx.Response(
                404,
                request=httpx.Request("GET", "https://huggingface.co/api/models/alice/missing"),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._plan(Path(tmp))
            with self.assertRaisesRegex(bs.SubmissionError, "challenge frontend"):
                bs.push_to_hub(plan, repo_id="alice/cascade-submission")

        api.create_repo.assert_not_called()
        api.upload_folder.assert_not_called()

    @patch("huggingface_hub.HfApi")
    def test_push_rejects_precreated_public_repository(self, api_type) -> None:
        api = api_type.return_value
        api.whoami.return_value = {"name": "alice"}
        api.repo_info.return_value = SimpleNamespace(private=False, sha="c" * 40)
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._plan(Path(tmp))
            with self.assertRaisesRegex(bs.SubmissionError, "must already be private"):
                bs.push_to_hub(plan, repo_id="alice/cascade-submission")

        api.update_repo_settings.assert_not_called()
        api.upload_folder.assert_not_called()

    @patch("huggingface_hub.HfApi")
    def test_push_rejects_repository_outside_authenticated_namespace(self, api_type) -> None:
        api = api_type.return_value
        api.whoami.return_value = {"name": "alice"}
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._plan(Path(tmp))
            with self.assertRaisesRegex(bs.SubmissionError, "authenticated Hugging Face user"):
                bs.push_to_hub(
                    plan,
                    repo_id="some-org/cascade-submission",
                    repo_type=bs.DEFAULT_REPO_TYPE,
                    private=True,
                )
        api.create_repo.assert_not_called()

    @patch("huggingface_hub.HfApi")
    def test_push_rejects_nonprivate_or_nonmodel_target(self, api_type) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._plan(Path(tmp))
            for repo_type, private in (("space", True), ("model", False)):
                with self.subTest(repo_type=repo_type, private=private):
                    with self.assertRaisesRegex(bs.SubmissionError, "private Hugging Face model"):
                        bs.push_to_hub(
                            plan,
                            repo_id="alice/cascade-submission",
                            repo_type=repo_type,
                            private=private,
                        )
        api_type.assert_not_called()


if __name__ == "__main__":
    unittest.main()
