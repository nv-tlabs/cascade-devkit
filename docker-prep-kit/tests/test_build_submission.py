from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import build_submission as bs  # noqa: E402


SHA256_A = "sha256:" + "a" * 64
SHA256_B = "sha256:" + "b" * 64


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
            (save / "manifest.json").write_text(
                json.dumps([{"Layers": ["blobs/sha256/aa", "blobs/sha256/bb"]}]),
                encoding="utf-8",
            )
            self.assertEqual(
                bs.read_save_layer_paths(save),
                ["blobs/sha256/aa", "blobs/sha256/bb"],
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
        self.assertEqual(cmd[-1], "/ctx")


class Sha256Test(unittest.TestCase):
    def test_sha256_file(self) -> None:
        import hashlib

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

    @patch("huggingface_hub.HfApi")
    def test_push_creates_private_model_repo_and_uploads_artifact(self, api_type) -> None:
        api = api_type.return_value
        api.whoami.return_value = {"name": "alice"}
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._plan(Path(tmp))
            bs.push_to_hub(
                plan,
                repo_id="alice/cascade-submission",
                token="hf_test",
            )

        api_type.assert_called_once_with(token="hf_test")
        api.create_repo.assert_called_once_with(
            repo_id="alice/cascade-submission",
            repo_type="model",
            private=True,
            exist_ok=True,
            token="hf_test",
        )
        api.update_repo_settings.assert_called_once_with(
            repo_id="alice/cascade-submission",
            repo_type="model",
            private=True,
            token="hf_test",
        )
        self.assertEqual(api.upload_file.call_args.kwargs["repo_type"], "model")
        self.assertEqual(api.upload_folder.call_args.kwargs["repo_type"], "model")

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
                    token="hf_test",
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
                            token="hf_test",
                        )
        api_type.assert_not_called()


if __name__ == "__main__":
    unittest.main()
