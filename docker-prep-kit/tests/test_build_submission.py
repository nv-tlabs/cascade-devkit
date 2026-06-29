from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import build_submission as bs  # noqa: E402


class ParseConfigTest(unittest.TestCase):
    def test_parses_valid_config(self) -> None:
        cfg = bs.parse_submission_config(
            """
            entrypoint: ["/opt/app/.venv/bin/python", "/opt/app/run.py"]
            env:
              TOP_K: 100
              MODE: fast
            cuda: "13.0"
            notes: hello
            """
        )
        self.assertEqual(cfg.entrypoint, ["/opt/app/.venv/bin/python", "/opt/app/run.py"])
        self.assertEqual(cfg.env, {"TOP_K": "100", "MODE": "fast"})
        self.assertEqual(cfg.cuda, "13.0")
        self.assertEqual(cfg.notes, "hello")

    def test_missing_entrypoint_raises(self) -> None:
        with self.assertRaises(bs.SubmissionError):
            bs.parse_submission_config("env:\n  TOP_K: 1\n")

    def test_entrypoint_must_be_list(self) -> None:
        with self.assertRaises(bs.SubmissionError):
            bs.parse_submission_config('entrypoint: "python run.py"\n')


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
        import tempfile

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
            base_image="reg/base:tag",
            base_image_id="sha256:abc",
            base_layers=["a", "b", "c"],
            layer_files=[{"name": "layer-00.tar", "sha256": "deadbeef", "bytes": 10}],
            config=cfg,
            created_at=0,
        )
        self.assertEqual(manifest["schema_version"], bs.MANIFEST_SCHEMA_VERSION)
        self.assertEqual(manifest["base_image"]["ref"], "reg/base:tag")
        self.assertEqual(manifest["base_image"]["id"], "sha256:abc")
        self.assertEqual(manifest["base_image"]["diff_ids"], ["a", "b", "c"])
        self.assertEqual(manifest["layers"][0]["name"], "layer-00.tar")
        self.assertEqual(manifest["entrypoint"], ["python", "run.py"])
        self.assertEqual(manifest["env"], {"TOP_K": "100"})
        self.assertEqual(manifest["created_at"], "1970-01-01T00:00:00Z")
        json.dumps(manifest)


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
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blob"
            path.write_bytes(b"cascade")
            self.assertEqual(bs.sha256_file(path), hashlib.sha256(b"cascade").hexdigest())


if __name__ == "__main__":
    unittest.main()
