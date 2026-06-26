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
            entrypoint: ["/opt/submission/env/bin/python", "/opt/submission/run.py"]
            env:
              TOP_K: 100
              MODE: fast
            cuda: "12.1"
            notes: hello
            """
        )
        self.assertEqual(
            cfg.entrypoint,
            ["/opt/submission/env/bin/python", "/opt/submission/run.py"],
        )
        self.assertEqual(cfg.env, {"TOP_K": "100", "MODE": "fast"})
        self.assertEqual(cfg.cuda, "12.1")
        self.assertEqual(cfg.notes, "hello")

    def test_missing_entrypoint_raises(self) -> None:
        with self.assertRaises(bs.SubmissionError):
            bs.parse_submission_config("env:\n  TOP_K: 1\n")

    def test_entrypoint_must_be_list(self) -> None:
        with self.assertRaises(bs.SubmissionError):
            bs.parse_submission_config('entrypoint: "python run.py"\n')

    def test_env_must_be_mapping(self) -> None:
        with self.assertRaises(bs.SubmissionError):
            bs.parse_submission_config('entrypoint: ["python"]\nenv: [1, 2]\n')

    def test_load_config_missing_file(self) -> None:
        with self.assertRaises(bs.SubmissionError):
            bs.load_submission_config(ROOT / "tests")


class CommandConstructionTest(unittest.TestCase):
    def test_docker_build_command_passes_base_arg(self) -> None:
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

    def test_copy_prefix_uses_dot_suffix(self) -> None:
        cmd = bs.docker_copy_prefix_command("abc123", "/opt/submission", Path("/stage/submission"))
        self.assertEqual(cmd, ["docker", "cp", "abc123:/opt/submission/.", "/stage/submission"])

    def test_tar_command(self) -> None:
        cmd = bs.tar_command(prefix_dir=Path("/stage/submission"), artifact_path=Path("/stage/submission.tar.gz"))
        self.assertEqual(cmd, ["tar", "-C", "/stage/submission", "-czf", "/stage/submission.tar.gz", "."])


class ManifestTest(unittest.TestCase):
    def test_manifest_shape(self) -> None:
        cfg = bs.SubmissionConfig(
            entrypoint=["python", "run.py"],
            env={"TOP_K": "100"},
            cuda="12.1",
            notes="n",
        )
        manifest = bs.build_manifest(
            base_image="reg/base:tag",
            prefix="/opt/submission",
            config=cfg,
            artifact_sha256="deadbeef",
            artifact_bytes=42,
            created_at=0,
        )
        self.assertEqual(manifest["schema_version"], bs.MANIFEST_SCHEMA_VERSION)
        self.assertEqual(manifest["base_image"], "reg/base:tag")
        self.assertEqual(manifest["entrypoint"], ["python", "run.py"])
        self.assertEqual(manifest["env"], {"TOP_K": "100"})
        self.assertEqual(manifest["artifact"], {"path": bs.ARTIFACT_NAME, "sha256": "deadbeef", "bytes": 42})
        self.assertEqual(manifest["created_at"], "1970-01-01T00:00:00Z")
        # round-trips as JSON
        json.dumps(manifest)


class Sha256Test(unittest.TestCase):
    def test_sha256_file(self) -> None:
        import hashlib
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blob"
            path.write_bytes(b"cascade")
            self.assertEqual(bs.sha256_file(path), hashlib.sha256(b"cascade").hexdigest())


class BuildAndExtractTest(unittest.TestCase):
    def test_dry_run_does_not_invoke_runner(self) -> None:
        import tempfile

        calls: list[list[str]] = []

        def runner(cmd):
            calls.append(list(cmd))

        with tempfile.TemporaryDirectory() as tmp:
            plan = bs.BuildPlan(
                context=Path(tmp),
                dockerfile=Path(tmp) / "Dockerfile",
                base_image="reg/base:tag",
                prefix="/opt/submission",
                image_tag="sub:local",
                staging_dir=Path(tmp) / "stage",
                config=bs.SubmissionConfig(entrypoint=["python"]),
            )
            bs.build_and_extract(plan, dry_run=True, runner=runner)

        self.assertEqual(calls, [])

    def test_build_extract_invokes_expected_steps(self) -> None:
        import tempfile

        calls: list[list[str]] = []

        def runner(cmd):
            calls.append(list(cmd))

        def capture(cmd):
            calls.append(list(cmd))
            return "container42\n"

        with tempfile.TemporaryDirectory() as tmp:
            plan = bs.BuildPlan(
                context=Path(tmp),
                dockerfile=Path(tmp) / "Dockerfile",
                base_image="reg/base:tag",
                prefix="/opt/submission",
                image_tag="sub:local",
                staging_dir=Path(tmp) / "stage",
                config=bs.SubmissionConfig(entrypoint=["python"]),
            )
            bs.build_and_extract(plan, runner=runner, capture=capture)

        verbs = [c[:2] for c in calls]
        self.assertIn(["docker", "build"], verbs)
        self.assertIn(["docker", "create"], verbs)
        self.assertIn(["docker", "cp"], verbs)
        self.assertIn(["docker", "rm"], verbs)
        self.assertIn(["tar", "-C"], verbs)
        # container id from capture threads into cp/rm
        self.assertTrue(any("container42:/opt/submission/." in part for c in calls for part in c))


if __name__ == "__main__":
    unittest.main()
