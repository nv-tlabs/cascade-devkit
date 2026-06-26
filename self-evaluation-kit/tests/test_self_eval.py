from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


KIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT_ROOT))

import cascade_task  # noqa: E402
import self_eval  # noqa: E402


class SelfEvaluationKitTest(unittest.TestCase):
    def test_loads_named_split_and_requires_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root, video_root = write_fixture(root)

            data = cascade_task.load_cascade_evaluation_data(
                dataset_root,
                split_name="cascade-v0.1",
                split="val",
                video_root=video_root,
            )

            self.assertEqual(
                [query.query_id for query in data.queries],
                ["cascade-v0.1_val_batch_00001_00001"],
            )
            self.assertEqual(data.qrels["cascade-v0.1_val_batch_00001_00001"], {"clip-1"})
            self.assertEqual({video.video_id for video in data.videos}, {"clip-1", "clip-2"})

            (video_root / "clip-2.mp4").unlink()
            with self.assertRaises(FileNotFoundError):
                cascade_task.load_cascade_evaluation_data(
                    dataset_root,
                    split_name="cascade-v0.1",
                    split="val",
                    video_root=video_root,
                )

    def test_scores_existing_predictions_and_writes_evaluation_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root, video_root = write_fixture(root)
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "query_id": "cascade-v0.1_val_batch_00001_00001",
                        "video_ids": ["clip-1", "clip-2"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            out = root / "evaluation.json"

            result = self_eval.evaluate(
                argparse.Namespace(
                    image="test-image",
                    dataset_root=dataset_root,
                    video_root=video_root,
                    split_name="cascade-v0.1",
                    split="val",
                    out=out,
                    top_k=100,
                    video_extension=".mp4",
                    video_manifest=None,
                    workdir=root / "work",
                    predictions=predictions,
                )
            )

            self.assertTrue(out.is_file())
            self.assertEqual(result["primary_metric"], "average_r_precision")
            self.assertEqual(result["score"], 1.0)
            self.assertEqual(result["metrics"]["num_predicted"], 1)


def write_fixture(root: Path) -> tuple[Path, Path]:
    dataset_root = root / "cascade"
    video_root = root / "videos"
    (dataset_root / "tasks" / "retrieval" / "batch_00001").mkdir(parents=True)
    (dataset_root / "data" / "batch_00001").mkdir(parents=True)
    video_root.mkdir()

    (dataset_root / "tasks" / "retrieval" / "retrieval_split.yaml").write_text(
        "\n".join(
            [
                "- name: cascade-v0.1",
                "  date: 2026-05-14",
                "  train:",
                "    - tasks/retrieval/batch_00001/train_queries.json",
                "  val:",
                "    - tasks/retrieval/batch_00001/val_queries.json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (dataset_root / "tasks" / "retrieval" / "batch_00001" / "val_queries.json").write_text(
        json.dumps(
            {
                "task": "retrieval",
                "batch": "batch_00001",
                "items": [
                    {
                        "query": "Find the relevant clip.",
                        "matches": [
                            {
                                "clip_id": "clip-1",
                                "annotation_file": "data/batch_00001/a__clip-1.json",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (dataset_root / "tasks" / "retrieval" / "batch_00001" / "train_queries.json").write_text(
        json.dumps({"task": "retrieval", "batch": "batch_00001", "items": []}),
        encoding="utf-8",
    )
    for clip_id in ("clip-1", "clip-2"):
        (dataset_root / "data" / "batch_00001" / f"a__{clip_id}.json").write_text(
            json.dumps({"video": {"clip_id": clip_id}}),
            encoding="utf-8",
        )
        (video_root / f"{clip_id}.mp4").write_bytes(b"")

    return dataset_root, video_root


if __name__ == "__main__":
    unittest.main()
