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
    def test_loads_task_owned_versioned_split_and_requires_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root, video_root = write_fixture(root)

            data = cascade_task.load_cascade_evaluation_data(
                dataset_root,
                split_name="cascade-v1.0",
                split="val",
                video_root=video_root,
            )

            self.assertEqual(
                [query.query_id for query in data.queries],
                [
                    "cascade-v1.0_val_batch_00001_00001",
                    "cascade-v1.0_val_batch_00001_00002",
                ],
            )
            self.assertEqual(len(data.queries), 2)
            self.assertEqual(len(data.qrels), 2)
            self.assertEqual(data.qrels["cascade-v1.0_val_batch_00001_00001"], {"clip-1"})
            self.assertEqual(data.qrels["cascade-v1.0_val_batch_00001_00002"], {"clip-2"})
            self.assertEqual({video.video_id for video in data.videos}, {"clip-1", "clip-2"})

            train_data = cascade_task.load_cascade_evaluation_data(
                dataset_root,
                split_name="cascade-v1.0",
                split="train",
                video_root=video_root,
            )
            self.assertEqual(
                [query.query_id for query in train_data.queries],
                ["cascade-v1.0_train_batch_00001_00001"],
            )
            self.assertEqual(
                train_data.qrels["cascade-v1.0_train_batch_00001_00001"],
                {"clip-1", "clip-2"},
            )

            (video_root / "clip-2.mp4").unlink()
            with self.assertRaises(FileNotFoundError):
                cascade_task.load_cascade_evaluation_data(
                    dataset_root,
                    split_name="cascade-v1.0",
                    split="val",
                    video_root=video_root,
                )

    def test_scores_existing_predictions_and_writes_evaluation_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root, video_root = write_fixture(root)
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "query_id": "cascade-v1.0_val_batch_00001_00001",
                                "video_ids": ["clip-1", "clip-2"],
                            }
                        ),
                        json.dumps(
                            {
                                "query_id": "cascade-v1.0_val_batch_00001_00002",
                                "video_ids": ["clip-2", "clip-1"],
                            }
                        ),
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            out = root / "evaluation.json"

            result = self_eval.evaluate(
                argparse.Namespace(
                    artifact=None,
                    base_image="python:3.12",
                    dataset_root=dataset_root,
                    video_root=video_root,
                    split_name="cascade-v1.0",
                    split="val",
                    out=out,
                    top_k=100,
                    gpus=None,
                    video_extension=".mp4",
                    video_manifest=None,
                    workdir=root / "work",
                    predictions=predictions,
                )
            )

            self.assertTrue(out.is_file())
            self.assertEqual(result["primary_metric"], "average_r_precision")
            self.assertEqual(result["score"], 1.0)
            self.assertEqual(result["num_queries"], 2)
            self.assertEqual(result["metrics"]["num_predicted"], 2)


def write_fixture(root: Path) -> tuple[Path, Path]:
    dataset_root = root / "cascade"
    video_root = root / "videos"
    (dataset_root / "tasks" / "retrieval" / "batch_00001").mkdir(parents=True)
    (dataset_root / "data" / "batch_00001").mkdir(parents=True)
    video_root.mkdir()

    (dataset_root / "tasks" / "retrieval" / "retrieval_split.yaml").write_text(
        "\n".join(
            [
                "- name: cascade-v1.0",
                "  date: 2026-08-17",
                "  train:",
                "    - tasks/retrieval/batch_00001/train_queries_0817.json",
                "  val:",
                "    - tasks/retrieval/batch_00001/val_queries_0817.json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (dataset_root / "tasks" / "retrieval" / "batch_00001" / "val_queries_0817.json").write_text(
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
                    },
                    {
                        "query": "Find the other relevant clip.",
                        "matches": [
                            {
                                "clip_id": "clip-2",
                                "annotation_file": "data/batch_00001/a__clip-2.json",
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (dataset_root / "tasks" / "retrieval" / "batch_00001" / "train_queries_0817.json").write_text(
        json.dumps(
            {
                "task": "retrieval",
                "batch": "batch_00001",
                "items": [
                    {
                        "query": "Find either relevant clip.",
                        "matches": [
                            {
                                "clip_id": "clip-1",
                                "annotation_file": "data/batch_00001/a__clip-1.json",
                            },
                            {
                                "clip_id": "clip-2",
                                "annotation_file": "data/batch_00001/a__clip-2.json",
                            },
                        ],
                    }
                ],
            }
        ),
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
