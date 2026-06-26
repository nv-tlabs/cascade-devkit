from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


KIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT_ROOT))

import validate_submission  # noqa: E402


class ValidateSubmissionTest(unittest.TestCase):
    def test_warns_when_every_query_is_padded_to_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queries = write_jsonl(
                root / "queries.jsonl",
                [
                    {"query_id": "q1", "text": "one"},
                    {"query_id": "q2", "text": "two"},
                ],
            )
            videos = write_jsonl(
                root / "videos.jsonl",
                [
                    {"video_id": "v1", "path": "videos/v1.mp4"},
                    {"video_id": "v2", "path": "videos/v2.mp4"},
                ],
            )
            predictions = write_jsonl(
                root / "predictions.jsonl",
                [
                    {"query_id": "q1", "video_ids": ["v1", "v2"]},
                    {"query_id": "q2", "video_ids": ["v2", "v1"]},
                ],
            )

            errors, warnings = validate_submission.validate_predictions(
                predictions, queries, videos, top_k=2
            )

            self.assertEqual(errors, [])
            self.assertEqual(len(warnings), 1)
            self.assertIn("TOP_K is a cap", warnings[0])

    def test_allows_short_or_empty_prediction_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queries = write_jsonl(
                root / "queries.jsonl",
                [
                    {"query_id": "q1", "text": "one"},
                    {"query_id": "q2", "text": "two"},
                ],
            )
            videos = write_jsonl(root / "videos.jsonl", [{"video_id": "v1"}])
            predictions = write_jsonl(
                root / "predictions.jsonl",
                [
                    {"query_id": "q1", "video_ids": ["v1"]},
                    {"query_id": "q2", "video_ids": []},
                ],
            )

            errors, warnings = validate_submission.validate_predictions(
                predictions, queries, videos, top_k=2
            )

            self.assertEqual(errors, [])
            self.assertEqual(warnings, [])


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
