"""Minimal submission entrypoint for the AV Causal Scenario Retrieval Challenge.

Replace ``rank_videos`` with your text-to-video retrieval model. The baseline
returns the corpus order for every query.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path


INPUT = Path(os.environ.get("INPUT_DIR", "/input"))
OUTPUT = Path(os.environ.get("OUTPUT_DIR", "/output"))
TOP_K = int(os.environ.get("TOP_K", "100"))


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_video_ids(input_dir: Path) -> list[str]:
    return [str(row["video_id"]) for row in read_jsonl(input_dir / "videos.jsonl")]


def rank_videos(query_text: str, video_ids: list[str]) -> list[str]:
    """Return video IDs ranked best-first for one query."""
    del query_text
    return video_ids[:TOP_K]


def main() -> None:
    video_ids = load_video_ids(INPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    count = 0
    with (OUTPUT / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for query in read_jsonl(INPUT / "queries.jsonl"):
            ranked = rank_videos(str(query["text"]), video_ids)
            row = {"query_id": str(query["query_id"]), "video_ids": ranked[:TOP_K]}
            handle.write(json.dumps(row) + "\n")
            count += 1

    print(f"Wrote predictions for {count} queries to {OUTPUT / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
