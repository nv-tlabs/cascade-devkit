from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc


def validate_predictions(
    predictions: Path,
    queries: Path,
    videos: Path,
    top_k: int,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    query_ids = [str(row.get("query_id", "")) for row in read_jsonl(queries)]
    query_id_set = set(query_ids)
    video_ids = {str(row.get("video_id", "")) for row in read_jsonl(videos)}
    prediction_lengths: list[int] = []

    seen_queries: set[str] = set()
    for index, row in enumerate(read_jsonl(predictions), start=1):
        query_id = str(row.get("query_id", ""))
        ranked = row.get("video_ids")
        if query_id not in query_id_set:
            errors.append(f"row {index}: unknown query_id {query_id!r}")
        if query_id in seen_queries:
            errors.append(f"row {index}: duplicate query_id {query_id!r}")
        seen_queries.add(query_id)
        if not isinstance(ranked, list):
            errors.append(f"row {index}: video_ids must be a list")
            continue
        prediction_lengths.append(len(ranked))
        if len(ranked) > top_k:
            errors.append(f"row {index}: video_ids has {len(ranked)} entries, max is {top_k}")
        for video_id in ranked:
            if str(video_id) not in video_ids:
                errors.append(f"row {index}: unknown video_id {video_id!r}")

    missing = sorted(query_id_set - seen_queries)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", and {len(missing) - 10} more"
        errors.append(f"missing predictions for query_id values: {preview}{suffix}")

    if query_ids and top_k > 0 and len(prediction_lengths) == len(query_ids):
        if all(length == top_k for length in prediction_lengths):
            warnings.append(
                "every query returns exactly TOP_K videos; TOP_K is a cap, not a target. "
                "Do not pad with non-matching videos."
            )

    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate challenge predictions.jsonl shape.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    errors, warnings = validate_predictions(args.predictions, args.queries, args.videos, args.top_k)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print("Submission predictions are valid.")


if __name__ == "__main__":
    main()
