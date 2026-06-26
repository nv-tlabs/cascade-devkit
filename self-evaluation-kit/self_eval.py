from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

import cascade_task
import metrics


K_VALUES = (1, 3, 5, 10)
PRIMARY_METRIC = "average_r_precision"


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_predictions(path: Path) -> dict[str, list[str]]:
    return {
        str(row["query_id"]): [str(video_id) for video_id in row.get("video_ids", [])]
        for row in read_jsonl(path)
    }


def load_qrels(path: Path) -> dict[str, set[str]]:
    return {
        str(row["query_id"]): {str(video_id) for video_id in row.get("relevant_video_ids", [])}
        for row in read_jsonl(path)
    }


def run_submission_image(
    image: str,
    input_dir: Path,
    video_root: Path,
    output_dir: Path,
    top_k: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "-e",
        f"TOP_K={top_k}",
        "-v",
        f"{input_dir.resolve()}:/input:ro",
        "-v",
        f"{video_root.resolve()}:/input/videos:ro",
        "-v",
        f"{output_dir.resolve()}:/output",
        image,
    ]
    subprocess.run(command, check=True)


def score_predictions(predictions_path: Path, qrels_path: Path) -> dict:
    predictions = load_predictions(predictions_path)
    result = metrics.evaluate_run(predictions, load_qrels(qrels_path), K_VALUES)
    result["score"] = result[PRIMARY_METRIC]
    result["num_predicted"] = len(predictions)
    return result


def evaluate(args: argparse.Namespace) -> dict:
    data = cascade_task.load_cascade_evaluation_data(
        args.dataset_root,
        split_name=args.split_name,
        split=args.split,
        video_root=args.video_root,
        video_extension=args.video_extension,
        video_manifest=args.video_manifest,
    )

    workdir = args.workdir
    cleanup = None
    if workdir is None:
        cleanup = tempfile.TemporaryDirectory(prefix="crc-self-eval-")
        workdir = Path(cleanup.name)
    else:
        workdir.mkdir(parents=True, exist_ok=True)

    try:
        input_dir = workdir / "input"
        labels_dir = workdir / "labels"
        output_dir = workdir / "output"
        qrels_path = cascade_task.write_challenge_files(data, input_dir, labels_dir)

        if args.predictions:
            predictions_path = args.predictions
        else:
            run_submission_image(args.image, input_dir, args.video_root, output_dir, args.top_k)
            predictions_path = output_dir / "predictions.jsonl"

        if not predictions_path.is_file():
            raise FileNotFoundError(f"Expected predictions at {predictions_path}.")

        score = score_predictions(predictions_path, qrels_path)
        evaluation = {
            "split_name": data.split_name,
            "split": data.split,
            "image": args.image,
            "dataset_root": str(args.dataset_root),
            "video_root": str(args.video_root),
            "num_queries": len(data.queries),
            "num_videos": len(data.videos),
            "predictions": str(predictions_path),
            "metrics": score,
            "score": score["score"],
            "primary_metric": score["primary_metric"],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
        return evaluation
    finally:
        if cleanup is not None:
            cleanup.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local self-evaluation for a challenge Docker image."
    )
    parser.add_argument("--image", required=True, help="Docker image tag to run.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="CASCADE dataset root containing data/ and tasks/.",
    )
    parser.add_argument("--video-root", type=Path, required=True, help="Root containing videos.")
    parser.add_argument("--split-name", required=True, help="Named split, for example cascade-v0.1.")
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val"],
        help="Split key within the named split.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Path to write evaluation JSON.")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--video-extension", default=".mp4")
    parser.add_argument("--video-manifest", type=Path)
    parser.add_argument("--workdir", type=Path, help="Optional work directory to keep files.")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Score an existing predictions.jsonl instead of running Docker.",
    )
    args = parser.parse_args()

    evaluation = evaluate(args)
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
