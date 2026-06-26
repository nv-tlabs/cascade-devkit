from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QueryRow:
    query_id: str
    text: str


@dataclass(frozen=True)
class VideoRow:
    video_id: str
    path: str
    annotation_file: str


@dataclass(frozen=True)
class CascadeEvaluationData:
    split_name: str
    split: str
    queries: list[QueryRow]
    videos: list[VideoRow]
    qrels: dict[str, set[str]]


def load_cascade_evaluation_data(
    dataset_root: Path,
    *,
    split_name: str,
    split: str,
    video_root: Path,
    video_extension: str = ".mp4",
    video_manifest: Path | None = None,
) -> CascadeEvaluationData:
    dataset_root = dataset_root.resolve()
    video_root = video_root.resolve()
    split_entry = get_split_entry(dataset_root, split_name)
    split_files = split_entry.get(split)
    if not isinstance(split_files, list) or not split_files:
        raise ValueError(f"Split {split!r} was not found for split-name {split_name!r}.")

    queries, qrels = load_queries_and_qrels(dataset_root, split_name, split, split_files)
    videos = load_video_rows(
        dataset_root,
        video_root=video_root,
        video_extension=video_extension,
        video_manifest=video_manifest,
    )
    missing = missing_video_files(videos, video_root)
    if missing:
        formatted = "\n".join(f"- {video_id}: expected {path}" for video_id, path in missing[:50])
        suffix = "" if len(missing) <= 50 else f"\n... and {len(missing) - 50} more"
        raise FileNotFoundError(f"Missing video files:\n{formatted}{suffix}")

    return CascadeEvaluationData(
        split_name=split_name,
        split=split,
        queries=queries,
        videos=videos,
        qrels=qrels,
    )


def get_split_entry(dataset_root: Path, split_name: str) -> dict[str, Any]:
    split_path = dataset_root / "tasks" / "retrieval" / "retrieval_split.yaml"
    entries = _load_yaml_list(split_path)
    for entry in entries:
        if entry.get("name") == split_name:
            return entry
    names = ", ".join(str(entry.get("name")) for entry in entries)
    raise ValueError(f"Split-name {split_name!r} was not found. Available: {names}")


def load_queries_and_qrels(
    dataset_root: Path,
    split_name: str,
    split: str,
    split_files: Iterable[str],
) -> tuple[list[QueryRow], dict[str, set[str]]]:
    queries: list[QueryRow] = []
    qrels: dict[str, set[str]] = {}

    for split_file in split_files:
        path = dataset_root / split_file
        payload = json.loads(path.read_text(encoding="utf-8"))
        batch = str(payload.get("batch") or Path(split_file).parent.name)
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise ValueError(f"{path} has invalid 'items'; expected a list.")
        for index, item in enumerate(items, start=1):
            query_text = str(item["query"])
            query_id = f"{split_name}_{split}_{batch}_{index:05d}"
            matches = item.get("matches", [])
            relevant = {str(match["clip_id"]) for match in matches}
            queries.append(QueryRow(query_id=query_id, text=query_text))
            qrels[query_id] = relevant

    return queries, qrels


def load_video_rows(
    dataset_root: Path,
    *,
    video_root: Path,
    video_extension: str,
    video_manifest: Path | None,
) -> list[VideoRow]:
    manifest = load_video_manifest(video_manifest) if video_manifest else {}
    rows: list[VideoRow] = []
    seen: set[str] = set()

    for annotation_file in sorted((dataset_root / "data").glob("batch_*/*.json")):
        payload = json.loads(annotation_file.read_text(encoding="utf-8"))
        video_id = str(payload.get("video", {}).get("clip_id", ""))
        if not video_id or video_id in seen:
            continue
        seen.add(video_id)
        relative_annotation = annotation_file.relative_to(dataset_root).as_posix()
        video_path = manifest.get(video_id) or default_video_relative_path(
            video_root, video_id, video_extension
        )
        rows.append(
            VideoRow(
                video_id=video_id,
                path=f"videos/{video_path}",
                annotation_file=relative_annotation,
            )
        )

    if not rows:
        raise ValueError(f"No annotation JSON files found under {dataset_root / 'data'}.")
    return rows


def load_video_manifest(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if text.startswith("{"):
        raw = json.loads(text)
        return {str(key): str(value) for key, value in raw.items()}

    mapping: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        video_id = str(row.get("video_id") or row.get("clip_id"))
        rel_path = str(row["path"])
        mapping[video_id] = _strip_videos_prefix(rel_path)
    return mapping


def default_video_relative_path(video_root: Path, video_id: str, video_extension: str) -> str:
    suffix = video_extension if video_extension.startswith(".") else f".{video_extension}"
    direct = video_root / f"{video_id}{suffix}"
    if direct.exists():
        return direct.relative_to(video_root).as_posix()
    return f"{video_id}{suffix}"


def missing_video_files(videos: Iterable[VideoRow], video_root: Path) -> list[tuple[str, Path]]:
    missing = []
    for row in videos:
        rel_path = _strip_videos_prefix(row.path)
        expected = video_root / rel_path
        if not expected.is_file():
            missing.append((row.video_id, expected))
    return missing


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_challenge_files(data: CascadeEvaluationData, input_dir: Path, labels_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "videos").mkdir(exist_ok=True)
    write_jsonl(
        input_dir / "queries.jsonl",
        ({"query_id": row.query_id, "text": row.text} for row in data.queries),
    )
    write_jsonl(
        input_dir / "videos.jsonl",
        (
            {
                "video_id": row.video_id,
                "path": row.path,
                "annotation_file": row.annotation_file,
            }
            for row in data.videos
        ),
    )
    labels_path = labels_dir / "qrels.jsonl"
    write_jsonl(
        labels_path,
        (
            {"query_id": query_id, "relevant_video_ids": sorted(relevant)}
            for query_id, relevant in data.qrels.items()
        ),
    )
    return labels_path


def _strip_videos_prefix(path: str) -> str:
    return path[len("videos/") :] if path.startswith("videos/") else path


def _load_yaml_list(path: Path) -> list[dict[str, Any]]:
    try:
        import yaml
    except ImportError:
        return _load_simple_split_yaml(path)

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a list of split entries.")
    return payload


def _load_simple_split_yaml(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    active_list: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        stripped = line.strip()
        if stripped.startswith("- name:"):
            current = {"name": stripped.split(":", 1)[1].strip()}
            entries.append(current)
            active_list = None
        elif current is not None and ":" in stripped and not stripped.startswith("- "):
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                current[key] = value
                active_list = None
            else:
                current[key] = []
                active_list = key
        elif current is not None and active_list and stripped.startswith("- "):
            current[active_list].append(stripped[2:].strip())

    if not entries:
        raise ValueError(f"{path} did not contain any split entries.")
    return entries
