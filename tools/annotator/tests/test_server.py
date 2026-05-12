# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the annotator FastAPI server.

Covers: health, list, get/put round-trip, validation rejection, atomic
write, fresh-clip creation, read-only mode, and bbox round-trip (regression
against the upstream `stripLegacyBboxes` bug which we deliberately drop).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from annotator.server.app import create_app
from annotator.server.io_adapter import build_clip_index
from causal_ai_av.io import load_file


def _client(corpus_dir: Path, *, read_only: bool = False) -> TestClient:
    index = build_clip_index([corpus_dir])
    app = create_app(index, read_only=read_only, destination_dir=corpus_dir)
    return TestClient(app)


# -----------------------------------------------------------------------------
# Health / listing
# -----------------------------------------------------------------------------

def test_health(corpus_copy: Path) -> None:
    client = _client(corpus_copy.parent)
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["clip_count"] == 1
    assert body["read_only"] is False


def test_list_clips(corpus_copy: Path) -> None:
    # Annotated clip (the copied JSON) + an unlabelled video (touch a fake .mp4
    # whose stem will become a clip_id).
    video_dir = corpus_copy.parent
    (video_dir / "ZZ_some_unlabelled_clip.mp4").touch()

    client = _client(video_dir)
    r = client.get("/api/clips")
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 2

    # Sorted by clip_id.
    assert [it["clip_id"] for it in items] == sorted(it["clip_id"] for it in items)
    kinds = {it["clip_id"]: it["kind"] for it in items}
    has_video = {it["clip_id"]: it["has_video"] for it in items}
    assert "ZZ_some_unlabelled_clip" in kinds
    assert kinds["ZZ_some_unlabelled_clip"] == "unlabelled"
    assert has_video["ZZ_some_unlabelled_clip"] is True

    # The annotated entry's clip_id comes from the bundle, not the filename.
    annotated_clip = load_file(corpus_copy).video.clip_id
    assert kinds[annotated_clip] == "annotated"


# -----------------------------------------------------------------------------
# GET annotations
# -----------------------------------------------------------------------------

def test_get_annotations_round_trip(corpus_copy: Path) -> None:
    bundle = load_file(corpus_copy)
    clip_id = bundle.video.clip_id

    client = _client(corpus_copy.parent)
    r = client.get(f"/api/clips/{clip_id}/annotations")
    assert r.status_code == 200
    payload = r.json()
    assert payload["video"]["clip_id"] == clip_id
    # Server dump matches direct load + dump.
    assert payload == bundle.model_dump(by_alias=True, exclude_unset=True, mode="json")


def test_get_annotations_404_on_unknown_clip(corpus_copy: Path) -> None:
    client = _client(corpus_copy.parent)
    r = client.get("/api/clips/nope-does-not-exist/annotations")
    assert r.status_code == 404


# -----------------------------------------------------------------------------
# PUT validation / atomic write / read-only / fresh-clip / bbox round-trip
# -----------------------------------------------------------------------------

def test_put_annotations_validates(corpus_copy: Path) -> None:
    bundle = load_file(corpus_copy)
    clip_id = bundle.video.clip_id

    client = _client(corpus_copy.parent)
    r = client.put(
        f"/api/clips/{clip_id}/annotations",
        json={"definitely_not_a_bundle": 1},
    )
    assert r.status_code == 422
    body = r.json()
    assert "errors" in body
    assert isinstance(body["errors"], list)
    assert body["errors"]  # at least one validation error


def test_put_annotations_writes_atomically(corpus_copy: Path) -> None:
    bundle = load_file(corpus_copy)
    clip_id = bundle.video.clip_id
    original_status = bundle.status

    # Mutate something visible.
    new_status = "needs_revision" if original_status != "needs_revision" else "approved"
    payload = bundle.model_dump(by_alias=True, exclude_unset=True, mode="json")
    payload["status"] = new_status

    # Capture pre-state for the file.
    pre_bytes = corpus_copy.read_bytes()

    client = _client(corpus_copy.parent)
    r = client.put(f"/api/clips/{clip_id}/annotations", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["saved_to"] == str(corpus_copy)
    assert body["bundle"]["status"] == new_status

    # On-disk bytes differ from pre-state, and parse back to the new value.
    post_bytes = corpus_copy.read_bytes()
    assert post_bytes != pre_bytes
    reloaded = load_file(corpus_copy)
    assert reloaded.status == new_status

    # No leftover `.tmp` siblings (atomic save cleaned up).
    leftovers = [
        p for p in corpus_copy.parent.iterdir()
        if p.name.startswith(f".{corpus_copy.name}.") and p.suffix == ".tmp"
    ]
    assert leftovers == []


def test_put_annotations_read_only_409(corpus_copy: Path) -> None:
    bundle = load_file(corpus_copy)
    clip_id = bundle.video.clip_id

    client = _client(corpus_copy.parent, read_only=True)
    r = client.put(
        f"/api/clips/{clip_id}/annotations",
        json=bundle.model_dump(by_alias=True, exclude_unset=True, mode="json"),
    )
    assert r.status_code == 409
    assert r.json() == {"error": "server is read-only"}


def test_put_annotations_fresh_clip_creates_file(tmp_path: Path) -> None:
    # Video-only "source" — no JSON yet.
    video = tmp_path / "freshclip.mp4"
    video.touch()

    index = build_clip_index([tmp_path])
    assert index["freshclip"].kind == "unlabelled"
    assert index["freshclip"].path is None

    app = create_app(index, read_only=False, destination_dir=tmp_path)
    client = TestClient(app)

    # The empty bundle the server reports for an unlabelled clip is a fine
    # round-trip payload, but we tweak `status` so we can verify the save.
    seed = client.get("/api/clips/freshclip/annotations").json()
    assert seed["video"]["clip_id"] == "freshclip"
    seed["status"] = "annotating"

    r = client.put("/api/clips/freshclip/annotations", json=seed)
    assert r.status_code == 200, r.text
    body = r.json()
    expected_path = tmp_path / "freshclip.json"
    assert body["saved_to"] == str(expected_path)
    assert expected_path.is_file()

    # The index entry mutated to annotated.
    entry = index["freshclip"]
    assert entry.kind == "annotated"
    assert entry.path == expected_path


def test_put_annotations_url_clip_id_mismatch_422(corpus_copy: Path) -> None:
    bundle = load_file(corpus_copy)
    clip_id = bundle.video.clip_id
    payload = bundle.model_dump(by_alias=True, exclude_unset=True, mode="json")
    payload["video"]["clip_id"] = "some-other-clip-id"

    client = _client(corpus_copy.parent)
    r = client.put(f"/api/clips/{clip_id}/annotations", json=payload)
    assert r.status_code == 422


def test_get_bbox_round_trip(bbox_corpus_copy: Path) -> None:
    """Regression: ensure bbox frames survive load → dump → reload.

    The upstream tool's frontend had a `stripLegacyBboxes` step that
    silently dropped bbox lists; we deliberately do not carry that bug. If
    a future change reintroduces stripping, this test catches it.

    The `bbox_corpus_copy` fixture guarantees a bbox-bearing source file —
    no skip path — so a regression that drops bboxes will fail loudly.
    """
    bundle = load_file(bbox_corpus_copy)
    clip_id = bundle.video.clip_id

    bbox_agents = [a for a in bundle.annotation.agents if a.bounding_boxes]
    assert bbox_agents, "bbox fixture invariant violated: no agent has bboxes"
    sample_agent = bbox_agents[0]
    expected_bbox_count = sum(len(f.bounding_boxes) for f in sample_agent.bounding_boxes)
    assert expected_bbox_count > 0

    client = _client(bbox_corpus_copy.parent)
    r = client.get(f"/api/clips/{clip_id}/annotations")
    assert r.status_code == 200
    payload = r.json()

    # Round-trip: PUT it back, then reload from disk.
    put = client.put(f"/api/clips/{clip_id}/annotations", json=payload)
    assert put.status_code == 200, put.text

    reloaded = load_file(bbox_corpus_copy)
    reloaded_agent = next(a for a in reloaded.annotation.agents if a.id == sample_agent.id)
    actual_bbox_count = sum(len(f.bounding_boxes) for f in reloaded_agent.bounding_boxes)
    assert actual_bbox_count == expected_bbox_count


# -----------------------------------------------------------------------------
# .bak snapshot (taken once per clip per server session)
# -----------------------------------------------------------------------------

def test_put_creates_bak_on_first_save_in_session(corpus_copy: Path) -> None:
    bundle = load_file(corpus_copy)
    clip_id = bundle.video.clip_id
    payload = bundle.model_dump(by_alias=True, exclude_unset=True, mode="json")
    bak_path = corpus_copy.with_suffix(corpus_copy.suffix + ".bak")
    pre_bytes = corpus_copy.read_bytes()
    assert not bak_path.exists()

    client = _client(corpus_copy.parent)

    # First save → .bak with the original bytes appears.
    r1 = client.put(f"/api/clips/{clip_id}/annotations", json=payload)
    assert r1.status_code == 200, r1.text
    assert bak_path.is_file()
    assert bak_path.read_bytes() == pre_bytes

    # Mutate the .bak so we can detect whether it gets re-stamped.
    sentinel = b'{"sentinel": "do-not-overwrite"}'
    bak_path.write_bytes(sentinel)

    # Second save (same session) → .bak is *not* re-written.
    payload["status"] = (
        "needs_revision" if payload.get("status") != "needs_revision" else "approved"
    )
    r2 = client.put(f"/api/clips/{clip_id}/annotations", json=payload)
    assert r2.status_code == 200, r2.text
    assert bak_path.read_bytes() == sentinel, ".bak should only stamp once per session"


def test_put_fresh_clip_does_not_write_bak(tmp_path: Path) -> None:
    """An unlabelled clip's first save creates the new file; no `.bak` to write."""
    video = tmp_path / "freshclip.mp4"
    video.touch()
    index = build_clip_index([tmp_path])
    app = create_app(index, read_only=False, destination_dir=tmp_path)
    client = TestClient(app)

    seed = client.get("/api/clips/freshclip/annotations").json()
    r = client.put("/api/clips/freshclip/annotations", json=seed)
    assert r.status_code == 200

    expected_path = tmp_path / "freshclip.json"
    assert expected_path.is_file()
    assert not expected_path.with_suffix(".json.bak").exists()


# -----------------------------------------------------------------------------
# make_empty_bundle self-descriptive on disk
# -----------------------------------------------------------------------------

def test_make_empty_bundle_includes_schema_and_status(tmp_path: Path) -> None:
    """A fresh-clip JSON written through the server keeps schema_version + status.

    `exclude_unset=True` in save_file would otherwise drop the defaults, so
    `make_empty_bundle` must mark them as set. This test exercises the end-to-end
    flow (GET unlabelled → PUT back → reload from disk) and asserts the
    persisted JSON is self-describing.
    """
    import json as _json

    video = tmp_path / "newclip.mp4"
    video.touch()
    index = build_clip_index([tmp_path])
    app = create_app(index, read_only=False, destination_dir=tmp_path)
    client = TestClient(app)

    seed = client.get("/api/clips/newclip/annotations").json()
    assert seed.get("schema_version") == "2.0.0"
    assert seed.get("status") == "annotating"

    r = client.put("/api/clips/newclip/annotations", json=seed)
    assert r.status_code == 200, r.text

    saved_path = tmp_path / "newclip.json"
    on_disk = _json.loads(saved_path.read_text())
    assert on_disk.get("schema_version") == "2.0.0"
    assert on_disk.get("status") == "annotating"


# -----------------------------------------------------------------------------
# Source-resolution sanity (covers io_adapter directly).
# -----------------------------------------------------------------------------

def test_build_clip_index_dedupes_duplicate_clip_ids(tmp_path: Path) -> None:
    """Two JSONs with the same video.clip_id collapse to one entry with a warning."""
    from causal_ai_av.spec import AnnotationBundle, VideoMeta

    b = AnnotationBundle(video=VideoMeta(clip_id="dup-clip"))
    a_path = tmp_path / "aaa__dup-clip.json"
    z_path = tmp_path / "zzz__dup-clip.json"
    a_path.write_text(b.model_dump_json(by_alias=True))
    z_path.write_text(b.model_dump_json(by_alias=True))

    with pytest.warns(UserWarning, match="clip_id"):
        index = build_clip_index([tmp_path])

    assert list(index.keys()) == ["dup-clip"]
    # First-by-filename wins.
    assert index["dup-clip"].path == a_path


def test_build_clip_index_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_clip_index([tmp_path / "does-not-exist"])
