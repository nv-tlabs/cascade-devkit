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
from cascade_av.io import load_file


def _client(corpus_dir: Path, *, read_only: bool = False) -> TestClient:
    index = build_clip_index([corpus_dir])
    app = create_app(index, read_only=read_only, destination_dir=corpus_dir)
    return TestClient(app)


@pytest.fixture
def mock_ffprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ffprobe with a fixed (fps, duration_s) result.

    Tests that exercise the fresh-clip path (`make_empty_bundle` → probe)
    only care that a bundle materialises; they do not assert on probe
    output. Per AGENTS.md ("the annotator's video tests mock
    `subprocess.run` / `shutil.which`, so real ffmpeg is never invoked"),
    we patch at the ``subprocess.run`` boundary inside ``annotator.server.video``
    so the real probe logic runs end-to-end and hosts without ffmpeg on
    PATH still pass.
    """
    import json as _json
    import subprocess

    probe_payload = _json.dumps({
        "streams": [{"avg_frame_rate": "30/1"}],
        "format": {"duration": "10.0"},
    })

    def _fake_probe(cmd: list[str], **_kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(cmd, 0, stdout=probe_payload, stderr="")

    monkeypatch.setattr("annotator.server.video.subprocess.run", _fake_probe)


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


def test_put_annotations_fresh_clip_creates_file(
    tmp_path: Path, mock_ffprobe: None
) -> None:
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


# NOTE: `test_get_bbox_round_trip` was removed in schema 2.1.0 along with the
# typed bbox fields. Round-trip coverage for bbox-as-extension data will land
# alongside the bbox extension itself (deferred PR; see plan
# `/home/horde/.claude/plans/regarding-the-boundingboxes-my-woolly-plum.md`).


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


def test_put_fresh_clip_does_not_write_bak(
    tmp_path: Path, mock_ffprobe: None
) -> None:
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

def test_make_empty_bundle_includes_schema_and_status(
    tmp_path: Path, mock_ffprobe: None
) -> None:
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

    from cascade_av.spec import CURRENT_SCHEMA_VERSION

    seed = client.get("/api/clips/newclip/annotations").json()
    assert seed.get("schema_version") == CURRENT_SCHEMA_VERSION
    assert seed.get("status") == "annotating"

    r = client.put("/api/clips/newclip/annotations", json=seed)
    assert r.status_code == 200, r.text

    saved_path = tmp_path / "newclip.json"
    on_disk = _json.loads(saved_path.read_text())
    assert on_disk.get("schema_version") == CURRENT_SCHEMA_VERSION
    assert on_disk.get("status") == "annotating"


def test_make_empty_bundle_probes_real_video_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `make_empty_bundle` receives a `video_path`, it calls
    ffprobe and threads the real `fps` / `duration_s` into the bundle's
    `VideoMeta`. This is the phase-3 wiring that replaces the old
    schema-defaults stub."""
    import json as _json
    import subprocess

    from annotator.server.io_adapter import make_empty_bundle

    probe_payload = _json.dumps({
        "streams": [{"avg_frame_rate": "30000/1001"}],  # NTSC 29.97
        "format": {"duration": "12.345"},
    })

    def _fake_probe(cmd: list[str], **_kwargs):  # type: ignore[no-untyped-def]
        assert cmd[0] == "ffprobe"
        return subprocess.CompletedProcess(cmd, 0, stdout=probe_payload, stderr="")

    monkeypatch.setattr("annotator.server.video.subprocess.run", _fake_probe)

    video = tmp_path / "newclip.mp4"
    video.touch()
    bundle = make_empty_bundle("newclip", video_path=video)

    assert bundle.video.fps == pytest.approx(30000.0 / 1001.0)
    assert bundle.video.duration_s == pytest.approx(12.345)


def test_make_empty_bundle_falls_back_to_defaults_when_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Probe failure (missing ffprobe, malformed JSON, no video stream)
    must NOT crash — `make_empty_bundle` falls back to the schema
    defaults so the fresh-clip path still produces a valid bundle.
    Important because the annotator must open unreadable / corrupt
    clips to let the user fix them."""
    import subprocess

    from annotator.server.io_adapter import make_empty_bundle

    def _failing_probe(cmd: list[str], **_kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr("annotator.server.video.subprocess.run", _failing_probe)

    video = tmp_path / "broken.mp4"
    video.touch()
    bundle = make_empty_bundle("broken", video_path=video)

    # Schema defaults: fps=30.0, duration_s=0.0 (see VideoMeta in
    # src/cascade_av/spec/schema.py).
    assert bundle.video.fps == 30.0
    assert bundle.video.duration_s == 0.0
    # Bundle still valid; clip_id + status both set.
    assert bundle.video.clip_id == "broken"
    assert bundle.status == "annotating"


def test_make_empty_bundle_skips_probe_when_video_path_is_none() -> None:
    """`video_path=None` (the default) must short-circuit before any
    subprocess call — synthetic test fixtures and unit tests that don't
    care about real video metadata stay fast and side-effect-free."""
    from annotator.server.io_adapter import make_empty_bundle

    bundle = make_empty_bundle("synthetic")
    assert bundle.video.fps == 30.0
    assert bundle.video.duration_s == 0.0
    assert bundle.video.clip_id == "synthetic"


# -----------------------------------------------------------------------------
# Source-resolution sanity (covers io_adapter directly).
# -----------------------------------------------------------------------------

def test_build_clip_index_dedupes_duplicate_clip_ids(tmp_path: Path) -> None:
    """Two JSONs with the same video.clip_id collapse to one entry with a warning."""
    from cascade_av.spec import AnnotationBundle, VideoMeta

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


def test_build_clip_index_skips_extra_json_sidecars(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`<stem>.extra.json` sidecar files must not be treated as main bundles.

    They share the `.json` suffix (Path.suffix only sees the last
    dot-segment), so the naive `_classify` would route them through
    `load_file`, which then fails because the sidecar has no top-level
    `video` field. Before this fix every sidecar logged a
    `WARNING ... validation error for AnnotationBundle: video Field required`
    on annotator launch against a migrated corpus.
    """
    from cascade_av.spec import AnnotationBundle, VideoMeta

    main = tmp_path / "clip1.json"
    main.write_text(
        AnnotationBundle(video=VideoMeta(clip_id="clip1")).model_dump_json(
            by_alias=True
        )
    )
    sidecar = tmp_path / "clip1.extra.json"
    sidecar.write_text(
        '{"schema_version": "2.2.0", "main_file": "clip1.json", '
        '"extensions": {"ui/1.0": {"track_index": {}}}}'
    )

    import logging

    with caplog.at_level(logging.WARNING, logger="annotator.server.io_adapter"):
        index = build_clip_index([tmp_path])

    assert set(index.keys()) == {"clip1"}
    assert index["clip1"].path == main
    # The noisy "failed to parse ... validation error" warning is the
    # signature of the bug — its absence is the contract.
    assert not any(
        "failed to parse" in r.message and "extra.json" in r.message
        for r in caplog.records
    )


# -----------------------------------------------------------------------------
# POST /annotations/validate — server-side validation contract
# -----------------------------------------------------------------------------

def _clean_validate_payload(clip_id: str = "freshclip") -> dict:
    """A complete-but-minimal bundle payload that satisfies every rule.

    Timestamps are populated explicitly because `timestamps_have_value`
    (soft rule) warns on blank intervals; without them the validate endpoint
    returns warnings even though every hard rule passes.
    """
    return {
        "video": {"clip_id": clip_id, "duration_s": 10.0},
        "annotation": {
            "ego_vehicle": {
                "actions": [
                    {
                        "id": "EA1",
                        "type": "Drive Straight",
                        "start_timestamp": "0:0.0",
                        "end_timestamp": "0:5.0",
                    }
                ],
            },
        },
    }


def _dirty_validate_payload(clip_id: str = "freshclip") -> dict:
    """A bundle payload with NO ego actions — trips the ego rule."""
    return {
        "video": {"clip_id": clip_id, "duration_s": 10.0},
        "annotation": {"ego_vehicle": {"actions": []}},
    }


def test_validate_endpoint_clean_bundle_returns_ok_true(tmp_path: Path) -> None:
    """A bundle that satisfies every hard rule returns ``ok=True`` + no issues."""
    video = tmp_path / "freshclip.mp4"
    video.touch()
    index = build_clip_index([tmp_path])
    app = create_app(index, read_only=False, destination_dir=tmp_path)
    client = TestClient(app)

    r = client.post(
        "/api/clips/freshclip/annotations/validate", json=_clean_validate_payload()
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["issues"] == []


def test_validate_endpoint_dirty_bundle_returns_ok_false_and_issues(
    tmp_path: Path,
) -> None:
    """A bundle missing the required ego action returns ``ok=False`` + an issue."""
    video = tmp_path / "freshclip.mp4"
    video.touch()
    index = build_clip_index([tmp_path])
    app = create_app(index, read_only=False, destination_dir=tmp_path)
    client = TestClient(app)

    r = client.post(
        "/api/clips/freshclip/annotations/validate", json=_dirty_validate_payload()
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False
    assert isinstance(body["issues"], list)
    assert body["issues"]
    rules = {i["rule"] for i in body["issues"]}
    assert "ego_has_at_least_one_action" in rules
    # Every issue is the documented shape.
    for issue in body["issues"]:
        assert set(issue.keys()) == {
            "severity",
            "entity_path",
            "entity_id",
            "field",
            "rule",
            "message",
        }
        assert issue["severity"] in {"error", "warning"}


def test_validate_endpoint_unknown_clip_404(corpus_copy: Path) -> None:
    client = _client(corpus_copy.parent)
    r = client.post(
        "/api/clips/nope-does-not-exist/annotations/validate",
        json={"video": {"clip_id": "x"}},
    )
    assert r.status_code == 404


def test_validate_endpoint_invalid_json_422(tmp_path: Path) -> None:
    """Payloads that fail pydantic validation surface as 422 with errors."""
    video = tmp_path / "freshclip.mp4"
    video.touch()
    index = build_clip_index([tmp_path])
    app = create_app(index, read_only=False, destination_dir=tmp_path)
    client = TestClient(app)

    r = client.post(
        "/api/clips/freshclip/annotations/validate",
        json={"definitely_not_a_bundle": 1},
    )
    assert r.status_code == 422
    body = r.json()
    assert "errors" in body
    assert isinstance(body["errors"], list)
    assert body["errors"]


def test_validate_endpoint_does_not_write(corpus_copy: Path) -> None:
    """Validation is read-only — the on-disk bytes must not change."""
    bundle = load_file(corpus_copy)
    clip_id = bundle.video.clip_id
    payload = bundle.model_dump(by_alias=True, exclude_unset=True, mode="json")
    pre_bytes = corpus_copy.read_bytes()

    client = _client(corpus_copy.parent)
    r = client.post(f"/api/clips/{clip_id}/annotations/validate", json=payload)
    assert r.status_code == 200, r.text
    assert corpus_copy.read_bytes() == pre_bytes


def test_validate_endpoint_works_in_read_only_mode(corpus_copy: Path) -> None:
    """Validation must work when the server is read-only — it never writes."""
    bundle = load_file(corpus_copy)
    clip_id = bundle.video.clip_id
    payload = bundle.model_dump(by_alias=True, exclude_unset=True, mode="json")

    client = _client(corpus_copy.parent, read_only=True)
    r = client.post(f"/api/clips/{clip_id}/annotations/validate", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "ok" in body
    assert "issues" in body


# -----------------------------------------------------------------------------
# PUT /annotations defense-in-depth — status=complete + validation gating
# -----------------------------------------------------------------------------

def _bundle_payload_for(corpus_copy: Path, *, status: str | None) -> tuple[str, dict]:
    """Helper: build a (clip_id, wire-shape payload) pair from the corpus fixture.

    Picks the status off the loaded bundle and overrides it. Used to exercise
    the PUT gate without re-deriving the corpus shape in every test.
    """
    bundle = load_file(corpus_copy)
    clip_id = bundle.video.clip_id
    payload = bundle.model_dump(by_alias=True, exclude_unset=True, mode="json")
    if status is None:
        payload.pop("status", None)
    else:
        payload["status"] = status
    return clip_id, payload


def test_put_complete_with_clean_bundle_saves(corpus_copy: Path) -> None:
    """A clean bundle marked `status="complete"` saves with 200."""
    clip_id, payload = _bundle_payload_for(corpus_copy, status="complete")
    client = _client(corpus_copy.parent)
    r = client.put(f"/api/clips/{clip_id}/annotations", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["bundle"]["status"] == "complete"


def test_put_complete_with_dirty_bundle_returns_422(tmp_path: Path) -> None:
    """A bundle with hard-error issues + `status="complete"` is rejected 422.

    Body shape must mirror the /validate endpoint so the frontend reuses
    a single issues-panel renderer for both paths.
    """
    video = tmp_path / "freshclip.mp4"
    video.touch()
    index = build_clip_index([tmp_path])
    app = create_app(index, read_only=False, destination_dir=tmp_path)
    client = TestClient(app)

    # Empty ego.actions trips `ego_has_at_least_one_action`.
    payload = {
        "video": {"clip_id": "freshclip", "duration_s": 10.0},
        "annotation": {"ego_vehicle": {"actions": []}},
        "status": "complete",
    }
    r = client.put("/api/clips/freshclip/annotations", json=payload)
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["error"] == "validation failed"
    assert isinstance(body["issues"], list)
    assert body["issues"]
    rules = {i["rule"] for i in body["issues"]}
    assert "ego_has_at_least_one_action" in rules
    # Every issue is the wire shape the frontend already consumes from
    # /validate (defense-in-depth → reuse the issues-panel renderer).
    for issue in body["issues"]:
        assert set(issue.keys()) == {
            "severity",
            "entity_path",
            "entity_id",
            "field",
            "rule",
            "message",
        }
    # File on disk must NOT have been written — the gate runs before save.
    assert not (tmp_path / "freshclip.json").exists()


def test_put_dirty_with_status_annotating_still_saves(tmp_path: Path) -> None:
    """Progress save semantics: a bundle that fails validation still persists
    when its status is anything other than `complete`. Annotators must be
    able to checkpoint partial work.
    """
    video = tmp_path / "freshclip.mp4"
    video.touch()
    index = build_clip_index([tmp_path])
    app = create_app(index, read_only=False, destination_dir=tmp_path)
    client = TestClient(app)

    payload = {
        "video": {"clip_id": "freshclip", "duration_s": 10.0},
        "annotation": {"ego_vehicle": {"actions": []}},
        "status": "annotating",
    }
    r = client.put("/api/clips/freshclip/annotations", json=payload)
    assert r.status_code == 200, r.text
    assert (tmp_path / "freshclip.json").exists()


def test_put_complete_warnings_only_saves(tmp_path: Path) -> None:
    """Soft-rule warnings do NOT block `status="complete"` — only hard errors do."""
    video = tmp_path / "freshclip.mp4"
    video.touch()
    index = build_clip_index([tmp_path])
    app = create_app(index, read_only=False, destination_dir=tmp_path)
    client = TestClient(app)

    # One ego action (satisfies every hard rule) with a blank because_of
    # entry — trips `id_references_have_value` (warning only).
    payload = {
        "video": {"clip_id": "freshclip", "duration_s": 10.0},
        "annotation": {
            "ego_vehicle": {
                "actions": [
                    {
                        "id": "EA1",
                        "type": "Drive Straight",
                        "start_timestamp": "0:0.0",
                        "end_timestamp": "0:5.0",
                        "because_of": [""],
                    }
                ]
            }
        },
        "status": "complete",
    }
    r = client.put("/api/clips/freshclip/annotations", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["bundle"]["status"] == "complete"


def test_list_clips_includes_status(corpus_copy: Path) -> None:
    """`GET /api/clips` exposes per-clip status so the sidebar can badge it
    without loading every bundle. Annotated clips return their on-disk
    status string; unlabelled clips return null."""
    video_dir = corpus_copy.parent
    (video_dir / "ZZ_some_unlabelled_clip.mp4").touch()

    client = _client(video_dir)
    r = client.get("/api/clips")
    assert r.status_code == 200
    items = {it["clip_id"]: it for it in r.json()}
    for item in items.values():
        assert "status" in item
    assert items["ZZ_some_unlabelled_clip"]["status"] is None
    # Annotated entry's status mirrors the on-disk JSON.
    annotated_id = load_file(corpus_copy).video.clip_id
    expected_status = load_file(corpus_copy).status
    assert items[annotated_id]["status"] == expected_status


def test_validate_endpoint_empty_condition_type_string_coerces_and_fires_rule(
    tmp_path: Path,
) -> None:
    """A bare-string `condition.type: ""` survives the schema's `_str_to_list`
    coercion (Condition wraps bare strings → `[""]`) and trips
    ``condition_requires_type``. Locks the coercion + rule interaction
    end-to-end across the wire.
    """
    video = tmp_path / "freshclip.mp4"
    video.touch()
    index = build_clip_index([tmp_path])
    app = create_app(index, read_only=False, destination_dir=tmp_path)
    client = TestClient(app)

    payload = _clean_validate_payload()
    # Wire-shape: `type` arrives as a bare string. Pydantic coerces to [""].
    payload["annotation"]["conditions"] = [{"id": "C1", "type": ""}]

    r = client.post("/api/clips/freshclip/annotations/validate", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False
    rules = {i["rule"] for i in body["issues"]}
    assert "condition_requires_type" in rules
    # And the offending entity is the one we just added.
    cond_issues = [i for i in body["issues"] if i["rule"] == "condition_requires_type"]
    assert any(i["entity_id"] == "C1" for i in cond_issues)
