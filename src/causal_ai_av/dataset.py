"""`CausalAVDataset` + `Sequence` — the user-facing DevKit entry points.

`CausalAVDataset` is a thin subclass of `physical_ai_av.PhysicalAIAVDatasetInterface`.
It extends the parent with:
- annotation I/O (local directory of `*.json`, or eventually a HF repo),
- per-clip `Sequence` objects bundling the annotation with the parent's
  egomotion / video features,
- point-in-time and windowed state queries (`Sequence.state_at`),
- a placeholder `Sequence.visualize` that the annotator subsystem will replace.

The visualize implementation is intentionally minimal; the spec calls it out
as a placeholder that the annotator subsystem will rewrite.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from physical_ai_av import PhysicalAIAVDatasetInterface

from causal_ai_av.io import load_file
from causal_ai_av.query import (
    Interval,
    MatchSet,
    agents_visible_at,
    ego_relative_pose_at,
    extract_causal_triplets,
    filter_active_at,
    filter_active_in_range,
    parse_timestamp,
)
from causal_ai_av.query.api import (
    count_on_dataset,
    find_on_bundle,
    find_on_dataset,
    group_by_on_dataset,
)
from causal_ai_av.query.triplets import CausalTriplet
from causal_ai_av.spec import (
    AnnotationBundle,
    BoundingBox,
    BoundingBoxFrame,
)
from causal_ai_av.state import (
    ActorState,
    ActorStateRange,
    EgoState,
    EgoStateRange,
    SequenceState,
    SequenceStateRange,
)

if TYPE_CHECKING:  # pragma: no cover — typing only
    from physical_ai_av.egomotion import EgomotionState
    from physical_ai_av.video import SeekVideoReader


# Default canonical-camera feature name. Match the parent dataset's feature
# string for the wide-FOV front camera.
DEFAULT_ANNOTATION_CAMERA = "CAMERA_FRONT_WIDE_120FOV"


# -----------------------------------------------------------------------------
# Helpers — strict bbox matching, interval-set filters
# -----------------------------------------------------------------------------

def _strict_bbox_frame(
    frames: list[BoundingBoxFrame], t: float, fps: float
) -> BoundingBoxFrame | None:
    """Return the bbox frame whose timestamp is within 1/fps of `t`, or None.

    "Strict" per the spec: we tolerate a single-frame offset. If multiple
    frames satisfy the predicate, the closest one wins.
    """
    if fps <= 0:
        return None
    tol = 1.0 / fps
    best: tuple[float, BoundingBoxFrame] | None = None
    for frame in frames:
        ft = parse_timestamp(frame.timestamp)
        if ft is None:
            continue
        d = abs(ft - t)
        if d < tol and (best is None or d < best[0]):
            best = (d, frame)
    return best[1] if best else None


def _bbox_for_agent(frame: BoundingBoxFrame, agent_id: str) -> BoundingBox | None:
    """The single box drawn for `agent_id` in `frame`, or None."""
    for box in frame.bounding_boxes:
        if box.object_id == agent_id:
            return box
    return None


def _frames_in_window(
    frames: list[BoundingBoxFrame], t_start: float, t_end: float
) -> list[BoundingBoxFrame]:
    """Frames whose timestamp falls inside `[t_start, t_end]`."""
    out: list[BoundingBoxFrame] = []
    for frame in frames:
        ft = parse_timestamp(frame.timestamp)
        if ft is not None and t_start <= ft <= t_end:
            out.append(frame)
    return out


# -----------------------------------------------------------------------------
# CausalAVDataset
# -----------------------------------------------------------------------------

class CausalAVDataset(PhysicalAIAVDatasetInterface):
    """DevKit for the AV Causal Dataset.

    Extends `PhysicalAIAVDatasetInterface` with annotation I/O and per-clip
    state queries. Construct from a local directory of annotation JSONs:

        ds = CausalAVDataset("/path/to/01_json_annotations")
        seq = ds.get_sequence(ds.list_sequences()[0])

    Pass `annotations=None` to load from the HF repo — currently unimplemented;
    raises `NotImplementedError` so the polymorphism is wired for a future PR.
    """

    DEFAULT_HF_REPO: ClassVar[str | None] = None

    def __init__(
        self,
        annotations: str | Path | list[str | Path] | None = None,
        *,
        annotation_camera: str = DEFAULT_ANNOTATION_CAMERA,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_dir: str | Path | None = None,
        confirm_download_threshold_gb: float = 10.0,
    ) -> None:
        if annotations is None:
            # Polymorphism is wired; a future PR sets `DEFAULT_HF_REPO` and
            # dispatches to an HF-backed loader here.
            raise NotImplementedError(
                "HF repo for causal annotations not yet defined"
            )

        # Resolve the per-clip annotation index *before* touching the parent —
        # if the local path is bogus we want to fail loudly without paying for
        # the parent's metadata download.
        annotation_paths, annotation_root = self._resolve_annotation_paths(annotations)

        super().__init__(
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_dir=local_dir,
            confirm_download_threshold_gb=confirm_download_threshold_gb,
        )

        self.annotation_camera = annotation_camera
        # _by_clip[clip_id] = (path, batch_name_or_None, bundle)
        self._by_clip: dict[str, tuple[Path, str | None, AnnotationBundle]] = {}
        self._scan(annotation_paths, annotation_root)

    # -- annotation loading ---------------------------------------------------

    @staticmethod
    def _resolve_annotation_paths(
        annotations: str | Path | list[str | Path],
    ) -> tuple[list[Path], Path | None]:
        """Normalize the `annotations` argument to (paths, root).

        Validation only — no parsing — so we fail fast on bad inputs before
        spending time on parent metadata. The returned `root` is the
        user-supplied directory in dir mode, or `None` in list mode; downstream
        code uses it to decide whether `path.parent.name` is meaningful as a
        batch label.
        """
        if isinstance(annotations, list):
            paths = [Path(p) for p in annotations]
            for p in paths:
                if not p.is_file():
                    raise FileNotFoundError(f"annotation path is not a file: {p}")
            return paths, None

        p = Path(annotations)
        if p.is_dir():
            return sorted(p.rglob("*.json")), p
        raise FileNotFoundError(f"annotations path is not a directory: {p}")

    def _scan(self, paths: list[Path], root: Path | None) -> None:
        """Parse each path; key by `bundle.video.clip_id`. Warn on dup clip-ids.

        `root` is the user-supplied annotations directory (or `None` for the
        list-mode constructor). When a file lives directly under `root`, its
        `batch` is `None`; only files in a `root/<batch>/...` layout get a
        non-`None` batch label.
        """
        # First pass: collect (path, bundle) by clip_id so we can deterministically
        # keep the first by sorted filename when there are duplicates.
        per_clip: dict[str, list[tuple[Path, AnnotationBundle]]] = {}
        for path in sorted(paths):
            bundle = load_file(path)
            per_clip.setdefault(bundle.video.clip_id, []).append((path, bundle))

        # Track duplicates so we can emit one summary warning at end-of-scan.
        duplicate_summary: list[tuple[str, Path, list[Path]]] = []
        for clip_id, entries in per_clip.items():
            # `sorted(paths)` above already ordered entries; first wins.
            kept_path, kept_bundle = entries[0]
            if len(entries) > 1:
                dropped = [p for p, _ in entries[1:]]
                duplicate_summary.append((clip_id, kept_path, dropped))
            if root is not None and kept_path.parent != root:
                batch: str | None = kept_path.parent.name
            else:
                batch = None
            self._by_clip[clip_id] = (kept_path, batch, kept_bundle)

        if duplicate_summary:
            n = len(duplicate_summary)
            warnings.warn(
                f"{n} clip_id(s) had multiple annotation files; kept first by "
                "filename. (Set CAUSAL_AV_VERBOSE=1 for per-file detail.)",
                stacklevel=2,
            )
            if os.environ.get("CAUSAL_AV_VERBOSE"):
                import sys
                for clip_id, kept, dropped in duplicate_summary:
                    dropped_names = ", ".join(p.name for p in dropped)
                    print(
                        f"  clip {clip_id}: kept {kept.name}; dropped {dropped_names}",
                        file=sys.stderr,
                    )

    # -- public API -----------------------------------------------------------

    def list_sequences(self) -> list[str]:
        """Sorted list of clip IDs with an annotation in this dataset."""
        return sorted(self._by_clip.keys())

    def get_sequence(self, clip_id: str, *, download_clip: bool = False) -> Sequence:
        """Return the `Sequence` for `clip_id`.

        If `download_clip=True`, also eagerly opens the canonical-camera video
        reader so the first `.video` access is fast.
        """
        if clip_id not in self._by_clip:
            raise KeyError(f"clip_id not in dataset: {clip_id}")
        path, batch, bundle = self._by_clip[clip_id]
        seq = Sequence(
            annotation=bundle,
            parent=self,
            annotation_path=path,
            batch=batch,
            annotation_camera=self.annotation_camera,
        )
        if download_clip:
            # Touch the video property to warm it.
            _ = seq.video
        return seq

    # -- container protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._by_clip)

    def __contains__(self, clip_id: object) -> bool:
        return isinstance(clip_id, str) and clip_id in self._by_clip

    def __iter__(self) -> Iterator[Sequence]:
        for clip_id in self.list_sequences():
            yield self.get_sequence(clip_id)

    def __getitem__(self, clip_id: str) -> Sequence:
        return self.get_sequence(clip_id)

    # -- DSL search / aggregations -------------------------------------------

    def find(self, dsl: str) -> MatchSet:
        """Run a DSL query over every clip in the dataset.

        Returns a single MatchSet whose entries reference entities
        across the corpus. See `meta/07_query_language.md`.
        """
        return find_on_dataset(self, dsl)

    def count(self, dsl: str) -> int:
        """Number of clips with ≥1 match for `dsl`."""
        return count_on_dataset(self, dsl)

    def group_by(self, dsl: str, key: str) -> dict[object, int]:
        """Run `dsl`, bucket matches by `key` (dotted path), return `{value: n_clips}`."""
        return group_by_on_dataset(self, dsl, key)

    def histogram(self, dsl: str, key: str) -> dict[object, int]:
        """Alias of `group_by` (kept for API symmetry per spec §4.3)."""
        return group_by_on_dataset(self, dsl, key)


# -----------------------------------------------------------------------------
# Sequence
# -----------------------------------------------------------------------------

class _CamerasView(Mapping[str, "SeekVideoReader"]):
    """Lazy, cached mapping from camera-feature name to `SeekVideoReader`.

    Decoding is deferred until first access per camera. We don't enumerate
    the full feature set — the keys are whatever the parent dataset reports
    under `features.CAMERA.ALL`.
    """

    def __init__(self, sequence: "Sequence") -> None:
        self._seq = sequence
        self._cache: dict[str, SeekVideoReader] = {}

    def _camera_names(self) -> set[str]:
        parent = self._seq._require_parent()
        cam = getattr(parent.features, "CAMERA", None)
        if cam is None:
            return set()
        return set(getattr(cam, "ALL", set()))

    def __getitem__(self, name: str) -> "SeekVideoReader":
        if name in self._cache:
            return self._cache[name]
        if name not in self._camera_names():
            raise KeyError(f"unknown camera feature: {name}")
        parent = self._seq._require_parent()
        reader = parent.get_clip_feature(self._seq.clip_id, name)
        self._cache[name] = reader
        return reader

    def __iter__(self) -> Iterator[str]:
        return iter(self._camera_names())

    def __len__(self) -> int:
        return len(self._camera_names())


class Sequence:
    """One annotated clip plus access to the parent dataset's per-clip features.

    Constructed via `CausalAVDataset.get_sequence`. For test or notebook use
    without the parent dataset, use `Sequence.from_annotation(bundle)`; the
    parent-backed properties (`video`, `cameras`, `egomotion`,
    `egomotion_interpolator`) will raise a descriptive error.
    """

    def __init__(
        self,
        annotation: AnnotationBundle,
        *,
        parent: PhysicalAIAVDatasetInterface | None = None,
        annotation_path: Path | None = None,
        batch: str | None = None,
        annotation_camera: str = DEFAULT_ANNOTATION_CAMERA,
    ) -> None:
        self.annotation = annotation
        self.clip_id = annotation.video.clip_id
        self._parent = parent
        self.annotation_path = annotation_path
        self.batch = batch
        self.annotation_camera = annotation_camera

        # Lazy slots — populated on first access.
        self._egomotion_interpolator: Any | None = None
        self._cameras: _CamerasView | None = None
        self._triplets: list[CausalTriplet] | None = None

        # Eagerly load egomotion *if* a parent is available, per the spec.
        # `get_clip_feature(clip_id, "egomotion")` already returns an
        # `Interpolator[EgomotionState]` (see physical_ai_av/dataset.py),
        # so we cache it directly.
        if parent is not None:
            self._egomotion_interpolator = parent.get_clip_feature(
                self.clip_id, "egomotion"
            )

    # -- constructors ---------------------------------------------------------

    @classmethod
    def from_annotation(
        cls,
        annotation: AnnotationBundle,
        *,
        annotation_camera: str = DEFAULT_ANNOTATION_CAMERA,
    ) -> Sequence:
        """Build a Sequence with no parent dataset wired in.

        Useful for tests, notebooks, or offline analysis where the user only
        needs `state_at` / `triplets` and doesn't care about egomotion/video.
        """
        return cls(annotation=annotation, parent=None, annotation_camera=annotation_camera)

    # -- internal -------------------------------------------------------------

    def _require_parent(self) -> PhysicalAIAVDatasetInterface:
        if self._parent is None:
            raise RuntimeError(
                "Sequence has no parent dataset; this property requires a "
                "`PhysicalAIAVDatasetInterface` to load clip features. "
                "Construct via `CausalAVDataset.get_sequence` instead of "
                "`Sequence.from_annotation`."
            )
        return self._parent

    # -- annotation-derived properties ----------------------------------------

    @property
    def description(self) -> str:
        return self.annotation.annotation.brief_description

    @property
    def fps(self) -> float:
        return self.annotation.video.fps

    @property
    def duration_s(self) -> float:
        return self.annotation.video.duration_s

    # -- parent-backed properties --------------------------------------------

    @property
    def video(self) -> "SeekVideoReader":
        """The canonical-camera video reader (lazy).

        Delegates to `self.cameras[self.annotation_camera]` so there's a single
        underlying reader cache — no double-decode of the canonical camera.
        """
        return self.cameras[self.annotation_camera]

    @property
    def cameras(self) -> Mapping[str, "SeekVideoReader"]:
        """Lazy, cached mapping of camera-feature name → `SeekVideoReader`."""
        if self._cameras is None:
            self._cameras = _CamerasView(self)
        return self._cameras

    @property
    def egomotion(self) -> "EgomotionState":
        """The raw `EgomotionState` underlying the interpolator (arrays, not interpolated)."""
        # The parent's egomotion interpolator carries the original
        # `EgomotionState` on `.values`.
        return self.egomotion_interpolator.values

    @property
    def egomotion_interpolator(self) -> Any:
        """The eagerly-loaded `Interpolator[EgomotionState]` from the parent."""
        if self._egomotion_interpolator is None:
            # Parent must be present and egomotion fetch was deferred (e.g.
            # because no parent was wired in). Re-attempt to give a clear error.
            parent = self._require_parent()
            self._egomotion_interpolator = parent.get_clip_feature(
                self.clip_id, "egomotion"
            )
        return self._egomotion_interpolator

    # -- queries --------------------------------------------------------------

    def triplets(self) -> list[CausalTriplet]:
        """All causal triplets extracted from this clip (memoized)."""
        if self._triplets is None:
            self._triplets = extract_causal_triplets(self.annotation)
        return self._triplets

    def find(self, dsl: str) -> MatchSet:
        """Run a DSL query against this clip's annotation bundle."""
        return find_on_bundle(self.annotation, dsl)

    def state_at(
        self, t: float, t_end: float | None = None
    ) -> SequenceState | SequenceStateRange:
        """Snapshot the scene at `t` (instant) or over `[t, t_end]` (window)."""
        if t_end is None:
            return self._state_at_instant(t)
        return self._state_over_range(t, t_end)

    # -- visualize (placeholder) ---------------------------------------------

    def visualize(
        self, t: float | None = None, t_end: float | None = None
    ) -> dict:
        """Return a single-frame visualization payload.

        Placeholder implementation per the spec; the full multi-track playback
        will land with the annotator subsystem.
        """
        # Resolve the "target moment" per the spec.
        if t is None and t_end is None:
            target = self.duration_s / 2.0
        elif t is None and t_end is not None:
            target = t_end / 2.0  # shouldn't happen in practice; defensive
        elif t is not None and t_end is None:
            target = t
        else:
            assert t is not None and t_end is not None
            target = (t + t_end) / 2.0

        state = self.state_at(t, t_end) if t is not None else self.state_at(target)

        # Strict-match bboxes for every agent at the target moment.
        bboxes: list[BoundingBox] = []
        for agent in self.annotation.annotation.agents:
            frame = _strict_bbox_frame(agent.bounding_boxes, target, self.fps)
            if frame is None:
                continue
            box = _bbox_for_agent(frame, agent.id)
            if box is not None:
                bboxes.append(box)

        # Frame decode — best-effort. We never raise here: the parent stack
        # may be absent (notebook / test path) and the placeholder shouldn't
        # surface that to the caller.
        frame_image: Any | None = None
        try:
            import numpy as np  # local import; numpy is a transitive parent dep

            video = self.video
            # SeekVideoReader.timestamps are in the parent's native units
            # (microseconds). Convert seconds → microseconds.
            requested = np.asarray([int(round(target * 1_000_000))], dtype=np.int64)
            images, _ = video.decode_images_from_timestamps(requested)
            frame_image = images[0]
        except Exception:  # pragma: no cover — placeholder path
            frame_image = None

        return {
            "frame": frame_image,
            "state": state,
            "bboxes": bboxes,
            "note": "placeholder; full multi-track playback comes with the annotator subsystem",
        }

    # -- state_at implementation ---------------------------------------------

    def _state_at_instant(self, t: float) -> SequenceState:
        ann = self.annotation.annotation
        fps = self.fps

        # Ego state
        ego_actions = filter_active_at(ann.ego_vehicle.actions, t)
        ego_properties = filter_active_at(ann.ego_vehicle.properties, t)
        ego_containment = filter_active_at(ann.ego_vehicle.containment, t)
        ego_influences = filter_active_at(ann.ego_vehicle.influenced_by, t)

        ego_pose = self._interpolate_ego_pose(t)

        ego = EgoState(
            actions=ego_actions,
            properties=ego_properties,
            containment=ego_containment,
            influenced_by=ego_influences,
            pose=ego_pose,
        )

        # Actors — every agent visible at t.
        actors: list[ActorState] = []
        for agent in agents_visible_at(self.annotation, t):
            agent_actions = filter_active_at(agent.actions, t)
            pose = ego_relative_pose_at(agent, t)
            frame = _strict_bbox_frame(agent.bounding_boxes, t, fps)
            bbox = _bbox_for_agent(frame, agent.id) if frame is not None else None
            actors.append(ActorState(
                agent=agent,
                actions=agent_actions,
                pose_rel_to_ego=pose,
                bbox=bbox,
            ))

        environments = filter_active_at(ann.environments, t)
        conditions = filter_active_at(ann.conditions, t)
        traffic_objects = filter_active_at(
            ann.traffic_objects, t,
            start_attr="visibility_start_timestamp",
            end_attr="visibility_end_timestamp",
        )
        traffic_lights = filter_active_at(
            ann.traffic_lights, t,
            start_attr="visibility_start_timestamp",
            end_attr="visibility_end_timestamp",
        )
        triplets = [
            tri for tri in self.triplets()
            if tri.interval is not None and tri.interval.contains(t)
        ]

        return SequenceState(
            t=t,
            clip_id=self.clip_id,
            ego=ego,
            actors=actors,
            environments=environments,
            conditions=conditions,
            traffic_objects=traffic_objects,
            traffic_lights=traffic_lights,
            triplets=triplets,
        )

    def _state_over_range(self, t_start: float, t_end: float) -> SequenceStateRange:
        if t_end < t_start:
            t_start, t_end = t_end, t_start
        ann = self.annotation.annotation

        ego_actions = filter_active_in_range(ann.ego_vehicle.actions, t_start, t_end)
        ego_properties = filter_active_in_range(ann.ego_vehicle.properties, t_start, t_end)
        ego_containment = filter_active_in_range(ann.ego_vehicle.containment, t_start, t_end)
        ego_influences = filter_active_in_range(ann.ego_vehicle.influenced_by, t_start, t_end)

        pose_trajectory = self._interpolate_ego_trajectory(t_start, t_end)

        ego = EgoStateRange(
            actions=ego_actions,
            properties=ego_properties,
            containment=ego_containment,
            influenced_by=ego_influences,
            pose_trajectory=pose_trajectory,
        )

        window = Interval(t_start, t_end)
        visible_agents = filter_active_in_range(
            ann.agents, t_start, t_end,
            start_attr="visibility_start_timestamp",
            end_attr="visibility_end_timestamp",
        )
        actors: list[ActorStateRange] = []
        for agent in visible_agents:
            actions = filter_active_in_range(agent.actions, t_start, t_end)
            poses = filter_active_in_range(agent.ego_relative_pose, t_start, t_end)
            bbox_frames = _frames_in_window(agent.bounding_boxes, t_start, t_end)
            actors.append(ActorStateRange(
                agent=agent,
                actions=actions,
                pose_rel_to_ego=poses,
                bboxes=bbox_frames,
            ))

        environments = filter_active_in_range(ann.environments, t_start, t_end)
        conditions = filter_active_in_range(ann.conditions, t_start, t_end)
        traffic_objects = filter_active_in_range(
            ann.traffic_objects, t_start, t_end,
            start_attr="visibility_start_timestamp",
            end_attr="visibility_end_timestamp",
        )
        traffic_lights = filter_active_in_range(
            ann.traffic_lights, t_start, t_end,
            start_attr="visibility_start_timestamp",
            end_attr="visibility_end_timestamp",
        )
        triplets = [
            tri for tri in self.triplets()
            if tri.interval is not None and tri.interval.overlaps(window)
        ]

        return SequenceStateRange(
            t_start=t_start,
            t_end=t_end,
            clip_id=self.clip_id,
            ego=ego,
            actors=actors,
            environments=environments,
            conditions=conditions,
            traffic_objects=traffic_objects,
            traffic_lights=traffic_lights,
            triplets=triplets,
        )

    # -- egomotion helpers ----------------------------------------------------

    def _interpolate_ego_pose(self, t: float) -> "EgomotionState | None":
        """Best-effort ego-pose interpolation at clip-relative time `t` (seconds)."""
        if self._parent is None and self._egomotion_interpolator is None:
            return None
        try:
            interp = self.egomotion_interpolator
        except Exception:
            return None
        return self._call_interpolator(interp, t)

    def _interpolate_ego_trajectory(
        self, t_start: float, t_end: float
    ) -> "list[EgomotionState]":
        """Sample ego pose along `[t_start, t_end]`.

        The spec prefers the parent's native sample timestamps but accepts
        10 Hz across the window as a fallback. The native timestamps are
        always reachable via the interpolator's `.timestamps` attribute, so
        we use them when available and fall back to 10 Hz otherwise.
        """
        if self._parent is None and self._egomotion_interpolator is None:
            return []
        try:
            interp = self.egomotion_interpolator
        except Exception:
            return []

        timestamps = getattr(interp, "timestamps", None)
        try:
            import numpy as np
        except Exception:  # pragma: no cover — numpy is a transitive parent dep
            return []

        # Clip-relative microseconds: the parent's canonical convention (see
        # `physical_ai_av/notebooks/data_visualization.ipynb`). No anchoring.
        if timestamps is not None:
            t_start_us = int(round(t_start * 1_000_000))
            t_end_us = int(round(t_end * 1_000_000))
            mask = (timestamps >= t_start_us) & (timestamps <= t_end_us)
            sample_ts = timestamps[mask]
        else:
            # 10 Hz fallback — sample in seconds then convert per-call below.
            step = 0.1
            n = max(1, int(round((t_end - t_start) / step)) + 1)
            sample_ts = np.linspace(t_start, t_end, n)
            sample_ts = (sample_ts * 1_000_000).astype(np.int64)

        out: list[Any] = []
        for ts in sample_ts:
            try:
                state = interp(int(ts))
            except Exception:
                state = None
            if state is not None:
                out.append(state)
        return out

    @staticmethod
    def _call_interpolator(interp: Any, t: float) -> Any:
        """Call an `Interpolator[EgomotionState]` at clip-relative time `t` (seconds).

        Per the parent's convention (e.g. `prediction_timestamp_us = 5_100_000`
        passed directly in `physical_ai_av/notebooks/data_visualization.ipynb`),
        the interpolator takes clip-relative microseconds. We just convert and
        pass through — no anchoring.
        """
        try:
            return interp(int(round(t * 1_000_000)))
        except Exception:
            return None


__all__ = ["CausalAVDataset", "Sequence"]
