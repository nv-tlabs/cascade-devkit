# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`CascadeDataset` + `Sequence` — the user-facing DevKit entry points.

`CascadeDataset` is a thin subclass of `physical_ai_av.PhysicalAIAVDatasetInterface`.
It extends the parent with:
- annotation I/O (local directory of `*.json`, or eventually a HF repo),
- per-clip `Sequence` objects bundling the annotation with the parent's
  egomotion / video features,
- point-in-time and windowed state queries (`Sequence.state_at`),
- `Sequence.visualize` — a polymorphic dispatcher into the
  `cascade_av.viz` package; returns a ``ClipPlayer`` widget for
  scrub-and-play, a ``PIL.Image`` for headless single-frame use, or a
  static multi-frame Plotly figure in ``paper_figure`` mode.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from physical_ai_av import PhysicalAIAVDatasetInterface

from cascade_av.io import load_file
from cascade_av.query import (
    Interval,
    MatchSet,
    agents_visible_at,
    ego_relative_pose_at,
    extract_causal_triplets,
    filter_active_at,
    filter_active_in_range,
)
from cascade_av.query.api import (
    count_on_dataset,
    find_on_bundle,
    find_on_dataset,
    group_by_on_dataset,
)
from cascade_av.query.context import ContextWindow, context_at
from cascade_av.query.engine import Match
from cascade_av.query.triplets import CausalTriplet
from cascade_av.spec import AnnotationBundle
from cascade_av.state import (
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
DEFAULT_ANNOTATION_CAMERA = "camera_front_wide_120fov"


# -----------------------------------------------------------------------------
# CascadeDataset
# -----------------------------------------------------------------------------

class CascadeDataset(PhysicalAIAVDatasetInterface):
    """DevKit for the CASCADE dataset.

    Extends `PhysicalAIAVDatasetInterface` with annotation I/O and per-clip
    state queries. Construct from a local directory of annotation JSONs:

        ds = CascadeDataset("/path/to/01_json_annotations")
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

        # Defer the parent `PhysicalAIAVDatasetInterface.__init__` — it does
        # gated-repo network I/O (downloads `clip_index.parquet` /
        # `feature_presence.parquet` from `nvidia/PhysicalAI-Autonomous-Vehicles`)
        # that the pure-annotation query API (`find` / `count` / `group_by`)
        # never needs. We initialize the parent lazily on the first call that
        # actually reads clip features or video — `get_sequence` /
        # `download_clips` — via `_ensure_parent`. This lets query-only
        # workflows (e.g. notebooks 01–04) construct the dataset with no HF
        # token and no network. See `_ensure_parent`.
        self._parent_init_kwargs: dict[str, object] = {
            "revision": revision,
            "token": token,
            "cache_dir": cache_dir,
            "local_dir": local_dir,
            "confirm_download_threshold_gb": confirm_download_threshold_gb,
        }
        self._parent_ready = False

        self.annotation_camera = annotation_camera
        # _by_clip[clip_id] = (path, batch_name_or_None, bundle)
        self._by_clip: dict[str, tuple[Path, str | None, AnnotationBundle]] = {}
        self._scan(annotation_paths, annotation_root)

    def _ensure_parent(self) -> None:
        """Initialize the parent HF dataset interface on first use.

        Idempotent: the underlying `super().__init__` (which contacts the
        gated parent repo for clip metadata) runs at most once. The query
        API never calls this; feature/video access (`get_sequence`,
        `download_clips`) does, so a token/network is only required when you
        actually reach for clip data. Re-raises the parent's errors verbatim
        (e.g. `GatedRepoError` when the HF token is missing or lacks access).
        """
        if self._parent_ready:
            return
        super().__init__(**self._parent_init_kwargs)
        self._parent_ready = True

    def __repr__(self) -> str:
        # The parent's `__repr__` reads `repo_snapshot_info`, which doesn't
        # exist until `_ensure_parent` runs. Provide a parent-free repr so
        # `repr(ds)` is safe (and informative) for query-only datasets.
        state = "parent-initialized" if self._parent_ready else "parent-deferred"
        return (
            f"CascadeDataset({len(self._by_clip)} clips, "
            f"annotation_camera={self.annotation_camera!r}, {state})"
        )

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
                "filename. (Set CASCADE_AV_VERBOSE=1 for per-file detail.)",
                stacklevel=2,
            )
            if os.environ.get("CASCADE_AV_VERBOSE"):
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
        # `Sequence` eagerly reads egomotion off the parent, so the parent
        # interface must be live before we build it.
        self._ensure_parent()
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

    def find(self, dsl: str, *, strict_identity: bool = False) -> MatchSet:
        """Run a DSL query over every clip in the dataset.

        Returns a single MatchSet whose entries reference entities
        across the corpus. Set ``strict_identity=True`` to exclude records
        with blank IDs. See `docs/user/query_language.md`.
        """
        return find_on_dataset(self, dsl, strict_identity=strict_identity)

    def count(self, dsl: str, *, strict_identity: bool = False) -> int:
        """Number of clips with ≥1 match for `dsl`.

        Set ``strict_identity=True`` to ignore match candidates with blank IDs.
        """
        return count_on_dataset(self, dsl, strict_identity=strict_identity)

    def group_by(
        self, dsl: str, key: str, *, strict_identity: bool = False
    ) -> dict[object, int]:
        """Run `dsl`, bucket matches by `key` (dotted path), return `{value: n_clips}`."""
        return group_by_on_dataset(self, dsl, key, strict_identity=strict_identity)

    def histogram(
        self, dsl: str, key: str, *, strict_identity: bool = False
    ) -> dict[object, int]:
        """Alias of `group_by` (kept for API symmetry per spec §4.3)."""
        return group_by_on_dataset(self, dsl, key, strict_identity=strict_identity)

    DEFAULT_DOWNLOAD_FEATURES: ClassVar[tuple[str, ...]] = (
        "egomotion",
    )
    """Always pulled in addition to the annotation camera — `get_sequence`
    reads egomotion in its constructor."""

    def download_clips(
        self,
        clip_ids: Iterable[str] | None = None,
        *,
        features: Iterable[str] | None = None,
    ) -> None:
        """Prefetch clip features from HuggingFace.

        `clip_ids` defaults to every clip in the dataset; `features`
        defaults to the canonical annotation camera plus the egomotion
        feature — the minimum required for `get_sequence` to succeed.
        Pass an explicit `features` list to widen (e.g. extra cameras
        or LiDAR).

        Delegates to the parent `download_clip_features`, which
        downloads in parallel via huggingface_hub.
        """
        ids = list(clip_ids) if clip_ids is not None else self.list_sequences()
        unknown = [c for c in ids if c not in self._by_clip]
        if unknown:
            shown = ", ".join(unknown[:3])
            more = f" (+{len(unknown) - 3} more)" if len(unknown) > 3 else ""
            raise KeyError(f"clip_ids not in dataset: {shown}{more}")
        if features is None:
            feats = [self.annotation_camera, *self.DEFAULT_DOWNLOAD_FEATURES]
        else:
            feats = list(features)
        # Downloading reads the parent repo's feature index — bring it up now.
        self._ensure_parent()
        self.download_clip_features(ids, features=feats)

    def context_for(self, match: Match) -> ContextWindow:
        """Everything else annotated in the clip during `match`'s window.

        Resolves `match.clip_id` to its bundle and returns a
        `ContextWindow` of visible agents, ego actions, environments,
        conditions, traffic-light states, and persistent traffic
        objects that overlap `match.interval`. Whole-clip matches
        (`match.interval is None`) widen to the full clip span.
        """
        if match.clip_id not in self._by_clip:
            raise KeyError(f"clip_id not in dataset: {match.clip_id!r}")
        _path, _batch, bundle = self._by_clip[match.clip_id]
        if match.interval is None:
            window = Interval(0.0, max(0.0, bundle.video.duration_s))
        else:
            window = match.interval
        return context_at(bundle, window)


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

    Constructed via `CascadeDataset.get_sequence`. For test or notebook use
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

        # Eagerly load egomotion *if* a parent is available.
        # `get_clip_feature(clip_id, "egomotion")` already returns an
        # `Interpolator[EgomotionState]` (see physical_ai_av/dataset.py),
        # so we cache it directly.
        #
        # The egomotion chunk lives in the parent HF dataset and isn't
        # always cached locally — `ds.download_clips(...)` pulls per-clip
        # video, not per-chunk egomotion. If the cache is incomplete, the
        # parent's `open_file` raises `FileNotFoundError`. Swallow that
        # and leave the slot as `None`; consumers that actually need
        # egomotion raise a clear error via their own guards. The viz
        # API path never reads egomotion, so a carousel over
        # cache-incomplete clips Just Works.
        if parent is not None:
            try:
                self._egomotion_interpolator = parent.get_clip_feature(
                    self.clip_id, "egomotion"
                )
            except FileNotFoundError:
                self._egomotion_interpolator = None

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
                "Construct via `CascadeDataset.get_sequence` instead of "
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
        """The eagerly-loaded `Interpolator[EgomotionState]` from the parent.

        Raises `RuntimeError` with actionable guidance when the egomotion
        chunk wasn't cached locally (the eager load in `__init__` caught
        `FileNotFoundError` and left this slot as `None`). The viz API
        never reaches this property, so cache-incomplete clips can still
        be rendered; only direct consumers of ego-relative quantities
        hit the guard.
        """
        if self._egomotion_interpolator is not None:
            return self._egomotion_interpolator
        # Either no parent was wired in (`Sequence.from_annotation` path —
        # `_require_parent` raises the canonical "no parent" error), or
        # `__init__` swallowed a `FileNotFoundError` on the eager load.
        # Re-attempt the fetch so a cache that was populated between
        # construction and access just works; rewrap any remaining
        # `FileNotFoundError` as a clear, actionable `RuntimeError`.
        parent = self._require_parent()
        try:
            self._egomotion_interpolator = parent.get_clip_feature(
                self.clip_id, "egomotion"
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"egomotion data not loaded for clip {self.clip_id!r}: "
                "the egomotion chunk isn't in the parent HF cache. Call "
                f"`dataset.download_clips([{self.clip_id!r}])` (or otherwise "
                "ensure the egomotion chunk is cached) before requesting "
                "ego-relative quantities. The viz API does not require "
                "egomotion."
            ) from exc
        return self._egomotion_interpolator

    # -- queries --------------------------------------------------------------

    def triplets(self) -> list[CausalTriplet]:
        """All causal triplets extracted from this clip (memoized)."""
        if self._triplets is None:
            self._triplets = extract_causal_triplets(self.annotation)
        return self._triplets

    def find(self, dsl: str, *, strict_identity: bool = False) -> MatchSet:
        """Run a DSL query against this clip's annotation bundle.

        If this Sequence was constructed via `CascadeDataset.get_sequence`,
        the returned MatchSet carries a weakref to the parent dataset so
        follow-up operations like `matches.visualize()` can find it.
        Standalone Sequences (constructed via `from_annotation`) return a
        MatchSet whose `.dataset` is None. Set ``strict_identity=True`` to
        exclude records with blank IDs.
        """
        return find_on_bundle(
            self.annotation,
            dsl,
            dataset=self._parent,
            strict_identity=strict_identity,
        )

    def state_at(
        self, t: float, t_end: float | None = None
    ) -> SequenceState | SequenceStateRange:
        """Snapshot the scene at `t` (instant) or over `[t, t_end]` (window)."""
        if t_end is None:
            return self._state_at_instant(t)
        return self._state_over_range(t, t_end)

    # -- visualize ------------------------------------------------------------

    def visualize(
        self,
        t: float | tuple[float, float] | None = None,
        *,
        match: Match | None = None,
        context: ContextWindow | None = None,
        pad: float = 1.0,
        static: bool = False,
        mode: Literal["auto", "paper_figure"] = "auto",
        timestamps: list[float] | tuple[float, ...] | None = None,
        fps: float = 8.0,
        highlight: tuple[float, float] | None = None,
        arrows: dict[str, bool] | None = None,
        entity_kinds: list[str] | None = None,
        agent_ids: list[str] | None = None,
        track_groups: list[str] | None = None,
        families: list[str] | None = None,
        track_visibility: Mapping[
            str, bool | Mapping[str, bool]
        ] | None = None,
        height: int | None = None,
        width: int | None = None,
        show_inline_labels: bool = True,
        show_containments: bool | None = None,
    ) -> Any:
        """Visualize this clip's video + timeline.

        Dispatches based on which argument is set:

        - ``seq.visualize()`` → ``ClipPlayer`` over the full clip.
        - ``seq.visualize(t=2.5)`` → ``ClipPlayer`` centered on ``t=2.5`` s
          (±1 s window, clamped to the clip bounds).
        - ``seq.visualize(t=2.5, static=True)`` → ``PIL.Image`` of that
          single frame, decoded headlessly via ``viz.render_frame``.
        - ``seq.visualize(t=(2.0, 5.0))`` → ``ClipPlayer`` over the
          explicit window.
        - ``seq.visualize(match=m)`` → ``ClipPlayer`` over
          ``m.interval ± pad`` with ``m.interval`` painted as the yellow
          highlight band. The padded window is clamped to
          ``[0, duration_s]``.
        - ``seq.visualize(context=cw)`` → ``ClipPlayer`` over
          ``cw.interval``; the context window's interval is also
          painted as the highlight when no explicit highlight is passed.
        - ``seq.visualize(mode="paper_figure", timestamps=[1, 3, 5])``
          → a static Plotly figure with those frames above the timeline.

        Exactly one of ``t`` / ``match`` / ``context`` may be set
        (passing more than one raises ``ValueError``).

        Args:
            t: scalar timestamp to scrub to, tuple ``(t0, t1)`` window,
                or ``None`` for the whole clip.
            match: a ``Match`` whose ``interval`` selects the playback
                window (``pad`` widens it). ``clip_id`` must match this
                sequence.
            context: a ``ContextWindow``. Its ``interval`` selects the
                playback window. ``clip_id`` must match this sequence.
            pad: seconds of padding applied on either side of
                ``match.interval``. Ignored when ``match`` is ``None``.
            static: when ``True`` and ``t`` is a scalar, return a
                ``PIL.Image`` instead of building a ``ClipPlayer``.
                Combining ``static=True`` with a tuple or with ``t=None``
                raises ``ValueError``.
            mode: ``"auto"`` preserves the existing frame/player dispatch.
                ``"paper_figure"`` returns a static multi-frame Plotly
                figure and cannot be combined with ``t``, ``match``,
                ``context``, or ``static=True``.
            timestamps: zero to three explicit video timestamps for
                ``mode="paper_figure"``. Frames render in ascending
                chronological order from left to right regardless of input
                order; duplicate timestamps remain separate. Values must be
                non-negative and within the video's actual timestamp coverage;
                invalid values raise instead of being silently clamped.
            fps: forwarded to ``ClipPlayer`` (scrub rate, frames /
                second). Ignored in the static path.
            highlight: explicit highlight band passed through to
                ``ClipPlayer``. When ``match`` or ``context`` is set,
                the interval that came in alongside the match takes
                priority — matching the carousel's behavior.
            arrows: per-family arrow-on/off toggles forwarded to
                ``ClipPlayer``.
            entity_kinds: optional entity-kind whitelist forwarded to the
                timeline painter.
            agent_ids: optional backward-compatible Agent ID whitelist.
            track_groups: optional category whitelist.
            families: optional whitelist of family leaves to render
                (``"condition"``, ``"containment"``, ``"state"``,
                ``"pose"``, ``"action"``,
                ``"property"``, ``"signal_head"``, ``"env_control"``,
                ``"physical_containment"``). The removed ``"influence"``
                leaf remains accepted as a compatibility no-op. Parent entity
                headers auto-render for any entity whose sub-rows survive;
                entities with no surviving sub-rows drop completely.
                ``None`` = all families. Ignored in the static path.
            track_visibility: grouped whole-kind or stable per-entity
                switches. For example,
                ``{"agent": {"agent_4": False, "agent_5": False}}``
                hides those Agents and their child rows. Omitted switches
                remain visible.
            height: optional total figure/player height in pixels.
            width: optional total figure/player width in pixels.
            show_inline_labels: whether timeline bars include inline text.
                Paper figures preserve complete labels on every visible box;
                other timeline views retain their compact label policy.
            show_containments: paper-mode switch for containment-family rows
                and their connected arrows. ``None`` uses the paper default
                (False); pass True to restore them. Invalid outside paper mode.

        Returns:
            ``PIL.Image.Image`` when ``static=True`` and ``t`` is a scalar;
            a plain ``plotly.graph_objects.Figure`` in ``paper_figure``
            mode; otherwise ``cascade_av.viz.ClipPlayer``.

        Raises:
            ValueError: if more than one of ``t`` / ``match`` /
                ``context`` is set; if ``static=True`` is paired with a
                non-scalar ``t``; if ``match`` / ``context`` refer to a
                different ``clip_id``; or if paper-mode arguments are
                invalid.
            ImportError: if the optional ``[viz]`` extra is not
                installed (Pillow / Plotly / ipywidgets).
        """
        if mode not in {"auto", "paper_figure"}:
            raise ValueError(
                "mode must be 'auto' or 'paper_figure'; "
                f"got {mode!r}"
            )

        # 1. Mutual exclusion — at most one of t / match / context.
        provided = sum(arg is not None for arg in (t, match, context))
        if provided > 1:
            raise ValueError(
                "visualize() takes at most one of `t`, `match`, `context`; "
                "got multiple."
            )

        # 2. clip_id consistency for match / context.
        if match is not None and match.clip_id != self.clip_id:
            raise ValueError(
                f"match is from a different clip "
                f"(match.clip_id={match.clip_id!r}, sequence.clip_id={self.clip_id!r})"
            )
        if context is not None and context.clip_id != self.clip_id:
            raise ValueError(
                f"context is from a different clip "
                f"(context.clip_id={context.clip_id!r}, "
                f"sequence.clip_id={self.clip_id!r})"
            )

        if mode == "paper_figure" and provided:
            raise ValueError(
                "mode='paper_figure' uses `timestamps`; it cannot be "
                "combined with `t`, `match`, or `context`"
            )
        if mode == "paper_figure" and static:
            raise ValueError(
                "mode='paper_figure' cannot be combined with static=True"
            )
        if mode == "auto" and timestamps is not None:
            raise ValueError(
                "`timestamps` is only valid with mode='paper_figure'"
            )
        if mode == "auto" and show_containments is not None:
            raise ValueError(
                "`show_containments` is only valid with "
                "mode='paper_figure'"
            )

        # 3. static=True only makes sense for a scalar `t`.
        scalar_t = isinstance(t, (int, float)) and not isinstance(t, bool)
        if static and not scalar_t:
            raise ValueError(
                f"static=True requires a scalar `t` (single-frame render); "
                f"got t={t!r}"
            )

        # 4. Lazy import — viz is an optional extra, and the dataset
        # module must remain importable without Pillow / Plotly /
        # ipywidgets on the path. Failures surface at call time only.
        try:
            from cascade_av.viz import (
                ClipPlayer,
                render_frame,
                render_paper_figure,
            )
        except ImportError as exc:
            raise ImportError(
                "Sequence.visualize() requires the 'viz' extra. Install with: "
                "pip install 'cascade-av[viz]'"
            ) from exc

        if mode == "paper_figure":
            return render_paper_figure(
                self,
                timestamps=() if timestamps is None else timestamps,
                highlight=highlight,
                arrows=arrows,
                entity_kinds=entity_kinds,
                agent_ids=agent_ids,
                track_groups=track_groups,
                families=families,
                track_visibility=track_visibility,
                height=height,
                width=1400 if width is None else width,
                show_inline_labels=show_inline_labels,
                show_containments=(
                    False if show_containments is None else show_containments
                ),
            )

        duration = float(self.duration_s) if self.duration_s else 0.0

        # 5. Resolve the playback window + highlight from the dispatch matrix.
        t_start: float = 0.0
        t_end: float = duration
        resolved_highlight: tuple[float, float] | None = highlight

        if scalar_t:
            assert isinstance(t, (int, float))
            t_val = float(t)
            if static:
                # Headless single-frame path bypasses ClipPlayer.
                return render_frame(self, t_val)
            t_start = max(0.0, t_val - 1.0)
            t_end = min(duration, t_val + 1.0)
        elif isinstance(t, tuple):
            t0, t1 = t
            t_start = float(t0)
            t_end = float(t1)
        elif match is not None:
            if match.interval is not None:
                m0, m1 = float(match.interval.start), float(match.interval.end)
                t_start = max(0.0, m0 - float(pad))
                t_end = min(duration, m1 + float(pad))
                resolved_highlight = (m0, m1)
            else:
                # Whole-clip match — fall through to the full clip span.
                t_start, t_end = 0.0, duration
        elif context is not None:
            iv = context.interval
            t_start = float(iv.start)
            t_end = float(iv.end)
            if resolved_highlight is None:
                resolved_highlight = (float(iv.start), float(iv.end))

        # 6. Clamp the resolved window to the video's actual timestamp
        # range. The parent dataset's video manifests rarely start at
        # exactly t=0us (real clips have non-zero first-frame offsets),
        # so a window of `[0, duration_s]` will trip
        # `SeekVideoReader.decode_images_from_timestamps([0])` with
        # `ValueError: requested timestamps must be within the range
        # of timestamps`. We pull `seq.video.timestamps` (np.int64 μs)
        # and clamp `t_start` / `t_end` to the actual covered span.
        # If the video isn't reachable yet (no parent dataset wired in,
        # cache miss, etc.) we fall through with the raw window — the
        # eventual decode error is more informative than us papering
        # over an unloaded reader here.
        try:
            ts = self.video.timestamps
            video_t_min = float(ts.min()) / 1e6
            video_t_max = float(ts.max()) / 1e6
        except Exception:
            video_t_min, video_t_max = 0.0, duration

        # If the requested window sits entirely outside the video's
        # coverage, snapping silently to a 1-frame sliver would hide
        # the misconfiguration. Surface it instead.
        if t_end < video_t_min or t_start > video_t_max:
            raise ValueError(
                f"requested window [{t_start}, {t_end}] is outside the "
                f"video's timestamp coverage [{video_t_min}, {video_t_max}]"
            )

        t_start = max(t_start, video_t_min)
        t_end = min(t_end, video_t_max)

        return ClipPlayer(
            self,
            t_start=t_start,
            t_end=t_end,
            fps=fps,
            highlight=resolved_highlight,
            arrows=arrows,
            entity_kinds=entity_kinds,
            agent_ids=agent_ids,
            track_groups=track_groups,
            families=families,
            track_visibility=track_visibility,
            height=height,
            width=width,
            show_inline_labels=show_inline_labels,
        )

    # -- state_at implementation ---------------------------------------------

    def _state_at_instant(self, t: float) -> SequenceState:
        ann = self.annotation.annotation

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
            actors.append(ActorState(
                agent=agent,
                actions=agent_actions,
                pose_rel_to_ego=pose,
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
            actors.append(ActorStateRange(
                agent=agent,
                actions=actions,
                pose_rel_to_ego=poses,
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


__all__ = ["CascadeDataset", "Sequence"]
