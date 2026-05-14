# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace I/O for CASCADE annotation JSON files.

Requires the optional ``hf`` extra (``pip install cascade_av[hf]``), which
pulls in ``physical_ai_av`` and uses its ``HfRepoInterface`` so caching,
offline mode, and multi-threaded downloads come for free.

Expected repo layout — the canonical CASCADE structure::

    <repo_root>/
    ├── data/
    │   ├── batch_NNNNN/
    │   │   ├── <annotation_uuid>__<clip_id>.json
    │   │   └── ...
    │   └── dataset_split.yaml      # optional — named, versioned splits
    ├── docs/                       # ignored
    ├── LICENSE                     # ignored
    └── README.md                   # ignored

``path_in_repo`` defaults to auto-detect: if the repo has a ``data/``
directory, use that; otherwise fall back to the repo root (legacy /
flat layouts). Pass ``path_in_repo=""`` to force the root.

``dataset_split.yaml``, if present, declares named versioned splits —
see ``available_splits()``, ``iter_split()``, and ``load_split()``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

try:
    from physical_ai_av.utils.hf_interface import HfRepoInterface
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "cascade_av.io.hf requires the `hf` extra. "
        "Install with: pip install 'cascade_av[hf]'"
    ) from e

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "cascade_av.io.hf requires PyYAML for the split-aware loaders. "
        "Reinstall the `hf` extra: pip install 'cascade_av[hf]'"
    ) from e

from cascade_av.io.local import load_file
from cascade_av.spec import AnnotationBundle

# Filename of the split manifest at the root of `path_in_repo`. The
# canonical CASCADE dataset writes this; legacy / flat repos without a
# manifest simply can't use the split-aware loaders.
_DATASET_SPLIT_FILENAME = "dataset_split.yaml"


class CausalAnnotationsHfRepo(HfRepoInterface):
    """A HuggingFace *dataset* repo of CASCADE annotation JSON files."""

    def __init__(
        self,
        repo_id: str,
        *,
        path_in_repo: str | None = None,
        revision: str | None = None,
        **hf_kwargs,
    ) -> None:
        super().__init__(
            repo_id=repo_id, repo_type="dataset", revision=revision, **hf_kwargs
        )
        # `None` (the new default) triggers auto-detection — prefer the
        # canonical `data/` layout when the repo has it, fall back to
        # root otherwise. Explicit `""` forces root; explicit non-empty
        # string forces that exact subdirectory. Callers that bypass
        # autodetect (e.g., to avoid the extra `list_repo_files` round
        # trip) can pass the value directly.
        if path_in_repo is None:
            path_in_repo = self._auto_detect_path_in_repo()
        self.path_in_repo = path_in_repo.rstrip("/")

    def _auto_detect_path_in_repo(self) -> str:
        """Prefer ``data/`` if the repo has it (canonical CASCADE layout);
        else fall back to root (flat / legacy layouts).

        Failures (offline, private repo, transient HF error) fall through
        to ``""`` so the caller still gets a working object — they can
        always pass ``path_in_repo`` explicitly if autodetect guesses
        wrong.
        """
        from huggingface_hub import HfApi  # lazy — only touched on autodetect

        try:
            files = HfApi().list_repo_files(
                self.repo_id, repo_type="dataset", revision=self.revision,
            )
        except Exception:  # noqa: BLE001 — any HF failure means "give up, use root"
            return ""
        if any(f.startswith("data/") for f in files):
            return "data"
        return ""

    # ------------------------------------------------------------------
    # All-bundles methods. These ignore `dataset_split.yaml`; see the
    # split-aware methods below for the train/val/test subset path.
    # ------------------------------------------------------------------

    def download_all(self) -> list[Path]:
        """Download every ``*.json`` under ``path_in_repo`` and return local paths."""
        results = self.download_repo_tree(self.path_in_repo or ".", recursive=True)
        return [
            Path(p) for p in results if isinstance(p, str) and p.endswith(".json")
        ]

    def iter_annotations(self) -> Iterator[AnnotationBundle]:
        """Yield every annotation as an ``AnnotationBundle`` (downloads on first pass)."""
        for p in self.download_all():
            yield load_file(p)

    def load_all(self) -> list[AnnotationBundle]:
        """Eager: download + parse every annotation in the repo."""
        return list(self.iter_annotations())

    def load_annotation(self, filename: str) -> AnnotationBundle:
        """Download (or use cached) one annotation file by its filename.

        ``filename`` is relative to ``path_in_repo`` — pass
        ``"batch_00001/<uuid>__<clip_id>.json"`` for the canonical layout.
        """
        repo_path = (
            f"{self.path_in_repo}/{filename}" if self.path_in_repo else filename
        )
        local = self.download_file(repo_path)
        if not isinstance(local, str):
            raise RuntimeError(
                f"Unexpected return type from download_file({repo_path!r}): {type(local)}"
            )
        return load_file(local)

    # ------------------------------------------------------------------
    # Split-aware methods. Backed by `dataset_split.yaml`, which is a
    # YAML list of split entries::
    #
    #     - name: cascade-v0.1
    #       date: 2026-05-14
    #       train: [batch_00001/foo.json, ...]
    #       validation: [batch_00001/bar.json, ...]
    #       test: [...]   # optional
    #
    # New splits are appended; existing ones are never modified, so a
    # named split is a stable, citeable handle.
    # ------------------------------------------------------------------

    def _load_split_manifest(self) -> list[dict]:
        """Download + parse ``dataset_split.yaml``; return the list of entries.

        Raises ``RuntimeError`` / ``ValueError`` on the file being absent
        or malformed — surfaces upstream so callers see the real failure
        rather than a silent empty result.
        """
        repo_path = (
            f"{self.path_in_repo}/{_DATASET_SPLIT_FILENAME}"
            if self.path_in_repo
            else _DATASET_SPLIT_FILENAME
        )
        local = self.download_file(repo_path)
        if not isinstance(local, str):
            raise RuntimeError(
                f"Unexpected return type from download_file({repo_path!r}): {type(local)}"
            )
        with open(local) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, list):
            raise ValueError(
                f"{_DATASET_SPLIT_FILENAME} must be a YAML list of split entries; "
                f"got {type(data).__name__}"
            )
        return data

    def available_splits(self) -> dict[str, list[str]]:
        """Map each named split version to the sub-splits it defines.

        Example: ``{"cascade-v0.1": ["train", "validation"]}``. Reads
        ``dataset_split.yaml`` from the repo (one network round-trip,
        cached by ``huggingface_hub`` afterward).
        """
        manifest = self._load_split_manifest()
        out: dict[str, list[str]] = {}
        for entry in manifest:
            if not isinstance(entry, dict) or "name" not in entry:
                continue
            name = entry["name"]
            sub_splits = [
                k for k, v in entry.items()
                if k not in ("name", "date") and isinstance(v, list)
            ]
            out[name] = sub_splits
        return out

    def iter_split(self, name: str, split: str) -> Iterator[AnnotationBundle]:
        """Yield bundles belonging to one sub-split.

        Args:
            name: the named version, e.g. ``"cascade-v0.1"``.
            split: a key under that version, e.g. ``"train"``,
                ``"validation"``, or ``"test"``.

        Raises:
            KeyError: ``name`` is absent from the manifest, or ``split``
                isn't a list-valued key under it.
        """
        manifest = self._load_split_manifest()
        entry = next(
            (e for e in manifest if isinstance(e, dict) and e.get("name") == name),
            None,
        )
        if entry is None:
            available = sorted(
                e.get("name", "?")
                for e in manifest if isinstance(e, dict) and "name" in e
            )
            raise KeyError(
                f"split set {name!r} not found in {_DATASET_SPLIT_FILENAME}; "
                f"available: {available}"
            )
        filenames = entry.get(split)
        if not isinstance(filenames, list):
            available = [
                k for k, v in entry.items()
                if k not in ("name", "date") and isinstance(v, list)
            ]
            raise KeyError(
                f"sub-split {split!r} not found in split set {name!r}; "
                f"available: {available}"
            )
        for filename in filenames:
            yield self.load_annotation(filename)

    def load_split(self, name: str, split: str) -> list[AnnotationBundle]:
        """Eager variant of ``iter_split`` — download + parse all bundles in a split."""
        return list(self.iter_split(name, split))


__all__ = ["CausalAnnotationsHfRepo"]
