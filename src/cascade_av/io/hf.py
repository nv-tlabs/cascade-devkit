# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace I/O for CASCADE annotation JSON files.

Requires the optional ``hf`` extra (``pip install cascade_av[hf]``), which
pulls in ``physical_ai_av`` and uses its ``HfRepoInterface`` so caching,
offline mode, and multi-threaded downloads come for free.

Expected repo layout — the canonical CASCADE structure::

    <repo_root>/
    ├── data/
    │   └── batch_NNNNN/
    │       ├── <annotation_uuid>__<clip_id>.json
    │       └── ...
    ├── tasks/                      # task-owned artifacts and split manifests
    ├── docs/                       # ignored
    ├── LICENSE                     # ignored
    └── README.md                   # ignored

``path_in_repo`` defaults to auto-detect: if the repo has a ``data/``
directory, use that; otherwise fall back to the repo root (legacy /
flat layouts). Pass ``path_in_repo=""`` to force the root.

This adapter intentionally loads the complete annotation corpus. Dataset
splits belong to individual tasks under ``tasks/<task>/`` and are consumed by
the corresponding task tooling, not by the base annotation loader.
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

from cascade_av.io.local import load_file
from cascade_av.spec import AnnotationBundle


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


__all__ = ["CausalAnnotationsHfRepo"]
