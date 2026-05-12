# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace I/O for AV Causal annotation JSON files.

Requires the optional `hf` extra (`pip install causal_ai_av[hf]`), which pulls
in `physical_ai_av` and uses its `HfRepoInterface` so caching, offline mode,
and multi-threaded downloads come for free.

Expected repo layout: a flat directory at `path_in_repo` (default: repo root)
containing one `<annotation_uuid>__<clip_id>.json` per annotation.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

try:
    from physical_ai_av.utils.hf_interface import HfRepoInterface
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "causal_ai_av.io.hf requires the `hf` extra. "
        "Install with: pip install 'causal_ai_av[hf]'"
    ) from e

from causal_ai_av.io.local import load_file
from causal_ai_av.spec import AnnotationBundle


class CausalAnnotationsHfRepo(HfRepoInterface):
    """A HuggingFace *dataset* repo of AV Causal annotation JSON files."""

    def __init__(
        self,
        repo_id: str,
        *,
        path_in_repo: str = "",
        revision: str | None = None,
        **hf_kwargs,
    ) -> None:
        super().__init__(
            repo_id=repo_id, repo_type="dataset", revision=revision, **hf_kwargs
        )
        self.path_in_repo = path_in_repo.rstrip("/")

    def download_all(self) -> list[Path]:
        """Download every `*.json` under `path_in_repo` and return local paths."""
        results = self.download_repo_tree(self.path_in_repo or ".", recursive=True)
        return [
            Path(p) for p in results if isinstance(p, str) and p.endswith(".json")
        ]

    def iter_annotations(self) -> Iterator[AnnotationBundle]:
        """Yield every annotation as an `AnnotationBundle` (downloads on first pass)."""
        for p in self.download_all():
            yield load_file(p)

    def load_all(self) -> list[AnnotationBundle]:
        """Eager: download + parse every annotation in the repo."""
        return list(self.iter_annotations())

    def load_annotation(self, filename: str) -> AnnotationBundle:
        """Download (or use cached) one annotation file by its filename."""
        repo_path = (
            f"{self.path_in_repo}/{filename}" if self.path_in_repo else filename
        )
        local = self.download_file(repo_path)
        if not isinstance(local, str):
            raise RuntimeError(
                f"Unexpected return type from download_file({repo_path!r}): {type(local)}"
            )
        return load_file(local)
