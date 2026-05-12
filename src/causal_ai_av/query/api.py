# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin wrappers for `find` / `count` / `group_by` / `histogram`.

`Sequence.find` and `CausalAVDataset.find` delegate here. The
corpus-scope aggregations are clip-level: `count` returns the number of
*clips* with ≥1 match (not the number of match tuples).
"""

from __future__ import annotations

import weakref
from typing import Any, TYPE_CHECKING

from causal_ai_av.query.dsl import parse
from causal_ai_av.query.engine import Match, MatchSet, evaluate
from causal_ai_av.query.entities import ENTITIES
from causal_ai_av.spec import AnnotationBundle

if TYPE_CHECKING:
    from causal_ai_av.dataset import CausalAVDataset


def find_on_bundle(
    bundle: AnnotationBundle,
    dsl_text: str,
    *,
    dataset: Any | None = None,
) -> MatchSet:
    """Parse `dsl_text` and evaluate it against a single bundle.

    If `dataset` is provided (typically the parent dataset of the bundle's
    Sequence), the returned MatchSet carries a weakref back to it so
    follow-up operations like `matches.visualize()` or `matches.context()`
    can find the source. Without `dataset=`, `MatchSet.dataset` returns
    None.
    """
    expr = parse(dsl_text)
    ms = evaluate(expr, bundle)
    if dataset is None:
        return ms
    return MatchSet(ms.matches, _dataset=weakref.ref(dataset))


def find_on_dataset(dataset: "CausalAVDataset", dsl_text: str) -> MatchSet:
    """Run the query across every clip in a dataset; return a union MatchSet.

    The returned MatchSet carries a weakref to `dataset` so follow-up
    operations can find the source. Access via `MatchSet.dataset`.
    """
    expr = parse(dsl_text)
    all_matches: list[Match] = []
    for clip_id, (_path, _batch, bundle) in dataset._by_clip.items():
        ms = evaluate(expr, bundle)
        all_matches.extend(ms.matches)
    return MatchSet(tuple(all_matches), _dataset=weakref.ref(dataset))


def count_on_dataset(dataset: "CausalAVDataset", dsl_text: str) -> int:
    """Number of clips with ≥1 match for `dsl_text`."""
    expr = parse(dsl_text)
    n = 0
    for _clip_id, (_path, _batch, bundle) in dataset._by_clip.items():
        ms = evaluate(expr, bundle)
        if ms:
            n += 1
    return n


def group_by_on_dataset(
    dataset: "CausalAVDataset", dsl_text: str, key: str
) -> dict[object, int]:
    """Run `dsl_text` and bucket clips by the value of `key` on each
    matched entity. Returns `{value: count_of_clips}`.

    `key` is a dotted path resolved the same way attribute predicates
    are. Example: `"agent.type"`, `"ego.action.type"`.
    """
    key_path = tuple(s.strip().lower() for s in key.split("."))
    if not key_path:
        return {}
    root = key_path[0]
    rest = key_path[1:]
    if root not in ENTITIES:
        raise KeyError(f"unknown entity in key {key!r}: {root!r}")

    expr = parse(dsl_text)
    buckets: dict[object, set[str]] = {}

    for _clip_id, (_path, _batch, bundle) in dataset._by_clip.items():
        ms = evaluate(expr, bundle)
        if not ms:
            continue
        clip_id = bundle.video.clip_id
        for m in ms.matches:
            value = _read_key(m.entity, root, rest, bundle)
            if value is None:
                continue
            if isinstance(value, list):
                for v in value:
                    buckets.setdefault(v, set()).add(clip_id)
            else:
                buckets.setdefault(value, set()).add(clip_id)
    return {k: len(v) for k, v in buckets.items()}


def _read_key(
    entity: Any, root: str, rest: tuple[str, ...], bundle: AnnotationBundle
) -> object:
    """Read the dotted key off a Match.entity. The match entity may be
    the entity itself (e.g. an `Agent` for `agent.type`), a sub-entity
    (e.g. an `EgoAction` for `ego.action.type`), or a tuple (from
    `while`/`then`/`because_of`).
    """
    if isinstance(entity, tuple) and entity:
        entity = entity[0]

    desc = ENTITIES.get(root)
    if desc is None:
        return None

    cur_desc = desc
    cur = entity
    for seg in rest:
        attr = cur_desc.attributes.get(seg)
        if attr is None:
            return None
        if attr.kind == "sub_entity":
            sub_desc = ENTITIES.get(attr.sub_entity or seg)
            if sub_desc is None:
                return None
            # If the matched entity is already at this sub-entity level
            # (e.g. a Match holds an EgoAction and the key descends through
            # `ego.action`), the parent reader will fail; just stay on cur.
            try:
                subs = attr.reader(cur, bundle)
            except (AttributeError, TypeError):
                cur_desc = sub_desc
                continue
            if not subs:
                return None
            cur_desc = sub_desc
            cur = subs[0]
            continue
        try:
            return attr.reader(cur, bundle)
        except (AttributeError, TypeError):
            return None
    return None


__all__ = [
    "find_on_bundle",
    "find_on_dataset",
    "count_on_dataset",
    "group_by_on_dataset",
]
