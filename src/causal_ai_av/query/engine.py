# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AST evaluator.

Walks an `Expr` (from `dsl.parse`) against an `AnnotationBundle` and
returns a `MatchSet`. Semantics per spec §3.

Match tuples are `(clip_id, entity, interval)`. Boolean and temporal
nodes operate over the union; `not` checks emptiness within the current
window; `because_of` walks the schema's `because_of` edge via `IdIndex`.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

from causal_ai_av.query.constants import ALIAS_FAMILIES, resolve_alias
from causal_ai_av.query.dsl import (
    And,
    AttrPredicate,
    BecauseOf,
    EntityClause,
    Expr,
    Not,
    Or,
    Then,
    While,
    Within,
)
from causal_ai_av.query.entities import (
    ENTITIES,
    EntityDescriptor,
    action_type_matches,
)
from causal_ai_av.query.index import IdIndex
from causal_ai_av.query.time import Interval
from causal_ai_av.spec import AgentAction, AnnotationBundle, EgoAction

if TYPE_CHECKING:  # pragma: no cover — typing only
    # Imported lazily for the `sequences()` return annotation; a runtime
    # import here would create a cycle (`dataset.py` already imports from
    # `causal_ai_av.query`). The runtime resolution happens dynamically
    # inside `MatchSet.sequences` via `self.dataset.get_sequence(...)`.
    from causal_ai_av.dataset import Sequence as _DatasetSequence


# ---------------------------------------------------------------------------
# Match types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    clip_id: str
    entity: object
    interval: Interval | None


@dataclass(frozen=True)
class MatchSet:
    matches: tuple[Match, ...]
    # Weakref back to the dataset (or other producer) that built this
    # MatchSet. Populated by the user-facing entry points
    # (`find_on_dataset`, `find_on_bundle` when called with `dataset=…`,
    # `Sequence.find`, `CausalAVDataset.find`); left None for intermediate
    # MatchSets created during boolean composition inside `evaluate`. Use
    # the `dataset` property to resolve; it raises if the referent has
    # been garbage-collected and returns None if never set.
    _dataset: "weakref.ReferenceType[Any] | None" = field(
        default=None, compare=False, repr=False
    )

    @property
    def dataset(self) -> Any | None:
        """Resolve the back-reference to the producing dataset.

        Returns `None` if this MatchSet was created without a dataset
        reference (e.g. via `find_on_bundle` with no `dataset=` kwarg).
        Raises `RuntimeError` if the dataset existed at construction
        time but has since been garbage-collected.
        """
        if self._dataset is None:
            return None
        obj = self._dataset()
        if obj is None:
            raise RuntimeError(
                "MatchSet's source dataset is no longer alive — the weakref "
                "expired. Keep the dataset bound in scope while you operate "
                "on the MatchSet."
            )
        return obj

    def clips(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for m in self.matches:
            if m.clip_id not in seen:
                seen.add(m.clip_id)
                out.append(m.clip_id)
        return out

    def entities(self) -> list[object]:
        return [m.entity for m in self.matches]

    def intervals(self) -> list[Interval]:
        return [m.interval for m in self.matches if m.interval is not None]

    def sequences(self) -> Iterator[tuple[Match, "_DatasetSequence"]]:
        """Yield ``(match, Sequence)`` pairs, one per match.

        The ``Sequence`` is constructed (or fetched) from the originating
        dataset; identical ``clip_id``s share a single ``Sequence``
        instance across yields (dedup-by-clip-id within a single call).

        Raises ``RuntimeError`` if this ``MatchSet`` has no dataset
        back-reference (e.g., produced by ``find_on_bundle`` without the
        ``dataset=`` kwarg). The wording mirrors the message used by the
        ``.dataset`` property when the weakref has expired.
        """
        if self._dataset is None:
            raise RuntimeError(
                "MatchSet has no dataset back-reference; cannot resolve "
                "Sequence objects. Use find_on_dataset(...) or "
                "find_on_bundle(..., dataset=ds) to attach one."
            )
        # `self.dataset` will raise its own RuntimeError if the weakref
        # has expired; we let that propagate.
        ds = self.dataset
        cache: dict[str, Any] = {}
        for m in self.matches:
            seq = cache.get(m.clip_id)
            if seq is None:
                seq = ds.get_sequence(m.clip_id)
                cache[m.clip_id] = seq
            yield m, seq

    def visualize(self, **kwargs: Any) -> Any:
        """Build a carousel of `ClipPlayer` widgets — one per match.

        See `causal_ai_av.viz.carousel.build_matchset_carousel` for the
        full keyword reference (`layout`, `cols`, `limit`, `pad`, `fps`,
        `arrows`). The return type is intentionally `Any` — the viz
        extra (`ipywidgets`, `plotly`) is optional, and an annotated
        return would force importing it at module top here.

        Raises:
            RuntimeError: if this `MatchSet` has no dataset
                back-reference. Bundle-level match sets (produced by
                `find_on_bundle` without `dataset=`) can't be
                visualized because there's no way to resolve a clip_id
                to a video reader.
        """
        # Lazy import: `viz` pulls `dataset`, and `dataset` already
        # imports from `causal_ai_av.query`. A top-level import here
        # would close the cycle. The viz extra is also optional — keep
        # the import inside the call so plain `from causal_ai_av.query
        # import MatchSet` doesn't require ipywidgets / plotly.
        from causal_ai_av.viz.carousel import build_matchset_carousel  # noqa: PLC0415

        return build_matchset_carousel(self, **kwargs)

    def __bool__(self) -> bool:
        return len(self.matches) > 0

    def __len__(self) -> int:
        return len(self.matches)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def _clip_span(bundle: AnnotationBundle) -> Interval:
    return Interval(0.0, max(0.0, bundle.video.duration_s))


def _intersect(a: Interval | None, b: Interval | None) -> Interval | None:
    if a is None or b is None:
        return a or b
    if not a.overlaps(b):
        return None
    return Interval(max(a.start, b.start), min(a.end, b.end))


def _intersect_window(
    iv: Interval | None, window: Sequence[Interval] | None
) -> Interval | None:
    """Intersect `iv` with the union of `window` intervals (closest single piece).

    Returns the first non-empty intersection encountered, or None.
    """
    if window is None:
        return iv
    if iv is None:
        return None
    for w in window:
        x = _intersect(iv, w)
        if x is not None:
            return x
    return None


def _values_for(atom: object, family: str | None) -> frozenset[Any]:
    """Resolve a single value atom to a set of schema strings, using the
    alias family if available. Non-strings (bool, int, float) pass through.
    """
    if isinstance(atom, bool) or not isinstance(atom, str):
        return frozenset({atom})
    if family is None or family not in ALIAS_FAMILIES:
        return frozenset({atom})
    try:
        return resolve_alias(family, atom)
    except KeyError:
        return frozenset({atom})


def _flatten_values(value: object, family: str | None) -> frozenset[Any]:
    """If `value` is a frozenset (from `in (a, b, c)`), expand each atom;
    otherwise expand the single atom.
    """
    if isinstance(value, frozenset):
        out: set[Any] = set()
        for v in value:
            out |= _values_for(v, family)
        return frozenset(out)
    return _values_for(value, family)


def _compare(left: object, op: str, right_set: frozenset[Any], right_raw: object) -> bool:
    """Apply comparison `op` between `left` and `right_set`/`right_raw`."""
    if op in ("=", "in"):
        return left in right_set
    if op == "!=":
        return left not in right_set
    # Numeric ops
    try:
        lv = float(left)  # type: ignore[arg-type]
        rv = float(right_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if op == ">":
        return lv > rv
    if op == ">=":
        return lv >= rv
    if op == "<":
        return lv < rv
    if op == "<=":
        return lv <= rv
    return False


def evaluate(expr: Expr, bundle: AnnotationBundle) -> MatchSet:
    """Top-level entry: evaluate `expr` against one bundle. Returns a
    MatchSet whose entries reference entities inside this bundle only.
    """
    ctx = _Context(bundle=bundle, window=None, id_index=IdIndex(bundle))
    return _eval(expr, ctx)


@dataclass(frozen=True)
class _Context:
    bundle: AnnotationBundle
    window: tuple[Interval, ...] | None
    id_index: IdIndex


def _eval(expr: Expr, ctx: _Context) -> MatchSet:
    if isinstance(expr, AttrPredicate):
        return _eval_attr(expr, ctx)
    if isinstance(expr, EntityClause):
        return _eval_entity(expr, ctx)
    if isinstance(expr, And):
        return _eval_and(expr, ctx)
    if isinstance(expr, Or):
        return _eval_or(expr, ctx)
    if isinstance(expr, Not):
        return _eval_not(expr, ctx)
    if isinstance(expr, While):
        return _eval_while(expr, ctx)
    if isinstance(expr, Then):
        return _eval_then(expr, ctx)
    if isinstance(expr, BecauseOf):
        return _eval_because_of(expr, ctx)
    if isinstance(expr, Within):
        return _eval_within(expr, ctx)
    raise TypeError(f"unknown AST node: {type(expr).__name__}")


# ---------------------------------------------------------------------------
# Attribute / entity-clause evaluation
# ---------------------------------------------------------------------------


def _entity_chain_for_predicate(
    pred: AttrPredicate, ctx: _Context
) -> tuple[EntityDescriptor, list[Any]]:
    """Resolve the candidate list for an attribute predicate.

    Walks `pred.path` until the *last* segment, descending into
    sub-entities along the way. Returns the final descriptor + list of
    candidate entities.
    """
    desc = ENTITIES[pred.entity]
    candidates = desc.candidates(ctx.bundle)
    if not pred.path:
        return desc, candidates

    # Descend through sub-entity segments (all but the last segment of path).
    for i, seg in enumerate(pred.path[:-1]):
        attr = desc.attributes[seg]
        if attr.kind != "sub_entity":
            # path bottomed out — last segment is supposed to be a scalar.
            break
        sub_desc = ENTITIES[attr.sub_entity or seg]
        new_cands: list[Any] = []
        for c in candidates:
            new_cands.extend(attr.reader(c, ctx.bundle))
        candidates = new_cands
        desc = sub_desc

    return desc, candidates


def _eval_attr(pred: AttrPredicate, ctx: _Context) -> MatchSet:
    desc, candidates = _entity_chain_for_predicate(pred, ctx)

    if pred.path:
        leaf_name = pred.path[-1]
        attr = desc.attributes.get(leaf_name)
        if attr is None:
            raise KeyError(f"no attribute {leaf_name!r} on {desc.name!r}")
    else:
        # Shouldn't happen — every attr predicate has at least one path
        # segment after the entity. Defensive.
        return MatchSet(())

    family = attr.alias_family
    right_set = _flatten_values(pred.value, family)

    matches: list[Match] = []
    for c in candidates:
        if _candidate_matches(c, desc, attr, leaf_name, pred.op, right_set, pred.value, ctx):
            iv = desc.interval(c, ctx.bundle)
            iv = _intersect_window(iv, ctx.window)
            if iv is None and ctx.window is not None:
                continue
            matches.append(Match(ctx.bundle.video.clip_id, c, iv))
    return MatchSet(tuple(matches))


def _candidate_matches(
    candidate: Any,
    desc: EntityDescriptor,
    attr: Any,
    leaf_name: str,
    op: str,
    right_set: frozenset[Any],
    raw_value: object,
    ctx: _Context,
) -> bool:
    value = attr.reader(candidate, ctx.bundle)

    # Special handling for `type` on action-typed sub-entities (uses the
    # parenthesized-suffix base-form rule).
    if leaf_name == "type" and desc.name in ("agent.action", "ego.action"):
        if op in ("=", "in"):
            return action_type_matches(value, right_set)
        if op == "!=":
            return not action_type_matches(value, right_set)
        # Numeric ops on a string field — falls through to generic.

    # List-valued attributes: existential match.
    if attr.kind == "list":
        if op == "!=":
            # "no element equals right" — vacuously true on empty list.
            return all(v not in right_set for v in value or [])
        if op == "=" or op == "in":
            return any(v in right_set for v in value or [])
        return False

    return _compare(value, op, right_set, raw_value)


def _eval_entity(node: EntityClause, ctx: _Context) -> MatchSet:
    desc = ENTITIES[node.entity]
    candidates = desc.candidates(ctx.bundle)
    matches: list[Match] = []
    for c in candidates:
        if _candidate_satisfies(node.entity, c, node.inner, ctx):
            iv = desc.interval(c, ctx.bundle)
            iv = _intersect_window(iv, ctx.window)
            if iv is None and ctx.window is not None:
                continue
            matches.append(Match(ctx.bundle.video.clip_id, c, iv))
    return MatchSet(tuple(matches))


def _candidate_satisfies(
    entity_name: str, candidate: Any, expr: Expr, ctx: _Context
) -> bool:
    """Evaluate `expr` *rooted at* a single candidate entity. The inner
    predicates may reference `entity_name` explicitly (no-op) or be
    "unqualified" — both forms work.

    We return True if at least one match exists for `candidate`.
    """
    if isinstance(expr, AttrPredicate):
        return _attr_matches_for_candidate(entity_name, candidate, expr, ctx)
    if isinstance(expr, EntityClause):
        # Sub-entity clause: e.g. `agent(type=ped, action(jaywalk=true))`.
        # Resolve via the parent's attribute table.
        return _sub_entity_clause_matches(entity_name, candidate, expr, ctx)
    if isinstance(expr, And):
        return _candidate_satisfies(entity_name, candidate, expr.left, ctx) and \
               _candidate_satisfies(entity_name, candidate, expr.right, ctx)
    if isinstance(expr, Or):
        return _candidate_satisfies(entity_name, candidate, expr.left, ctx) or \
               _candidate_satisfies(entity_name, candidate, expr.right, ctx)
    if isinstance(expr, Not):
        return not _candidate_satisfies(entity_name, candidate, expr.expr, ctx)
    # Temporal/relational/within: not legal *inside* an entity clause for
    # v1. Fall back to evaluating against the whole bundle and treating
    # the clip-level emptiness as the answer.
    sub = _eval(expr, ctx)
    return bool(sub)


def _resolve_entity_path(
    root_entity: str, candidate: Any, path: tuple[str, ...], ctx: _Context
) -> tuple[EntityDescriptor, list[Any]]:
    """Walk `path` starting from `candidate` (rooted at `root_entity`)."""
    desc = ENTITIES[root_entity]
    candidates: list[Any] = [candidate]
    for i, seg in enumerate(path[:-1] if path else ()):
        attr = desc.attributes[seg]
        if attr.kind != "sub_entity":
            break
        sub_desc = ENTITIES[attr.sub_entity or seg]
        new_cands: list[Any] = []
        for c in candidates:
            new_cands.extend(attr.reader(c, ctx.bundle))
        candidates = new_cands
        desc = sub_desc
    return desc, candidates


def _attr_matches_for_candidate(
    entity_name: str, candidate: Any, pred: AttrPredicate, ctx: _Context
) -> bool:
    """Inside an entity-clause: the predicate's `entity` may match the
    bound entity (in which case it applies to *this* candidate) or be a
    different entity (then it's a clip-level filter).
    """
    # If the predicate is unqualified (uses the same root entity), apply
    # to `candidate`. Otherwise it's a global predicate.
    if pred.entity != entity_name:
        return bool(_eval_attr(pred, ctx))

    path = pred.path
    if not path:
        return False

    desc, sub_candidates = _resolve_entity_path(entity_name, candidate, path, ctx)
    leaf_name = path[-1]
    attr = desc.attributes.get(leaf_name)
    if attr is None:
        return False

    right_set = _flatten_values(pred.value, attr.alias_family)
    for c in sub_candidates:
        if _candidate_matches(c, desc, attr, leaf_name, pred.op, right_set, pred.value, ctx):
            return True
    return False


def _sub_entity_clause_matches(
    parent_entity: str, candidate: Any, clause: EntityClause, ctx: _Context
) -> bool:
    """Resolve `agent(...action(type=walk)...)` against a single agent."""
    parent_desc = ENTITIES[parent_entity]
    attr = parent_desc.attributes.get(clause.entity)
    if attr is None:
        # Top-level entity nested directly — treat as clip-level.
        return bool(_eval(clause, ctx))
    if attr.kind != "sub_entity":
        return False
    sub_name = attr.sub_entity or clause.entity
    for sub in attr.reader(candidate, ctx.bundle):
        if _candidate_satisfies(sub_name, sub, clause.inner, ctx):
            return True
    return False


# ---------------------------------------------------------------------------
# Boolean operators
# ---------------------------------------------------------------------------


def _eval_and(node: And, ctx: _Context) -> MatchSet:
    left = _eval(node.left, ctx)
    right = _eval(node.right, ctx)
    if not left or not right:
        return MatchSet(())
    # Both non-empty for this clip → union of matches.
    return MatchSet(tuple(left.matches) + tuple(right.matches))


def _eval_or(node: Or, ctx: _Context) -> MatchSet:
    left = _eval(node.left, ctx)
    right = _eval(node.right, ctx)
    if not left and not right:
        return MatchSet(())
    return MatchSet(tuple(left.matches) + tuple(right.matches))


def _eval_not(node: Not, ctx: _Context) -> MatchSet:
    inner = _eval(node.expr, ctx)
    if inner:
        return MatchSet(())
    # Emit a single whole-window match for the clip.
    iv: Interval | None
    if ctx.window:
        iv = ctx.window[0]
    else:
        iv = _clip_span(ctx.bundle)
    return MatchSet((Match(ctx.bundle.video.clip_id, ctx.bundle, iv),))


# ---------------------------------------------------------------------------
# Temporal operators
# ---------------------------------------------------------------------------


def _eval_while(node: While, ctx: _Context) -> MatchSet:
    left = _eval(node.left, ctx)
    right = _eval(node.right, ctx)
    out: list[Match] = []
    for a in left.matches:
        for b in right.matches:
            if a.clip_id != b.clip_id:
                continue
            iv = _intersect(a.interval, b.interval)
            if iv is None:
                continue
            out.append(Match(a.clip_id, (a.entity, b.entity), iv))
    return MatchSet(tuple(out))


def _eval_then(node: Then, ctx: _Context) -> MatchSet:
    left = _eval(node.left, ctx)
    right = _eval(node.right, ctx)
    k = node.k
    out: list[Match] = []
    for a in left.matches:
        for b in right.matches:
            if a.clip_id != b.clip_id:
                continue
            ai, bi = a.interval, b.interval
            if ai is None or bi is None:
                continue
            if bi.start < ai.start:
                continue
            overlap = ai.overlaps(bi)
            gap = bi.start - ai.end
            if not overlap and not (0 <= gap <= k):
                continue
            iv = Interval(min(ai.start, bi.start), max(ai.end, bi.end))
            out.append(Match(a.clip_id, (a.entity, b.entity), iv))
    return MatchSet(tuple(out))


def _eval_because_of(node: BecauseOf, ctx: _Context) -> MatchSet:
    left = _eval(node.left, ctx)
    right = _eval(node.right, ctx)
    out: list[Match] = []
    for a in left.matches:
        action = a.entity
        if not isinstance(action, (AgentAction, EgoAction)):
            continue
        if not action.because_of:
            continue
        for b in right.matches:
            if a.clip_id != b.clip_id:
                continue
            # Find the schema id of b's entity (best effort).
            b_id = getattr(b.entity, "id", None)
            if b_id is None or b_id not in action.because_of:
                continue
            out.append(Match(a.clip_id, action, a.interval))
            out.append(Match(b.clip_id, b.entity, b.interval))
    return MatchSet(tuple(out))


# ---------------------------------------------------------------------------
# Within
# ---------------------------------------------------------------------------


def _eval_within(node: Within, ctx: _Context) -> MatchSet:
    win = _eval(node.window, ctx)
    if not win:
        return MatchSet(())
    windows = tuple(m.interval for m in win.matches if m.interval is not None)
    if not windows:
        return MatchSet(())
    sub_ctx = _Context(
        bundle=ctx.bundle, window=windows, id_index=ctx.id_index,
    )
    return _eval(node.body, sub_ctx)


__all__ = ["Match", "MatchSet", "evaluate"]
