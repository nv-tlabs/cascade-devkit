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
from typing import TYPE_CHECKING, Any, Final, Iterator, Sequence

from cascade_av.query.constants import ALIAS_FAMILIES, resolve_alias
from cascade_av.query.dsl import (
    And,
    AttrPredicate,
    AttrRef,
    Before,
    BecauseOf,
    EntityClause,
    EntityRef,
    Expr,
    InfluencedBy,
    Not,
    Or,
    QueryParseError,
    Then,
    While,
    WhileStrict,
    Within,
)
from cascade_av.query.entities import (
    ENTITIES,
    EntityDescriptor,
    action_type_matches,
)
from cascade_av.query.index import EGO_ID, IdIndex
from cascade_av.query.time import Interval
from cascade_av.spec import AgentAction, AnnotationBundle, EgoAction, EgoVehicle

if TYPE_CHECKING:  # pragma: no cover — typing only
    # Imported lazily for the `sequences()` return annotation; a runtime
    # import here would create a cycle (`dataset.py` already imports from
    # `cascade_av.query`). The runtime resolution happens dynamically
    # inside `MatchSet.sequences` via `self.dataset.get_sequence(...)`.
    from cascade_av.dataset import Sequence as _DatasetSequence


# ---------------------------------------------------------------------------
# Match types
# ---------------------------------------------------------------------------


IDLESS_RECORD_ID: Final[None] = None
"""Public marker used by :attr:`Match.record_ids` for a record with no ID."""


def _record_ids(entity: object) -> tuple[str | None, ...]:
    """Project the schema record IDs represented by a match entity.

    Temporal operators build nested tuples of their operand entities, so the
    projection recursively flattens tuples while preserving operand order and
    duplicates. The two query roots without a schema ``id`` field have stable
    public identities: the bundle uses its clip ID and the ego vehicle uses the
    same synthetic ``"Ego"`` anchor as :class:`IdIndex`.
    """
    if isinstance(entity, tuple):
        return tuple(record_id for item in entity for record_id in _record_ids(item))
    if isinstance(entity, AnnotationBundle):
        return (entity.video.clip_id,)
    if isinstance(entity, EgoVehicle):
        return (EGO_ID,)

    record_id = getattr(entity, "id", IDLESS_RECORD_ID)
    if isinstance(record_id, str) and record_id:
        return (record_id,)
    return (IDLESS_RECORD_ID,)


@dataclass(frozen=True)
class Match:
    clip_id: str
    entity: object
    interval: Interval | None

    @property
    def record_ids(self) -> tuple[str | None, ...]:
        """IDs of the records represented by this match.

        ``IDLESS_RECORD_ID`` (``None``) marks a schema record whose ID is
        absent or blank. Composite temporal matches are flattened recursively
        in left-to-right operand order.
        """
        return _record_ids(self.entity)


@dataclass(frozen=True)
class MatchSet:
    matches: tuple[Match, ...]
    # Weakref back to the dataset (or other producer) that built this
    # MatchSet. Populated by the user-facing entry points
    # (`find_on_dataset`, `find_on_bundle` when called with `dataset=…`,
    # `Sequence.find`, `CascadeDataset.find`); left None for intermediate
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

        Pass `unique_clips=True` to flip the carousel into one player
        per distinct clip (matches in the same clip get unioned and
        rendered once). See
        `cascade_av.viz.carousel.build_matchset_carousel` for the
        full keyword reference (`layout`, `cols`, `limit`, `pad`, `fps`,
        `arrows`, `families`, `unique_clips`). The return type is
        intentionally `Any` — the viz extra (`ipywidgets`, `plotly`)
        is optional, and an annotated return would force importing it
        at module top here.

        Raises:
            RuntimeError: if this `MatchSet` has no dataset
                back-reference. Bundle-level match sets (produced by
                `find_on_bundle` without `dataset=`) can't be
                visualized because there's no way to resolve a clip_id
                to a video reader.
        """
        # Lazy import: `viz` pulls `dataset`, and `dataset` already
        # imports from `cascade_av.query`. A top-level import here
        # would close the cycle. The viz extra is also optional — keep
        # the import inside the call so plain `from cascade_av.query
        # import MatchSet` doesn't require ipywidgets / plotly.
        from cascade_av.viz.carousel import build_matchset_carousel  # noqa: PLC0415

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


def _compare_attr_ref(
    candidate: Any,
    desc: EntityDescriptor,
    attr: Any,
    op: str,
    ref: AttrRef,
    ctx: "_Context",
) -> bool:
    """Same-entity attribute-to-attribute comparison.

    The LHS value comes from ``attr`` on ``candidate``; the RHS value
    comes from the attribute named by ``ref.path[-1]`` on the same
    ``desc``. Implements the "None-as-equal" rule from
    ``docs/user/query_language.md`` §4.2: when either side resolves to
    ``None`` for the same candidate, the comparison treats the two sides
    as equal — so ``=``, ``>=``, and ``<=`` all match, but ``!=``,
    ``>``, and ``<`` do not.
    """
    lhs = attr.reader(candidate, ctx.bundle)
    rhs_attr = desc.attributes[ref.path[-1]]
    rhs = rhs_attr.reader(candidate, ctx.bundle)

    # "None means same" per the user-stated rule. Equal satisfies =,
    # >=, <=; not !=, >, <.
    if lhs is None or rhs is None:
        return op in ("=", ">=", "<=")

    if op == "=":
        return lhs == rhs
    if op == "!=":
        return lhs != rhs
    try:
        lv = float(lhs)
        rv = float(rhs)
    except (TypeError, ValueError):
        return False
    return {">": lv > rv, ">=": lv >= rv, "<": lv < rv, "<=": lv <= rv}[op]


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


def evaluate(
    expr: Expr, bundle: AnnotationBundle, *, strict_identity: bool = False
) -> MatchSet:
    """Top-level entry: evaluate `expr` against one bundle. Returns a
    MatchSet whose entries reference entities inside this bundle only.

    When ``strict_identity`` is true, schema records that declare an ``id``
    field but have a blank/missing value are excluded from matching. The
    default remains permissive for backward compatibility.
    """
    ctx = _Context(
        bundle=bundle,
        window=None,
        id_index=IdIndex(bundle),
        strict_identity=strict_identity,
    )
    return _eval(expr, ctx)


@dataclass(frozen=True)
class _Context:
    bundle: AnnotationBundle
    window: tuple[Interval, ...] | None
    id_index: IdIndex
    strict_identity: bool = False


_NO_ID_FIELD = object()


def _candidate_is_eligible(candidate: object, ctx: _Context) -> bool:
    """Whether ``candidate`` may participate under the identity policy.

    Bundle and ego roots intentionally have no schema ``id`` field and remain
    eligible in strict mode. Records that do expose ``id`` must carry a
    truthy value, matching :meth:`IdIndex._add`'s addressability rule.
    """
    if not ctx.strict_identity:
        return True
    record_id = getattr(candidate, "id", _NO_ID_FIELD)
    return record_id is _NO_ID_FIELD or bool(record_id)


def _eval(expr: Expr, ctx: _Context) -> MatchSet:
    if isinstance(expr, AttrPredicate):
        return _eval_attr(expr, ctx)
    if isinstance(expr, EntityClause):
        return _eval_entity(expr, ctx)
    if isinstance(expr, EntityRef):
        return _eval_entity_ref(expr, ctx)
    if isinstance(expr, And):
        return _eval_and(expr, ctx)
    if isinstance(expr, Or):
        return _eval_or(expr, ctx)
    if isinstance(expr, Not):
        return _eval_not(expr, ctx)
    if isinstance(expr, While):
        return _eval_while(expr, ctx)
    if isinstance(expr, WhileStrict):
        return _eval_while_strict(expr, ctx)
    if isinstance(expr, Then):
        return _eval_then(expr, ctx)
    if isinstance(expr, Before):
        return _eval_before(expr, ctx)
    if isinstance(expr, BecauseOf):
        return _eval_because_of(expr, ctx)
    if isinstance(expr, InfluencedBy):
        return _eval_influenced_by(expr, ctx)
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
    if isinstance(pred.value, AttrRef):
        # AttrRef bypasses alias flattening — handled by
        # `_compare_attr_ref` directly. `right_set` is unused on that
        # path.
        right_set: frozenset[Any] = frozenset()
    else:
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
    if not _candidate_is_eligible(candidate, ctx):
        return False

    # Attribute-to-attribute comparison short-circuits before reading the
    # LHS through the alias / list / action-type machinery — both sides
    # are scalar values pulled from the same candidate, and the None
    # substitution rule lives in `_compare_attr_ref`.
    if isinstance(raw_value, AttrRef):
        return _compare_attr_ref(candidate, desc, attr, op, raw_value, ctx)

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
    if not _candidate_is_eligible(candidate, ctx):
        return False
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

    if isinstance(pred.value, AttrRef):
        # AttrRef bypasses alias flattening — handled by
        # `_compare_attr_ref` directly. `right_set` is unused on that
        # path.
        right_set: frozenset[Any] = frozenset()
    else:
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


def _eval_while_strict(node: WhileStrict, ctx: _Context) -> MatchSet:
    left = _eval(node.left, ctx)
    right = _eval(node.right, ctx)
    out: list[Match] = []
    for a in left.matches:
        for b in right.matches:
            if a.clip_id != b.clip_id:
                continue
            if a.interval is None or b.interval is None:
                continue
            iv = _intersect(a.interval, b.interval)
            if iv is None or iv.end - iv.start <= 0:
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


def _eval_before(node: Before, ctx: _Context) -> MatchSet:
    left = _eval(node.left, ctx)
    right = _eval(node.right, ctx)
    out: list[Match] = []
    for a in left.matches:
        for b in right.matches:
            if a.clip_id != b.clip_id:
                continue
            ai, bi = a.interval, b.interval
            if ai is None or bi is None:
                continue
            gap = bi.start - ai.end
            if gap <= 0:
                continue
            if node.k is not None and gap > node.k:
                continue
            out.append(
                Match(
                    a.clip_id,
                    (a.entity, b.entity),
                    Interval(ai.start, bi.end),
                )
            )
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
# influenced_by — bare-entity LHS, schema influencer dispatch
# ---------------------------------------------------------------------------


def _is_valid_influenced_by_lhs(expr: Expr) -> bool:
    """Validate that ``expr`` reduces to a shape the influenced_by
    operator can read: a bare-entity reference or an entity clause
    rooted at ``ego`` / ``agent``. Sub-entity clauses, action / property
    predicates, etc. are rejected — the operator's contract is "pull the
    influenced_by list off the subject," and that field only lives on
    EgoVehicle and Agent.
    """
    if isinstance(expr, EntityRef) and expr.entity in ("ego", "agent"):
        return True
    if isinstance(expr, EntityClause) and expr.entity in ("ego", "agent"):
        return True
    return False


def _rhs_target_entities(expr: Expr) -> frozenset[str]:
    """Return the set of top-level entity names referenced in any
    ``AttrPredicate.entity`` or ``EntityClause.entity`` within ``expr``.

    Used to skip influencer candidates whose kind doesn't match the RHS
    predicate (e.g. don't evaluate ``light.color = red`` against a
    ``TrafficObject``). Returning an empty set means "no constraint" —
    every candidate kind is fair game.
    """
    if isinstance(expr, AttrPredicate):
        return frozenset({expr.entity})
    if isinstance(expr, EntityClause):
        return frozenset({expr.entity})
    if isinstance(expr, (And, Or)):
        return _rhs_target_entities(expr.left) | _rhs_target_entities(expr.right)
    if isinstance(expr, Not):
        return _rhs_target_entities(expr.expr)
    if isinstance(expr, (While, WhileStrict, Then, Before, BecauseOf, InfluencedBy)):
        return _rhs_target_entities(expr.left) | _rhs_target_entities(expr.right)
    if isinstance(expr, Within):
        return _rhs_target_entities(expr.body)
    return frozenset()


def _light_state_overlaps_window(state: Any, window: Interval) -> bool:
    """True if ``state``'s own time interval overlaps ``window``.
    Permissive on unparseable / missing timestamps — returns True so
    the state is still considered, matching the convention used
    elsewhere in the engine for time-bearing schema entries with
    partial annotations.
    """
    iv = Interval.from_strings(
        getattr(state, "start_timestamp", None),
        getattr(state, "end_timestamp", None),
    )
    if iv is None:
        return True
    return iv.overlaps(window)


def _resolve_influencer_candidates(
    subject: Any,
    ctx: _Context,
    head_to_tl: dict[str, Any],
    infl_iv: Interval,
) -> list[tuple[str, Any]]:
    """Expand a referenced influencer ``Subject`` into ``(entity_name,
    candidate)`` pairs the RHS predicate can be evaluated against.

    Most kinds map 1:1: a ``traffic_object`` becomes ``("obj", obj)``;
    an ``agent`` becomes ``("agent", agent)``; an ``ego`` becomes
    ``("ego", ego)``; a ``light_state`` is already a ``LightStates``
    instance ready to evaluate as ``("light", state)``. The interesting
    cases are ``signal_head`` and ``traffic_light`` — both fan out to
    only those ``LightStates`` whose own interval overlaps
    ``infl_iv`` (the Influence window we're currently evaluating). A
    light cycling G→Y→R must not match `light.color = red` AND
    `light.color = green` AND `light.color = yellow` for the same
    Influence window — only the colour that was actually showing
    during the window should count. Each expanded state is stamped
    with the same ``_owner_signal_head`` (and ``_owner_traffic_light``)
    backref that ``_light_candidates`` sets, so attributes like
    ``affects_ego`` resolve correctly.

    Kinds the corpus doesn't currently use as influencers (env / cond /
    properties / actions / containments / object states) return an
    empty list — the evaluator simply skips them. Future schema
    additions extend the dispatch here.
    """
    if subject.kind == "traffic_object":
        return [("obj", subject.obj)]
    if subject.kind == "agent":
        return [("agent", subject.obj)]
    if subject.kind == "ego":
        return [("ego", subject.obj)]
    if subject.kind == "light_state":
        # Bare LightStates influencer: only contribute if its own
        # interval overlaps the Influence window (permissive on
        # unparseable timestamps).
        if _light_state_overlaps_window(subject.obj, infl_iv):
            return [("light", subject.obj)]
        return []
    if subject.kind == "signal_head":
        head = subject.obj
        tl = head_to_tl.get(head.id)
        out: list[tuple[str, Any]] = []
        for state in head.state_sequence:
            if not _light_state_overlaps_window(state, infl_iv):
                continue
            object.__setattr__(state, "_owner_signal_head", head)
            if tl is not None:
                object.__setattr__(state, "_owner_traffic_light", tl)
            out.append(("light", state))
        return out
    if subject.kind == "traffic_light":
        tl = subject.obj
        out = []
        for head in tl.signal_heads:
            for state in head.state_sequence:
                if not _light_state_overlaps_window(state, infl_iv):
                    continue
                object.__setattr__(state, "_owner_signal_head", head)
                object.__setattr__(state, "_owner_traffic_light", tl)
                out.append(("light", state))
        return out
    return []


def _build_head_to_tl(ctx: _Context) -> dict[str, Any]:
    """One-pass index mapping signal-head ID to its owning traffic-light
    Pydantic model. Cheap to rebuild per call; the bundle's traffic-light
    tree is small.
    """
    out: dict[str, Any] = {}
    for tl in ctx.bundle.annotation.traffic_lights:
        for head in tl.signal_heads:
            if head.id:
                out[head.id] = tl
    return out


def _influenced_by_owner_pairs(
    node_left: Expr, ctx: _Context
) -> list[tuple[Any, list[Any]]]:
    """Gather ``(owner_entity, influenced_by_list)`` tuples from the LHS
    shape. ``owner_entity`` is the schema EgoVehicle / Agent instance
    whose ``influenced_by`` field supplies the windows; the second
    element is that field's value (a list of ``Influence`` records).

    For ``EntityClause(entity='agent', …)`` we filter the agent list
    through the inner constraint via ``_candidate_satisfies`` — same
    machinery used by ``_eval_entity`` — so ``agent(type=vehicle)
    influenced_by …`` only walks vehicle agents. For ``ego`` clauses we
    confirm the single ego candidate satisfies the inner predicate
    before pairing.
    """
    ann = ctx.bundle.annotation
    if isinstance(node_left, EntityRef):
        if node_left.entity == "ego":
            return [(ann.ego_vehicle, list(ann.ego_vehicle.influenced_by))]
        # agent
        return [
            (a, list(a.influenced_by))
            for a in ann.agents
            if _candidate_is_eligible(a, ctx)
        ]

    # EntityClause path — entity guaranteed to be 'ego' or 'agent'
    assert isinstance(node_left, EntityClause)
    if node_left.entity == "agent":
        out: list[tuple[Any, list[Any]]] = []
        for agent in ann.agents:
            if _candidate_satisfies("agent", agent, node_left.inner, ctx):
                out.append((agent, list(agent.influenced_by)))
        return out
    # ego clause: filter on the inner predicate against the ego vehicle.
    ego = ann.ego_vehicle
    if _candidate_satisfies("ego", ego, node_left.inner, ctx):
        return [(ego, list(ego.influenced_by))]
    return []


def _eval_influenced_by(node: InfluencedBy, ctx: _Context) -> MatchSet:
    """Match where the LHS subject is influenced over a window by some
    entity satisfying the RHS predicate.

    The LHS must reduce to a bare ``ego``/``agent`` or an entity clause
    rooted at one of them; anything else raises ``QueryParseError``
    (different from ``because_of``'s silent-skip policy because the
    operator's contract on the LHS is a small fixed contract).

    For each ``Influence`` window on each LHS owner we resolve every
    ``influencer`` ID through :class:`IdIndex` and dispatch it via
    :func:`_resolve_influencer_candidates` into one or more
    ``(entity_name, candidate)`` pairs. The RHS expression is evaluated
    against each candidate via the existing
    :func:`_candidate_satisfies` helper; the first satisfying candidate
    in a window emits one ``Match`` and the window is done.

    Match interval is the **Influence window** (intersected with
    ``ctx.window`` when ``within`` scopes the query) — *not* the
    influencer's own lifetime. That makes ``within X: ego influenced_by
    Y`` and ``ego influenced_by Y while Z`` compose correctly: the
    influencer is a static traffic sign or fixed light; the period
    where it actually modulates ego behaviour is the Influence window
    the annotator authored.
    """
    if not _is_valid_influenced_by_lhs(node.left):
        raise QueryParseError(
            1, 1,
            f"influenced_by LHS must be `ego`, `agent`, or `agent(...)` — "
            f"got {type(node.left).__name__}",
        )

    expected_entities = _rhs_target_entities(node.right)
    head_to_tl = _build_head_to_tl(ctx)
    owners = _influenced_by_owner_pairs(node.left, ctx)

    matches: list[Match] = []
    for owner, infl_list in owners:
        for infl in infl_list:
            if not _candidate_is_eligible(infl, ctx):
                continue
            infl_iv = _interval_from_strings_or_clip_influence(infl, ctx.bundle)
            iv = _intersect_window(infl_iv, ctx.window)
            if iv is None or iv.duration <= 0:
                continue
            for influencer_id in infl.influencers:
                subject = ctx.id_index.get(influencer_id)
                if subject is None:
                    continue
                # Pass the raw Influence window (pre-`within` intersection)
                # so light-state expansion filters by whether the state's
                # own interval overlaps the Influence window — keeps a
                # cycling head from matching every colour at once.
                candidates = _resolve_influencer_candidates(
                    subject, ctx, head_to_tl, infl_iv
                )
                hit = False
                for entity_name, candidate in candidates:
                    if expected_entities and entity_name not in expected_entities:
                        continue
                    if _candidate_satisfies(entity_name, candidate, node.right, ctx):
                        matches.append(
                            Match(ctx.bundle.video.clip_id, owner, iv)
                        )
                        hit = True
                        break
                if hit:
                    break
    return MatchSet(tuple(matches))


def _interval_from_strings_or_clip_influence(
    infl: Any, bundle: AnnotationBundle
) -> Interval:
    """Influence-record interval reader. Falls back to the clip span
    when ``start_timestamp`` / ``end_timestamp`` are missing or
    unparseable — mirrors the convention used by other temporal-bearing
    schema entries.
    """
    iv = Interval.from_strings(
        getattr(infl, "start_timestamp", None),
        getattr(infl, "end_timestamp", None),
    )
    return iv if iv is not None else _clip_span(bundle)


def _eval_entity_ref(node: EntityRef, ctx: _Context) -> MatchSet:
    """Defensive evaluator for a top-level ``EntityRef``. The parser
    only emits ``EntityRef`` as ``influenced_by``'s LHS today, so this
    path is unreachable from a well-formed query; we still cover it so
    a future operator can reuse the AST node without surprises.

    Semantics: every candidate of the named entity matches over its
    natural interval (mirroring a trivially-true entity clause).
    """
    desc = ENTITIES[node.entity]
    matches: list[Match] = []
    for c in desc.candidates(ctx.bundle):
        if not _candidate_is_eligible(c, ctx):
            continue
        iv = desc.interval(c, ctx.bundle)
        iv = _intersect_window(iv, ctx.window)
        if iv is None and ctx.window is not None:
            continue
        if iv is not None and iv.duration <= 0:
            continue
        matches.append(Match(ctx.bundle.video.clip_id, c, iv))
    return MatchSet(tuple(matches))


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
        bundle=ctx.bundle,
        window=windows,
        id_index=ctx.id_index,
        strict_identity=ctx.strict_identity,
    )
    return _eval(node.body, sub_ctx)


__all__ = ["IDLESS_RECORD_ID", "Match", "MatchSet", "evaluate"]
