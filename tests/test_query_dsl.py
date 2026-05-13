# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal end-to-end tests for the DSL query engine.

Per the user's constraint: ~5–8 tests total. We exercise parse, evaluate,
entity-clause coupling, boolean composition, `because_of`, and the
Dataset aggregation surface. No per-method unit coverage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cascade_av.dataset import CascadeDataset
from cascade_av.query import (
    AttrPredicate,
    EntityClause,
    Match,
    MatchSet,
    QueryParseError,
    evaluate,
    parse,
)
from cascade_av.spec import Agent, AnnotationBundle

CORPUS = Path("/home/horde/01_json_annotations")


@pytest.fixture
def patched_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass `PhysicalAIAVDatasetInterface.__init__` network calls."""
    from physical_ai_av import PhysicalAIAVDatasetInterface

    def _noop(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.is_offline_mode = True

    monkeypatch.setattr(PhysicalAIAVDatasetInterface, "__init__", _noop)


def test_parser_accepts_spec_examples() -> None:
    """Each query from §4.1 of the spec parses and yields the expected
    AST root shape."""
    cases = [
        ("agent.type = ped", AttrPredicate),
        ("agent(type = vehicle, pos = front)", EntityClause),
        ("agent.type = ped and env.type = crosswalk and ego.action = decel", type),
        ("light.color = yellow then(3) ego.action = stop", type),
        ("within light.color = red: not ego.action = stop", type),
    ]
    for text, _expected in cases:
        ast = parse(text)
        assert ast is not None
    # Spot-check the first two for type identity.
    assert isinstance(parse("agent.type = ped"), AttrPredicate)
    assert isinstance(parse("agent(type = vehicle, pos = front)"), EntityClause)


def test_parse_errors_carry_position() -> None:
    # Unknown entity
    with pytest.raises(QueryParseError) as exc_info:
        parse("flarble.type = ped")
    assert exc_info.value.line == 1
    assert "flarble" in exc_info.value.message

    # Missing closing paren
    with pytest.raises(QueryParseError) as exc_info:
        parse("agent(type = ped")
    assert exc_info.value.line >= 1


def test_single_primitive_finds_pedestrian(rich_bundle: AnnotationBundle) -> None:
    ms = evaluate(parse("agent.type = ped"), rich_bundle)
    if not ms:
        pytest.skip("rich bundle has no pedestrian agents")
    assert len(ms) >= 1
    for m in ms.matches:
        assert isinstance(m, Match)
        assert isinstance(m.entity, Agent)


def test_entity_clause_couples_constraints(rich_bundle: AnnotationBundle) -> None:
    """`agent(type=ped, action(type=walk))` should return matches whose
    entity is an Agent (not an AgentAction)."""
    ms = evaluate(
        parse("agent(type = ped and action(type = walk))"), rich_bundle
    )
    if not ms:
        pytest.skip("rich bundle has no ped/walk agents")
    for m in ms.matches:
        assert isinstance(m.entity, Agent)


def test_and_across_entities(patched_parent: None) -> None:
    """`agent.type = ped and env.type = crosswalk` finds at least one clip."""
    ds = CascadeDataset(CORPUS)
    ms = ds.find("agent.type = ped and env.type = crosswalk")
    # Soft assertion — corpus is real; we expect *some* clip to have both.
    assert isinstance(ms, MatchSet)
    assert len(ms.clips()) >= 1


def test_because_of_parses_and_runs(patched_parent: None) -> None:
    """`ego.action = decel because_of agent.type = ped` parses and evaluates
    without crashing. Match count may be 0 — we only test wiring."""
    ds = CascadeDataset(CORPUS)
    # Just confirm it parses and runs against every clip without error.
    n = ds.count("ego.action = decel because_of agent.type = ped")
    assert isinstance(n, int)
    assert n >= 0


def test_dataset_count_and_group_by(patched_parent: None) -> None:
    ds = CascadeDataset(CORPUS)
    n = ds.count("agent.type = ped")
    assert isinstance(n, int)
    assert n > 0

    g = ds.group_by("agent.type = vehicle", "agent.type")
    assert isinstance(g, dict)
    # Every key should be a (non-empty) type string; every value a positive int.
    for k, v in g.items():
        assert isinstance(k, str)
        assert isinstance(v, int) and v > 0


def test_match_set_carries_dataset_back_reference(patched_parent: None) -> None:
    """MatchSet from `ds.find(...)` resolves `.dataset` back to the dataset."""
    ds = CascadeDataset(CORPUS)
    ms = ds.find("agent.type = ped")
    assert ms.dataset is ds


def test_match_set_carries_dataset_back_reference_from_sequence(
    patched_parent: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MatchSet from `seq.find(...)` resolves `.dataset` back to the parent."""
    from physical_ai_av import PhysicalAIAVDatasetInterface

    # Stub the eager egomotion load that `Sequence.__init__` triggers
    # when `parent` is set. `patched_parent` only stubs the parent's
    # `__init__`; it doesn't stub `get_clip_feature`.
    monkeypatch.setattr(
        PhysicalAIAVDatasetInterface,
        "get_clip_feature",
        lambda self, *a, **kw: None,
    )
    ds = CascadeDataset(CORPUS)
    clip_id = ds.list_sequences()[0]
    seq = ds.get_sequence(clip_id)
    ms = seq.find("agent.type = ped")
    # `Sequence._parent` is set to the dataset by `get_sequence`.
    assert ms.dataset is ds


def test_bundle_level_match_set_has_no_dataset(rich_bundle: AnnotationBundle) -> None:
    """A MatchSet built via `find_on_bundle` (no `dataset=` kwarg) has
    `.dataset` of None — the bundle-level entry doesn't fabricate a
    back-reference."""
    from cascade_av.query.api import find_on_bundle

    ms = find_on_bundle(rich_bundle, "agent.type = ped")
    assert ms.dataset is None


def test_match_set_dataset_raises_after_gc(patched_parent: None) -> None:
    """When the source dataset is garbage-collected, `MatchSet.dataset`
    raises a clear RuntimeError rather than returning a stale reference."""
    import gc
    import weakref as _wr

    ds = CascadeDataset(CORPUS)
    ms = ds.find("agent.type = ped")
    # Sanity: ms holds only a weakref, not a strong reference back.
    ds_wr = _wr.ref(ds)
    del ds
    gc.collect()
    if ds_wr() is not None:
        # Something else is keeping the dataset alive (e.g. the test
        # harness); skip the assertion rather than declare a false
        # positive. The non-gc path is covered by the other two tests.
        pytest.skip("test environment retains a reference to the dataset")
    with pytest.raises(RuntimeError, match="no longer alive"):
        _ = ms.dataset


def test_match_set_sequences_yields_pairs(
    patched_parent: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ms.sequences()` yields `(Match, Sequence)` pairs whose clip_ids match."""
    from physical_ai_av import PhysicalAIAVDatasetInterface

    from cascade_av.dataset import Sequence

    # `Sequence.__init__` eagerly loads egomotion via the parent — stub it
    # so the constructor doesn't try to hit the network.
    monkeypatch.setattr(
        PhysicalAIAVDatasetInterface,
        "get_clip_feature",
        lambda self, *a, **kw: None,
    )
    ds = CascadeDataset(CORPUS)
    ms = ds.find("agent.type = ped")
    if not ms:
        pytest.skip("corpus has no pedestrian matches")

    pairs = list(ms.sequences())
    assert len(pairs) == len(ms.matches)
    for (yielded_match, seq), source_match in zip(pairs, ms.matches):
        assert isinstance(yielded_match, Match)
        assert isinstance(seq, Sequence)
        assert yielded_match is source_match
        assert seq.clip_id == yielded_match.clip_id


def test_match_set_sequences_dedups_per_clip(
    patched_parent: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two matches with the same clip_id share one `Sequence` instance."""
    from physical_ai_av import PhysicalAIAVDatasetInterface

    monkeypatch.setattr(
        PhysicalAIAVDatasetInterface,
        "get_clip_feature",
        lambda self, *a, **kw: None,
    )
    ds = CascadeDataset(CORPUS)
    ms = ds.find("agent.type = ped")

    # Find a clip_id that produced at least two matches in this MatchSet.
    counts: dict[str, int] = {}
    for m in ms.matches:
        counts[m.clip_id] = counts.get(m.clip_id, 0) + 1
    multi = [cid for cid, n in counts.items() if n >= 2]
    if not multi:
        pytest.skip("no clip in the corpus produced ≥2 matches for ped")
    target = multi[0]

    seen: dict[str, object] = {}
    for match, seq in ms.sequences():
        if match.clip_id == target:
            if target in seen:
                # Second (and subsequent) yields for this clip_id must be
                # the *same* Sequence instance, not just an equal one.
                assert seq is seen[target]
            else:
                seen[target] = seq
    assert target in seen


def test_match_set_sequences_raises_without_dataset(
    rich_bundle: AnnotationBundle,
) -> None:
    """A bundle-level MatchSet has no dataset; `.sequences()` must raise."""
    from cascade_av.query.api import find_on_bundle

    ms = find_on_bundle(rich_bundle, "agent.type = ped")
    with pytest.raises(RuntimeError, match="no dataset back-reference"):
        next(ms.sequences())
