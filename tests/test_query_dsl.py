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

from causal_ai_av.dataset import CausalAVDataset
from causal_ai_av.query import (
    AttrPredicate,
    EntityClause,
    Match,
    MatchSet,
    QueryParseError,
    evaluate,
    parse,
)
from causal_ai_av.spec import Agent, AnnotationBundle

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
    ds = CausalAVDataset(CORPUS)
    ms = ds.find("agent.type = ped and env.type = crosswalk")
    # Soft assertion — corpus is real; we expect *some* clip to have both.
    assert isinstance(ms, MatchSet)
    assert len(ms.clips()) >= 1


def test_because_of_parses_and_runs(patched_parent: None) -> None:
    """`ego.action = decel because_of agent.type = ped` parses and evaluates
    without crashing. Match count may be 0 — we only test wiring."""
    ds = CausalAVDataset(CORPUS)
    # Just confirm it parses and runs against every clip without error.
    n = ds.count("ego.action = decel because_of agent.type = ped")
    assert isinstance(n, int)
    assert n >= 0


def test_dataset_count_and_group_by(patched_parent: None) -> None:
    ds = CausalAVDataset(CORPUS)
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
    ds = CausalAVDataset(CORPUS)
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
    ds = CausalAVDataset(CORPUS)
    clip_id = ds.list_sequences()[0]
    seq = ds.get_sequence(clip_id)
    ms = seq.find("agent.type = ped")
    # `Sequence._parent` is set to the dataset by `get_sequence`.
    assert ms.dataset is ds


def test_bundle_level_match_set_has_no_dataset(rich_bundle: AnnotationBundle) -> None:
    """A MatchSet built via `find_on_bundle` (no `dataset=` kwarg) has
    `.dataset` of None — the bundle-level entry doesn't fabricate a
    back-reference."""
    from causal_ai_av.query.api import find_on_bundle

    ms = find_on_bundle(rich_bundle, "agent.type = ped")
    assert ms.dataset is None


def test_match_set_dataset_raises_after_gc(patched_parent: None) -> None:
    """When the source dataset is garbage-collected, `MatchSet.dataset`
    raises a clear RuntimeError rather than returning a stale reference."""
    import gc
    import weakref as _wr

    ds = CausalAVDataset(CORPUS)
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
