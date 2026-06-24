# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal end-to-end tests for the DSL query engine.

Per the user's constraint: ~5–8 tests total. We exercise parse, evaluate,
entity-clause coupling, boolean composition, `because_of`, and the
Dataset aggregation surface. No per-method unit coverage.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cascade_av.dataset import CascadeDataset
from cascade_av.query import (
    AttrPredicate,
    AttrRef,
    EntityClause,
    EntityRef,
    InfluencedBy,
    Match,
    MatchSet,
    QueryParseError,
    While,
    evaluate,
    parse,
)
from cascade_av.spec import Agent, AnnotationBundle, Environment

_root = os.environ.get("CASCADE_AV_DATASET_ROOT")
CORPUS = Path(_root) if _root else None

# Tests that construct a real CascadeDataset call `_skip_if_no_corpus()`
# at the top of the body; unit tests with fabricated bundles run
# unconditionally. The module-wide guard is intentionally absent so the
# parser/registry tests stay green on machines without the corpus.


def _skip_if_no_corpus() -> Path:
    if CORPUS is None or not CORPUS.is_dir():
        pytest.skip("set CASCADE_AV_DATASET_ROOT to run corpus-backed tests")
    return CORPUS


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


def test_parser_accepts_contained_in_alias() -> None:
    """The `contained_in` alias reaches the containment attribute that is
    otherwise unreachable in the grammar because the bare name `in` is
    also the IN set-membership operator."""
    cases = [
        ("agent.contained_in = road", AttrPredicate),
        ("agent(contained_in = sidewalk)", EntityClause),
        ("ego.contained_in in (road, intersection)", AttrPredicate),
        ("agent(type = ped, contained_in = crosswalk)", EntityClause),
    ]
    for text, expected_root in cases:
        ast = parse(text)
        assert isinstance(ast, expected_root), f"{text} -> {type(ast).__name__}"


def test_contained_in_is_alias_of_in() -> None:
    """`contained_in` and `in` are registered as the same Attribute on
    both agent and ego descriptors, so they share alias_family and reader."""
    from cascade_av.query.entities import AGENT_DESCRIPTOR, EGO_DESCRIPTOR

    for desc in (AGENT_DESCRIPTOR, EGO_DESCRIPTOR):
        assert desc.attributes["in"] is desc.attributes["contained_in"]


def test_other_aliases_for_four_unaliased_families() -> None:
    """The four families that previously lacked an `other` alias now
    resolve `"other"` to the right `Vocab.OTHER` literal so users
    don't have to fall back to the schema-literal escape hatch for
    8 % of the audit corpus's agent/action/env/signaling-source
    entities."""
    from cascade_av.query.constants import resolve_alias
    from cascade_av.spec.schema import (
        AgentActionTypeVocab,
        AgentTypeVocab,
        EnvironmentTypeVocab,
        SignalSourceVocab,
    )

    cases = [
        ("agent_type", AgentTypeVocab.OTHER),
        ("action_type", AgentActionTypeVocab.OTHER),
        ("env_type", EnvironmentTypeVocab.OTHER),
        ("signaling_source", SignalSourceVocab.OTHER),
    ]
    for family, expected in cases:
        resolved = resolve_alias(family, "other")
        assert expected in resolved, (
            f"{family}.other resolved to {resolved}, missing {expected!r}"
        )


def test_drift_0_6x_vocab_aliases_resolve() -> None:
    """The 0.6.x schema-drift vocab additions each resolve from their
    alias to the right `Vocab` constant. `agent_property_type` /
    `ego_property_type` gain `stopped`; `action_type` gains the
    standalone `jaywalk_action` leaf; `env_type` gains `speed_bump` /
    `light_rail_lane`; `obj_type` gains `boom_gate`; `signaling_intent`
    gains `signal_left` / `signal_right`; and the brand-new
    `eventful_reason` family covers `ego_adapts` and friends."""
    from cascade_av.query.constants import resolve_alias
    from cascade_av.spec.schema import (
        AgentActionTypePedestrianVocab,
        AgentPropertyTypeVocab,
        EgoPropertyTypeVocab,
        EnvironmentTypeVocab,
        EventfulReasonVocab,
        SignalIntentVocab,
        TrafficObjectTypeVocab,
    )

    # Property-type `stopped` lands on both the agent and ego families.
    assert AgentPropertyTypeVocab.STOPPED in resolve_alias("agent_property_type", "stopped")
    assert EgoPropertyTypeVocab.STOPPED in resolve_alias("ego_property_type", "stopped")

    # `jaywalk_action` is a single-value leaf, so it resolves to exactly
    # that one literal (it must NOT pull in the deprecated `(jaywalk)`
    # suffix forms).
    assert resolve_alias("action_type", "jaywalk_action") == frozenset(
        {AgentActionTypePedestrianVocab.JAYWALK}
    )

    # New environment types.
    assert EnvironmentTypeVocab.SPEED_BUMP in resolve_alias("env_type", "speed_bump")
    assert EnvironmentTypeVocab.LIGHT_RAIL_LANE in resolve_alias("env_type", "light_rail_lane")

    # New traffic-object type.
    assert TrafficObjectTypeVocab.BOOM_GATE in resolve_alias("obj_type", "boom_gate")

    # Left/Right indicator superseded the deprecated combined `Turn` intent.
    assert SignalIntentVocab.LEFT_INDICATOR in resolve_alias("signaling_intent", "signal_left")
    assert SignalIntentVocab.RIGHT_INDICATOR in resolve_alias("signaling_intent", "signal_right")

    # Brand-new `eventful_reason` family.
    assert EventfulReasonVocab.EGO_ADAPTS in resolve_alias("eventful_reason", "ego_adapts")
    assert EventfulReasonVocab.SPECIAL_ENVIRONMENT in resolve_alias(
        "eventful_reason", "special_env"
    )
    assert EventfulReasonVocab.AGENT_ADAPTS in resolve_alias("eventful_reason", "agent_adapts")
    assert EventfulReasonVocab.OTHER in resolve_alias("eventful_reason", "other")


def test_clip_eventful_reason_uses_eventful_reason_family() -> None:
    """`clip.eventful_reason` wires the new `eventful_reason` alias family
    onto `AnnotationBundle.annotation.eventful_reason`, so callers can
    write `clip.eventful_reason = ego_adapts` instead of the raw string."""
    from cascade_av.query.entities import CLIP_DESCRIPTOR

    assert CLIP_DESCRIPTOR.attributes["eventful_reason"].alias_family == "eventful_reason"

    for q in (
        "clip.eventful_reason = ego_adapts",
        "clip.eventful_reason in (ego_adapts, agent_adapts)",
    ):
        assert parse(q) is not None


def test_standalone_jaywalk_distinct_from_deprecated_suffix() -> None:
    """The real matcher distinguishes the standalone `Jaywalk` base verb
    from the deprecated `oxd:Walk (jaywalk)` suffix form. Because
    `Jaywalk` has no parens, prefix matching is disabled, so it matches
    only the exact literal — not `oxd:Walk (jaywalk)`."""
    from cascade_av.query.entities import action_type_matches

    assert action_type_matches("Jaywalk", frozenset({"Jaywalk"})) is True
    assert action_type_matches("oxd:Walk (jaywalk)", frozenset({"Jaywalk"})) is False


def test_agent_amount_aliases_resolve_and_descriptor_uses_them() -> None:
    """`agent.amount` was registered without an alias_family, so users
    had to spell `agent.amount = "Single"` exactly. The new
    `agent_amount` family lets them write `agent.amount = single`,
    `... = group`, `... = traffic`, etc."""
    from cascade_av.query.constants import resolve_alias
    from cascade_av.query.entities import AGENT_DESCRIPTOR
    from cascade_av.spec.schema import AgentAmountVocab

    # Leaves resolve to the canonical Vocab values.
    assert resolve_alias("agent_amount", "single") == frozenset(
        {AgentAmountVocab.SINGLE}
    )
    assert resolve_alias("agent_amount", "row") == frozenset(
        {AgentAmountVocab.ROW_GROUP}
    )
    assert resolve_alias("agent_amount", "group") == frozenset(
        {AgentAmountVocab.ROW_GROUP}
    )

    # `traffic` parent covers the three density tiers (which the 100json
    # corpus does not exercise yet, but the schema declares).
    traffic = resolve_alias("agent_amount", "traffic")
    assert AgentAmountVocab.LIGHT_TRAFFIC in traffic
    assert AgentAmountVocab.MEDIUM_TRAFFIC in traffic
    assert AgentAmountVocab.HEAVY_TRAFFIC in traffic
    assert AgentAmountVocab.SINGLE not in traffic

    # `multiple` is the "not alone" parent.
    multiple = resolve_alias("agent_amount", "multiple")
    assert AgentAmountVocab.ROW_GROUP in multiple
    assert AgentAmountVocab.SINGLE not in multiple

    # Descriptor wires the family in.
    assert AGENT_DESCRIPTOR.attributes["amount"].alias_family == "agent_amount"

    # Parser accepts the natural call-site forms.
    for q in (
        "agent.amount = single",
        "agent.amount = group",
        "agent.amount in (row, single)",
        "agent(type = ped, amount = group)",
    ):
        assert parse(q) is not None


def test_vehicle_parent_includes_generic_vehicle_literal() -> None:
    """`agent.type = vehicle` matches the bare `"Vehicle"` literal in
    addition to the specific subtypes (car, truck, bus, motorcycle,
    emergency). The leaf alias `generic_vehicle` is the named handle
    for the literal on its own."""
    from cascade_av.query.constants import resolve_alias
    from cascade_av.spec.schema import AgentTypeVocab

    assert AgentTypeVocab.VEHICLE in resolve_alias("agent_type", "vehicle")
    assert AgentTypeVocab.VEHICLE in resolve_alias("agent_type", "generic_vehicle")


def test_cond_env_attrs_register_and_resolve_join() -> None:
    """`cond.env_type` / `env_lanes` / `env_one_way` / `env_id`
    denormalize the `Condition.env_id → Environment.*` join the DSL
    can't express directly. Verifies descriptor shape, parser
    acceptance, and the reader's behaviour against a fabricated
    bundle with two environments + one condition pointing at one of
    them."""
    from cascade_av.query.entities import COND_DESCRIPTOR

    # Descriptor shape: env_type reuses the existing env_type family;
    # the others are scalar-no-family.
    assert COND_DESCRIPTOR.attributes["env_type"].alias_family == "env_type"
    for name in ("env_lanes", "env_one_way", "env_id"):
        assert COND_DESCRIPTOR.attributes[name].alias_family is None
        assert COND_DESCRIPTOR.attributes[name].kind == "scalar"

    for q in (
        "cond.env_type = road",
        "cond.env_type in (road, crossroad)",
        "cond.env_lanes > 2",
        "cond.env_one_way = true",
        "cond.type = construction and cond.env_type = road",
    ):
        assert parse(q) is not None

    # Behavioural: fabricate a bundle with two envs and one cond
    # pointing at the first, verify the join resolves.
    from cascade_av.query.entities import _cond_env_id, _cond_env_lanes, _cond_env_type
    from cascade_av.spec import Condition
    from cascade_av.spec.schema import (
        AnnotationBundle as _AB,
    )
    from cascade_av.spec.schema import (
        EgoVehicle as _Ego,
    )
    from cascade_av.spec.schema import (
        SilAvAnnotation,
        VideoMeta,
    )

    env_a = Environment(id="env-A", type="oxd:Road", num_lanes=3, one_way=True)
    env_b = Environment(id="env-B", type="oxd:CrossRoad", num_lanes=2)
    cond_in_a = Condition(id="c-1", env_id="env-A", type=["Construction Zone"])
    cond_orphan = Condition(id="c-2", env_id="env-missing", type=["Other"])

    bundle = _AB(
        schema_version="2.0.0",
        video=VideoMeta(clip_id="test", fps=30.0, duration_s=10.0),
        annotation=SilAvAnnotation(
            environments=[env_a, env_b],
            conditions=[cond_in_a, cond_orphan],
            ego_vehicle=_Ego(),
        ),
    )

    # Resolved join — cond_in_a → env_a's type.
    assert _cond_env_type(cond_in_a, bundle) == "oxd:Road"
    assert _cond_env_lanes(cond_in_a, bundle) == 3
    assert _cond_env_id(cond_in_a, bundle) == "env-A"
    # Dangling env_id resolves to None across the board so predicates
    # fall through cleanly.
    assert _cond_env_type(cond_orphan, bundle) is None
    assert _cond_env_lanes(cond_orphan, bundle) is None
    # `env_id` passthrough preserves the raw value (even when dangling)
    # so callers can spot orphans via inequality.
    assert _cond_env_id(cond_orphan, bundle) == "env-missing"


def test_clip_brief_description_register_and_parse() -> None:
    """`clip.brief_description` exposes the annotator's one-line
    summary on `AnnotationBundle.brief_description` (100 % populated
    in the 100json audit). Scalar string, no alias family — equality
    only until a regex-match surface arrives."""
    from cascade_av.query.entities import CLIP_DESCRIPTOR

    attr = CLIP_DESCRIPTOR.attributes["brief_description"]
    assert attr.kind == "scalar"
    assert attr.alias_family is None

    for q in (
        'clip.brief_description = "ego stops at a red light"',
        'clip.brief_description in ("a", "b")',
    ):
        assert parse(q) is not None


def test_action_id_link_attrs() -> None:
    """`{agent,ego}.action.{link_to,action_target}` expose the companion
    ID lists to the `because_of` operator. Both are `kind="list"` so
    the DSL matches existentially."""
    from cascade_av.query.entities import (
        AGENT_ACTION_DESCRIPTOR,
        EGO_ACTION_DESCRIPTOR,
    )

    for desc in (AGENT_ACTION_DESCRIPTOR, EGO_ACTION_DESCRIPTOR):
        for name in ("link_to", "action_target"):
            assert desc.attributes[name].kind == "list"
            assert desc.attributes[name].alias_family is None

    for q in (
        # IDs in the corpus carry hyphens (UUIDs), so callers quote them
        # to bypass the bare-identifier lexer rule.
        'agent.action.link_to = "abc-def"',
        "agent.action.link_to = abc",
        "ego.action.action_target in (a, b)",
        "agent(action.link_to = abc)",
    ):
        assert parse(q) is not None


def test_containment_flag_attrs_register_on_agent_and_obj() -> None:
    """`{agent,obj}.{lane_edge,near_lane,illegal_lane}` expose the three
    boolean flags carried in `Containment.{edge,near_flag,illegal_flag}`.
    Boolean attributes (no alias family), read existentially across the
    entity's containment list."""
    from cascade_av.query.entities import AGENT_DESCRIPTOR, OBJ_DESCRIPTOR

    for desc in (AGENT_DESCRIPTOR, OBJ_DESCRIPTOR):
        for name in ("lane_edge", "near_lane", "illegal_lane"):
            attr = desc.attributes[name]
            assert attr.alias_family is None
            assert attr.kind == "scalar"

    for q in (
        "agent.lane_edge = true",
        "agent.near_lane = true",
        "agent.illegal_lane = true",
        "agent(type = car, illegal_lane = true)",
        "obj.lane_edge = true",
        "obj(type = cone, lane_edge = true)",
    ):
        assert parse(q) is not None


def test_signaling_details_attrs_on_props() -> None:
    """`agent.prop.source` and `agent.prop.not_facing_ego` (and the
    same on `ego.prop`) read through `AgentProperty.signaling_details`.
    Populated for `Signal`-typed properties (65 % of Signal props in
    100json); ``None`` for non-Signal properties so the predicates
    cleanly fall through."""
    from cascade_av.query.entities import AGENT_PROP_DESCRIPTOR, EGO_PROP_DESCRIPTOR

    for desc in (AGENT_PROP_DESCRIPTOR, EGO_PROP_DESCRIPTOR):
        assert desc.attributes["source"].alias_family == "signaling_source"
        assert desc.attributes["not_facing_ego"].alias_family is None

    for q in (
        "agent.prop.source = flashing_light",
        "agent.prop.source in (flashing_light, holding_sign)",
        "agent.prop.not_facing_ego = true",
        "agent(prop.source = flashing_light)",
        "ego.prop.not_facing_ego = false",
    ):
        assert parse(q) is not None


def test_pos_any_dir_any_parse_and_register() -> None:
    """`pos_any` / `dir_any` are reachable from the parser and registered
    as `kind="list"` attributes on AGENT_DESCRIPTOR sharing the alias
    families of `pos` / `dir`."""
    from cascade_av.query.entities import AGENT_DESCRIPTOR

    for q in (
        "agent.pos_any = front",
        "agent.pos_any in (front, left)",
        "agent(type = vehicle, dir_any = opposite)",
        "agent(type = vehicle, dir_any in (perpendicular_rl, perpendicular_lr))",
    ):
        assert parse(q) is not None

    pa = AGENT_DESCRIPTOR.attributes["pos_any"]
    da = AGENT_DESCRIPTOR.attributes["dir_any"]
    assert pa.kind == "list"
    assert da.kind == "list"
    assert pa.alias_family == "position"
    assert da.alias_family == "direction"


def test_pos_any_iterates_intervals(rich_bundle: AnnotationBundle) -> None:
    """`agent.pos_any` returns all distinct positions across pose intervals;
    `agent.pos` returns one value (the midpoint sample).  For an agent
    whose pose changes during its visibility, the two will differ."""
    from cascade_av.query.entities import AGENT_DESCRIPTOR

    pos_reader = AGENT_DESCRIPTOR.attributes["pos"].reader
    pos_any_reader = AGENT_DESCRIPTOR.attributes["pos_any"].reader

    found_multi = False
    for agent in rich_bundle.annotation.agents:
        positions = pos_any_reader(agent, rich_bundle)
        if len(positions) >= 2:
            # Found an agent whose pose changes across intervals.
            midpoint = pos_reader(agent, rich_bundle)
            assert isinstance(positions, list)
            # pos_any must be a superset of (or equal to) what midpoint
            # sampling returns — anything the midpoint sees is in some
            # interval.
            if midpoint is not None:
                assert midpoint in positions
            found_multi = True
            break

    if not found_multi:
        pytest.skip("rich bundle has no agent with multi-interval pose")


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
    ds = CascadeDataset(_skip_if_no_corpus())
    ms = ds.find("agent.type = ped and env.type = crosswalk")
    # Soft assertion — corpus is real; we expect *some* clip to have both.
    assert isinstance(ms, MatchSet)
    assert len(ms.clips()) >= 1


def test_because_of_parses_and_runs(patched_parent: None) -> None:
    """`ego.action = decel because_of agent.type = ped` parses and evaluates
    without crashing. Match count may be 0 — we only test wiring."""
    ds = CascadeDataset(_skip_if_no_corpus())
    # Just confirm it parses and runs against every clip without error.
    n = ds.count("ego.action = decel because_of agent.type = ped")
    assert isinstance(n, int)
    assert n >= 0


# ---------------------------------------------------------------------------
# influenced_by — parser acceptance + LHS rejection + live-corpus smoke
# ---------------------------------------------------------------------------


def test_influenced_by_parser_accepts_target_queries() -> None:
    """Every shape the operator advertises (bare ego / bare agent / agent
    clause LHS; atomic and compound RHS via `or`, `in`, parenthesised
    groups; composition with `while` and `and`) parses and roots at the
    expected AST node. Covers the 11 motivating queries from the design
    doc one-by-one so a future grammar regression is loud."""
    bare_lhs_cases = [
        "ego influenced_by obj.type = yield_sign",
        "ego influenced_by obj.type = stop_sign",
        "ego influenced_by light.color = red",
        "ego influenced_by light.color = green",
        "ego influenced_by obj.type in (yield_sign, stop_sign)",
        "ego influenced_by (obj.type = stop_sign or light.color = red)",
        "agent influenced_by obj.type = stop_sign",
        "agent influenced_by light.color = red",
        "agent(type = vehicle) influenced_by light.color = red",
    ]
    for q in bare_lhs_cases:
        ast = parse(q)
        assert isinstance(ast, InfluencedBy), f"{q!r} -> {type(ast).__name__}"

    # `while` and `and` bind looser than `influenced_by`, so the root is
    # the boolean / temporal node and InfluencedBy lives inside.
    composed = parse("ego influenced_by obj.type = yield_sign while agent.type = ped")
    assert isinstance(composed, While)
    assert isinstance(composed.left, InfluencedBy)

    ast_and = parse("ego influenced_by light.color = red and ego.action = stop")
    from cascade_av.query.dsl import And as _And  # local import to avoid top-level noise

    assert isinstance(ast_and, _And)
    assert isinstance(ast_and.left, InfluencedBy)

    # Bare LHS rewrites to `EntityRef(...)`; the parser MUST NOT emit an
    # `EntityClause` here (no `(...)` was written).
    ast = parse("ego influenced_by light.color = red")
    assert isinstance(ast.left, EntityRef) and ast.left.entity == "ego"
    ast = parse("agent influenced_by light.color = red")
    assert isinstance(ast.left, EntityRef) and ast.left.entity == "agent"

    # Clause LHS keeps the EntityClause shape — the short-circuit only
    # fires when no `(` follows the head.
    ast = parse("agent(type = vehicle) influenced_by light.color = red")
    assert isinstance(ast.left, EntityClause) and ast.left.entity == "agent"


def _expect_lhs_rejection(query: str) -> None:
    """Helper: ``query`` must be rejected with ``QueryParseError`` either
    at parse time (the parser's existing "expected comparison op…" path
    catches the bare-LHS pseudo-shapes like ``clip influenced_by …``
    before they reach the evaluator) or at evaluation time (the
    evaluator's ``_is_valid_influenced_by_lhs`` guard rejects shapes
    that parse but aren't a valid subject).
    """
    try:
        ast = parse(query)
    except QueryParseError:
        return  # parse-time rejection is sufficient
    with pytest.raises(QueryParseError) as exc_info:
        evaluate(ast, _empty_bundle())
    assert "influenced_by LHS" in exc_info.value.message, (
        f"{query!r}: expected influenced_by LHS rejection, "
        f"got {exc_info.value.message!r}"
    )


def test_influenced_by_rejects_invalid_lhs() -> None:
    """LHS shapes that aren't a bare ego / agent or an ego/agent clause
    must be rejected — the operator's dispatch is rooted at those two
    subject kinds and silently filtering an invalid shape would mask a
    typo. Some shapes are caught by the existing parser machinery (e.g.
    ``clip influenced_by …`` reads ``clip`` as the head of an attribute
    path and expects a comparison op next); the rest are caught by the
    evaluator-side validator. The literal-RHS case bottoms out in the
    unary parser before the operator's right-hand side ever resolves."""

    # `clip influenced_by ...` — clip isn't a subject; the parser
    # catches it as "expected comparison op after 'clip'".
    _expect_lhs_rejection("clip influenced_by light.color = red")

    # RHS-shaped LHS — an attribute predicate isn't a valid subject.
    _expect_lhs_rejection("obj.type = stop_sign influenced_by ego")

    # Sub-entity descent on the LHS — `agent.action.type = walk` reduces
    # to an attribute predicate, not a bare agent / agent clause.
    _expect_lhs_rejection("agent.action.type = walk influenced_by ego")

    # Light predicate as LHS — not a subject kind that carries
    # influencer windows.
    _expect_lhs_rejection("light.color = red influenced_by ego")

    # Literal RHS — parser-time error: the unary parser tries to read a
    # primary and hits a NUMBER token where an entity-name / '(' is
    # expected.
    with pytest.raises(QueryParseError):
        parse("ego influenced_by 5")


def test_influenced_by_evaluator_guard_rejects_paren_attr_lhs() -> None:
    """A parenthesised attribute predicate is a syntactically-valid
    expression that the parser happily reads as `InfluencedBy`'s LHS —
    so `(agent.type = ped) influenced_by light.color = red` parses
    fine but must be rejected by the evaluator's
    `_is_valid_influenced_by_lhs` guard. Without this test the guard
    is reachable from no test case; the other LHS-rejection cases all
    fail at parse time."""
    ast = parse("(agent.type = ped) influenced_by light.color = red")
    # Sanity: the parser produced an InfluencedBy whose LHS is an
    # AttrPredicate, not an EntityRef / EntityClause — exactly the
    # shape the evaluator guard is meant to reject.
    assert isinstance(ast, InfluencedBy)
    from cascade_av.query.dsl import AttrPredicate as _AP

    assert isinstance(ast.left, _AP)
    with pytest.raises(QueryParseError, match="influenced_by LHS must be"):
        evaluate(ast, _empty_bundle())


def _empty_bundle() -> AnnotationBundle:
    """A minimal bundle wired up enough for the evaluator to run; used
    by the LHS-rejection tests where we only care that the validator
    raises before any real work happens."""
    return AnnotationBundle.model_validate({"video": {"clip_id": "lhs-reject"}})


def test_influenced_by_runs_on_live_corpus(patched_parent: None) -> None:
    """Smoke test mirroring `test_because_of_parses_and_runs` —
    `ds.count(...)` must execute without raising for both bare-LHS and
    clause-LHS forms. We don't pin to a specific count; the only
    contract here is "the operator wires through the dataset surface."
    """
    ds = CascadeDataset(_skip_if_no_corpus())
    for q in (
        "ego influenced_by light.color = red",
        "agent influenced_by obj.type = stop_sign",
        "agent(type = vehicle) influenced_by light.color = red",
        "ego influenced_by (obj.type = stop_sign or light.color = red)",
    ):
        n = ds.count(q)
        assert isinstance(n, int)
        assert n >= 0


def test_dataset_count_and_group_by(patched_parent: None) -> None:
    ds = CascadeDataset(_skip_if_no_corpus())
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
    ds = CascadeDataset(_skip_if_no_corpus())
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
    ds = CascadeDataset(_skip_if_no_corpus())
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

    ds = CascadeDataset(_skip_if_no_corpus())
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
    ds = CascadeDataset(_skip_if_no_corpus())
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
    ds = CascadeDataset(_skip_if_no_corpus())
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


# ---------------------------------------------------------------------------
# Same-entity attribute-to-attribute comparisons (AttrRef)
# ---------------------------------------------------------------------------


def _inner_predicate(ast: object) -> AttrPredicate:
    """Pull the inner AttrPredicate out of an EntityClause (or assume the
    AST is already an AttrPredicate). Helper for the parser-acceptance
    test below."""
    if isinstance(ast, EntityClause):
        inner = ast.inner
        assert isinstance(inner, AttrPredicate), (
            f"expected AttrPredicate inside clause, got {type(inner).__name__}"
        )
        return inner
    assert isinstance(ast, AttrPredicate), (
        f"expected AttrPredicate, got {type(ast).__name__}"
    )
    return ast


def test_parser_accepts_same_entity_attr_ref() -> None:
    """Parser builds `AttrRef` for same-entity attr-to-attr comparisons,
    keeps literal RHS as scalar atoms, and rejects cross-entity refs +
    unknown attributes with clear errors."""
    # Same-entity comparisons inside an entity clause.
    for op_query, expected_op in (
        ("env(out_lanes > lanes)", ">"),
        ("env(out_lanes = lanes)", "="),
        ("env(out_lanes != lanes)", "!="),
        ("env(out_lanes < lanes)", "<"),
        ("env(out_lanes >= lanes)", ">="),
        ("env(out_lanes <= lanes)", "<="),
    ):
        pred = _inner_predicate(parse(op_query))
        assert pred.entity == "env"
        assert pred.path == ("out_lanes",)
        assert pred.op == expected_op
        assert isinstance(pred.value, AttrRef), (
            f"{op_query!r}: expected AttrRef RHS, got {type(pred.value).__name__}"
        )
        assert pred.value.path == ("lanes",)

    # Qualified form outside the entity clause.
    pred = _inner_predicate(parse("env.out_lanes >= env.lanes"))
    assert pred.entity == "env"
    assert pred.path == ("out_lanes",)
    assert pred.op == ">="
    assert isinstance(pred.value, AttrRef)
    assert pred.value.path == ("lanes",)

    # Composition with a sibling predicate via `and`.
    ast = parse("env.type = lane_fork and env(out_lanes > lanes)")
    # Right side of `and` is the entity clause carrying the AttrRef.
    from cascade_av.query.dsl import And

    assert isinstance(ast, And)
    rhs_pred = _inner_predicate(ast.right)
    assert isinstance(rhs_pred.value, AttrRef)
    assert rhs_pred.value.path == ("lanes",)

    # Regressions: literal RHS still parses as a scalar atom, not AttrRef.
    pred = _inner_predicate(parse("env.lanes >= 2"))
    assert pred.value == 2
    assert not isinstance(pred.value, AttrRef)

    pred = _inner_predicate(parse("agent.type = ped"))
    assert pred.value == "ped"
    assert not isinstance(pred.value, AttrRef)

    # Cross-entity references are rejected with a `cross-entity` hint.
    # (The architect's original example `agent.in = ego.in` can't reach
    # the AttrRef parser — the bare `in` token is reserved for the IN
    # set-membership operator, so the LHS path stops at `agent.` before
    # the RHS even tokenises. Use `env.lanes = agent.amount` instead,
    # which clears the lexer.)
    with pytest.raises(QueryParseError) as exc_info:
        parse("env.lanes = agent.amount")
    assert "cross-entity" in exc_info.value.message

    with pytest.raises(QueryParseError) as exc_info:
        parse("env(out_lanes > agent.lanes)")
    assert "cross-entity" in exc_info.value.message

    # Unknown attribute on the RHS still goes through the did-you-mean
    # path that `_validate_path` provides — but only when the parser
    # commits to the path branch. A bare unknown ident like `lanez`
    # falls through to a literal value atom (so legacy alias-style
    # queries keep working); the qualified form is what triggers the
    # attribute-name validator.
    with pytest.raises(QueryParseError) as exc_info:
        parse("env.lanes = env.lanez")
    assert "unknown attribute" in exc_info.value.message


def _make_env(
    *, env_id: str, type_str: str, lanes: int | None, out_lanes: int | None
) -> Environment:
    """Build a minimal Environment for the AttrRef semantics test."""
    return Environment(
        id=env_id,
        type=type_str,
        num_lanes=lanes,
        num_out_lanes=out_lanes,
    )


def _bundle_with_envs(envs: list[Environment]) -> AnnotationBundle:
    bundle = AnnotationBundle.model_validate({"video": {"clip_id": "attr-ref-test"}})
    bundle.annotation.environments.extend(envs)
    return bundle


def test_attr_ref_lane_geometry_semantics() -> None:
    """Lane-geometry attr-to-attr comparisons honour the None-substitution
    rule from `docs/user/query_language.md` §4.2: when either side is
    None, substitute the other side's value into the None side before
    comparing. Both None → equal."""
    e1 = _make_env(env_id="e1", type_str="lane_fork", lanes=2, out_lanes=3)
    e2 = _make_env(env_id="e2", type_str="lane_merge", lanes=3, out_lanes=2)
    e3 = _make_env(env_id="e3", type_str="lane_merge", lanes=3, out_lanes=None)
    e4 = _make_env(env_id="e4", type_str="lane_merge", lanes=3, out_lanes=3)
    e5 = _make_env(env_id="e5", type_str="lane_fork", lanes=None, out_lanes=None)
    bundle = _bundle_with_envs([e1, e2, e3, e4, e5])

    def ids_for(query: str) -> set[str]:
        ms = evaluate(parse(query), bundle)
        return {m.entity.id for m in ms.matches}

    assert ids_for("env(out_lanes > lanes)") == {"e1"}
    assert ids_for("env(out_lanes < lanes)") == {"e2"}
    # `=` matches both the symmetric pair (e4) and the None-substituted
    # pairs (e3, e5).
    assert ids_for("env(out_lanes = lanes)") == {"e3", "e4", "e5"}
    # `!=` is strict — None matches never count as inequal.
    assert ids_for("env(out_lanes != lanes)") == {"e1", "e2"}
    # `>=` / `<=` both succeed whenever either side is None (the
    # None-as-equal rule satisfies both directions); for envs where
    # both sides are non-None the usual numeric comparison applies.
    # e1 (3>2): >=, not <=. e2 (2<3): <=, not >=. e4 (3=3): both.
    # e3 (3, None) and e5 (None, None): both via None-as-equal.
    assert ids_for("env(out_lanes >= lanes)") == {"e1", "e3", "e4", "e5"}
    assert ids_for("env(out_lanes <= lanes)") == {"e2", "e3", "e4", "e5"}

    # Both-None env (e5) specifically: matches =, >=, <=; fails !=, >, <.
    only_e5 = _bundle_with_envs([e5])

    def ids_for_e5(query: str) -> set[str]:
        ms = evaluate(parse(query), only_e5)
        return {m.entity.id for m in ms.matches}

    assert ids_for_e5("env(out_lanes = lanes)") == {"e5"}
    assert ids_for_e5("env(out_lanes >= lanes)") == {"e5"}
    assert ids_for_e5("env(out_lanes <= lanes)") == {"e5"}
    assert ids_for_e5("env(out_lanes != lanes)") == set()
    assert ids_for_e5("env(out_lanes > lanes)") == set()
    assert ids_for_e5("env(out_lanes < lanes)") == set()


def test_attr_ref_qualified_form_matches_clause_form() -> None:
    """`env.out_lanes > env.lanes` (outside the entity clause) and
    `env(out_lanes > lanes)` (inside) return the same match set."""
    e1 = _make_env(env_id="e1", type_str="lane_fork", lanes=2, out_lanes=3)
    e2 = _make_env(env_id="e2", type_str="lane_merge", lanes=3, out_lanes=2)
    bundle = _bundle_with_envs([e1, e2])

    clause = evaluate(parse("env(out_lanes > lanes)"), bundle)
    qualified = evaluate(parse("env.out_lanes > env.lanes"), bundle)
    assert {m.entity.id for m in clause.matches} == {m.entity.id for m in qualified.matches}


def test_attr_ref_rejects_list_valued_lhs() -> None:
    """Symmetric to the RHS list check: a list-valued LHS in an
    attr-to-attr comparison is rejected at parse time, rather than
    silently never matching. Regression: a list-valued LHS paired with a
    literal RHS (the existing existential-match shape) still parses."""
    # `cond.type` is `kind="list"`; comparing it against itself as an
    # attr-ref would be undefined under v1 semantics.
    with pytest.raises(QueryParseError) as exc_info:
        parse("cond(type = type)")
    assert "list-valued" in exc_info.value.message

    # Regression: literal RHS on a list-valued LHS continues to parse
    # (the engine resolves it as an existential-match against the list).
    pred = _inner_predicate(parse("cond.type = construction"))
    assert pred.entity == "cond"
    assert pred.path == ("type",)
    assert pred.value == "construction"
    assert not isinstance(pred.value, AttrRef)


def test_attr_ref_rejects_sub_entity_scoped_cross_entity_rhs() -> None:
    """The qualified-RHS guard fires when the LHS is sub-entity-scoped
    (e.g. `agent.action.illegal`) and the RHS names a different
    top-level entity. Even though `ego.judgment` is a real attribute,
    cross-entity references are out of scope for v1 attr-to-attr
    comparisons."""
    with pytest.raises(QueryParseError) as exc_info:
        parse("agent.action.illegal = ego.judgment")
    # The cross-entity branch fires first because the RHS head
    # (`ego`) differs from the LHS top-level entity (`agent`).
    assert "cross-entity" in exc_info.value.message

    # When the RHS uses the bare top-level form *but* the LHS is
    # sub-entity-scoped (same top-level entity, different leaf
    # descriptor), the dedicated sub-entity-scope guard fires instead.
    with pytest.raises(QueryParseError) as exc_info:
        parse("agent.action.illegal = agent.amount")
    assert "sub-entity scope" in exc_info.value.message
