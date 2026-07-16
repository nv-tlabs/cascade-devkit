# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict temporal query-operator acceptance tests."""

from __future__ import annotations

import pytest

from cascade_av.query import (
    And,
    AttrPredicate,
    Before,
    Interval,
    QueryParseError,
    Then,
    While,
    WhileStrict,
    evaluate,
    parse,
)
from cascade_av.spec import Agent, AnnotationBundle, TrafficObject


def _timestamp(seconds: float) -> str:
    return f"0:{seconds:.6f}"


def _object(
    record_id: str,
    type_name: str,
    interval: tuple[float, float],
) -> TrafficObject:
    return TrafficObject(
        id=record_id,
        type=type_name,
        visibility_start_timestamp=_timestamp(interval[0]),
        visibility_end_timestamp=_timestamp(interval[1]),
    )


def _bundle(
    a_interval: tuple[float, float],
    b_interval: tuple[float, float],
    *,
    extra_objects: list[TrafficObject] | None = None,
    with_agent: bool = False,
) -> AnnotationBundle:
    bundle = AnnotationBundle.model_validate(
        {"video": {"clip_id": "strict-temporal-test", "duration_s": 20.0}}
    )
    bundle.annotation.traffic_objects.extend(
        [
            _object("a", "fst:StopSign", a_interval),
            _object("b", "fst:YieldSign", b_interval),
            *(extra_objects or []),
        ]
    )
    if with_agent:
        bundle.annotation.agents.append(
            Agent(
                id="agent",
                type="oxd:Car",
                visibility_start_timestamp="0:0.0",
                visibility_end_timestamp="0:20.0",
            )
        )
    return bundle


_A = "(obj.type = stop_sign)"
_B = "(obj.type = yield_sign)"


def _matches(query: str, bundle: AnnotationBundle) -> bool:
    return bool(evaluate(parse(query), bundle))


@pytest.mark.parametrize(
    ("a_interval", "b_interval", "then_3", "before_3", "before_unbounded"),
    [
        ((0.0, 5.0), (3.0, 9.0), True, False, False),
        ((0.0, 5.0), (5.0, 9.0), True, False, False),
        ((0.0, 5.0), (6.0, 9.0), True, True, True),
        ((0.0, 5.0), (9.0, 12.0), False, False, True),
        ((3.0, 9.0), (0.0, 5.0), False, False, False),
    ],
)
def test_before_truth_table_and_legacy_then_behavior(
    a_interval: tuple[float, float],
    b_interval: tuple[float, float],
    then_3: bool,
    before_3: bool,
    before_unbounded: bool,
) -> None:
    bundle = _bundle(a_interval, b_interval)

    assert _matches(f"{_A} then(3) {_B}", bundle) is then_3
    assert _matches(f"{_A} before(3) {_B}", bundle) is before_3
    assert _matches(f"{_A} before {_B}", bundle) is before_unbounded


@pytest.mark.parametrize(
    ("a_interval", "b_interval", "while_matches", "strict_matches"),
    [
        ((0.0, 5.0), (3.0, 9.0), True, True),
        ((0.0, 5.0), (5.0, 9.0), True, False),
        ((0.0, 5.0), (6.0, 9.0), False, False),
    ],
)
def test_while_strict_truth_table_and_legacy_while_behavior(
    a_interval: tuple[float, float],
    b_interval: tuple[float, float],
    while_matches: bool,
    strict_matches: bool,
) -> None:
    bundle = _bundle(a_interval, b_interval)

    assert _matches(f"{_A} while {_B}", bundle) is while_matches
    assert _matches(f"{_A} while_strict {_B}", bundle) is strict_matches


def test_before_bound_edges_and_emitted_interval() -> None:
    exact = _bundle((0.0, 5.0), (8.0, 9.0))
    just_over = _bundle((0.0, 5.0), (8.0001, 9.0))

    bounded = evaluate(parse(f"{_A} before(3) {_B}"), exact)
    assert len(bounded) == 1
    assert bounded.matches[0].interval == Interval(0.0, 9.0)
    assert _matches(f"{_A} then(3) {_B}", exact)

    assert not _matches(f"{_A} before(3) {_B}", just_over)
    assert not _matches(f"{_A} then(3) {_B}", just_over)
    assert _matches(f"{_A} before {_B}", just_over)
    assert not _matches(f"{_A} before(0) {_B}", exact)


def test_while_strict_emits_positive_duration_intersection() -> None:
    matches = evaluate(
        parse(f"{_A} while_strict {_B}"),
        _bundle((0.0, 5.0), (3.0, 9.0)),
    )

    assert len(matches) == 1
    assert matches.matches[0].interval == Interval(3.0, 5.0)


def test_parser_distinguishes_before_bound_from_parenthesized_rhs() -> None:
    bounded = parse(f"{_A} before(5) {_B}")
    unbounded = parse(f"{_A} before {_B}")
    unbounded_group = parse(f"{_A} before (obj.type = yield_sign)")
    seconds_suffix = parse(f"{_A} before(5s) {_B}")

    assert isinstance(bounded, Before) and bounded.k == 5.0
    assert isinstance(unbounded, Before) and unbounded.k is None
    assert isinstance(unbounded_group, Before) and unbounded_group.k is None
    assert isinstance(seconds_suffix, Before) and seconds_suffix.k == 5.0
    assert isinstance(parse(f"{_A} while_strict {_B}"), WhileStrict)


@pytest.mark.parametrize(
    "query",
    [
        f"{_A} before() {_B}",
        f"{_A} before(5 {_B}",
        f"{_A} before",
        f"{_A} while_strict",
    ],
)
def test_new_temporal_operators_reject_malformed_or_missing_operands(
    query: str,
) -> None:
    with pytest.raises(QueryParseError):
        parse(query)


def test_new_temporal_operators_compose_and_remain_left_associative() -> None:
    chain = parse(f"{_A} before {_B} while_strict {_B}")
    before_compound = parse(
        "obj(type = stop_sign) before(3) obj(type = yield_sign) and agent(type = car)"
    )
    strict_overlap_compound = parse(
        "obj(type = stop_sign) while_strict obj(type = yield_sign) "
        "and agent(type = car)"
    )

    assert isinstance(chain, WhileStrict)
    assert isinstance(chain.left, Before)
    assert isinstance(before_compound, And)
    assert isinstance(before_compound.left, Before)
    assert isinstance(strict_overlap_compound, And)
    assert isinstance(strict_overlap_compound.left, WhileStrict)

    for compound, bundle in (
        (before_compound, _bundle((0.0, 5.0), (6.0, 9.0), with_agent=True)),
        (
            strict_overlap_compound,
            _bundle((0.0, 5.0), (3.0, 9.0), with_agent=True),
        ),
    ):
        matches = evaluate(compound, bundle)
        assert any(isinstance(match.entity, tuple) for match in matches.matches)
        assert any(isinstance(match.entity, Agent) for match in matches.matches)


def test_new_operator_words_remain_valid_literal_values() -> None:
    assert parse("clip.brief_description = before") == parse('clip.brief_description = "before"')
    assert parse("clip.brief_description = while_strict") == parse(
        'clip.brief_description = "while_strict"'
    )
    assert parse("clip.brief_description in (before, while_strict)") == parse(
        'clip.brief_description in ("before", "while_strict")'
    )


def test_existing_temporal_ast_types_are_unchanged() -> None:
    assert isinstance(parse(f"{_A} then(3) {_B}"), Then)
    assert isinstance(parse(f"{_A} while {_B}"), While)
    assert isinstance(parse("obj.type = stop_sign"), AttrPredicate)


def test_before_evaluation_order_is_deterministic() -> None:
    bundle = _bundle(
        (0.0, 1.0),
        (5.0, 6.0),
        extra_objects=[
            _object("a2", "fst:StopSign", (0.5, 1.5)),
            _object("b2", "fst:YieldSign", (7.0, 8.0)),
        ],
    )
    expr = parse(f"{_A} before {_B}")

    def signature() -> tuple[tuple[str, str, Interval | None], ...]:
        matches = evaluate(expr, bundle)
        return tuple(
            (match.entity[0].id, match.entity[1].id, match.interval) for match in matches.matches
        )

    expected = (
        ("a", "b", Interval(0.0, 6.0)),
        ("a", "b2", Interval(0.0, 8.0)),
        ("a2", "b", Interval(0.5, 6.0)),
        ("a2", "b2", Interval(0.5, 8.0)),
    )
    assert signature() == expected
    assert all(signature() == expected for _ in range(10))
