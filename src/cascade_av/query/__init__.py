# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Query primitives over `AnnotationBundle`.

- `time`      — timestamp parsing / `Interval`
- `index`     — `IdIndex`: by-ID lookup of every addressable entity in a bundle
- `triplets`  — causal triplet extraction (`(subject, predicate, cause)` from `because_of`)
- `temporal`  — action-at-time, action overlaps, interval predicates
- `spatial`   — agent visibility, ego-relative pose, agents-in-position
- `constants` — DSL value aliases, grounded against the corpus
- `entities`  — DSL entity descriptors
- `dsl`       — DSL lexer + parser
- `engine`    — AST evaluator → MatchSet
- `api`       — thin wrappers (`find_on_bundle`, `find_on_dataset`, …)
"""

from cascade_av.query.constants import (
    ActionKind,
    AgentKind,
    EnvKind,
    LightColor,
    Position,
    resolve_alias,
)
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
    parse,
)
from cascade_av.query.api import (
    count_on_dataset,
    find_on_bundle,
    find_on_dataset,
    group_by_on_dataset,
)
from cascade_av.query.context import (
    AgentInWindow,
    ContextWindow,
    LightStateInWindow,
    context_at,
)
from cascade_av.query.engine import IDLESS_RECORD_ID, Match, MatchSet, evaluate
from cascade_av.query.index import IdIndex, Subject
from cascade_av.query.spatial import (
    agent_visibility_interval,
    agents_in_position,
    agents_visible_at,
    ego_relative_pose_at,
)
from cascade_av.query.temporal import (
    action_interval,
    actions_at,
    filter_active_at,
    filter_active_in_range,
    overlapping_actions,
)
from cascade_av.query.time import (
    Interval,
    format_timestamp,
    parse_timestamp,
    parse_timestamp_or,
)
from cascade_av.query.triplets import CausalTriplet, extract_causal_triplets, iter_triplets

__all__ = [
    "Interval",
    "parse_timestamp",
    "parse_timestamp_or",
    "format_timestamp",
    "IdIndex",
    "Subject",
    "CausalTriplet",
    "extract_causal_triplets",
    "iter_triplets",
    "action_interval",
    "actions_at",
    "filter_active_at",
    "filter_active_in_range",
    "overlapping_actions",
    "agent_visibility_interval",
    "agents_visible_at",
    "ego_relative_pose_at",
    "agents_in_position",
    # DSL surface
    "parse",
    "evaluate",
    "find_on_bundle",
    "find_on_dataset",
    "count_on_dataset",
    "group_by_on_dataset",
    "Match",
    "MatchSet",
    "IDLESS_RECORD_ID",
    "QueryParseError",
    "AttrPredicate",
    "AttrRef",
    "EntityClause",
    "EntityRef",
    "And",
    "Or",
    "Not",
    "While",
    "WhileStrict",
    "Then",
    "Before",
    "BecauseOf",
    "InfluencedBy",
    "Within",
    "Expr",
    # Context windows
    "AgentInWindow",
    "ContextWindow",
    "LightStateInWindow",
    "context_at",
    "resolve_alias",
    "AgentKind",
    "ActionKind",
    "EnvKind",
    "LightColor",
    "Position",
]
