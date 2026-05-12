# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entity descriptors for the DSL.

Each "entity kind" in the DSL (`clip`, `agent`, `ego`, `env`, `cond`,
`light`, `obj`, plus sub-entities `agent.action`, `agent.prop`,
`ego.action`, `ego.prop`) is described by:

- A `candidates(bundle)` function — yields the schema instances of that
  kind inside a bundle.
- An `attributes` table — DSL attribute name → reader function
  `(entity, bundle) -> value`, plus the alias family the parser should
  consult for value normalization.
- An `interval(entity, bundle)` function — the entity's lifetime, or the
  clip span when no parseable interval is available (per spec §3.1).

The evaluator walks an AST against these descriptors; the parser uses
them for attribute-name validation and "did you mean?" hints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from causal_ai_av.query.constants import (
    ACTION_FLAG_SCHEMA_FIELDS,
    ACTION_FLAG_TOKENS,
)
from causal_ai_av.query.time import Interval
from causal_ai_av.query.spatial import (
    agent_visibility_interval,
    ego_relative_pose_at,
)
from causal_ai_av.spec import (
    Agent,
    AgentAction,
    AgentProperty,
    AnnotationBundle,
    Condition,
    EgoAction,
    EgoVehicle,
    Environment,
    LightStates,
    TrafficObject,
)


# ---------------------------------------------------------------------------
# Attribute descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Attribute:
    """One attribute on an entity kind.

    - `reader(entity, bundle) -> value` — pulls the attribute value.
    - `alias_family` — the family name in `constants.ALIAS_FAMILIES` that
      should resolve DSL value tokens. `None` means the attribute takes
      raw values (numbers, booleans, free strings).
    - `kind` — `"scalar"` (one value), `"list"` (existential match across
      a list), `"sub_entity"` (the attribute is itself an entity kind,
      enabling `<entity>.<sub>(...)` clauses or `<entity>.<sub>.<attr>`
      predicates).
    - `sub_entity` — name of the sub-entity descriptor (only when
      `kind == "sub_entity"`).
    """

    reader: Callable[[Any, AnnotationBundle], Any]
    alias_family: str | None = None
    kind: str = "scalar"
    sub_entity: str | None = None


@dataclass(frozen=True)
class EntityDescriptor:
    """One entity kind (`agent`, `light`, `agent.action`, …)."""

    name: str
    candidates: Callable[[AnnotationBundle], list[Any]]
    interval: Callable[[Any, AnnotationBundle], Interval | None]
    attributes: Mapping[str, Attribute]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip_span(bundle: AnnotationBundle) -> Interval:
    return Interval(0.0, max(0.0, bundle.video.duration_s))


def _interval_from_strings_or_clip(
    obj: Any, bundle: AnnotationBundle,
    start: str = "start_timestamp", end: str = "end_timestamp",
) -> Interval | None:
    """Try to parse `obj.<start>` / `obj.<end>` into an Interval; fall back
    to the clip span if either field is missing or unparseable.
    """
    iv = Interval.from_strings(
        getattr(obj, start, None), getattr(obj, end, None)
    )
    if iv is not None:
        return iv
    return _clip_span(bundle)


def _parenthesized_suffix(type_str: str) -> str | None:
    """Return the lowercased parenthesized suffix of an action type, or None.

    `oxd:Walk (jaywalk, erratic)` -> `"jaywalk, erratic"`.
    """
    if "(" not in type_str or ")" not in type_str:
        return None
    a = type_str.rfind("(")
    b = type_str.rfind(")")
    if b <= a:
        return None
    return type_str[a + 1 : b].lower()


def _suffix_tokens(type_str: str) -> frozenset[str]:
    """Lowercased comma-separated tokens inside the parenthesized suffix."""
    suffix = _parenthesized_suffix(type_str)
    if suffix is None:
        return frozenset()
    return frozenset(t.strip() for t in suffix.split(","))


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def _clip_candidates(bundle: AnnotationBundle) -> list[Any]:
    return [bundle]


def _agent_candidates(bundle: AnnotationBundle) -> list[Any]:
    return list(bundle.annotation.agents)


def _ego_candidates(bundle: AnnotationBundle) -> list[Any]:
    return [bundle.annotation.ego_vehicle]


def _env_candidates(bundle: AnnotationBundle) -> list[Any]:
    return list(bundle.annotation.environments)


def _cond_candidates(bundle: AnnotationBundle) -> list[Any]:
    return list(bundle.annotation.conditions)


def _light_candidates(bundle: AnnotationBundle) -> list[Any]:
    """Flattened: every `LightStates` from every signal head of every
    traffic light. The DSL `light` entity is the user-facing collapsed
    view of `TrafficLight × SignalHead × LightStates` (per QA Q1).
    """
    out: list[LightStates] = []
    for tl in bundle.annotation.traffic_lights:
        for head in tl.signal_heads:
            for state in head.state_sequence:
                # Decorate each LightStates with its owning head for
                # affects_ego lookup; we attach a private attribute.
                # Pydantic models are mutable so this is fine.
                object.__setattr__(state, "_owner_signal_head", head)
                object.__setattr__(state, "_owner_traffic_light", tl)
                out.append(state)
    return out  # type: ignore[return-value]


def _obj_candidates(bundle: AnnotationBundle) -> list[Any]:
    return list(bundle.annotation.traffic_objects)


# ---------------------------------------------------------------------------
# Interval functions
# ---------------------------------------------------------------------------


def _clip_interval(_obj: Any, bundle: AnnotationBundle) -> Interval:
    return _clip_span(bundle)


def _agent_interval(agent: Agent, bundle: AnnotationBundle) -> Interval:
    iv = agent_visibility_interval(agent)
    return iv if iv is not None else _clip_span(bundle)


def _generic_interval(obj: Any, bundle: AnnotationBundle) -> Interval:
    iv = _interval_from_strings_or_clip(obj, bundle)
    return iv if iv is not None else _clip_span(bundle)


def _object_visibility_interval(
    obj: TrafficObject, bundle: AnnotationBundle
) -> Interval:
    iv = Interval.from_strings(
        obj.visibility_start_timestamp, obj.visibility_end_timestamp
    )
    return iv if iv is not None else _clip_span(bundle)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _clip_id(b: AnnotationBundle, _bundle: AnnotationBundle) -> str:
    return b.video.clip_id


def _clip_fps(b: AnnotationBundle, _bundle: AnnotationBundle) -> float:
    return b.video.fps


def _clip_duration(b: AnnotationBundle, _bundle: AnnotationBundle) -> float:
    return b.video.duration_s


def _clip_eventful(b: AnnotationBundle, _bundle: AnnotationBundle) -> bool | None:
    return b.annotation.eventful


def _agent_type(a: Agent, _bundle: AnnotationBundle) -> str:
    return a.type


def _agent_amount(a: Agent, _bundle: AnnotationBundle) -> str:
    return a.amount


def _agent_vis(a: Agent, bundle: AnnotationBundle) -> Interval:
    return _agent_interval(a, bundle)


def _agent_pos(a: Agent, bundle: AnnotationBundle) -> str | None:
    """Position relative to ego, sampled at the midpoint of agent visibility."""
    iv = agent_visibility_interval(a)
    if iv is None:
        # Use any pose's position string.
        for pose in a.ego_relative_pose:
            if pose.position_rel_to_ego:
                return pose.position_rel_to_ego
        return None
    t = (iv.start + iv.end) / 2
    pose = ego_relative_pose_at(a, t)
    if pose is None and a.ego_relative_pose:
        return a.ego_relative_pose[0].position_rel_to_ego
    return pose.position_rel_to_ego if pose else None


def _agent_dir(a: Agent, bundle: AnnotationBundle) -> str | None:
    iv = agent_visibility_interval(a)
    if iv is None:
        for pose in a.ego_relative_pose:
            if pose.direction_rel_to_ego:
                return pose.direction_rel_to_ego
        return None
    t = (iv.start + iv.end) / 2
    pose = ego_relative_pose_at(a, t)
    if pose is None and a.ego_relative_pose:
        return a.ego_relative_pose[0].direction_rel_to_ego
    return pose.direction_rel_to_ego if pose else None


def _agent_in(a: Agent, bundle: AnnotationBundle) -> list[str]:
    """List of environment types this agent is contained in."""
    env_by_id = {e.id: e for e in bundle.annotation.environments}
    out: list[str] = []
    for cont in a.containment:
        env = env_by_id.get(cont.env_id)
        if env is not None and env.type:
            out.append(env.type)
    return out


def _agent_signaling(a: Agent, _bundle: AnnotationBundle) -> list[str]:
    """List of signaling-intent strings from this agent's Signal properties."""
    out: list[str] = []
    for prop in a.properties:
        if prop.property_type == "Signal" and prop.signaling_details is not None:
            intent = prop.signaling_details.intent
            if intent:
                out.append(intent)
    return out


def _ego_judgment(e: EgoVehicle, _bundle: AnnotationBundle) -> str | None:
    return e.driving_judgment


def _ego_in(e: EgoVehicle, bundle: AnnotationBundle) -> list[str]:
    env_by_id = {env.id: env for env in bundle.annotation.environments}
    out: list[str] = []
    for cont in e.containment:
        env = env_by_id.get(cont.env_id)
        if env is not None and env.type:
            out.append(env.type)
    return out


def _env_type(e: Environment, _bundle: AnnotationBundle) -> str:
    return e.type


def _env_lanes(e: Environment, _bundle: AnnotationBundle) -> int | None:
    return e.num_lanes


def _env_out_lanes(e: Environment, _bundle: AnnotationBundle) -> int | None:
    return e.num_out_lanes


def _env_one_way(e: Environment, _bundle: AnnotationBundle) -> bool:
    return e.one_way


def _cond_type(c: Condition, _bundle: AnnotationBundle) -> list[str]:
    return list(c.type)


def _light_color(ls: LightStates, _bundle: AnnotationBundle) -> str | None:
    return ls.color


def _light_state(ls: LightStates, _bundle: AnnotationBundle) -> str | None:
    return ls.type


def _light_shape(ls: LightStates, _bundle: AnnotationBundle) -> str | None:
    return ls.shape


def _light_on_ego_path(ls: LightStates, _bundle: AnnotationBundle) -> bool | None:
    return ls.yellow_on_ego_path


def _light_ego_in_on_yellow(
    ls: LightStates, _bundle: AnnotationBundle
) -> bool | None:
    return ls.ego_in_intersection_on_yellow


def _light_could_have_cleared(
    ls: LightStates, _bundle: AnnotationBundle
) -> bool | None:
    return ls.ego_could_have_cleared_safely


def _light_affects_ego(ls: LightStates, _bundle: AnnotationBundle) -> bool | None:
    head = getattr(ls, "_owner_signal_head", None)
    if head is None:
        return None
    raw = head.affects_ego
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    return None


def _obj_type(o: TrafficObject, _bundle: AnnotationBundle) -> str:
    return o.type


def _obj_state(o: TrafficObject, _bundle: AnnotationBundle) -> list[str]:
    """Existential over the object's state sequence: motion_state strings."""
    return [s.motion_state for s in o.state_sequence if s.motion_state]


def _obj_open(o: TrafficObject, _bundle: AnnotationBundle) -> list[str]:
    return [s.open_state for s in o.state_sequence if s.open_state]


# ---------------------------------------------------------------------------
# Sub-entity readers: action / property
# ---------------------------------------------------------------------------


def _agent_actions(a: Agent, _bundle: AnnotationBundle) -> list[AgentAction]:
    return list(a.actions)


def _agent_props(a: Agent, _bundle: AnnotationBundle) -> list[AgentProperty]:
    return list(a.properties)


def _ego_actions(e: EgoVehicle, _bundle: AnnotationBundle) -> list[EgoAction]:
    return list(e.actions)


def _ego_props(e: EgoVehicle, _bundle: AnnotationBundle) -> list[AgentProperty]:
    return list(e.properties)


def _action_type_attr(action: AgentAction | EgoAction, _bundle: AnnotationBundle) -> str:
    if isinstance(action, AgentAction):
        return action.action_type
    return action.type


def _action_flag_reader(flag: str) -> Callable[[Any, AnnotationBundle], bool]:
    """Reader for a parenthesized-suffix flag like `jaywalk` on actions.

    Returns True if (a) the schema flag field is True, OR (b) the
    parenthesized-suffix of the type string contains the flag token.
    """
    schema_field = ACTION_FLAG_SCHEMA_FIELDS.get(flag)

    def _read(action: Any, _bundle: AnnotationBundle) -> bool:
        type_str = _action_type_attr(action, _bundle)
        if schema_field is not None:
            val = getattr(action, schema_field, None)
            if val is True:
                return True
        tokens = _suffix_tokens(type_str)
        # Suffix tokens are e.g. {"jaywalk"}, {"out of lane: into ego lane"}.
        # Match by membership or by canonicalized DSL token name.
        dsl_token = flag.replace("_", " ")
        if dsl_token in tokens or flag in tokens:
            return True
        # Special-case: some action type strings carry suffixes like
        # `(out of lane: into ego lane)` — keep loose containment.
        for tok in tokens:
            if dsl_token in tok:
                return True
        return False

    return _read


def _prop_type_attr(p: AgentProperty, _bundle: AnnotationBundle) -> str:
    return p.property_type


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# Build the sub-entity descriptors first; they're referenced by the
# parent descriptors via name.

_action_flag_attrs: dict[str, Attribute] = {}
for _flag in ACTION_FLAG_SCHEMA_FIELDS:
    _action_flag_attrs[_flag] = Attribute(
        reader=_action_flag_reader(_flag), alias_family=None, kind="scalar"
    )


def _action_interval_reader(a: AgentAction | EgoAction, bundle: AnnotationBundle) -> Interval:
    return _generic_interval(a, bundle)


def _prop_interval_reader(p: AgentProperty, bundle: AnnotationBundle) -> Interval:
    return _generic_interval(p, bundle)


AGENT_ACTION_DESCRIPTOR = EntityDescriptor(
    name="agent.action",
    candidates=lambda b: [a for ag in b.annotation.agents for a in ag.actions],
    interval=_action_interval_reader,
    attributes={
        "type": Attribute(reader=_action_type_attr, alias_family="action_type"),
        **_action_flag_attrs,
    },
)


EGO_ACTION_DESCRIPTOR = EntityDescriptor(
    name="ego.action",
    candidates=lambda b: list(b.annotation.ego_vehicle.actions),
    interval=_action_interval_reader,
    attributes={
        "type": Attribute(reader=_action_type_attr, alias_family="action_type"),
        **_action_flag_attrs,
    },
)


AGENT_PROP_DESCRIPTOR = EntityDescriptor(
    name="agent.prop",
    candidates=lambda b: [p for ag in b.annotation.agents for p in ag.properties],
    interval=_prop_interval_reader,
    attributes={
        "type": Attribute(reader=_prop_type_attr, alias_family="agent_property_type"),
    },
)


EGO_PROP_DESCRIPTOR = EntityDescriptor(
    name="ego.prop",
    candidates=lambda b: list(b.annotation.ego_vehicle.properties),
    interval=_prop_interval_reader,
    attributes={
        "type": Attribute(reader=_prop_type_attr, alias_family="ego_property_type"),
    },
)


CLIP_DESCRIPTOR = EntityDescriptor(
    name="clip",
    candidates=_clip_candidates,
    interval=_clip_interval,
    attributes={
        "id": Attribute(reader=_clip_id),
        "fps": Attribute(reader=_clip_fps),
        "duration": Attribute(reader=_clip_duration),
        "eventful": Attribute(reader=_clip_eventful),
    },
)


AGENT_DESCRIPTOR = EntityDescriptor(
    name="agent",
    candidates=_agent_candidates,
    interval=_agent_interval,
    attributes={
        "type": Attribute(reader=_agent_type, alias_family="agent_type"),
        "amount": Attribute(reader=_agent_amount),
        "vis": Attribute(reader=_agent_vis),
        "pos": Attribute(reader=_agent_pos, alias_family="position"),
        "dir": Attribute(reader=_agent_dir, alias_family="direction"),
        "in": Attribute(reader=_agent_in, alias_family="env_type", kind="list"),
        "signaling": Attribute(
            reader=_agent_signaling, alias_family="signaling_intent", kind="list"
        ),
        "action": Attribute(
            reader=_agent_actions, kind="sub_entity", sub_entity="agent.action"
        ),
        "prop": Attribute(
            reader=_agent_props, kind="sub_entity", sub_entity="agent.prop"
        ),
    },
)


EGO_DESCRIPTOR = EntityDescriptor(
    name="ego",
    candidates=_ego_candidates,
    interval=_clip_interval,
    attributes={
        "action": Attribute(
            reader=_ego_actions, kind="sub_entity", sub_entity="ego.action"
        ),
        "prop": Attribute(
            reader=_ego_props, kind="sub_entity", sub_entity="ego.prop"
        ),
        "judgment": Attribute(reader=_ego_judgment, alias_family="driving_judgment"),
        "in": Attribute(reader=_ego_in, alias_family="env_type", kind="list"),
    },
)


ENV_DESCRIPTOR = EntityDescriptor(
    name="env",
    candidates=_env_candidates,
    interval=_generic_interval,
    attributes={
        "type": Attribute(reader=_env_type, alias_family="env_type"),
        "lanes": Attribute(reader=_env_lanes),
        "out_lanes": Attribute(reader=_env_out_lanes),
        "one_way": Attribute(reader=_env_one_way),
    },
)


COND_DESCRIPTOR = EntityDescriptor(
    name="cond",
    candidates=_cond_candidates,
    interval=_generic_interval,
    attributes={
        "type": Attribute(reader=_cond_type, alias_family="cond_type", kind="list"),
    },
)


LIGHT_DESCRIPTOR = EntityDescriptor(
    name="light",
    candidates=_light_candidates,
    interval=_generic_interval,
    attributes={
        "color": Attribute(reader=_light_color, alias_family="light_color"),
        "state": Attribute(reader=_light_state, alias_family="light_state"),
        "shape": Attribute(reader=_light_shape, alias_family="light_shape"),
        "on_ego_path": Attribute(reader=_light_on_ego_path),
        "ego_in_on_yellow": Attribute(reader=_light_ego_in_on_yellow),
        "could_have_cleared": Attribute(reader=_light_could_have_cleared),
        "affects_ego": Attribute(reader=_light_affects_ego),
    },
)


OBJ_DESCRIPTOR = EntityDescriptor(
    name="obj",
    candidates=_obj_candidates,
    interval=_object_visibility_interval,
    attributes={
        "type": Attribute(reader=_obj_type, alias_family="obj_type"),
        "state": Attribute(
            reader=_obj_state, alias_family="obj_motion_state", kind="list"
        ),
        "open": Attribute(
            reader=_obj_open, alias_family="obj_open_state", kind="list"
        ),
    },
)


ENTITIES: Mapping[str, EntityDescriptor] = {
    "clip": CLIP_DESCRIPTOR,
    "agent": AGENT_DESCRIPTOR,
    "ego": EGO_DESCRIPTOR,
    "env": ENV_DESCRIPTOR,
    "cond": COND_DESCRIPTOR,
    "light": LIGHT_DESCRIPTOR,
    "obj": OBJ_DESCRIPTOR,
    "agent.action": AGENT_ACTION_DESCRIPTOR,
    "agent.prop": AGENT_PROP_DESCRIPTOR,
    "ego.action": EGO_ACTION_DESCRIPTOR,
    "ego.prop": EGO_PROP_DESCRIPTOR,
}


# Top-level entities the parser exposes (without the "agent.action" forms).
TOP_LEVEL_ENTITIES: tuple[str, ...] = (
    "clip", "agent", "ego", "env", "cond", "light", "obj",
)


def get_entity(name: str) -> EntityDescriptor:
    """Return the descriptor for `name`. Raises KeyError if unknown."""
    return ENTITIES[name]


def candidate_entity_names() -> list[str]:
    """Names a parser-level "did you mean?" search should look at."""
    return list(TOP_LEVEL_ENTITIES)


def candidate_attribute_names(entity: str) -> list[str]:
    """Attribute names defined on `entity`. For "did you mean?" hints."""
    desc = ENTITIES.get(entity)
    return list(desc.attributes.keys()) if desc else []


# ---------------------------------------------------------------------------
# Action type matching — public helper (used by engine)
# ---------------------------------------------------------------------------


def action_type_matches(action_type_str: str, candidates: frozenset[str]) -> bool:
    """Match `action_type_str` against `candidates` using the corpus's
    base-verb-with-suffix convention.

    A candidate `oxd:Walk` matches any string equal to `"oxd:Walk"` or
    starting with `"oxd:Walk ("`. An exact-literal candidate
    `oxd:ChangeLane (left)` matches only the exact string.
    """
    if not action_type_str:
        return False
    for c in candidates:
        if action_type_str == c:
            return True
        # If the candidate is a bare base form (no parens), allow prefix
        # match with " (".
        if "(" not in c and action_type_str.startswith(c + " ("):
            return True
    return False


__all__ = [
    "Attribute",
    "EntityDescriptor",
    "ENTITIES",
    "TOP_LEVEL_ENTITIES",
    "get_entity",
    "candidate_entity_names",
    "candidate_attribute_names",
    "action_type_matches",
    "ACTION_FLAG_TOKENS",
]
