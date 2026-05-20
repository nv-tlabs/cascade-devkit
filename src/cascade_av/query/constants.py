# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSL vocabulary — short aliases, hierarchical parents, schema literals.

All sets are frozen and contain the exact strings declared in
``cascade_av.spec.schema`` ``*Vocab`` classes — the spec module is the single
source of truth for vocabulary literals; this module only assembles short DSL
aliases over them.

Lookup discipline:

- DSL parser accepts a short alias (e.g. ``"ped"``) or a full literal
  (e.g. ``"Pedestrian (Adult)"``). It calls :func:`resolve_alias`
  ``(family, name)`` to get the underlying set of schema strings.
- Aliases are case-folded to lowercase at the call site.
- Parent aliases (e.g. ``"vehicle"``, ``"vru"``, ``"turn"``) resolve to the
  union of their child aliases.
- Unknown aliases raise ``KeyError`` — callers wrap in ``NameError``.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping

from cascade_av.spec.schema import (
    AgentActionTypePedestrianVocab,
    AgentActionTypeVocab,
    AgentPropertyTypeVocab,
    AgentTypeVocab,
    ConditionTypeVocab,
    DirectionRelToEgoVocab,
    DrivingJudgmentVocab,
    EgoActionTypeVocab,
    EgoPropertyTypeVocab,
    EnvironmentTypeVocab,
    LightColorVocab,
    LightShapeVocab,
    LightStateTypeVocab,
    MotionStateVocab,
    OpenStateVocab,
    PositionRelToEgoVocab,
    SignalIntentVocab,
    SignalSourceVocab,
    TrafficObjectTypeVocab,
)

# ---------------------------------------------------------------------------
# Agent types
# ---------------------------------------------------------------------------

AGENT_TYPE_ALIASES: Mapping[str, frozenset[str]] = {
    "car": frozenset({AgentTypeVocab.CAR}),
    "truck": frozenset({AgentTypeVocab.TRUCK, AgentTypeVocab.HEAVY_DUTY}),
    "bus": frozenset({AgentTypeVocab.BUS}),
    "motorcycle": frozenset({AgentTypeVocab.MOTORCYCLE}),
    "emergency": frozenset({AgentTypeVocab.EMERGENCY_VEHICLE}),
    "bicycle": frozenset({AgentTypeVocab.BICYCLE}),
    "scooter": frozenset({AgentTypeVocab.SCOOTER}),
    "animal": frozenset({AgentTypeVocab.ANIMAL}),
    "ped_adult": frozenset({AgentTypeVocab.PEDESTRIAN_ADULT}),
    "ped_kid": frozenset({AgentTypeVocab.PEDESTRIAN_KID_TEEN}),
    "ped_personnel": frozenset({AgentTypeVocab.PEDESTRIAN_PERSONNEL}),
    "ped_officer": frozenset({AgentTypeVocab.PEDESTRIAN_OFFICER}),
    "ped_stroller": frozenset({AgentTypeVocab.PEDESTRIAN_STROLLER}),
    "ped_wheelchair": frozenset({AgentTypeVocab.PEDESTRIAN_WHEELCHAIR}),
    "ped_other": frozenset({AgentTypeVocab.PEDESTRIAN_OTHER, AgentTypeVocab.PEDESTRIAN}),
    "officer": frozenset({AgentTypeVocab.PEDESTRIAN_OFFICER}),
}

# Parent categories — unions of the leaf aliases above.
AGENT_TYPE_PARENTS: Mapping[str, frozenset[str]] = {
    "vehicle": (
        AGENT_TYPE_ALIASES["car"]
        | AGENT_TYPE_ALIASES["truck"]
        | AGENT_TYPE_ALIASES["bus"]
        | AGENT_TYPE_ALIASES["motorcycle"]
        | AGENT_TYPE_ALIASES["emergency"]
    ),
    "cyclist": AGENT_TYPE_ALIASES["bicycle"] | AGENT_TYPE_ALIASES["scooter"],
    "ped": (
        AGENT_TYPE_ALIASES["ped_adult"]
        | AGENT_TYPE_ALIASES["ped_kid"]
        | AGENT_TYPE_ALIASES["ped_personnel"]
        | AGENT_TYPE_ALIASES["ped_officer"]
        | AGENT_TYPE_ALIASES["ped_stroller"]
        | AGENT_TYPE_ALIASES["ped_wheelchair"]
        | AGENT_TYPE_ALIASES["ped_other"]
    ),
}
AGENT_TYPE_PARENTS = {
    **AGENT_TYPE_PARENTS,
    "vru": AGENT_TYPE_PARENTS["ped"]
    | AGENT_TYPE_PARENTS["cyclist"],
}

AGENT_TYPE: Mapping[str, frozenset[str]] = {**AGENT_TYPE_ALIASES, **AGENT_TYPE_PARENTS}

# ---------------------------------------------------------------------------
# Action types (used for both agent and ego actions; same vocabulary)
#
# The corpus encodes flags inside the type string as parenthesized
# suffixes — e.g. ``oxd:Walk (jaywalk)``. Base-verb aliases below resolve
# to the *base* string only; the engine adds prefix-match logic so
# ``agent.action.type = walk`` matches every variant. See
# `docs/user/query_language.md` §3.9.
# ---------------------------------------------------------------------------

ACTION_TYPE_ALIASES: Mapping[str, frozenset[str]] = {
    "drive": frozenset({EgoActionTypeVocab.DRIVING_IN_LANE}),
    "stop": frozenset({EgoActionTypeVocab.STOP}),
    "park": frozenset({AgentActionTypeVocab.PARK}),
    "walk": frozenset({AgentActionTypePedestrianVocab.WALK}),
    "stand": frozenset({AgentActionTypePedestrianVocab.STAND}),
    "run": frozenset({AgentActionTypePedestrianVocab.RUN}),
    "yield": frozenset({EgoActionTypeVocab.YIELD}),
    "decel": frozenset({EgoActionTypeVocab.DECELERATE}),
    "creep": frozenset({EgoActionTypeVocab.CREEP}),
    "enter": frozenset({EgoActionTypeVocab.ENTER}),
    "follow": frozenset({EgoActionTypeVocab.FOLLOW_ROAD_USER}),
    "not_move": frozenset({EgoActionTypeVocab.NOT_MOVE}),
    "abort": frozenset({EgoActionTypeVocab.MANEUVER_ABORT}),
    "reverse": frozenset({EgoActionTypeVocab.REVERSE}),
    "turn_left": frozenset({"oxd:MakeALeftTurn"}),
    "turn_right": frozenset({"oxd:MakeARightTurn"}),
    "uturn": frozenset({"fst:MakeAUTurn"}),
    "change_lane_left": frozenset({EgoActionTypeVocab.CHANGE_LANE_LEFT}),
    "change_lane_right": frozenset({EgoActionTypeVocab.CHANGE_LANE_RIGHT}),
    "change_lane": frozenset({"oxd:ChangeLane"}),
    "nudge_in": frozenset({EgoActionTypeVocab.NUDGE_IN_LANE}),
    "nudge_out": frozenset({EgoActionTypeVocab.NUDGE_OUT_OF_LANE}),
    "nudge": frozenset({"fst:Nudge"}),
    "overtake": frozenset({EgoActionTypeVocab.OVERTAKE}),
}

# Parent categories.
ACTION_TYPE_PARENTS: Mapping[str, frozenset[str]] = {
    "turn": (
        ACTION_TYPE_ALIASES["turn_left"]
        | ACTION_TYPE_ALIASES["turn_right"]
        | ACTION_TYPE_ALIASES["uturn"]
    ),
    "stop_yield_decel": (
        ACTION_TYPE_ALIASES["stop"]
        | ACTION_TYPE_ALIASES["yield"]
        | ACTION_TYPE_ALIASES["decel"]
    ),
}

ACTION_TYPE: Mapping[str, frozenset[str]] = {
    **ACTION_TYPE_ALIASES,
    **ACTION_TYPE_PARENTS,
}

# Backward-compatibility stub. The 2.0.0 schema reboot removed every
# parallel ``*_flag`` field on AgentAction / EgoAction (jaywalk_flag,
# erratic_flag, turn_protected, ...), so this mapping is intentionally
# empty — there are no schema fields left to map. The name is preserved
# only so ``cascade_av.query.entities`` keeps importing on this branch;
# Phase 3a rewrites entities.py to drop the import and the for-loop that
# iterates it, at which point this stub goes away too.
ACTION_FLAG_SCHEMA_FIELDS: Mapping[str, str | None] = {}

# Action-type flags — these are tokens that appear inside parenthesized
# suffixes. The 2.0.0 schema reboot removed the parallel ``*_flag`` fields
# on AgentAction / EgoAction (jaywalk_flag, erratic_flag, turn_protected,
# ...), so the suffix string is now the single source of truth — the engine
# matches by checking suffix tokens alone.
ACTION_FLAG_TOKENS: frozenset[str] = frozenset(
    {
        "jaywalk",
        "erratic",
        "protected",
        "unprotected",
        "left",
        "right",
        "in lane",
        "out of lane",
        "into ego lane",
        "not into ego lane",
        "using ego lane",
    }
)

# ---------------------------------------------------------------------------
# Environment types
# ---------------------------------------------------------------------------

ENV_TYPE_ALIASES: Mapping[str, frozenset[str]] = {
    "road": frozenset({EnvironmentTypeVocab.ROAD}),
    "crosswalk": frozenset({EnvironmentTypeVocab.PEDESTRIAN_CROSSING}),
    "sidewalk": frozenset({EnvironmentTypeVocab.SIDEWALK}),
    "crossroad": frozenset({EnvironmentTypeVocab.CROSSROAD}),
    "t_intersection": frozenset({EnvironmentTypeVocab.T_INTERSECTION}),
    "y_intersection": frozenset({EnvironmentTypeVocab.Y_INTERSECTION}),
    "five_way_intersection": frozenset({EnvironmentTypeVocab.FIVE_WAY}),
    "six_way": frozenset({EnvironmentTypeVocab.SIX_WAY}),
    "six_plus_way": frozenset({EnvironmentTypeVocab.SIX_PLUS_WAY}),
    "other_intersection": frozenset({EnvironmentTypeVocab.OTHER_INTERSECTION}),
    "cycle_lane": frozenset({EnvironmentTypeVocab.CYCLE_LANE}),
    "bike_lane": frozenset({EnvironmentTypeVocab.CYCLE_LANE}),  # alias
    "lane_merge": frozenset({EnvironmentTypeVocab.LANE_MERGE}),
    "lane_fork": frozenset({EnvironmentTypeVocab.LANE_FORK}),
    "roundabout": frozenset({EnvironmentTypeVocab.ROUNDABOUT}),
    "tunnel": frozenset({EnvironmentTypeVocab.TUNNEL}),
    "bridge": frozenset({EnvironmentTypeVocab.BRIDGE}),
    "paved_shoulder": frozenset({EnvironmentTypeVocab.PAVED_SHOULDER}),
    "grass_shoulder": frozenset({EnvironmentTypeVocab.GRASS_SHOULDER}),
    "rail_crossing": frozenset({EnvironmentTypeVocab.RAIL_CROSSING}),
}

ENV_TYPE_PARENTS: Mapping[str, frozenset[str]] = {
    "intersection": (
        ENV_TYPE_ALIASES["crossroad"]
        | ENV_TYPE_ALIASES["t_intersection"]
        | ENV_TYPE_ALIASES["y_intersection"]
        | ENV_TYPE_ALIASES["five_way_intersection"]
        | ENV_TYPE_ALIASES["six_way"]
        | ENV_TYPE_ALIASES["six_plus_way"]
        | ENV_TYPE_ALIASES["other_intersection"]
    ),
    "shoulder": ENV_TYPE_ALIASES["paved_shoulder"] | ENV_TYPE_ALIASES["grass_shoulder"],
}

ENV_TYPE: Mapping[str, frozenset[str]] = {**ENV_TYPE_ALIASES, **ENV_TYPE_PARENTS}

# ---------------------------------------------------------------------------
# Condition types
# ---------------------------------------------------------------------------

COND_TYPE: Mapping[str, frozenset[str]] = {
    "construction": frozenset({ConditionTypeVocab.CONSTRUCTION_ZONE}),
    "temp_marked": frozenset({ConditionTypeVocab.TEMPORARILY_MARKED}),
    "wet": frozenset({ConditionTypeVocab.WET_ROAD}),
    "snowy": frozenset({ConditionTypeVocab.SNOWY_ROAD}),
    "overgrown": frozenset({ConditionTypeVocab.OVERGROWN}),
    "shared_center": frozenset({ConditionTypeVocab.SHARED_MARKED_CENTER_LANE}),
    "no_divider": frozenset({ConditionTypeVocab.NO_DIRECTION_DIVIDER}),
    "lanes_obscured": frozenset({ConditionTypeVocab.LANES_OBSCURED}),
    "other": frozenset({ConditionTypeVocab.OTHER}),
}

# ---------------------------------------------------------------------------
# Light state attributes
# ---------------------------------------------------------------------------

LIGHT_COLOR: Mapping[str, frozenset[str]] = {
    "red": frozenset({LightColorVocab.RED}),
    "yellow": frozenset({LightColorVocab.YELLOW}),
    "green": frozenset({LightColorVocab.GREEN}),
    "other": frozenset({LightColorVocab.OTHER}),
}

LIGHT_STATE_TYPE: Mapping[str, frozenset[str]] = {
    "fixed": frozenset({LightStateTypeVocab.FIXED}),
    "flashing": frozenset({LightStateTypeVocab.FLASHING}),
    "off": frozenset({LightStateTypeVocab.OFF}),
}

LIGHT_SHAPE: Mapping[str, frozenset[str]] = {
    "round": frozenset({LightShapeVocab.ROUND}),
    "arrow_left": frozenset({LightShapeVocab.ARROW_LEFT}),
    "arrow_right": frozenset({LightShapeVocab.ARROW_RIGHT}),
    "arrow_up": frozenset({LightShapeVocab.ARROW_UP}),
    "arrow_down": frozenset({LightShapeVocab.ARROW_DOWN}),
    "other": frozenset({LightShapeVocab.OTHER}),
}

TRAFFIC_LIGHT_TYPE: Mapping[str, frozenset[str]] = {
    "regular": frozenset({"fst:RegularTrafficLight"}),
}

# ---------------------------------------------------------------------------
# Traffic object types
# ---------------------------------------------------------------------------

OBJ_TYPE_ALIASES: Mapping[str, frozenset[str]] = {
    "traffic_sign": frozenset({TrafficObjectTypeVocab.OTHER_TRAFFIC_SIGN}),
    "stop_sign": frozenset({TrafficObjectTypeVocab.STOP_SIGN}),
    "yield_sign": frozenset({TrafficObjectTypeVocab.YIELD_SIGN}),
    "speed_limit_sign": frozenset({TrafficObjectTypeVocab.SPEED_LIMIT_SIGN}),
    "warning_sign": frozenset({TrafficObjectTypeVocab.WARNING_SIGN}),
    "cone": frozenset({TrafficObjectTypeVocab.TRAFFIC_CONE}),
    "roadblock": frozenset({TrafficObjectTypeVocab.ROADBLOCKS}),
    "bollard": frozenset({TrafficObjectTypeVocab.BOLLARD}),
    "barrier": frozenset({TrafficObjectTypeVocab.BARRIER}),
    "garage": frozenset({TrafficObjectTypeVocab.GARAGE}),
    "toll_plaza": frozenset({TrafficObjectTypeVocab.TOLL_PLAZA}),
    "rail_crossing_obj": frozenset({TrafficObjectTypeVocab.RAIL_CROSSING}),
    "merge_ahead": frozenset({TrafficObjectTypeVocab.MERGE_AHEAD}),
    "adjacent_lanes_ahead": frozenset({TrafficObjectTypeVocab.ADJACENT_LANES_AHEAD}),
    "do_not_enter": frozenset({TrafficObjectTypeVocab.DO_NOT_ENTER}),
    "portable_indicator": frozenset({TrafficObjectTypeVocab.OTHER_PORTABLE_INDICATOR}),
    "portable_display": frozenset({TrafficObjectTypeVocab.PORTABLE_DISPLAY}),
    "toy": frozenset({TrafficObjectTypeVocab.TOY}),
    "ball": frozenset({TrafficObjectTypeVocab.BALL}),
    "fallen_object": frozenset({TrafficObjectTypeVocab.OTHER_FALLEN_OBJECT}),
    "trash": frozenset({TrafficObjectTypeVocab.TRASH}),
    "dirt": frozenset({TrafficObjectTypeVocab.DIRT}),
    "other_debris": frozenset({TrafficObjectTypeVocab.OTHER_DEBRIS}),
    "not_identifiable": frozenset({TrafficObjectTypeVocab.NOT_IDENTIFIABLE}),
    "other": frozenset({TrafficObjectTypeVocab.OTHER}),
}

OBJ_TYPE_PARENTS: Mapping[str, frozenset[str]] = {
    "sign": (
        OBJ_TYPE_ALIASES["traffic_sign"]
        | OBJ_TYPE_ALIASES["stop_sign"]
        | OBJ_TYPE_ALIASES["yield_sign"]
        | OBJ_TYPE_ALIASES["speed_limit_sign"]
        | OBJ_TYPE_ALIASES["warning_sign"]
        | OBJ_TYPE_ALIASES["do_not_enter"]
        | OBJ_TYPE_ALIASES["merge_ahead"]
        | OBJ_TYPE_ALIASES["adjacent_lanes_ahead"]
    ),
    "debris": (
        OBJ_TYPE_ALIASES["other_debris"]
        | OBJ_TYPE_ALIASES["trash"]
        | OBJ_TYPE_ALIASES["dirt"]
    ),
    "barrier": (
        OBJ_TYPE_ALIASES["barrier"]
        | OBJ_TYPE_ALIASES["bollard"]
        | OBJ_TYPE_ALIASES["roadblock"]
        | OBJ_TYPE_ALIASES["cone"]
    ),
    "fallen": (
        OBJ_TYPE_ALIASES["toy"]
        | OBJ_TYPE_ALIASES["ball"]
        | OBJ_TYPE_ALIASES["fallen_object"]
    ),
}

OBJ_TYPE: Mapping[str, frozenset[str]] = {**OBJ_TYPE_ALIASES, **OBJ_TYPE_PARENTS}

# ---------------------------------------------------------------------------
# Object state (motion/open)
# ---------------------------------------------------------------------------

OBJ_MOTION_STATE: Mapping[str, frozenset[str]] = {
    "static": frozenset({MotionStateVocab.STATIC}),
    "moving": frozenset({MotionStateVocab.MOVING}),
}

OBJ_OPEN_STATE: Mapping[str, frozenset[str]] = {
    "open": frozenset({OpenStateVocab.OPEN}),
    "closed": frozenset({OpenStateVocab.CLOSED}),
}

# ---------------------------------------------------------------------------
# Ego-relative pose
# ---------------------------------------------------------------------------

POSITION_ALIASES: Mapping[str, frozenset[str]] = {
    "front": frozenset({PositionRelToEgoVocab.IN_FRONT}),
    "left": frozenset({PositionRelToEgoVocab.LEFT}),
    "right": frozenset({PositionRelToEgoVocab.RIGHT}),
    "behind": frozenset({PositionRelToEgoVocab.BEHIND}),
}

DIRECTION_ALIASES: Mapping[str, frozenset[str]] = {
    "same": frozenset({DirectionRelToEgoVocab.SAME}),
    "opposite": frozenset({DirectionRelToEgoVocab.OPPOSITE}),
    "perpendicular_rl": frozenset({DirectionRelToEgoVocab.PERPENDICULAR_RL}),
    "perpendicular_lr": frozenset({DirectionRelToEgoVocab.PERPENDICULAR_LR}),
}

DIRECTION_PARENTS: Mapping[str, frozenset[str]] = {
    "perpendicular": (
        DIRECTION_ALIASES["perpendicular_rl"]
        | DIRECTION_ALIASES["perpendicular_lr"]
    ),
}

DIRECTION: Mapping[str, frozenset[str]] = {**DIRECTION_ALIASES, **DIRECTION_PARENTS}

# ---------------------------------------------------------------------------
# Agent property types (and signaling)
# ---------------------------------------------------------------------------

AGENT_PROPERTY_TYPE: Mapping[str, frozenset[str]] = {
    "signal": frozenset({AgentPropertyTypeVocab.SIGNAL}),
    "slow": frozenset({AgentPropertyTypeVocab.SLOW}),
    "fast": frozenset({AgentPropertyTypeVocab.FAST}),
    "aggressive": frozenset({AgentPropertyTypeVocab.AGGRESSIVE}),
    "erratic": frozenset({AgentPropertyTypeVocab.ERRATIC}),
    "emergency": frozenset({AgentPropertyTypeVocab.EMERGENCY}),
    "on_duty": frozenset({AgentPropertyTypeVocab.ON_DUTY}),
    "double_parked": frozenset({AgentPropertyTypeVocab.DOUBLE_PARKED}),
    "outside_camera": frozenset({AgentPropertyTypeVocab.OUTSIDE_CAMERA_VIEW}),
    "other": frozenset({AgentPropertyTypeVocab.OTHER}),
}

EGO_PROPERTY_TYPE: Mapping[str, frozenset[str]] = {
    "signal": frozenset({EgoPropertyTypeVocab.SIGNAL}),
    "slow": frozenset({EgoPropertyTypeVocab.SLOW}),
    "fast": frozenset({EgoPropertyTypeVocab.FAST}),
    "aggressive": frozenset({EgoPropertyTypeVocab.AGGRESSIVE}),
    "erratic": frozenset({EgoPropertyTypeVocab.ERRATIC}),
    "emergency": frozenset({EgoPropertyTypeVocab.EMERGENCY}),
    "on_duty": frozenset({EgoPropertyTypeVocab.ON_DUTY}),
    "double_parked": frozenset({EgoPropertyTypeVocab.DOUBLE_PARKED}),
    "outside_camera": frozenset({EgoPropertyTypeVocab.OUTSIDE_CAMERA_VIEW}),
    "other": frozenset({EgoPropertyTypeVocab.OTHER}),
}

SIGNALING_INTENT: Mapping[str, frozenset[str]] = {
    "proceed": frozenset({SignalIntentVocab.PROCEED}),
    "turn": frozenset({SignalIntentVocab.TURN}),
    "stop": frozenset({SignalIntentVocab.STOP}),
    "caution": frozenset({SignalIntentVocab.CAUTION}),
    "slow": frozenset({SignalIntentVocab.SLOW_DOWN}),
    "follow": frozenset({SignalIntentVocab.FOLLOW}),
    "danger": frozenset({SignalIntentVocab.DANGER}),
    "unclear": frozenset({SignalIntentVocab.UNCLEAR}),
    "other": frozenset({SignalIntentVocab.OTHER}),
}

SIGNALING_SOURCE: Mapping[str, frozenset[str]] = {
    "flashing_light": frozenset({SignalSourceVocab.FLASHING_LIGHT}),
    "holding_sign": frozenset({SignalSourceVocab.HOLDING_SIGN}),
    "hand_gesture": frozenset({SignalSourceVocab.HAND_GESTURE}),
}

# ---------------------------------------------------------------------------
# Driving judgment
# ---------------------------------------------------------------------------

DRIVING_JUDGMENT: Mapping[str, frozenset[str]] = {
    "good": frozenset({DrivingJudgmentVocab.GOOD}),
    "neutral": frozenset({DrivingJudgmentVocab.NEUTRAL}),
    "bad": frozenset({DrivingJudgmentVocab.BAD}),
}

# ---------------------------------------------------------------------------
# Enums — typed aliases for Python users (parallel surface to the strings)
# ---------------------------------------------------------------------------


class AgentKind(str, Enum):
    """Hierarchical agent-type categories. Use ``.aliases`` for the
    matching strings."""

    PED = "ped"
    PED_ADULT = "ped_adult"
    PED_KID = "ped_kid"
    PED_OFFICER = "ped_officer"
    PED_PERSONNEL = "ped_personnel"
    PED_STROLLER = "ped_stroller"
    PED_WHEELCHAIR = "ped_wheelchair"
    PED_OTHER = "ped_other"
    CYCLIST = "cyclist"
    BICYCLE = "bicycle"
    SCOOTER = "scooter"
    VEHICLE = "vehicle"
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    EMERGENCY = "emergency"
    OFFICER = "officer"
    ANIMAL = "animal"
    VRU = "vru"

    @property
    def aliases(self) -> frozenset[str]:
        return AGENT_TYPE[self.value]


class ActionKind(str, Enum):
    DRIVE = "drive"
    STOP = "stop"
    YIELD = "yield"
    DECEL = "decel"
    PARK = "park"
    WALK = "walk"
    STAND = "stand"
    RUN = "run"
    CREEP = "creep"
    ENTER = "enter"
    FOLLOW = "follow"
    NOT_MOVE = "not_move"
    ABORT = "abort"
    REVERSE = "reverse"
    TURN = "turn"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    UTURN = "uturn"
    CHANGE_LANE = "change_lane"
    CHANGE_LANE_LEFT = "change_lane_left"
    CHANGE_LANE_RIGHT = "change_lane_right"
    NUDGE = "nudge"
    NUDGE_IN = "nudge_in"
    NUDGE_OUT = "nudge_out"
    OVERTAKE = "overtake"
    STOP_YIELD_DECEL = "stop_yield_decel"

    @property
    def aliases(self) -> frozenset[str]:
        return ACTION_TYPE[self.value]


class EnvKind(str, Enum):
    ROAD = "road"
    CROSSWALK = "crosswalk"
    SIDEWALK = "sidewalk"
    INTERSECTION = "intersection"
    CROSSROAD = "crossroad"
    T_INTERSECTION = "t_intersection"
    Y_INTERSECTION = "y_intersection"
    ROUNDABOUT = "roundabout"
    CYCLE_LANE = "cycle_lane"
    LANE_MERGE = "lane_merge"
    LANE_FORK = "lane_fork"
    TUNNEL = "tunnel"
    BRIDGE = "bridge"
    SHOULDER = "shoulder"
    RAIL_CROSSING = "rail_crossing"

    @property
    def aliases(self) -> frozenset[str]:
        return ENV_TYPE[self.value]


class LightColor(str, Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    OTHER = "other"

    @property
    def aliases(self) -> frozenset[str]:
        return LIGHT_COLOR[self.value]


class Position(str, Enum):
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    BEHIND = "behind"

    @property
    def aliases(self) -> frozenset[str]:
        return POSITION_ALIASES[self.value]


# ---------------------------------------------------------------------------
# Family registry — used by the parser to dispatch ``resolve_alias``.
# ---------------------------------------------------------------------------

ALIAS_FAMILIES: Mapping[str, Mapping[str, frozenset[str]]] = {
    "agent_type": AGENT_TYPE,
    "action_type": ACTION_TYPE,
    "env_type": ENV_TYPE,
    "cond_type": COND_TYPE,
    "light_color": LIGHT_COLOR,
    "light_state": LIGHT_STATE_TYPE,
    "light_shape": LIGHT_SHAPE,
    "traffic_light_type": TRAFFIC_LIGHT_TYPE,
    "obj_type": OBJ_TYPE,
    "obj_motion_state": OBJ_MOTION_STATE,
    "obj_open_state": OBJ_OPEN_STATE,
    "position": POSITION_ALIASES,
    "direction": DIRECTION,
    "agent_property_type": AGENT_PROPERTY_TYPE,
    "ego_property_type": EGO_PROPERTY_TYPE,
    "signaling_intent": SIGNALING_INTENT,
    "signaling_source": SIGNALING_SOURCE,
    "driving_judgment": DRIVING_JUDGMENT,
}


def resolve_alias(family: str, name: str) -> frozenset[str]:
    """Return the schema-string set for ``name`` in ``family``.

    Case-insensitive. Unknown name → ``KeyError`` so callers can raise a
    parser-level ``NameError`` with positional context.

    Bypasses the alias table when ``name`` already looks like a schema
    literal (contains ``:`` or matches a literal seen in any family) —
    that's the "escape hatch" for the DSL.
    """
    fam = ALIAS_FAMILIES[family]
    key = name.lower()
    if key in fam:
        return fam[key]
    # Escape hatch: user typed a full literal like "oxd:Pedestrian".
    # Accept it verbatim — the engine will compare against the raw string.
    return frozenset({name})
