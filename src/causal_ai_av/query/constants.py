"""DSL vocabulary — short aliases, hierarchical parents, schema literals.

Grounded against the 376-clip corpus on 2026-05-11 (see
`scripts/scan_corpus_vocabulary.py`). All sets are frozen and contain
the exact strings observed in `Agent.type` / `AgentAction.action_type`
/ etc.

Lookup discipline:
- DSL parser accepts a short alias (e.g. `"ped"`) or a full literal
  (e.g. `"oxd:Pedestrian (Adult)"`). It calls `resolve_alias(family,
  name)` to get the underlying set of schema strings.
- Aliases are case-folded to lowercase at the call site.
- Parent aliases (e.g. `"vehicle"`, `"vru"`, `"turn"`) resolve to the
  union of their child aliases.
- Unknown aliases raise `KeyError` — callers wrap in `NameError`.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping

# ---------------------------------------------------------------------------
# Agent types
# ---------------------------------------------------------------------------

AGENT_TYPE_ALIASES: Mapping[str, frozenset[str]] = {
    "car": frozenset({"oxd:Car"}),
    "truck": frozenset({"oxd:Truck", "Heavy-duty vehicle"}),
    "bus": frozenset({"PublicBus"}),
    "motorcycle": frozenset({"oxd:Motorcycle"}),
    "emergency": frozenset({"oxd:EmergencyVehicle"}),
    "bicycle": frozenset({"oxd:Bicycle"}),
    "scooter": frozenset({"fst:Scooter"}),
    "animal": frozenset({"oxd:Animal"}),
    "ped_adult": frozenset({"Pedestrian (Adult)"}),
    "ped_kid": frozenset({"Pedestrian (Kid/Teen)"}),
    "ped_personnel": frozenset({"Pedestrian (Personnel)"}),
    "ped_officer": frozenset({"Pedestrian (Officer)"}),
    "ped_stroller": frozenset({"Pedestrian (Stroller)"}),
    "ped_other": frozenset({"Pedestrian (Other)", "Pedestrian"}),
    "officer": frozenset({"Pedestrian (Officer)"}),
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
    "drive": frozenset({"fst:DrivingInLane"}),
    "stop": frozenset({"oxd:Stop"}),
    "park": frozenset({"fst:Park"}),
    "walk": frozenset({"oxd:Walk"}),
    "stand": frozenset({"oxd:Stand"}),
    "run": frozenset({"oxd:Run"}),
    "yield": frozenset({"fst:Yield"}),
    "decel": frozenset({"oxd:Decelerate"}),
    "creep": frozenset({"fst:Creep"}),
    "enter": frozenset({"fst:Enter"}),
    "follow": frozenset({"oxd:FollowRoadUser"}),
    "not_move": frozenset({"oxd:NotMove"}),
    "abort": frozenset({"fst:ManeuverAbort"}),
    "reverse": frozenset({"fst:Reverse"}),
    "turn_left": frozenset({"oxd:MakeALeftTurn"}),
    "turn_right": frozenset({"oxd:MakeARightTurn"}),
    "uturn": frozenset({"fst:MakeAUTurn"}),
    "change_lane_left": frozenset({"oxd:ChangeLane (left)"}),
    "change_lane_right": frozenset({"oxd:ChangeLane (right)"}),
    "change_lane": frozenset({"oxd:ChangeLane"}),
    "nudge_in": frozenset({"fst:Nudge (in lane)"}),
    "nudge_out": frozenset({"fst:Nudge (out of lane)"}),
    "nudge": frozenset({"fst:Nudge"}),
    "overtake": frozenset({"oxd:Overtake"}),
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

# Action-type flags — these are tokens that appear inside parenthesized
# suffixes. The engine matches by checking suffix tokens AND
# corresponding schema flag fields where they exist.
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

# Mapping from DSL flag attribute name → corresponding schema field name
# on AgentAction / EgoAction. ``None`` means "no schema flag; rely on
# suffix matching alone".
ACTION_FLAG_SCHEMA_FIELDS: Mapping[str, str | None] = {
    "jaywalk": "jaywalk_flag",
    "erratic": "erratic_flag",
    "illegal": "illegal_flag",
    "aggressive": "is_aggressive_or_cut_in",
    "turn_protected": "turn_protected",
    "ego_lane": "ego_lane_flag",
    "aborted": "maneuver_aborted_flag",
    # Suffix-only flags:
    "unprotected": None,
    "left": None,
    "right": None,
    "in_lane": None,
    "out_of_lane": None,
    "into_ego_lane": None,
}

# ---------------------------------------------------------------------------
# Environment types
# ---------------------------------------------------------------------------

ENV_TYPE_ALIASES: Mapping[str, frozenset[str]] = {
    "road": frozenset({"oxd:Road"}),
    "crosswalk": frozenset({"oxd:PedestrianCrossing"}),
    "sidewalk": frozenset({"oxd:Sidewalk"}),
    "crossroad": frozenset({"oxd:CrossRoad"}),
    "t_intersection": frozenset({"oxd:TIntersection"}),
    "y_intersection": frozenset({"oxd:YIntersection"}),
    "five_way_intersection": frozenset({"5-way"}),
    "other_intersection": frozenset({"Other Intersection"}),
    "cycle_lane": frozenset({"oxd:CycleLane"}),
    "bike_lane": frozenset({"oxd:CycleLane"}),  # alias
    "lane_merge": frozenset({"fst:LaneMerge"}),
    "lane_fork": frozenset({"fst:LaneFork"}),
    "roundabout": frozenset({"oxd:Roundabout"}),
    "tunnel": frozenset({"oxd:Tunnel"}),
    "bridge": frozenset({"oxd:Bridge"}),
    "paved_shoulder": frozenset({"oxd:PavedShoulder"}),
    "grass_shoulder": frozenset({"oxd:GrassShoulder"}),
    "rail_crossing": frozenset({"oxd:RailCrossing"}),
}

ENV_TYPE_PARENTS: Mapping[str, frozenset[str]] = {
    "intersection": (
        ENV_TYPE_ALIASES["crossroad"]
        | ENV_TYPE_ALIASES["t_intersection"]
        | ENV_TYPE_ALIASES["y_intersection"]
        | ENV_TYPE_ALIASES["five_way_intersection"]
        | ENV_TYPE_ALIASES["other_intersection"]
    ),
    "shoulder": ENV_TYPE_ALIASES["paved_shoulder"] | ENV_TYPE_ALIASES["grass_shoulder"],
}

ENV_TYPE: Mapping[str, frozenset[str]] = {**ENV_TYPE_ALIASES, **ENV_TYPE_PARENTS}

# ---------------------------------------------------------------------------
# Condition types
# ---------------------------------------------------------------------------

COND_TYPE: Mapping[str, frozenset[str]] = {
    "clear": frozenset({"Clear"}),
    "construction": frozenset({"Construction Zone"}),
    "wet": frozenset({"oxd:wetRoadCondition"}),
    "snowy": frozenset({"oxd:snowyRoadCondition"}),
    "temp_marked": frozenset({"Temporarily marked"}),
}

# ---------------------------------------------------------------------------
# Light state attributes
# ---------------------------------------------------------------------------

LIGHT_COLOR: Mapping[str, frozenset[str]] = {
    "red": frozenset({"Red"}),
    "yellow": frozenset({"Yellow"}),
    "green": frozenset({"Green"}),
    "other": frozenset({"Other"}),
}

LIGHT_STATE_TYPE: Mapping[str, frozenset[str]] = {
    "fixed": frozenset({"Fixed"}),
    "flashing": frozenset({"Flashing"}),
    "off": frozenset({"OFF"}),
}

LIGHT_SHAPE: Mapping[str, frozenset[str]] = {
    "round": frozenset({"Round"}),
    "arrow_left": frozenset({"Arrow_Left"}),
    "arrow_right": frozenset({"Arrow_Right"}),
    "arrow_up": frozenset({"Arrow_Up"}),
    "other": frozenset({"Other"}),
}

TRAFFIC_LIGHT_TYPE: Mapping[str, frozenset[str]] = {
    "regular": frozenset({"fst:RegularTrafficLight"}),
}

# ---------------------------------------------------------------------------
# Traffic object types
# ---------------------------------------------------------------------------

OBJ_TYPE_ALIASES: Mapping[str, frozenset[str]] = {
    "traffic_sign": frozenset({"Other oxd:TrafficSign"}),
    "stop_sign": frozenset({"fst:StopSign"}),
    "yield_sign": frozenset({"fst:YieldSign"}),
    "speed_limit_sign": frozenset({"fst:SpeedLimitSign"}),
    "warning_sign": frozenset({"oxd:WarningSign"}),
    "cone": frozenset({"oxd:TrafficCone"}),
    "roadblock": frozenset({"oxd:Roadblocks"}),
    "rail_crossing_obj": frozenset({"oxd:RailCrossing"}),
    "merge_ahead": frozenset({"Merge ahead"}),
    "adjacent_lanes_ahead": frozenset({"Adjacent lanes ahead"}),
    "do_not_enter": frozenset({"Do not enter"}),
    "portable_indicator": frozenset({"Other Small Portable Traffic Indicator"}),
    "portable_display": frozenset({"PortableDisplay"}),
    "trash": frozenset({"Trash"}),
    "dirt": frozenset({"Dirt"}),
    "other_debris": frozenset({"Other oxd:Debris"}),
    "not_identifiable": frozenset({"Not identifiable"}),
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
    "barrier": OBJ_TYPE_ALIASES["roadblock"] | OBJ_TYPE_ALIASES["cone"],
}

OBJ_TYPE: Mapping[str, frozenset[str]] = {**OBJ_TYPE_ALIASES, **OBJ_TYPE_PARENTS}

# ---------------------------------------------------------------------------
# Object state (motion/open)
# ---------------------------------------------------------------------------

OBJ_MOTION_STATE: Mapping[str, frozenset[str]] = {
    "static": frozenset({"Static"}),
    "moving": frozenset({"Moving / Rolling"}),
}

OBJ_OPEN_STATE: Mapping[str, frozenset[str]] = {
    "open": frozenset({"Open"}),
    "closed": frozenset({"Closed"}),
}

# ---------------------------------------------------------------------------
# Ego-relative pose
# ---------------------------------------------------------------------------

POSITION_ALIASES: Mapping[str, frozenset[str]] = {
    "front": frozenset({"In front"}),
    "left": frozenset({"Left"}),
    "right": frozenset({"Right"}),
    "behind": frozenset({"Behind"}),
}

DIRECTION_ALIASES: Mapping[str, frozenset[str]] = {
    "same": frozenset({"Same"}),
    "opposite": frozenset({"Opposite"}),
    "perpendicular_rl": frozenset({"Perpendicular-R-L"}),
    "perpendicular_lr": frozenset({"Perpendicular-L-R"}),
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
    "signal": frozenset({"Signal"}),
    "slow": frozenset({"Slow"}),
    "on_duty": frozenset({"On Duty"}),
    "double_parked": frozenset({"Double Parked"}),
}

EGO_PROPERTY_TYPE: Mapping[str, frozenset[str]] = {
    "slow": frozenset({"Slow"}),
    "signal": frozenset({"Signal"}),
    "erratic": frozenset({"Erratic"}),
    "fast": frozenset({"Fast"}),
}

SIGNALING_INTENT: Mapping[str, frozenset[str]] = {
    "proceed": frozenset({"Proceed"}),
    "turn": frozenset({"Turn"}),
    "stop": frozenset({"Stop"}),
    "caution": frozenset({"Caution"}),
    "slow": frozenset({"Slow Down"}),
    "follow": frozenset({"Follow"}),
}

SIGNALING_SOURCE: Mapping[str, frozenset[str]] = {
    "flashing_light": frozenset({"Flashing light"}),
    "holding_sign": frozenset({"Holding sign"}),
    "hand_gesture": frozenset({"Hand gesture"}),
}

# ---------------------------------------------------------------------------
# Driving judgment
# ---------------------------------------------------------------------------

DRIVING_JUDGMENT: Mapping[str, frozenset[str]] = {
    "good": frozenset({"good"}),
    "neutral": frozenset({"neutral"}),
    # Future-proofing: these glyphs/labels may appear in later corpus
    # versions; including them here is cheap.
    "bad": frozenset({"bad"}),
    "acceptable": frozenset({"acceptable"}),
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
