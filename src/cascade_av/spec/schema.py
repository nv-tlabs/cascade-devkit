# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical Pydantic v2 schema for the CASCADE annotation format.

The current schema version is :data:`cascade_av.spec.versions.CURRENT_SCHEMA_VERSION`
(``2.0.0`` — the format was rebooted in May 2026 to align byte-for-byte with the
upstream ``sil-dense-annotation-tool 0.4.5`` JSON shape). See
:mod:`cascade_av.spec.versions` for the registry and
``docs/dev/schema-history.md`` for the human-readable changelog.

The on-disk corpus is ground truth. Vocabulary-typed fields (action types,
agent types, etc.) are declared as plain ``str`` rather than ``Enum``, because
the upstream tool's frontend writes free-form strings and unknown values must
round-trip rather than fail validation. Advisory ``*Vocab`` namespace classes
near the bottom of this file enumerate the known values for autocomplete and
for use in queries.

Annotator timeline-layout indices (``_track_index``, ``_cond_track_index``,
``_state_track_index``, ``_influence_track_index``, ``_cont_track_index``,
``_prop_track_index``, ``_state_lane_count``) live as Pydantic ``Field``
aliases on the relevant typed models, so they round-trip through
``model_dump(by_alias=True)`` to the underscore-prefixed on-disk keys. The
top-level ``_ui_config`` block on :class:`AnnotationBundle` carries
annotator-only viewport state the same way.

``model_config = ConfigDict(extra="allow")`` is set on every model so anything
new added by the frontend (or any other producer) is preserved on read and
dump without code change. Schema-extension data lives in a sidecar
``<stem>.extra.json`` next to the main file; see :mod:`cascade_av.extensions`.
"""

from __future__ import annotations

import warnings
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from cascade_av.spec.versions import CURRENT_SCHEMA_VERSION

# One-shot per process: schema versions we have already warned about as
# "older than CURRENT_SCHEMA_VERSION". Reset by
# `_reset_version_warn_cache_for_tests`.
_warned_schema_versions: set[str] = set()

# Shared model config: permit unknown extras, accept both Python-name and
# JSON-alias inputs.
_MC = ConfigDict(extra="allow", populate_by_name=True)


# -----------------------------------------------------------------------------
# Shared atoms
# -----------------------------------------------------------------------------


class Containment(BaseModel):
    """Containment of an entity within an environment / lane over a window."""

    model_config = _MC

    id: str = ""
    env_id: str = ""
    lane_number: str = ""
    illegal_flag: bool = False
    near_flag: bool = False
    edge: str | None = None  # "left" | "right" | None
    other: str | None = None
    start_timestamp: str = ""
    end_timestamp: str = ""
    # Upstream emits `_cont_track_index` (not `_track_index`) for containment
    # entries — separate prefix per nested family. See issue #124.
    track_index: int | None = Field(default=None, alias="_cont_track_index")


class Influence(BaseModel):
    """A set of entities that modulate ego behaviour over a time window."""

    model_config = _MC

    id: str = ""
    influencers: list[str] = Field(default_factory=list)
    comment: str = ""
    start_timestamp: str = ""
    end_timestamp: str = ""
    influence_track_index: int | None = Field(default=None, alias="_influence_track_index")


class Keypoint(BaseModel):
    """Sparse trajectory point (rare in this corpus)."""

    model_config = _MC

    timestamp: str = ""
    x: float = 0.0
    y: float = 0.0


class SignalingDetails(BaseModel):
    """Details for `Signal*` properties: source, intent, sign type."""

    model_config = _MC

    source: str | None = None
    intent: str | None = None
    sign_type: str | None = None
    other_source_description: str = ""
    other_intent_description: str = ""
    other_sign_description: str = ""
    # Intentionally KEPT active: signaling targets still link here (not deprecated,
    # unlike the action-level `link_to` on AgentAction / EgoAction).
    link_to: list[str] = Field(default_factory=list)
    not_facing_ego: bool | None = None
    target_agent_ids: list[str] | None = None  # deprecated; superseded by link_to


class EgoRelativePose(BaseModel):
    """Qualitative pose of an entity relative to ego over a time window."""

    model_config = _MC

    position_rel_to_ego: str | None = None  # e.g. "In front" | "Left" | "Right" | "Behind"
    direction_rel_to_ego: str | None = None  # e.g. "Same" | "Opposite" | "Perpendicular-L-R"
    start_timestamp: str = ""
    end_timestamp: str = ""


# -----------------------------------------------------------------------------
# Environment & condition
# -----------------------------------------------------------------------------


class Environment(BaseModel):
    """A semantic road-scene element (road, intersection, sidewalk, ...)."""

    model_config = _MC

    id: str
    name: str = ""
    type: str = ""  # see EnvironmentTypeVocab
    type_other_description: str = ""
    num_lanes: int | None = None
    num_out_lanes: int | None = None
    one_way: bool = False
    start_timestamp: str = ""
    end_timestamp: str = ""
    keypoints: list[Keypoint] = Field(default_factory=list)
    track_index: int | None = Field(default=None, alias="_track_index")

    @field_validator("num_lanes", "num_out_lanes", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: object) -> object:
        # The corpus carries `num_lanes: ""` when the annotator left it blank.
        return None if v == "" else v


class Condition(BaseModel):
    """An environmental state. `type` is a list because a single window may
    carry multiple condition labels
    (e.g. ``["Construction Zone", "Lanes obscured / unmarked"]``)."""

    model_config = _MC

    id: str
    env_id: str = ""
    type: list[str] = Field(default_factory=list)
    condition_other_description: str = ""
    start_timestamp: str = ""
    end_timestamp: str = ""
    cond_track_index: int | None = Field(default=None, alias="_cond_track_index")

    @field_validator("type", mode="before")
    @classmethod
    def _str_to_list(cls, v: object) -> object:
        # Some corpora (notably the 02json export) write a bare string
        # like "Construction Zone" instead of ["Construction Zone"].
        # Wrap it so downstream consumers always see the canonical list.
        return [v] if isinstance(v, str) else v


# -----------------------------------------------------------------------------
# Traffic objects (signs, cones, debris, barriers, ...)
# -----------------------------------------------------------------------------


class ObjectStateEntry(BaseModel):
    """One contiguous state interval for a traffic object."""

    model_config = _MC

    id: str = ""
    start_timestamp: str = ""
    end_timestamp: str = ""
    open_state: str | None = None  # "Open" | "Closed" | None
    motion_state: str | None = None  # "Static" | "Moving / Rolling" | None
    other_condition_description: str = ""


class TrafficObject(BaseModel):
    """A static or minor-dynamic non-light object (sign, cone, debris, ...)."""

    model_config = _MC

    id: str
    name: str = ""
    type: str = ""  # see TrafficObjectTypeVocab
    other_type_description: str = ""
    visibility_start_timestamp: str = ""
    visibility_end_timestamp: str = ""
    lane_number: str = ""
    quantity: str | None = None  # "Single" | "Line / Row" | "Channelizing Line" | ...
    containment: list[Containment] = Field(default_factory=list)
    state_sequence: list[ObjectStateEntry] = Field(default_factory=list)
    keypoints: list[Keypoint] = Field(default_factory=list)
    track_index: int | None = Field(default=None, alias="_track_index")


# -----------------------------------------------------------------------------
# Traffic lights
# -----------------------------------------------------------------------------


class LightStates(BaseModel):
    """One state interval for a signal head (color + shape + state type)."""

    model_config = _MC

    id: str = ""
    type: str | None = None  # "Fixed" | "Flashing" | "OFF" | None
    color: str | None = None  # "Green" | "Yellow" | "Red" | "Other" | None
    shape: str | None = None  # "Round" | "Arrow_Left" | "Arrow_Right" | ...
    start_timestamp: str = ""
    end_timestamp: str = ""
    other_condition_description: str = ""
    yellow_on_ego_path: bool | None = None
    ego_in_intersection_on_yellow: bool | None = None
    ego_could_have_cleared_safely: bool | None = None
    state_track_index: int | None = Field(default=None, alias="_state_track_index")


class SignalHead(BaseModel):
    """One signal head on a traffic light (an individual lamp)."""

    model_config = _MC

    id: str = ""
    state_sequence: list[LightStates] = Field(default_factory=list)
    env_controlled: list[Containment] = Field(default_factory=list)
    influenced_agent_ids: list[str] = Field(default_factory=list)
    affects_ego: str | None = None  # "True" | "False" | None
    start_timestamp: str = ""
    end_timestamp: str = ""
    keypoints: list[Keypoint] = Field(default_factory=list)
    state_lane_count: int | None = Field(default=None, alias="_state_lane_count")


class TrafficLight(BaseModel):
    """A traffic-light fixture with one or more signal heads."""

    model_config = _MC

    id: str
    name: str = ""
    type: str = "fst:RegularTrafficLight"
    other_type_description: str = ""
    visibility_start_timestamp: str = ""
    visibility_end_timestamp: str = ""
    containment: list[Containment] = Field(default_factory=list)
    signal_heads: list[SignalHead] = Field(default_factory=list)
    track_index: int | None = Field(default=None, alias="_track_index")


# -----------------------------------------------------------------------------
# Actions, properties, agents
# -----------------------------------------------------------------------------


class AgentProperty(BaseModel):
    """A property of an agent or the ego vehicle (Signal*, Parked, ...)."""

    model_config = _MC

    id: str = ""
    property_type: str = ""
    other_description: str = ""
    start_timestamp: str = ""
    end_timestamp: str = ""
    signaling_details: SignalingDetails | None = None
    track_index: int | None = Field(default=None, alias="_prop_track_index")


class AgentAction(BaseModel):
    """An action performed by an agent over a time window.

    ``action_type`` is a combined-string suffix carrying both the base verb
    and any flags — e.g. ``"oxd:Walk (jaywalk, erratic)"`` or
    ``"oxd:MakeALeftTurn (unprotected)"``. The legacy ``jaywalk_flag`` /
    ``erratic_flag`` / ``turn_protected`` / etc. side-channel fields are
    gone in the 2.0.0 reboot — the suffix is now the single source of truth.
    """

    model_config = _MC

    id: str = ""
    action_type: str = ""  # see AgentActionTypeVocab / AgentActionTypePedestrianVocab
    other_description: str = ""
    because_of: list[str] = Field(default_factory=list)
    # DEPRECATED on actions (annotator <=0.6.1): use because_of / action_target.
    link_to: list[str] = Field(default_factory=list)
    action_target: list[str] = Field(default_factory=list)
    start_timestamp: str = ""
    end_timestamp: str = ""
    illegal_flag: bool = False
    signaling_details: SignalingDetails | None = None


class Agent(BaseModel):
    """A moving actor in the scene (vehicle, pedestrian, cyclist, ...)."""

    model_config = _MC

    id: str = ""
    name: str = ""
    amount: str = "Single"  # see AgentAmountVocab
    type: str = ""  # see AgentTypeVocab
    other_type_description: str = ""
    visibility_start_timestamp: str = ""
    visibility_end_timestamp: str = ""
    actions: list[AgentAction] = Field(default_factory=list)
    properties: list[AgentProperty] = Field(default_factory=list)
    ego_relative_pose: list[EgoRelativePose] = Field(default_factory=list)
    containment: list[Containment] = Field(default_factory=list)
    influenced_by: list[Influence] = Field(default_factory=list)
    keypoints: list[Keypoint] = Field(default_factory=list)
    track_index: int | None = Field(default=None, alias="_track_index")


class EgoAction(BaseModel):
    """An action performed by the ego vehicle over a time window."""

    model_config = _MC

    id: str = ""
    type: str = ""  # see EgoActionTypeVocab
    turn_other_description: str = ""
    action_other_description: str = ""
    because_of: list[str] = Field(default_factory=list)
    # DEPRECATED on actions (annotator <=0.6.1): use because_of / action_target.
    link_to: list[str] = Field(default_factory=list)
    action_target: list[str] = Field(default_factory=list)
    illegal_flag: bool = False
    start_timestamp: str = ""
    end_timestamp: str = ""


class EgoVehicle(BaseModel):
    """The ego vehicle's actions, properties, containment, and influences."""

    model_config = _MC

    name: str = ""
    actions: list[EgoAction] = Field(default_factory=list)
    properties: list[AgentProperty] = Field(default_factory=list)
    containment: list[Containment] = Field(default_factory=list)
    influenced_by: list[Influence] = Field(default_factory=list)
    driving_judgment: str | None = None  # see DrivingJudgmentVocab


# -----------------------------------------------------------------------------
# Top-level annotation container
# -----------------------------------------------------------------------------


class SilAvAnnotation(BaseModel):
    """The annotation payload for one clip."""

    model_config = _MC

    eventful: bool | None = None
    # Added post-2.0.0 (annotator internal 0.6.x). Categorical rationale for `eventful`;
    # see EventfulReasonVocab. Advisory vocab only (field stays an open str).
    eventful_reason: str | None = None
    eventful_reason_other: str = ""  # free text when eventful_reason == "other"
    brief_description: str = ""
    environments: list[Environment] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    traffic_objects: list[TrafficObject] = Field(default_factory=list)
    traffic_lights: list[TrafficLight] = Field(default_factory=list)
    ego_vehicle: EgoVehicle = Field(default_factory=EgoVehicle)
    agents: list[Agent] = Field(default_factory=list)


class VideoMeta(BaseModel):
    """Reference to the clip the annotation applies to."""

    model_config = _MC

    clip_id: str
    source: str = "huggingface"
    fps: float = 30.0
    duration_s: float = 0.0


class QAIssue(BaseModel):
    """One QA finding raised against the annotation bundle."""

    model_config = _MC

    type: str = ""
    severity: str = ""
    message: str = ""


class UiConfig(BaseModel):
    """Annotator-only viewport state. Round-trips on disk as ``_ui_config``.

    Carries timeline track-count totals, current zoom level, and current
    scroll offset so the annotator restores the same layout the user left.
    Not part of the semantic annotation; downstream consumers can ignore it.
    """

    model_config = _MC

    env_track_count: int | None = Field(default=None, alias="envTrackCount")
    object_track_count: int | None = Field(default=None, alias="objectTrackCount")
    light_track_count: int | None = Field(default=None, alias="lightTrackCount")
    agent_track_count: int | None = Field(default=None, alias="agentTrackCount")
    ego_cont_track_count: int | None = Field(default=None, alias="egoContTrackCount")
    zoom_level: float | None = Field(default=None, alias="zoomLevel")
    scroll_offset: float | None = Field(default=None, alias="scrollOffset")


class AnnotationBundle(BaseModel):
    """Root model — one serialized JSON file equals one AnnotationBundle."""

    model_config = _MC

    schema_version: str = CURRENT_SCHEMA_VERSION
    video: VideoMeta
    annotation: SilAvAnnotation = Field(default_factory=SilAvAnnotation)
    status: str = "annotating"  # see AnnotationStatusVocab
    provenance: dict[str, Any] = Field(default_factory=lambda: {"generated_by": "human"})
    qa_issues: list[QAIssue] = Field(default_factory=list)
    review_round: int | None = None
    ui_config: UiConfig | None = Field(default=None, alias="_ui_config")

    # Schema-extension surface (see `cascade_av.extensions`). Both are
    # PrivateAttr so they are absent from `model_dump()` / `model_dump_json()`
    # and never pollute the main on-disk JSON. Extension payloads live in the
    # `<stem>.extra.json` sidecar managed by `cascade_av.io`.
    #
    # `_extensions` is where registered extensions stash their typed view; the
    # public `ext()` accessor reads from it.
    #
    # `_sidecar_raw` holds verbatim payloads for sidecar keys with no
    # registered extension, so they round-trip through load → save without
    # being silently dropped.
    _extensions: dict[str, Any] = PrivateAttr(default_factory=dict)
    _sidecar_raw: dict[str, Any] = PrivateAttr(default_factory=dict)

    def ext(self, key: str) -> Any | None:
        """Return the typed view a registered extension attached under ``key``.

        Returns ``None`` when no extension has populated ``key`` on this
        bundle (either the extension is not registered, or it found
        nothing to load).
        """
        return self._extensions.get(key)

    @model_validator(mode="after")
    def _warn_on_old_schema(self) -> "AnnotationBundle":
        """Emit one ``DeprecationWarning`` per non-current schema_version per process.

        The loader still accepts older bundles — this is advisory only. After
        the 2.0.0 reboot there is no in-tree migration path; downstream
        producers need to re-export against the current schema.
        """
        v = self.schema_version
        if v != CURRENT_SCHEMA_VERSION and v not in _warned_schema_versions:
            _warned_schema_versions.add(v)
            warnings.warn(
                f"AnnotationBundle has schema_version={v!r}; current is "
                f"{CURRENT_SCHEMA_VERSION!r}. Re-export the bundle against "
                f"the current schema.",
                DeprecationWarning,
                stacklevel=3,
            )
        return self


def _reset_version_warn_cache_for_tests() -> None:
    """Wipe the schema-version warn cache. Test fixtures only."""
    _warned_schema_versions.clear()


# -----------------------------------------------------------------------------
# Advisory vocabularies — the fields above are open `str`; these enumerate
# the known values for autocomplete and for use in queries. All strings are
# verbatim from the upstream ``sil-dense-annotation-tool 0.4.5`` reference
# and must be kept byte-for-byte in sync.
# -----------------------------------------------------------------------------


class AgentTypeVocab:
    """Known values for ``Agent.type``. Advisory, not exhaustive; field is ``str``."""

    VEHICLE = "Vehicle"
    CAR = "oxd:Car"
    HEAVY_DUTY = "Heavy-duty vehicle"
    EMERGENCY_VEHICLE = "oxd:EmergencyVehicle"
    TRUCK = "oxd:Truck"
    BUS = "PublicBus"
    BICYCLE = "oxd:Bicycle"
    MOTORCYCLE = "oxd:Motorcycle"
    SCOOTER = "fst:Scooter"
    PEDESTRIAN = "Pedestrian"
    PEDESTRIAN_OFFICER = "Pedestrian (Officer)"
    PEDESTRIAN_PERSONNEL = "Pedestrian (Personnel)"
    PEDESTRIAN_ADULT = "Pedestrian (Adult)"
    PEDESTRIAN_KID_TEEN = "Pedestrian (Kid/Teen)"
    PEDESTRIAN_STROLLER = "Pedestrian (Stroller)"
    PEDESTRIAN_WHEELCHAIR = "Pedestrian (oxd:Wheelchair)"
    PEDESTRIAN_OTHER = "Pedestrian (Other)"
    ANIMAL = "oxd:Animal"
    OTHER = "Other"


class AgentActionTypeVocab:
    """Known values for ``AgentAction.action_type`` on vehicle/cyclist agents.

    These are combined-suffix strings: the base verb plus any qualifier in
    parentheses, e.g. ``"oxd:MakeALeftTurn (unprotected)"``.
    """

    MANEUVER_ABORT = "fst:ManeuverAbort"
    PARK = "fst:Park"
    STOP = "oxd:Stop"
    NOT_MOVE = "oxd:NotMove"  # DEPRECATED (annotator <=0.6.1): retired; producers emit oxd:Stop
    ENTER = "fst:Enter"
    EXIT = "fst:Exit"
    CREEP = "fst:Creep"
    YIELD = "fst:Yield"
    DECELERATE = "oxd:Decelerate"
    DRIVING_IN_LANE = "fst:DrivingInLane"
    FOLLOW_ROAD_USER = "oxd:FollowRoadUser"
    NUDGE_IN_LANE = "fst:Nudge (in lane)"
    NUDGE_OUT_OF_LANE_NOT_INTO_EGO_LANE = "fst:Nudge (out of lane: not into ego lane)"
    NUDGE_OUT_OF_LANE_INTO_EGO_LANE = "fst:Nudge (out of lane: into ego lane)"
    OVERTAKE_USING_EGO_LANE = "oxd:Overtake (using ego lane)"
    OVERTAKE_NOT_USING_EGO_LANE = "oxd:Overtake (not using ego lane)"
    CHANGE_LANE_LEFT = "oxd:ChangeLane (left)"
    CHANGE_LANE_RIGHT = "oxd:ChangeLane (right)"
    LEFT_TURN_UNPROTECTED = "oxd:MakeALeftTurn (unprotected)"
    RIGHT_TURN_UNPROTECTED = "oxd:MakeARightTurn (unprotected)"
    UTURN_UNPROTECTED = "fst:MakeAUTurn (unprotected)"
    LEFT_TURN_PROTECTED = "oxd:MakeALeftTurn (protected)"
    RIGHT_TURN_PROTECTED = "oxd:MakeARightTurn (protected)"
    UTURN_PROTECTED = "fst:MakeAUTurn (protected)"
    OTHER_MAKE_A_TURN = "Other oxd:MakeATurn"
    REVERSE = "fst:Reverse"
    OTHER = "Other"


class AgentActionTypePedestrianVocab:
    """Known values for ``AgentAction.action_type`` on pedestrian agents."""

    MANEUVER_ABORT = "fst:ManeuverAbort"
    STOP = "oxd:Stop"
    STAND = "oxd:Stand"
    ENTER = "fst:Enter"
    EXIT = "fst:Exit"
    CREEP = "fst:Creep"
    YIELD = "fst:Yield"
    DECELERATE = "oxd:Decelerate"
    FOLLOW_ROAD_USER = "oxd:FollowRoadUser"
    WALK = "oxd:Walk"
    JAYWALK = "Jaywalk"
    # DEPRECATED (annotator <=0.6.1): combined erratic/jaywalk forms; standalone
    # Jaywalk + Erratic property now used.
    WALK_JAYWALK = (
        "oxd:Walk (jaywalk)"  # DEPRECATED (annotator <=0.6.1): combined erratic/jaywalk form
    )
    WALK_ERRATIC = (
        "oxd:Walk (erratic)"  # DEPRECATED (annotator <=0.6.1): combined erratic/jaywalk form
    )
    WALK_JAYWALK_ERRATIC = "oxd:Walk (jaywalk, erratic)"  # DEPRECATED (annotator <=0.6.1): combined erratic/jaywalk form
    RUN = "oxd:Run"
    RUN_JAYWALK = (
        "oxd:Run (jaywalk)"  # DEPRECATED (annotator <=0.6.1): combined erratic/jaywalk form
    )
    RUN_ERRATIC = (
        "oxd:Run (erratic)"  # DEPRECATED (annotator <=0.6.1): combined erratic/jaywalk form
    )
    RUN_JAYWALK_ERRATIC = "oxd:Run (jaywalk, erratic)"  # DEPRECATED (annotator <=0.6.1): combined erratic/jaywalk form
    OTHER = "Other"


class EgoActionTypeVocab:
    """Known values for ``EgoAction.type``."""

    MANEUVER_ABORT = "fst:ManeuverAbort"
    STOP = "oxd:Stop"
    NOT_MOVE = "oxd:NotMove"  # DEPRECATED (annotator <=0.6.1): retired; producers emit oxd:Stop
    ENTER = "fst:Enter"
    EXIT = "fst:Exit"
    CREEP = "fst:Creep"
    YIELD = "fst:Yield"
    DECELERATE = "oxd:Decelerate"
    DRIVING_IN_LANE = "fst:DrivingInLane"
    FOLLOW_ROAD_USER = "oxd:FollowRoadUser"
    NUDGE_IN_LANE = "fst:Nudge (in lane)"
    NUDGE_OUT_OF_LANE = "fst:Nudge (out of lane)"
    OVERTAKE = "oxd:Overtake"
    CHANGE_LANE_LEFT = "oxd:ChangeLane (left)"
    CHANGE_LANE_RIGHT = "oxd:ChangeLane (right)"
    LEFT_TURN_UNPROTECTED = "oxd:MakeALeftTurn (unprotected)"
    RIGHT_TURN_UNPROTECTED = "oxd:MakeARightTurn (unprotected)"
    UTURN_UNPROTECTED = "fst:MakeAUTurn (unprotected)"
    LEFT_TURN_PROTECTED = "oxd:MakeALeftTurn (protected)"
    RIGHT_TURN_PROTECTED = "oxd:MakeARightTurn (protected)"
    UTURN_PROTECTED = "fst:MakeAUTurn (protected)"
    OTHER_MAKE_A_TURN = "Other oxd:MakeATurn"
    REVERSE = "fst:Reverse"
    OTHER = "Other"


class EnvironmentTypeVocab:
    """Known values for ``Environment.type``."""

    ROAD = "oxd:Road"
    LANE_MERGE = "fst:LaneMerge"
    LANE_FORK = "fst:LaneFork"
    T_INTERSECTION = "oxd:TIntersection"
    Y_INTERSECTION = "oxd:YIntersection"
    CROSSROAD = "oxd:CrossRoad"
    FIVE_WAY = "5-way"
    SIX_WAY = "6-way"
    SIX_PLUS_WAY = "6+-way"
    OTHER_INTERSECTION = "Other Intersection"
    ROUNDABOUT = "oxd:Roundabout"
    TUNNEL = "oxd:Tunnel"
    BRIDGE = "oxd:Bridge"
    PAVED_SHOULDER = "oxd:PavedShoulder"
    GRASS_SHOULDER = "oxd:GrassShoulder"
    SIDEWALK = "oxd:Sidewalk"
    PEDESTRIAN_CROSSING = "oxd:PedestrianCrossing"
    RAIL_CROSSING = "oxd:RailCrossing"
    CYCLE_LANE = "oxd:CycleLane"
    SPEED_BUMP = "fst:SpeedBump"
    LIGHT_RAIL_LANE = "fst:LightRailLane"
    OTHER = "Other"


class ConditionTypeVocab:
    """Known values for ``Condition.type`` (list-valued field)."""

    CONSTRUCTION_ZONE = "Construction Zone"
    TEMPORARILY_MARKED = "Temporarily marked"
    SNOWY_ROAD = "oxd:snowyRoadCondition"
    WET_ROAD = "oxd:wetRoadCondition"
    OVERGROWN = "Overgrown"
    SHARED_MARKED_CENTER_LANE = "Shared marked center lane"
    NO_DIRECTION_DIVIDER = "No direction divider"
    LANES_OBSCURED = "Lanes obscured / unmarked"
    OTHER = "Other"


class TrafficObjectTypeVocab:
    """Known values for ``TrafficObject.type``."""

    STOP_SIGN = "fst:StopSign"
    YIELD_SIGN = "fst:YieldSign"
    SPEED_LIMIT_SIGN = "fst:SpeedLimitSign"
    MERGE_AHEAD = "Merge ahead"
    ADJACENT_LANES_AHEAD = "Adjacent lanes ahead"
    DO_NOT_ENTER = "Do not enter"
    OTHER_TRAFFIC_SIGN = "Other oxd:TrafficSign"
    RAIL_CROSSING = "oxd:RailCrossing"
    GARAGE = "Garage"
    TOLL_PLAZA = "oxd:TollPlaza"
    BOLLARD = "Bollard"
    ROADBLOCKS = "oxd:Roadblocks"
    TRAFFIC_CONE = "oxd:TrafficCone"
    WARNING_SIGN = "oxd:WarningSign"
    PORTABLE_DISPLAY = "PortableDisplay"
    BARRIER = "Barrier"
    OTHER_PORTABLE_INDICATOR = "Other Small Portable Traffic Indicator"
    TOY = "Toy"
    BALL = "Ball"
    OTHER_FALLEN_OBJECT = "Other fst:FallenObject"
    DIRT = "Dirt"
    TRASH = "Trash"
    OTHER_DEBRIS = "Other oxd:Debris"
    NOT_IDENTIFIABLE = "Not identifiable"
    BOOM_GATE = "Boom gate"
    OTHER = "Other"


class LightColorVocab:
    """Known values for ``LightStates.color``."""

    GREEN = "Green"
    YELLOW = "Yellow"
    RED = "Red"
    OTHER = "Other"


class LightShapeVocab:
    """Known values for ``LightStates.shape``."""

    ROUND = "Round"
    ARROW_LEFT = "Arrow_Left"
    ARROW_RIGHT = "Arrow_Right"
    ARROW_UP = "Arrow_Up"
    ARROW_DOWN = "Arrow_Down"
    OTHER = "Other"


class LightStateTypeVocab:
    """Known values for ``LightStates.type``."""

    FIXED = "Fixed"
    FLASHING = "Flashing"
    OFF = "OFF"


class AgentPropertyTypeVocab:
    """Known values for ``AgentProperty.property_type`` on non-ego agents."""

    SLOW = "Slow"
    FAST = "Fast"
    AGGRESSIVE = "Aggressive"
    ERRATIC = "Erratic"
    EMERGENCY = "Emergency"
    ON_DUTY = "On Duty"
    DOUBLE_PARKED = "Double Parked"
    SIGNAL = "Signal"
    OUTSIDE_CAMERA_VIEW = "Outside Camera View"
    STOPPED = "Stopped"
    OTHER = "Other"


class EgoPropertyTypeVocab:
    """Known values for ``AgentProperty.property_type`` on the ego vehicle.

    Same string vocabulary as :class:`AgentPropertyTypeVocab`; this class
    exists for symmetry with the producer UI, which exposes the two
    property pickers separately.
    """

    SLOW = "Slow"
    FAST = "Fast"
    AGGRESSIVE = "Aggressive"
    ERRATIC = "Erratic"
    EMERGENCY = "Emergency"
    ON_DUTY = "On Duty"
    DOUBLE_PARKED = "Double Parked"
    SIGNAL = "Signal"
    OUTSIDE_CAMERA_VIEW = "Outside Camera View"
    STOPPED = "Stopped"
    OTHER = "Other"


class SignalSourceVocab:
    """Known values for ``SignalingDetails.source``."""

    FLASHING_LIGHT = "Flashing light"
    HAND_GESTURE = "Hand gesture"
    HOLDING_SIGN = "Holding sign"
    OTHER = "Other"


class SignalIntentVocab:
    """Known values for ``SignalingDetails.intent``."""

    TURN = "Turn"  # DEPRECATED (annotator <=0.6.1): superseded by Left/Right Indicator
    LEFT_INDICATOR = "Left Indicator"
    RIGHT_INDICATOR = "Right Indicator"
    STOP = "Stop"
    SLOW_DOWN = "Slow Down"
    PROCEED = "Proceed"
    FOLLOW = "Follow"
    CAUTION = "Caution"
    DANGER = "Danger"
    UNCLEAR = "Unclear / Incorrectly used"
    OTHER = "Other"


class SignTypeVocab:
    """Known values for ``SignalingDetails.sign_type``."""

    STOP_SIGN = "Stop Sign"
    YIELD_SIGN = "Yield Sign"
    SLOW_SIGN = "Slow Sign"
    NOT_IDENTIFIABLE = "Not identifiable"
    OTHER = "Other"


class MotionStateVocab:
    """Known values for ``ObjectStateEntry.motion_state``."""

    STATIC = "Static"
    MOVING = "Moving / Rolling"


class OpenStateVocab:
    """Known values for ``ObjectStateEntry.open_state``."""

    OPEN = "Open"
    CLOSED = "Closed"


class TrafficObjectQuantityVocab:
    """Known values for ``TrafficObject.quantity``."""

    SINGLE = "Single"
    LINE = "Line / Row"
    CHANNELIZING = "Channelizing Line"
    PERIMETER = "Perimeter"
    GROUP = "Group"


class AgentAmountVocab:
    """Known values for ``Agent.amount``."""

    SINGLE = "Single"
    ROW_GROUP = "Row/group"
    LIGHT_TRAFFIC = (
        "Light traffic"  # DEPRECATED (annotator <=0.6.1): traffic-density tier no longer produced
    )
    MEDIUM_TRAFFIC = (
        "Medium traffic"  # DEPRECATED (annotator <=0.6.1): traffic-density tier no longer produced
    )
    HEAVY_TRAFFIC = (
        "Heavy traffic"  # DEPRECATED (annotator <=0.6.1): traffic-density tier no longer produced
    )


class PositionRelToEgoVocab:
    """Known values for ``EgoRelativePose.position_rel_to_ego``."""

    IN_FRONT = "In front"
    LEFT = "Left"
    RIGHT = "Right"
    BEHIND = "Behind"


class DirectionRelToEgoVocab:
    """Known values for ``EgoRelativePose.direction_rel_to_ego``."""

    SAME = "Same"
    OPPOSITE = "Opposite"
    PERPENDICULAR_LR = "Perpendicular-L-R"
    PERPENDICULAR_RL = "Perpendicular-R-L"


class AnnotationStatusVocab:
    """Known status values across producers. Field is ``str``."""

    # From the upstream tool
    PENDING = "pending"
    ANNOTATING = "annotating"
    APPROVED = "approved"
    DISAPPROVED = "disapproved"
    # From the corpus
    SUBMITTED = "submitted"
    NEEDS_REVISION = "needs_revision"


class DrivingJudgmentVocab:
    """Known values for ``EgoVehicle.driving_judgment``."""

    GOOD = "good"
    NEUTRAL = "neutral"
    BAD = "bad"


class EventfulReasonVocab:
    """Known values for ``SilAvAnnotation.eventful_reason``. Advisory; field is ``str``."""

    EGO_ADAPTS = "ego_adapts"
    SPECIAL_ENVIRONMENT = "special_environment"
    AGENT_ADAPTS = "agent_adapts"
    OTHER = "other"


__all__ = [
    "Agent",
    "AgentAction",
    "AgentActionTypePedestrianVocab",
    "AgentActionTypeVocab",
    "AgentAmountVocab",
    "AgentProperty",
    "AgentPropertyTypeVocab",
    "AgentTypeVocab",
    "AnnotationBundle",
    "AnnotationStatusVocab",
    "Condition",
    "ConditionTypeVocab",
    "Containment",
    "DirectionRelToEgoVocab",
    "DrivingJudgmentVocab",
    "EgoAction",
    "EgoActionTypeVocab",
    "EgoPropertyTypeVocab",
    "EgoRelativePose",
    "EgoVehicle",
    "Environment",
    "EnvironmentTypeVocab",
    "EventfulReasonVocab",
    "Influence",
    "Keypoint",
    "LightColorVocab",
    "LightShapeVocab",
    "LightStateTypeVocab",
    "LightStates",
    "MotionStateVocab",
    "ObjectStateEntry",
    "OpenStateVocab",
    "PositionRelToEgoVocab",
    "QAIssue",
    "SignTypeVocab",
    "SignalHead",
    "SignalIntentVocab",
    "SignalSourceVocab",
    "SignalingDetails",
    "SilAvAnnotation",
    "TrafficLight",
    "TrafficObject",
    "TrafficObjectQuantityVocab",
    "TrafficObjectTypeVocab",
    "UiConfig",
    "VideoMeta",
]
