"""Canonical Pydantic v2 schema for the AV Causal annotation format
(schema_version "2.0.0").

The on-disk corpus is ground truth. Vocabulary-typed fields (action types,
agent types, etc.) are declared as plain `str` rather than `Enum`, because the
upstream tool's frontend writes free-form strings and unknown values must
round-trip rather than fail validation. Advisory `*Vocab` namespace classes
near the bottom of this file enumerate the known values for autocomplete and
for use in queries.

Internal frontend indices (`_track_index`, `_cond_track_index`,
`_state_track_index`, `_influence_track_index`, `_prop_track_index`,
`_cont_track_index`) are explicitly modelled so they round-trip cleanly. They
are exposed in Python without the leading underscore (e.g. `track_index`) via
Pydantic aliases.

`model_config = ConfigDict(extra="allow")` is set on every model so anything
new added by the frontend (or any other producer) is preserved on read and
dump without code change.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Shared model config: permit unknown extras, accept both Python-name and
# JSON-alias inputs.
_MC = ConfigDict(extra="allow", populate_by_name=True)


# -----------------------------------------------------------------------------
# Geometry: bounding boxes
# -----------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """One axis-aligned 2-D bounding box in normalized image coordinates."""

    model_config = _MC

    x: float
    y: float
    w: float
    h: float
    object_id: str = ""
    object_class: str = ""


class BoundingBoxFrame(BaseModel):
    """One frame on the clip timeline carrying zero or more bounding boxes."""

    model_config = _MC

    timestamp: str
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)


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
    track_index: int | None = Field(
        default=None,
        validation_alias="_track_index",
        serialization_alias="_track_index",
    )
    cont_track_index: int | None = Field(
        default=None,
        validation_alias="_cont_track_index",
        serialization_alias="_cont_track_index",
    )


class Influence(BaseModel):
    """A set of entities that modulate ego behaviour over a time window."""

    model_config = _MC

    id: str = ""
    influencers: list[str] = Field(default_factory=list)
    comment: str = ""
    start_timestamp: str = ""
    end_timestamp: str = ""
    influence_track_index: int | None = Field(
        default=None,
        validation_alias="_influence_track_index",
        serialization_alias="_influence_track_index",
    )


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
    track_index: int | None = Field(
        default=None,
        validation_alias="_track_index",
        serialization_alias="_track_index",
    )

    @field_validator("num_lanes", "num_out_lanes", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: object) -> object:
        # The corpus carries `num_lanes: ""` when the annotator left it blank.
        return None if v == "" else v


class Condition(BaseModel):
    """An environmental state. `type` is a list because a single window may
    carry multiple condition labels (e.g. ["Construction Zone", "Clear"])."""

    model_config = _MC

    id: str
    env_id: str = ""
    type: list[str] = Field(default_factory=list)
    condition_other_description: str = ""
    start_timestamp: str = ""
    end_timestamp: str = ""
    track_index: int | None = Field(
        default=None,
        validation_alias="_track_index",
        serialization_alias="_track_index",
    )
    cond_track_index: int | None = Field(
        default=None,
        validation_alias="_cond_track_index",
        serialization_alias="_cond_track_index",
    )


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
    bounding_boxes: list[BoundingBoxFrame] = Field(default_factory=list)
    track_index: int | None = Field(
        default=None,
        validation_alias="_track_index",
        serialization_alias="_track_index",
    )


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
    state_track_index: int | None = Field(
        default=None,
        validation_alias="_state_track_index",
        serialization_alias="_state_track_index",
    )


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
    bounding_boxes: list[BoundingBoxFrame] = Field(default_factory=list)
    track_index: int | None = Field(
        default=None,
        validation_alias="_track_index",
        serialization_alias="_track_index",
    )


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
    prop_track_index: int | None = Field(
        default=None,
        validation_alias="_prop_track_index",
        serialization_alias="_prop_track_index",
    )


class AgentAction(BaseModel):
    """An action performed by an agent over a time window."""

    model_config = _MC

    id: str = ""
    action_type: str = ""  # see AgentActionTypeVocab
    other_description: str = ""
    because_of: list[str] = Field(default_factory=list)
    link_to: list[str] = Field(default_factory=list)
    action_target: list[str] = Field(default_factory=list)
    start_timestamp: str = ""
    end_timestamp: str = ""
    # Conditional flags — only set for specific action types. Most are unused
    # in the current corpus but the producer UI exposes them.
    turn_protected: bool | None = None
    change_where: str | None = None  # "Left" | "Right"
    jaywalk_flag: bool | None = None
    erratic_flag: bool | None = None
    ego_lane_flag: bool | None = None
    nudge_magnitude: str | None = None  # "In Lane" | "Out-of-Lane"
    is_aggressive_or_cut_in: bool | None = None
    illegal_flag: bool = False
    maneuver_aborted_flag: bool = False
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
    bounding_boxes: list[BoundingBoxFrame] = Field(default_factory=list)
    track_index: int | None = Field(
        default=None,
        validation_alias="_track_index",
        serialization_alias="_track_index",
    )


class EgoAction(BaseModel):
    """An action performed by the ego vehicle over a time window."""

    model_config = _MC

    id: str = ""
    type: str = ""  # see EgoActionTypeVocab
    turn_other_description: str = ""
    action_other_description: str = ""
    because_of: list[str] = Field(default_factory=list)
    link_to: list[str] = Field(default_factory=list)
    action_target: list[str] = Field(default_factory=list)
    illegal_flag: bool = False
    is_aggressive_or_cut_in: bool | None = None
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
    driving_judgment: str | None = None  # emoji glyph; see DrivingJudgmentVocab


# -----------------------------------------------------------------------------
# Top-level annotation container
# -----------------------------------------------------------------------------

class SilAvAnnotation(BaseModel):
    """The annotation payload for one clip."""

    model_config = _MC

    eventful: bool | None = None
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


class AnnotationBundle(BaseModel):
    """Root model — one serialized JSON file equals one AnnotationBundle."""

    model_config = _MC

    schema_version: str = "2.0.0"
    video: VideoMeta
    annotation: SilAvAnnotation = Field(default_factory=SilAvAnnotation)
    status: str = "annotating"  # see AnnotationStatusVocab
    provenance: dict[str, Any] = Field(default_factory=lambda: {"generated_by": "human"})


# -----------------------------------------------------------------------------
# Advisory vocabularies — the fields above are open `str`; these enumerate
# the known values for autocomplete and for use in queries.
# -----------------------------------------------------------------------------

class AgentTypeVocab:
    """Known values for Agent.type. Advisory, not exhaustive; field is `str`."""

    CAR = "oxd:Car"
    TRUCK = "oxd:Truck"
    BUS = "PublicBus"
    BICYCLE = "oxd:Bicycle"
    MOTORCYCLE = "oxd:Motorcycle"
    SCOOTER = "fst:Scooter"
    HEAVY_DUTY = "Heavy-duty vehicle"
    EMERGENCY_VEHICLE = "oxd:EmergencyVehicle"
    ANIMAL = "oxd:Animal"
    PEDESTRIAN_ADULT = "Pedestrian (Adult)"
    PEDESTRIAN_KID_TEEN = "Pedestrian (Kid/Teen)"
    PEDESTRIAN_PERSONNEL = "Pedestrian (Personnel)"
    PEDESTRIAN_OFFICER = "Pedestrian (Officer)"
    PEDESTRIAN_STROLLER = "Pedestrian (Stroller)"
    PEDESTRIAN_WHEELCHAIR = "Pedestrian (oxd:Wheelchair)"
    PEDESTRIAN_OTHER = "Pedestrian (Other)"
    OTHER = "Other"


class AnnotationStatusVocab:
    """Known status values across producers. Field is `str`."""

    # From the upstream tool
    PENDING = "pending"
    ANNOTATING = "annotating"
    APPROVED = "approved"
    DISAPPROVED = "disapproved"
    # From the corpus
    SUBMITTED = "submitted"
    NEEDS_REVISION = "needs_revision"


class DrivingJudgmentVocab:
    """Emoji glyphs used for ego_vehicle.driving_judgment."""

    GOOD = "\U0001f642"  # 🙂
    NEUTRAL = "\U0001f610"  # 😐
    BAD = "\U0001f641"  # 🙁


__all__ = [
    "Agent",
    "AgentAction",
    "AgentProperty",
    "AgentTypeVocab",
    "AnnotationBundle",
    "AnnotationStatusVocab",
    "BoundingBox",
    "BoundingBoxFrame",
    "Condition",
    "Containment",
    "DrivingJudgmentVocab",
    "EgoAction",
    "EgoRelativePose",
    "EgoVehicle",
    "Environment",
    "Influence",
    "Keypoint",
    "LightStates",
    "ObjectStateEntry",
    "SignalHead",
    "SignalingDetails",
    "SilAvAnnotation",
    "TrafficLight",
    "TrafficObject",
    "VideoMeta",
]
