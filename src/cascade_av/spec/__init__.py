# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic v2 schema for the CASCADE annotation format.

Single entry point: `AnnotationBundle`. Everything is reachable from it.

Current schema version is :data:`CURRENT_SCHEMA_VERSION` (see
:mod:`cascade_av.spec.versions` for the full registry).
"""

from cascade_av.spec.schema import (
    Agent,
    AgentAction,
    AgentProperty,
    AgentTypeVocab,
    AnnotationBundle,
    AnnotationStatusVocab,
    Condition,
    Containment,
    DrivingJudgmentVocab,
    EgoAction,
    EgoRelativePose,
    EgoVehicle,
    Environment,
    Influence,
    Keypoint,
    LightStates,
    ObjectStateEntry,
    SignalHead,
    SignalingDetails,
    SilAvAnnotation,
    TrafficLight,
    TrafficObject,
    VideoMeta,
)
from cascade_av.spec.versions import (
    CURRENT_SCHEMA_VERSION,
    SCHEMA_HISTORY,
    SUPPORTED_SCHEMA_VERSIONS,
    SchemaVersion,
    changelog_for,
    is_known,
)

__all__ = [
    "AnnotationBundle",
    "VideoMeta",
    "SilAvAnnotation",
    "EgoVehicle",
    "EgoAction",
    "Agent",
    "AgentAction",
    "AgentProperty",
    "EgoRelativePose",
    "SignalingDetails",
    "Containment",
    "Influence",
    "Keypoint",
    "Environment",
    "Condition",
    "TrafficObject",
    "ObjectStateEntry",
    "TrafficLight",
    "SignalHead",
    "LightStates",
    "AgentTypeVocab",
    "AnnotationStatusVocab",
    "DrivingJudgmentVocab",
    # Schema-version registry
    "CURRENT_SCHEMA_VERSION",
    "SCHEMA_HISTORY",
    "SUPPORTED_SCHEMA_VERSIONS",
    "SchemaVersion",
    "changelog_for",
    "is_known",
]
