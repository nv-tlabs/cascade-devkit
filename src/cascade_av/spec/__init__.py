# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic v2 schema for the CASCADE annotation format (schema_version 2.0.0).

Single entry point: `AnnotationBundle`. Everything is reachable from it.
"""

from cascade_av.spec.schema import (
    Agent,
    AgentAction,
    AgentProperty,
    AgentTypeVocab,
    AnnotationBundle,
    AnnotationStatusVocab,
    BoundingBox,
    BoundingBoxFrame,
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
    "BoundingBox",
    "BoundingBoxFrame",
    "AgentTypeVocab",
    "AnnotationStatusVocab",
    "DrivingJudgmentVocab",
]
