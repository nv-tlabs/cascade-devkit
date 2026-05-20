# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`Segment` dataclass + `annotation_to_segments` — the annotator's timeline
contract ported to Python.

This is the single load-bearing porting effort identified in
`meta/10_visualization_api_plan.md`. The annotator's TypeScript
`annotationToSegments()` (in `tools/annotator/web/src/lib/timeline-utils.ts`,
function near line 833) folds an `SilAvAnnotation` into a flat list of
`TimelineSegment` rows — one per agent action, ego action, environment,
condition, traffic-object state, signal-head state, containment, influence,
or property. Both the annotator's Timeline rasterizer and the viz layer's
upcoming Plotly renderer consume the same shape, so the two stay aligned
by construction.

Field parity with `TimelineSegment` (TS, `tools/annotator/web/src/lib/types.ts:289`)::

    interface TimelineSegment {
      id: string; trackId: string; label: string
      t0: number; t1: number
      illegal?: boolean; because_of?: string[]; meta?: unknown
    }

The Python `Segment` is the immutable equivalent. `because_of` is a tuple
(hashable, matches the dataclass `frozen=True` contract). `meta` is a plain
dict of provenance keys carried through from the TS port — callers should
treat its contents as advisory; the load-bearing fields are the six above.

The TS port performs several in-place migrations on the annotation before
building segments (`migrateEgoActions`, `migrateAgentActions`,
`migrateLanesObscuredToCondition`, etc.). Those normalize older corpus
forms into the current schema. The Python port deliberately skips them:
the corpus is parsed via the canonical `AnnotationBundle` schema 2.0.0
(`src/cascade_av/spec/schema.py`) which is already in the target form,
and we never mutate the user's annotation here.

Lane indexing (timeline rows / sub-lanes) is read inline from the typed
model: each entity carries an optional ``track_index`` (and, where the
producer emits one, ``cond_track_index`` / ``state_track_index`` /
``influence_track_index``). Missing → row 0. ``AgentProperty`` carries no
inline lane index in 2.0.0; property subtracks fall back to greedy lane
assignment in :func:`assign_lanes`. The annotator's
``autoAssignOverlappingTracks`` is its own concern; viz only reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from cascade_av.query.time import parse_timestamp_or
from cascade_av.spec import (
    Agent,
    AnnotationBundle,
    Condition,
    Containment,
    Environment,
    Influence,
    SignalHead,
    SilAvAnnotation,
    TrafficLight,
    TrafficObject,
)

# Family taxonomy — one of the eleven values below. Mirrors the
# annotator's family bands in `Timeline.tsx`: a top-level `parent` bar
# per entity, plus per-family sub-rows (`containment`, `pose`, …). The
# painter groups segments by `(group, family, sh_index)` to lay out
# distinct labeled bands instead of mixing every sub-row into one row.
SegmentFamily = Literal[
    "parent",
    "action",
    "property",
    "containment",
    "influence",
    "pose",
    "condition",
    "state",
    "signal_head",
    "env_control",
    "physical_containment",
]

# Numeric-coordinate-required parser. Delegates to the single
# `cascade_av.query.time.parse_timestamp_or` helper so segment t0/t1
# never silently disagree with the rest of the DevKit. The TS port's
# `parseTs` returns 0.0 on empty / unparseable input; we preserve that
# contract by passing `default=0.0`. The DevKit's strict variant
# (`parse_timestamp`) returns `None` on the same inputs — the two used to
# be implemented twice with subtly different regex anchoring (`search` vs
# `match`); they're now one parser, two boundary contracts.
def _parse_ts(ts: str | None) -> float:
    """Parse a `M:S.D` timestamp string to seconds; `0.0` on bad input.

    Thin alias for `parse_timestamp_or(ts, default=0.0)`. Kept for
    intra-module readability and back-compat with `tests/test_viz_segments.py`,
    which asserts the TS-parity contract.
    """
    return parse_timestamp_or(ts, default=0.0)


@dataclass(frozen=True, slots=True)
class Segment:
    """One timeline row — the Python mirror of TS `TimelineSegment`.

    Attributes:
        id: stable string id for the segment within its annotation.
        track_id: the timeline lane this segment belongs to
            (`env_<n>`, `ego_act`, `obj_<n>`, `light_<n>`, `agent_<n>`).
        label: human-readable label as the annotator renders it.
        t0: segment start, seconds.
        t1: segment end, seconds.
        illegal: True for actions / containment flagged illegal by the
            annotator. False otherwise; never None.
        because_of: tuple of ids this segment cites as causes
            (for `because_of` arrow rendering).
        meta: provenance dict — entity kind, source indices, source object.
            Callers should treat the keys as advisory; only the six fields
            above are load-bearing.
        family: which annotator sub-row this segment paints onto — one of
            the eleven `SegmentFamily` values. Drives the family-band
            layout in `_paint_timeline_onto`. Defaults to `"parent"` so
            historical `Segment(...)` constructions in tests keep working
            without an explicit family.
    """

    id: str
    track_id: str
    label: str
    t0: float
    t1: float
    illegal: bool = False
    because_of: tuple[str, ...] = ()
    meta: dict[str, Any] | None = field(default=None)
    family: SegmentFamily = "parent"


# -----------------------------------------------------------------------------
# Label helpers — port the TS label-building heuristics with the same
# fall-through order, but without the annotator's display-name lookup tables.
# The display-name maps live only in TS; here we fall back to the raw type
# string when no humanized variant is present. This keeps the port simple
# and lossless — callers (and tests) can compare against the schema strings.
# -----------------------------------------------------------------------------


def _env_lane_suffix(env: Environment) -> str:
    """Mirror of the TS `environmentLaneLabel` helper."""
    if env.type in ("fst:LaneMerge", "fst:LaneFork"):
        in_lanes = env.num_lanes or 0
        out_lanes = env.num_out_lanes or 0
        if in_lanes > 0 and out_lanes > 0:
            return f" [{in_lanes}->{out_lanes}L]"
        if in_lanes > 0:
            return f" [in {in_lanes}L]"
        if out_lanes > 0:
            return f" [out {out_lanes}L]"
        return ""
    if (env.num_lanes or 0) > 1:
        return f" [{env.num_lanes}L]"
    return ""


def _env_label(env: Environment) -> str:
    base = (
        f"Other ({env.type_other_description})"
        if env.type == "Other" and env.type_other_description
        else (env.type or "Environment")
    )
    suffix = _env_lane_suffix(env)
    name = f' · "{env.name}"' if env.name else ""
    return f"{base}{suffix} ({env.id}{name})"


def _cont_label(
    envs_by_id: dict[str, Environment],
    cont: Containment,
    *,
    omit_lane: bool = False,
) -> str:
    """Mirror of the TS `contLabel` helper."""
    prefix_parts: list[str] = []
    if cont.near_flag:
        prefix_parts.append("Near")
    if cont.edge == "left":
        prefix_parts.append("Left edge")
    elif cont.edge == "right":
        prefix_parts.append("Right edge")
    prefix = f"{', '.join(prefix_parts)}: " if prefix_parts else ""
    lane_suffix = "" if omit_lane else f" Lane {cont.lane_number or '?'}"
    env = envs_by_id.get(cont.env_id)
    if env is not None:
        env_type = (env.type or "Environment").replace(" (specify)", "")
        name_suffix = f' · "{env.name}"' if env.name else ""
        env_label = f"{env_type} ({env.id}{name_suffix})"
        return f"{prefix}{env_label}{lane_suffix}"
    return f"{prefix}Env {cont.env_id or '?'}{lane_suffix}"


def _cond_label(cond: Condition) -> str:
    if cond.type and cond.type[0] == "Other" and cond.condition_other_description:
        type_str = f"Other ({cond.condition_other_description})"
    elif cond.type:
        type_str = cond.type[0]
    else:
        type_str = ""
    return f"{type_str} ({cond.id})"


def _ego_action_label(act: Any) -> str:
    if act.type == "Other" and act.action_other_description:
        return f"Other ({act.action_other_description})"
    if act.type == "Other oxd:MakeATurn" and act.turn_other_description:
        return f"Other turn ({act.turn_other_description})"
    return act.type or "Action"


def _agent_action_label(act: Any) -> str:
    if act.action_type == "Other" and act.other_description:
        return f"Other ({act.other_description})"
    if act.action_type == "Other oxd:MakeATurn" and act.other_description:
        return f"Other turn ({act.other_description})"
    return act.action_type or "Action"


def _property_label(prop: Any) -> str:
    if prop.property_type == "Signal":
        intent = prop.signaling_details.intent if prop.signaling_details else None
        return f"Signal: {intent or '?'}"
    if prop.property_type == "Other" and prop.other_description:
        return f"Other ({prop.other_description})"
    return prop.property_type or "Property"


def _agent_prefix(agent: Agent) -> str:
    type_label = (
        f"Other ({agent.other_type_description})"
        if agent.type == "Other" and agent.other_type_description
        else agent.type
    )
    amount = "" if agent.amount == "Single" else f"{agent.amount} "
    return f"{amount}{type_label}"


def _influence_label(infl: Influence) -> str:
    return f"Infl: {', '.join(infl.influencers) if infl.influencers else '?'}"


# -----------------------------------------------------------------------------
# annotation_to_segments — the main entry point.
# -----------------------------------------------------------------------------


def annotation_to_segments(bundle: AnnotationBundle) -> list[Segment]:
    """Flatten an `AnnotationBundle` to the annotator's timeline shape.

    Ports `annotationToSegments()` from
    `tools/annotator/web/src/lib/timeline-utils.ts` (function near line 833).
    Covers the same families: environments, conditions, ego actions /
    containment / influences / properties, traffic objects + their state
    and containment subtracks, traffic lights + signal heads + light
    states + env_controlled, and agents + actions / properties / pose /
    containment / influences.

    Args:
        bundle: a parsed annotation bundle.

    Returns:
        A flat list of `Segment`s in the same emission order as the TS port:
        environments → conditions → ego (action / containment / influence
        / property) → traffic objects (parent → state → containment) →
        traffic lights (parent → physical containment → signal head →
        light states → env_controlled) → agents (parent → action →
        property → pose → containment → influence).
    """
    ann: SilAvAnnotation = bundle.annotation
    segs: list[Segment] = []
    next_id = 0

    def _next() -> int:
        nonlocal next_id
        nid = next_id
        next_id += 1
        return nid

    # Layout indices live inline on the typed model (see schema 2.0.0).
    # Missing → 0, matching the pre-reboot "no index set" behaviour.
    envs_by_id: dict[str, Environment] = {e.id: e for e in ann.environments}

    # --- Environments ---
    for ei, env in enumerate(ann.environments):
        track_idx = env.track_index or 0
        segs.append(
            Segment(
                id=f"env_{ei}_{_next()}",
                track_id=f"env_{track_idx}",
                label=_env_label(env),
                t0=_parse_ts(env.start_timestamp),
                t1=_parse_ts(env.end_timestamp),
                meta={"_envIndex": ei, "_objKind": "environment"},
                family="parent",
            )
        )

    # --- Conditions (subtracks under parent env) ---
    for ci, cond in enumerate(ann.conditions):
        parent_env = envs_by_id.get(cond.env_id)
        parent_track_idx = (parent_env.track_index if parent_env else 0) or 0
        segs.append(
            Segment(
                id=f"cond_{ci}_{_next()}",
                track_id=f"env_{parent_track_idx}",
                label=_cond_label(cond),
                t0=_parse_ts(cond.start_timestamp),
                t1=_parse_ts(cond.end_timestamp),
                meta={
                    "_condIndex": ci,
                    "_isCondSubtrack": True,
                    "_cond_track_index": cond.cond_track_index or 0,
                },
                family="condition",
            )
        )

    # --- Ego parent ---
    # Ego is a singleton (no track_index, no name field) but emits a
    # parent band like every other category so the families filter
    # retains an "Ego" header above whichever sub-rows survive. Only
    # emitted when Ego has at least one sub-row — mirrors the "no
    # orphan parent" rule used in the families-filter pre-pass and
    # keeps `annotation_to_segments` empty-bundle-clean. The parent
    # spans the full clip duration; Ego is present the whole clip by
    # definition. Carries `_isEgoParent` in meta so the painter can
    # distinguish it from regular ego_act subtracks if needed.
    ego = ann.ego_vehicle
    if ego.actions or ego.containment or ego.influenced_by or ego.properties:
        duration_s = (
            float(bundle.video.duration_s or 0.0) if bundle.video else 0.0
        )
        segs.append(
            Segment(
                id=f"ego_parent_{_next()}",
                track_id="ego_act",
                label="Ego",
                t0=0.0,
                t1=duration_s,
                meta={"_isEgoParent": True},
                family="parent",
            )
        )
    # --- Ego actions ---
    for i, act in enumerate(ego.actions):
        segs.append(
            Segment(
                id=f"ego_act_{_next()}",
                track_id="ego_act",
                label=_ego_action_label(act),
                t0=_parse_ts(act.start_timestamp),
                t1=_parse_ts(act.end_timestamp),
                illegal=bool(act.illegal_flag),
                because_of=tuple(act.because_of),
                meta={"_egoActIndex": i, "_isEgoActionSubtrack": True},
                family="action",
            )
        )
    # --- Ego containment ---
    for ci, cont in enumerate(ego.containment):
        segs.append(
            Segment(
                id=f"ego_cont_{ci}_{_next()}",
                track_id="ego_act",
                label=_cont_label(envs_by_id, cont),
                t0=_parse_ts(cont.start_timestamp),
                t1=_parse_ts(cont.end_timestamp),
                illegal=bool(cont.illegal_flag),
                meta={
                    "_contIndex": ci,
                    "_isEgoContSubtrack": True,
                    "_cont_track_index": cont.track_index or 0,
                },
                family="containment",
            )
        )
    # --- Ego influences ---
    for ii, infl in enumerate(ego.influenced_by):
        segs.append(
            Segment(
                id=f"ego_infl_{ii}_{_next()}",
                track_id="ego_act",
                label=_influence_label(infl),
                t0=_parse_ts(infl.start_timestamp),
                t1=_parse_ts(infl.end_timestamp),
                meta={
                    "_inflIndex": ii,
                    "_isEgoInfluenceSubtrack": True,
                    "_influence_track_index": infl.influence_track_index or 0,
                },
                family="influence",
            )
        )
    # --- Ego properties ---
    # `AgentProperty` carries no inline lane index in 2.0.0; `assign_lanes`
    # falls through to greedy assignment when no hint is present in meta.
    for pi, prop in enumerate(ego.properties):
        segs.append(
            Segment(
                id=f"ego_prop_{pi}_{_next()}",
                track_id="ego_act",
                label=_property_label(prop),
                t0=_parse_ts(prop.start_timestamp),
                t1=_parse_ts(prop.end_timestamp),
                meta={
                    "_propIndex": pi,
                    "_isEgoPropertySubtrack": True,
                },
                family="property",
            )
        )

    # --- Traffic objects ---
    obj: TrafficObject
    for oi, obj in enumerate(ann.traffic_objects):
        track_idx = obj.track_index or 0
        name_prefix = f"{obj.name} · " if obj.name else ""
        type_label = obj.type
        if obj.type.startswith("Other") and obj.other_type_description:
            type_label = obj.type.replace(
                "(specify)", f"({obj.other_type_description})"
            )
        segs.append(
            Segment(
                id=f"obj_{oi}_{_next()}",
                track_id=f"obj_{track_idx}",
                label=f"{name_prefix}{type_label} ({obj.id})",
                t0=_parse_ts(obj.visibility_start_timestamp),
                t1=_parse_ts(obj.visibility_end_timestamp),
                meta={"_objIndex": oi, "_objKind": "traffic_object"},
                family="parent",
            )
        )
        # State subtracks
        for si, st in enumerate(obj.state_sequence):
            label = "/".join(p for p in (st.motion_state, st.open_state) if p) or "State"
            segs.append(
                Segment(
                    id=f"obj_state_{oi}_{si}_{_next()}",
                    track_id=f"obj_{track_idx}",
                    label=label,
                    t0=_parse_ts(st.start_timestamp),
                    t1=_parse_ts(st.end_timestamp),
                    meta={
                        "_objIndex": oi,
                        "_stateIndex": si,
                        "_isObjStateSubtrack": True,
                    },
                    family="state",
                )
            )
        # Object containment
        for ci, cont in enumerate(obj.containment):
            segs.append(
                Segment(
                    id=f"obj_cont_{oi}_{ci}_{_next()}",
                    track_id=f"obj_{track_idx}",
                    label=_cont_label(envs_by_id, cont),
                    t0=_parse_ts(cont.start_timestamp),
                    t1=_parse_ts(cont.end_timestamp),
                    illegal=bool(cont.illegal_flag),
                    meta={
                        "_objIndex": oi,
                        "_contIndex": ci,
                        "_isObjContSubtrack": True,
                        "_cont_track_index": cont.track_index or 0,
                    },
                    family="containment",
                )
            )

    # --- Traffic lights ---
    light: TrafficLight
    for li, light in enumerate(ann.traffic_lights):
        track_idx = light.track_index or 0
        name_prefix = f"{light.name} · " if light.name else ""
        segs.append(
            Segment(
                id=f"light_{li}_{_next()}",
                track_id=f"light_{track_idx}",
                label=f"{name_prefix}TrafficLight ({light.id})",
                t0=_parse_ts(light.visibility_start_timestamp),
                t1=_parse_ts(light.visibility_end_timestamp),
                meta={"_lightIndex": li, "_objKind": "traffic_light"},
                family="parent",
            )
        )
        # Physical containment (omit_lane=True in TS) — labeled
        # "TL control" in the annotator's y-axis. Lives at the light's
        # top level, not nested under any signal head, so `_sh_index`
        # is omitted from meta.
        for ci, cont in enumerate(light.containment):
            segs.append(
                Segment(
                    id=f"light_phys_cont_{li}_{ci}_{_next()}",
                    track_id=f"light_{track_idx}",
                    label=_cont_label(envs_by_id, cont, omit_lane=True),
                    t0=_parse_ts(cont.start_timestamp),
                    t1=_parse_ts(cont.end_timestamp),
                    illegal=bool(cont.illegal_flag),
                    meta={
                        "_lightIndex": li,
                        "_contIndex": ci,
                        "_isLightPhysContSubtrack": True,
                        "_cont_track_index": cont.track_index or 0,
                    },
                    family="physical_containment",
                )
            )
        # Signal heads. The annotator paints a per-head stripe:
        # the SH bar, the env_controlled "containment" rows under it,
        # and the state rows. We tag each per-head segment with
        # `_sh_index = hi` so the painter can lay out one band per head
        # per family.
        sh: SignalHead
        for hi, sh in enumerate(light.signal_heads):
            segs.append(
                Segment(
                    id=f"light_sh_{li}_{hi}_{_next()}",
                    track_id=f"light_{track_idx}",
                    label=f"SignalHead ({sh.id})",
                    t0=_parse_ts(sh.start_timestamp),
                    t1=_parse_ts(sh.end_timestamp),
                    meta={
                        "_lightIndex": li,
                        "_headIndex": hi,
                        "_sh_index": hi,
                        "_isSignalHeadSubtrack": True,
                    },
                    family="signal_head",
                )
            )
            for si, st in enumerate(sh.state_sequence):
                st_label = " ".join(p for p in (st.type, st.color, st.shape) if p) or "Light"
                segs.append(
                    Segment(
                        id=f"light_state_{li}_{hi}_{si}_{_next()}",
                        track_id=f"light_{track_idx}",
                        label=st_label,
                        t0=_parse_ts(st.start_timestamp),
                        t1=_parse_ts(st.end_timestamp),
                        meta={
                            "_lightIndex": li,
                            "_headIndex": hi,
                            "_sh_index": hi,
                            "_stateIndex": si,
                            "_isLightStateSubtrack": True,
                        },
                        family="state",
                    )
                )
            for ci, cont in enumerate(sh.env_controlled):
                segs.append(
                    Segment(
                        id=f"light_cont_{li}_{hi}_{ci}_{_next()}",
                        track_id=f"light_{track_idx}",
                        label=_cont_label(envs_by_id, cont),
                        t0=_parse_ts(cont.start_timestamp),
                        t1=_parse_ts(cont.end_timestamp),
                        illegal=bool(cont.illegal_flag),
                        meta={
                            "_lightIndex": li,
                            "_headIndex": hi,
                            "_sh_index": hi,
                            "_contIndex": ci,
                            "_isLightContSubtrack": True,
                        },
                        family="env_control",
                    )
                )

    # --- Agents ---
    for ai, agent in enumerate(ann.agents):
        track_idx = agent.track_index or 0
        prefix = _agent_prefix(agent)
        # Parent visibility bounds — fall back to action / pose envelope.
        if agent.visibility_start_timestamp and agent.visibility_end_timestamp:
            vis_t0 = _parse_ts(agent.visibility_start_timestamp)
            vis_t1 = _parse_ts(agent.visibility_end_timestamp)
        else:
            cands_lo: list[float] = []
            cands_hi: list[float] = []
            for act in agent.actions:
                cands_lo.append(_parse_ts(act.start_timestamp))
                cands_hi.append(_parse_ts(act.end_timestamp))
            for sp in agent.ego_relative_pose:
                cands_lo.append(_parse_ts(sp.start_timestamp))
                cands_hi.append(_parse_ts(sp.end_timestamp))
            if cands_lo:
                vis_t0, vis_t1 = min(cands_lo), max(cands_hi)
            else:
                vis_t0, vis_t1 = 0.0, 0.0
        name_prefix = f"{agent.name} · " if agent.name else ""
        segs.append(
            Segment(
                id=f"agent_{ai}_{_next()}",
                track_id=f"agent_{track_idx}",
                label=f"{name_prefix}{prefix} [Agent]",
                t0=vis_t0,
                t1=vis_t1,
                meta={"_agentIndex": ai, "_objKind": "agent"},
                family="parent",
            )
        )
        # Action subtracks
        for act_idx, act in enumerate(agent.actions):
            t0 = _parse_ts(act.start_timestamp)
            t1 = _parse_ts(act.end_timestamp)
            segs.append(
                Segment(
                    id=f"agent_action_{ai}_{act_idx}_{_next()}",
                    track_id=f"agent_{track_idx}",
                    label=_agent_action_label(act),
                    t0=t0,
                    t1=t1,
                    illegal=bool(act.illegal_flag),
                    because_of=tuple(act.because_of),
                    meta={
                        "_agentIndex": ai,
                        "_actIndex": act_idx,
                        "_isAgentActionSubtrack": True,
                        "_segType": "action",
                        "agent_type": agent.type,
                        "agent_prefix": prefix,
                    },
                    family="action",
                )
            )
        # Property subtracks.
        # `AgentProperty` has no inline lane index in 2.0.0; greedy
        # assignment in `assign_lanes` handles the layout.
        for pi, prop in enumerate(agent.properties):
            segs.append(
                Segment(
                    id=f"agent_prop_{ai}_{pi}_{_next()}",
                    track_id=f"agent_{track_idx}",
                    label=_property_label(prop),
                    t0=_parse_ts(prop.start_timestamp),
                    t1=_parse_ts(prop.end_timestamp),
                    meta={
                        "_agentIndex": ai,
                        "_propIndex": pi,
                        "_isAgentPropertySubtrack": True,
                        "_segType": "property",
                    },
                    family="property",
                )
            )
        # Pose subtracks
        for pi, pose in enumerate(agent.ego_relative_pose):
            pose_label = " / ".join(
                p for p in (pose.position_rel_to_ego, pose.direction_rel_to_ego) if p
            ) or "Pose"
            segs.append(
                Segment(
                    id=f"agent_pose_{ai}_{pi}_{_next()}",
                    track_id=f"agent_{track_idx}",
                    label=pose_label,
                    t0=_parse_ts(pose.start_timestamp),
                    t1=_parse_ts(pose.end_timestamp),
                    meta={
                        "_agentIndex": ai,
                        "_poseIndex": pi,
                        "_isAgentPoseSubtrack": True,
                        "_segType": "pose",
                    },
                    family="pose",
                )
            )
        # Containment subtracks
        for ci, cont in enumerate(agent.containment):
            segs.append(
                Segment(
                    id=f"agent_cont_{ai}_{ci}_{_next()}",
                    track_id=f"agent_{track_idx}",
                    label=_cont_label(envs_by_id, cont),
                    t0=_parse_ts(cont.start_timestamp),
                    t1=_parse_ts(cont.end_timestamp),
                    illegal=bool(cont.illegal_flag),
                    meta={
                        "_agentIndex": ai,
                        "_contIndex": ci,
                        "_isAgentContSubtrack": True,
                        "_cont_track_index": cont.track_index or 0,
                    },
                    family="containment",
                )
            )
        # Influence subtracks
        for ii, infl in enumerate(agent.influenced_by):
            segs.append(
                Segment(
                    id=f"agent_infl_{ai}_{ii}_{_next()}",
                    track_id=f"agent_{track_idx}",
                    label=_influence_label(infl),
                    t0=_parse_ts(infl.start_timestamp),
                    t1=_parse_ts(infl.end_timestamp),
                    meta={
                        "_agentIndex": ai,
                        "_inflIndex": ii,
                        "_isAgentInfluenceSubtrack": True,
                        "_influence_track_index": infl.influence_track_index or 0,
                    },
                    family="influence",
                )
            )

    return segs


# -----------------------------------------------------------------------------
# Lane assignment — Python port of the annotator's `assignTracks`.
# -----------------------------------------------------------------------------

# Meta keys we consult for a producer-supplied lane index, in priority order
# matching the annotator's `assignTracks(..., key)` call sites in
# `tools/annotator/web/src/lib/timeline-utils.ts`. We try the most specific
# subtrack lane key first, then fall back to `_track_index`. Stable: missing
# keys default to None (no hint) — never crash.
_LANE_HINT_KEYS: tuple[str, ...] = (
    "_cond_track_index",
    "_cont_track_index",
    "_influence_track_index",
    "_prop_track_index",
    "_track_index",
)


def _lane_hint(seg: Segment) -> int | None:
    """Return the producer-supplied lane index for `seg`, or None.

    Mirrors the TS `assignTracks(items, key)` "existing lane" preservation
    behaviour: if the annotator already wrote a lane index onto the entity
    (e.g., during an edit session), prefer it over a re-computed value.
    """
    if not seg.meta:
        return None
    for key in _LANE_HINT_KEYS:
        v = seg.meta.get(key)
        if isinstance(v, int) and v >= 0:
            return v
    return None


def assign_lanes(segments: list[Segment]) -> dict[str, int]:
    """Greedy lane assignment within each `track_id`.

    Per `track_id`, sort segments by `(t0, t1)`. For each segment: if it
    carries a producer-supplied lane hint (`_cond_track_index`,
    `_cont_track_index`, `_influence_track_index`, `_prop_track_index`,
    `_track_index`) and that lane is free at this `t0`, keep it.
    Otherwise, place the segment on the lowest-index lane whose last
    segment's `t1` is `<= t0` (no overlap). The output is the smallest
    lane count that accommodates the worst-case overlap stack.

    Ports `assignTracks` from
    `tools/annotator/web/src/lib/timeline-utils.ts:197`. The TS version
    mutates `item[key]` in place; we instead return a mapping so
    `Segment` can remain immutable.

    Args:
        segments: a flat list of `Segment`s. Caller controls grouping —
            we group by `track_id` ourselves.

    Returns:
        Mapping `segment.id -> lane_index` (0-based, dense within each
        `track_id`). Lane indices are not coordinated across track ids;
        the renderer is responsible for stacking lanes within a group
        row.
    """
    # Group by track_id while preserving stable per-group ordering.
    by_track: dict[str, list[Segment]] = {}
    for seg in segments:
        by_track.setdefault(seg.track_id, []).append(seg)

    lanes: dict[str, int] = {}
    for items in by_track.values():
        # Sort by (t0, t1). Ties on t0 break by t1 so longer-overlap
        # items get placed first — matches the TS `a.start - b.start ||
        # a.end - b.end` comparator.
        ordered = sorted(items, key=lambda s: (s.t0, s.t1))
        track_ends: list[float] = []
        for seg in ordered:
            hint = _lane_hint(seg)
            if hint is not None:
                # Extend `track_ends` so `hint` is addressable, then
                # accept the hint iff the lane is free at `seg.t0`.
                while len(track_ends) <= hint:
                    track_ends.append(float("-inf"))
                if track_ends[hint] <= seg.t0:
                    track_ends[hint] = seg.t1
                    lanes[seg.id] = hint
                    continue
            # Otherwise: lowest free lane.
            placed = False
            for lane_idx, end in enumerate(track_ends):
                if end <= seg.t0:
                    track_ends[lane_idx] = seg.t1
                    lanes[seg.id] = lane_idx
                    placed = True
                    break
            if not placed:
                lanes[seg.id] = len(track_ends)
                track_ends.append(seg.t1)
    return lanes


__all__ = ["Segment", "SegmentFamily", "annotation_to_segments", "assign_lanes"]
