# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scan the corpus and print every distinct value for query-relevant fields.

Output is grouped by entity + attribute, sorted, with counts. Drives the
alias tables in `src/cascade_av/query/constants.py`.

Usage:
    uv run python scripts/scan_corpus_vocabulary.py [CORPUS_DIR]
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from cascade_av.io import load_dir
from cascade_av.spec import AnnotationBundle

DEFAULT_CORPUS = Path("/home/horde/01_json_annotations")


def _collect(bundles: list[AnnotationBundle]) -> dict[str, Counter]:
    out: dict[str, Counter] = {
        "agent.type": Counter(),
        "agent_action.action_type": Counter(),
        "agent_property.property_type": Counter(),
        "agent_property.signaling.intent": Counter(),
        "agent_property.signaling.source": Counter(),
        "agent_property.signaling.sign_type": Counter(),
        "ego_action.type": Counter(),
        "ego_property.type": Counter(),
        "ego.driving_judgment": Counter(),
        "environment.type": Counter(),
        "condition.type": Counter(),
        "traffic_light.type": Counter(),
        "light_state.color": Counter(),
        "light_state.type": Counter(),
        "light_state.shape": Counter(),
        "traffic_object.type": Counter(),
        "traffic_object.quantity": Counter(),
        "object_state.open_state": Counter(),
        "object_state.motion_state": Counter(),
        "ego_relative_pose.position": Counter(),
        "ego_relative_pose.direction": Counter(),
        "containment.edge": Counter(),
    }

    for b in bundles:
        ann = b.annotation

        for agent in ann.agents:
            if agent.type:
                out["agent.type"][agent.type] += 1
            for a in agent.actions:
                if a.action_type:
                    out["agent_action.action_type"][a.action_type] += 1
            for p in agent.properties:
                if p.property_type:
                    out["agent_property.property_type"][p.property_type] += 1
                sd = getattr(p, "signaling_details", None)
                if sd is not None:
                    if getattr(sd, "intent", None):
                        out["agent_property.signaling.intent"][sd.intent] += 1
                    if getattr(sd, "source", None):
                        out["agent_property.signaling.source"][sd.source] += 1
                    if getattr(sd, "sign_type", None):
                        out["agent_property.signaling.sign_type"][sd.sign_type] += 1
            for pose in getattr(agent, "ego_relative_pose", []):
                if getattr(pose, "position_rel_to_ego", None):
                    out["ego_relative_pose.position"][pose.position_rel_to_ego] += 1
                if getattr(pose, "direction_rel_to_ego", None):
                    out["ego_relative_pose.direction"][pose.direction_rel_to_ego] += 1
            for cont in getattr(agent, "containment", []):
                edge = getattr(cont, "edge", None)
                if edge:
                    out["containment.edge"][edge] += 1

        ego = ann.ego_vehicle
        if getattr(ego, "driving_judgment", None):
            out["ego.driving_judgment"][ego.driving_judgment] += 1
        for a in ego.actions:
            t = getattr(a, "type", None) or getattr(a, "action_type", None)
            if t:
                out["ego_action.type"][t] += 1
        for p in getattr(ego, "properties", []):
            t = getattr(p, "type", None) or getattr(p, "property_type", None)
            if t:
                out["ego_property.type"][t] += 1

        for env in ann.environments:
            if env.type:
                out["environment.type"][env.type] += 1

        for cond in ann.conditions:
            for t in cond.type or []:
                out["condition.type"][t] += 1

        for tl in ann.traffic_lights:
            if getattr(tl, "type", None):
                out["traffic_light.type"][tl.type] += 1
            for head in getattr(tl, "signal_heads", []):
                for ls in getattr(head, "state_sequence", []):
                    if getattr(ls, "color", None):
                        out["light_state.color"][ls.color] += 1
                    if getattr(ls, "type", None):
                        out["light_state.type"][ls.type] += 1
                    if getattr(ls, "shape", None):
                        out["light_state.shape"][ls.shape] += 1

        for to in ann.traffic_objects:
            if to.type:
                out["traffic_object.type"][to.type] += 1
            if getattr(to, "quantity", None):
                out["traffic_object.quantity"][to.quantity] += 1
            for s in getattr(to, "state_sequence", []):
                if getattr(s, "open_state", None):
                    out["object_state.open_state"][s.open_state] += 1
                if getattr(s, "motion_state", None):
                    out["object_state.motion_state"][s.motion_state] += 1

    return out


def main() -> None:
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CORPUS
    print(f"# scanning {corpus}", file=sys.stderr)
    bundles = load_dir(corpus)
    print(f"# loaded {len(bundles)} bundles", file=sys.stderr)

    counters = _collect(bundles)

    for key, ctr in counters.items():
        print(f"\n## {key}  ({len(ctr)} distinct, {sum(ctr.values())} total)")
        for value, n in sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {n:>6}  {value!r}")


if __name__ == "__main__":
    main()
