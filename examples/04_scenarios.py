# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Driving-scenario detection.

Encodes ~20 representative driving scenarios as named DSL queries.
Runs them all against the corpus and prints the candidate-clip count
for each.

    CASCADE_AV_DATASET_ROOT=/path/to/json_annotations \\
        uv run python examples/04_scenarios.py

The queries here are *necessary-condition* filters — they reject
clearly-not-this-scenario candidates. A real classifier (VLM, human,
heuristic) would then disambiguate the survivors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from cascade_av.dataset import CascadeDataset


@dataclass(frozen=True)
class Scenario:
    id: str       # short descriptive slug — also the column key in the printout
    title: str    # one-line human description
    query: str    # DSL — copy-paste-able into the API


# ---------------------------------------------------------------------------
# Scenario catalog — representative sample, by category.
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    # ===== VRU Interactions =====
    Scenario(
        id="ped-crosswalk-drive",
        title="Ego drives through crosswalk while pedestrian present",
        query="agent.type = ped and env.type = crosswalk and ego.action = drive",
    ),
    Scenario(
        id="ped-crosswalk-yield",
        title="Ego yields/stops at crosswalk for a pedestrian",
        query="agent.type = ped and env.type = crosswalk "
              "and ego.action in (stop, yield, decel)",
    ),
    Scenario(
        id="ped-jaywalk-yield",
        title="Pedestrian jaywalks; ego brakes / yields",
        # Post schema-2.0.0, `jaywalk` is a parenthesized suffix on
        # `action_type` rather than a flag — enumerate the four combined
        # strings (walk/run × with/without erratic). Same canon as
        # examples/02_query_operators.py.
        query='agent(type = ped, action.type in ('
              '"oxd:Walk (jaywalk)", "oxd:Walk (jaywalk, erratic)", '
              '"oxd:Run (jaywalk)", "oxd:Run (jaywalk, erratic)")) '
              "and ego.action in (stop, yield, decel)",
    ),
    Scenario(
        id="ped-crosswalk-turn",
        title="Pedestrian crosses while ego is turning",
        query="agent.type = ped and env.type = crosswalk "
              "and ego.action in (turn_left, turn_right)",
    ),
    Scenario(
        id="cyclist-defensive",
        title="Cyclist mid-block in ego's path; ego defensive",
        query="agent.type = cyclist and ego.action in (stop, yield, decel)",
    ),

    # ===== Signalized Intersections =====
    Scenario(
        id="red-light-stop",
        title="Ego stops at a red traffic light (nominal)",
        query="light.color = red and ego.action = stop and env.type = intersection",
    ),
    Scenario(
        id="signal-blackout-stop",
        title="Signal blackout — ego treats as all-way stop",
        query="light.state = off and ego.action = stop and env.type = intersection",
    ),
    Scenario(
        id="yellow-in-intersection",
        title="Ego already in intersection when yellow begins, proceeds",
        query="light(color = yellow, ego_in_on_yellow = true) "
              "and ego.action in (drive, enter, creep)",
    ),
    Scenario(
        id="yellow-approach-stop",
        title="Ego approaches intersection, yellow appears, ego stops",
        query="light(color = yellow, on_ego_path = true) and ego.action = stop",
    ),
    Scenario(
        id="yellow-could-have-cleared",
        title="Ego stops on yellow it could have cleared safely",
        query="light(color = yellow, could_have_cleared = true) and ego.action = stop",
    ),

    # ===== Stop / Right-of-Way =====
    Scenario(
        id="aws-basic",
        title="Ego stops at a stop sign",
        query="obj.type = stop_sign and ego.action = stop",
    ),
    Scenario(
        id="yield-basic",
        title="Ego yields at a yield sign",
        query="obj.type = yield_sign and ego.action = yield",
    ),

    # ===== Obstacle Avoidance & Lane Adjustment =====
    Scenario(
        id="vehicle-stopped-front-nudge",
        title="Vehicle stopped in front of ego, ego nudges or changes lane",
        query="agent(type = vehicle, pos = front, action(type in (stop, not_move))) "
              "and ego.action in (nudge, change_lane_left, change_lane_right)",
    ),
    Scenario(
        id="vehicle-front-signaling-left",
        title="Vehicle in front of ego is signaling left",
        query="agent(type = vehicle, pos = front, signaling = turn)",
    ),

    # ===== Highway & Lane Dynamics =====
    Scenario(
        id="lane-change-multi-lane",
        title="Multi-lane road, ego changes lane",
        query="env(type = road, lanes >= 2) "
              "and ego.action in (change_lane_left, change_lane_right)",
    ),

    # ===== Authority & Emergency =====
    Scenario(
        id="officer-stop",
        title="Officer signaling stop; ego stops",
        query="agent(type = officer, signaling = stop) and ego.action = stop",
    ),

    # ===== Roundabouts =====
    Scenario(
        id="roundabout-transit",
        title="Ego transits roundabout without yielding",
        query="env.type = roundabout and ego.action = drive",
    ),
    Scenario(
        id="roundabout-cut-in",
        title="Vehicle cuts in at roundabout exit; ego brakes",
        query="env.type = roundabout "
              "and agent(type = vehicle, action(type in (change_lane, change_lane_left, change_lane_right))) "
              "and ego.action in (decel, stop, nudge)",
    ),

    # ===== Causal queries (DSL distinguishing feature) =====
    Scenario(
        id="ego-decel-because-ped",
        title="Ego decelerates BECAUSE OF a pedestrian (causal edge)",
        query="ego.action = decel because_of agent.type = ped",
    ),
    Scenario(
        id="ego-drive-because-ped",
        title="Ego drives BECAUSE OF a pedestrian (causal edge — pedestrian-aware nominal)",
        query="ego.action = drive because_of agent.type = ped",
    ),
    Scenario(
        id="ego-yield-because-vehicle",
        title="Ego yields BECAUSE OF another vehicle",
        query="ego.action = yield because_of agent.type = vehicle",
    ),

    # ===== Temporal queries =====
    Scenario(
        id="yellow-then-stop",
        title="Yellow light immediately followed by ego stop (within 3s)",
        query="light.color = yellow then(3) ego.action = stop",
    ),
    Scenario(
        id="red-without-stop",
        title="During red light, ego never stops (potential violation)",
        query="within light.color = red: not ego.action = stop",
    ),
]


def main() -> None:
    try:
        dataset_root = Path(os.environ["CASCADE_AV_DATASET_ROOT"])
    except KeyError:
        raise SystemExit(
            "CASCADE_AV_DATASET_ROOT is not set. Point it at a directory of "
            "annotation JSON bundles. See the 'Getting the data' section of "
            "the repo's README.md for how to obtain them."
        )

    ds = CascadeDataset(dataset_root)

    header = f"{'#id':>10}  {'count':>5}  scenario"
    print(header)
    print("-" * 80)
    for s in SCENARIOS:
        try:
            n = ds.count(s.query)
            print(f"{s.id:>10}  {n:>5}  {s.title}")
        except Exception as e:  # noqa: BLE001 — example script
            print(f"{s.id:>10}  ERROR  {s.title}: {e}")

    print()
    print("# To inspect a scenario's matching clips:")
    print('# matches = ds.find(\'agent(type = ped, action.type = "oxd:Walk (jaywalk)")\')')
    print("# print(set(matches.clips()))")


if __name__ == "__main__":
    main()
