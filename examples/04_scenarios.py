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
        # Post-0.6.1, `Jaywalk` is a standalone action (alias
        # `jaywalk_action`); the combined-suffix forms `oxd:Walk
        # (jaywalk)` / `oxd:Run (jaywalk)` are deprecated but still
        # match. Enumerate the standalone action alongside the four
        # combined strings (walk/run × with/without erratic). Same canon
        # as examples/02_query_operators.py.
        query='agent(type = ped, action.type in ('
              '"Jaywalk", '
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
    # `because_of` walks `action.because_of`, which holds the IDs of *actions,
    # signal states, properties, and traffic objects* that caused this action —
    # never agent IDs directly. So `because_of agent.type = X` never matches;
    # point at the agent's *action* instead, or at a signal state / object /
    # property. The RHS can be any parenthesized expression — boolean
    # combinations, `in (...)`, and cross-modal compounds are all valid.
    Scenario(
        id="ego-decel-because-ped",
        title="Ego decelerates because a pedestrian was walking/running",
        query='ego.action = decel because_of agent.action.type in ("oxd:Walk", "oxd:Run")',
    ),
    Scenario(
        id="ego-decel-because-vehicle-in-lane",
        title="Ego decelerates because a vehicle was driving in lane",
        query="ego.action = decel because_of agent.action.type = fst:DrivingInLane",
    ),
    Scenario(
        id="ego-decel-because-vehicle-stopped-or-driving",
        title="Ego decelerates because another vehicle was stopped or driving in lane",
        query="ego.action = decel because_of agent.action.type in (fst:Stop, fst:DrivingInLane)",
    ),
    Scenario(
        id="ego-decel-because-vehicle-changing-lane",
        title="Ego decelerates because a vehicle was changing lane",
        query="ego.action = decel because_of agent.action.type = oxd:ChangeLane",
    ),
    Scenario(
        id="ego-decel-because-vehicle-turning",
        title="Ego decelerates because a vehicle was turning",
        query="ego.action = decel because_of "
              "agent.action.type in (oxd:MakeARightTurn, oxd:MakeALeftTurn)",
    ),
    Scenario(
        id="ego-stop-because-red",
        title="Ego stops because of a red light (causal edge into signal state)",
        query="ego.action = stop because_of light.color = red",
    ),
    Scenario(
        id="ego-stop-because-stop-sign",
        title="Ego stops because of a stop sign (causal edge into traffic object)",
        query="ego.action = stop because_of obj.type = stop_sign",
    ),
    Scenario(
        id="ego-yield-because-yield-sign",
        title="Ego yields because of a yield sign",
        query="ego.action = yield because_of obj.type = yield_sign",
    ),
    Scenario(
        id="ego-stop-because-any-control",
        title="Ego stops because of ANY traffic control (cross-modal RHS)",
        query="ego.action = stop because_of "
              "(light.color = red or obj.type = stop_sign or obj.type = yield_sign)",
    ),
    Scenario(
        id="ego-defensive-because-red",
        title="Ego stops OR decelerates because of red light (compound LHS + single RHS)",
        query="(ego.action = stop or ego.action = decel) because_of light.color = red",
    ),
    Scenario(
        id="ego-defensive-because-red-or-stop-sign",
        title="Ego stops OR decels because of red OR stop sign (compound LHS + compound RHS)",
        query="(ego.action = stop or ego.action = decel) because_of "
              "(light.color = red or obj.type = stop_sign)",
    ),
    Scenario(
        id="ego-decel-because-debris",
        title="Ego decelerates because of cone / barrier / debris",
        query="ego.action = decel because_of obj.type in (cone, barrier, debris)",
    ),
    Scenario(
        id="crosswalk-ped-causes-ego-decel",
        title="Ego decels because of ped walk/run AND pedestrian is in a crosswalk "
              "(combines causal edge with entity predicate)",
        query='ego.action = decel because_of agent.action.type in ("oxd:Walk", "oxd:Run") '
              "and agent.type = ped and env.type = crosswalk",
    ),
    Scenario(
        id="while-ped-ego-decel-causal",
        title="During a window where a pedestrian is present, ego decels because of an agent action",
        query="within agent.type = ped: ego.action = decel because_of "
              'agent.action.type in ("oxd:Walk", "oxd:Run", fst:Stop, fst:DrivingInLane)',
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
    # Standalone `Jaywalk` is the post-0.6.1 form; `oxd:Walk (jaywalk)`
    # is the deprecated combined suffix (still matched).
    print('# matches = ds.find(\'agent(type = ped, action.type = "Jaywalk")\')')
    print("# print(set(matches.clips()))")


if __name__ == "__main__":
    main()
