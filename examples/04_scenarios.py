"""Driving-scenario detection.

Encodes ~20 representative scenarios from `meta/scenarios_and_queries_reference.md`
as named queries. Runs them all against the corpus and prints the
candidate-clip count for each.

    uv run python examples/04_scenarios.py

The queries here are *necessary-condition* filters — they reject
clearly-not-this-scenario candidates. A real classifier (VLM, human,
heuristic) would then disambiguate the survivors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from causal_ai_av.dataset import CausalAVDataset

CORPUS = Path("/home/horde/01_json_annotations")


@dataclass(frozen=True)
class Scenario:
    id: str       # short id matching the reference doc's scenario number
    title: str    # one-line human description
    query: str    # DSL — copy-paste-able into the API


# ---------------------------------------------------------------------------
# Scenario catalog — representative sample, by category.
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    # ===== VRU Interactions =====
    Scenario(
        id="1",
        title="Ego drives through crosswalk while pedestrian present",
        query="agent.type = ped and env.type = crosswalk and ego.action = drive",
    ),
    Scenario(
        id="2",
        title="Ego yields/stops at crosswalk for a pedestrian",
        query="agent.type = ped and env.type = crosswalk "
              "and ego.action in (stop, yield, decel)",
    ),
    Scenario(
        id="8",
        title="Pedestrian jaywalks; ego brakes / yields",
        query="agent(type = ped, action(jaywalk = true)) "
              "and ego.action in (stop, yield, decel)",
    ),
    Scenario(
        id="9",
        title="Pedestrian crosses while ego is turning",
        query="agent.type = ped and env.type = crosswalk "
              "and ego.action in (turn_left, turn_right)",
    ),
    Scenario(
        id="12",
        title="Cyclist mid-block in ego's path; ego defensive",
        query="agent.type = cyclist and ego.action in (stop, yield, decel)",
    ),

    # ===== Signalized Intersections =====
    Scenario(
        id="14",
        title="Ego stops at a red traffic light (nominal)",
        query="light.color = red and ego.action = stop and env.type = intersection",
    ),
    Scenario(
        id="16",
        title="Signal blackout — ego treats as all-way stop",
        query="light.state = off and ego.action = stop and env.type = intersection",
    ),
    Scenario(
        id="20",
        title="Ego already in intersection when yellow begins, proceeds",
        query="light(color = yellow, ego_in_on_yellow = true) "
              "and ego.action in (drive, enter, creep)",
    ),
    Scenario(
        id="21",
        title="Ego approaches intersection, yellow appears, ego stops",
        query="light(color = yellow, on_ego_path = true) and ego.action = stop",
    ),
    Scenario(
        id="23",
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
        id="76",
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
        id="144",
        title="Ego transits roundabout without yielding",
        query="env.type = roundabout and ego.action = drive",
    ),
    Scenario(
        id="142",
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
    ds = CausalAVDataset(CORPUS)

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
    print('# matches = ds.find("agent(type = ped, action(jaywalk = true))")')
    print("# print(set(matches.clips()))")


if __name__ == "__main__":
    main()
