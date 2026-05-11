"""Dataset statistics via the query API.

Demonstrates count / group_by / histogram for human-facing analytics.

    CAUSAL_AV_DATASET_ROOT=/path/to/json_annotations \\
        uv run python examples/03_statistics.py
"""

from __future__ import annotations

import os
from pathlib import Path

from causal_ai_av.dataset import CausalAVDataset


def print_bar_chart(title: str, counts: dict[object, int], width: int = 40) -> None:
    print(f"\n# {title}")
    if not counts:
        print("  (no matches)")
        return
    max_n = max(counts.values())
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        bar = "█" * int(width * n / max_n) if max_n else ""
        print(f"  {str(key)[:30]:>30}  {n:>4}  {bar}")


def main() -> None:
    try:
        dataset_root = Path(os.environ["CAUSAL_AV_DATASET_ROOT"])
    except KeyError:
        raise SystemExit("set CAUSAL_AV_DATASET_ROOT to the directory of JSON annotations")

    ds = CausalAVDataset(dataset_root)
    print(f"corpus has {len(ds)} clips")

    # ----- count -----
    # count() returns the number of *clips* with at least one match.
    print(f"\nclips with a pedestrian:       {ds.count('agent.type = ped')}")
    print(f"clips with a vehicle:          {ds.count('agent.type = vehicle')}")
    print(f"clips with a VRU (ped/cyclist): {ds.count('agent.type = vru')}")
    print(f"clips with a stop sign:        {ds.count('obj.type = stop_sign')}")

    # ----- group_by -----
    # Group matches by an attribute path; values are clip counts.
    print_bar_chart(
        "ego action distribution (clips per action type)",
        ds.group_by(
            "ego.action.type = drive or ego.action.type = stop or ego.action.type = decel "
            "or ego.action.type = yield or ego.action.type = turn_left "
            "or ego.action.type = turn_right or ego.action.type = change_lane",
            key="ego.action.type",
        ),
    )

    print_bar_chart(
        "environment-type distribution",
        ds.group_by("env.type = road or env.type = intersection or env.type = crosswalk "
                    "or env.type = sidewalk or env.type = roundabout or env.type = cycle_lane "
                    "or env.type = tunnel",
                    key="env.type"),
    )

    print_bar_chart(
        "agent-type distribution",
        ds.group_by("agent.type = vehicle or agent.type = vru or agent.type = animal",
                    key="agent.type"),
    )

    print_bar_chart(
        "light-color distribution (clips per color)",
        ds.group_by("light.color = red or light.color = yellow or light.color = green",
                    key="light.color"),
    )

    # ----- composite stats -----
    print("\n# composite questions")
    n_red = ds.count("light.color = red")
    n_red_stop = ds.count("light.color = red and ego.action = stop")
    print(f"  ego stops at red light: {n_red_stop}/{n_red} red-light clips "
          f"({100 * n_red_stop / max(n_red, 1):.0f}%)")

    n_yellow_clear = ds.count("light(color = yellow, could_have_cleared = true)")
    print(f"  yellows ego could have cleared safely: {n_yellow_clear}")

    n_jaywalk = ds.count("agent(type = ped, action(jaywalk = true))")
    print(f"  clips with a jaywalking pedestrian: {n_jaywalk}")


if __name__ == "__main__":
    main()
