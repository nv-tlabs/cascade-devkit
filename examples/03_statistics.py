# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset statistics via the query API.

Demonstrates the two perspectives the API supports — clip-level
("how many clips contain X?") and entity-level ("how many X
instances are there in total?").

    CASCADE_AV_DATASET_ROOT=/path/to/json_annotations \\
        uv run python examples/03_statistics.py
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from cascade_av.dataset import CascadeDataset


def print_bar_chart(title: str, counts: dict[object, int], width: int = 40) -> None:
    print(f"\n# {title}")
    if not counts:
        print("  (no matches)")
        return
    max_n = max(counts.values())
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        bar = "█" * int(width * n / max_n) if max_n else ""
        print(f"  {str(key)[:30]:>30}  {n:>5}  {bar}")


def entity_count(ds: CascadeDataset, query: str) -> int:
    """Number of *matching entities* across the corpus (not clips).

    For single-entity predicates (`agent.type = ped`, `obj.type = stop_sign`,
    …) this equals the number of distinct entities. Boolean composition
    over different kinds (`and` / `or`) produces a flat union of matches;
    use single-entity predicates here for a clean count.
    """
    return len(ds.find(query).matches)


def entities_by_attr(ds: CascadeDataset, query: str, attr: str) -> dict[object, int]:
    """Bucket matching entities by a top-level attribute (e.g. `type`).

    Like `ds.group_by` but counts entities rather than clips.
    """
    out: Counter[object] = Counter()
    for m in ds.find(query).matches:
        val = getattr(m.entity, attr, None)
        if isinstance(val, list):
            for v in val:
                out[v] += 1
        else:
            out[val] += 1
    return dict(out)


def main() -> None:
    try:
        dataset_root = Path(os.environ["CASCADE_AV_DATASET_ROOT"])
    except KeyError:
        raise SystemExit("set CASCADE_AV_DATASET_ROOT to the directory of JSON annotations")

    ds = CascadeDataset(dataset_root)
    print(f"corpus has {len(ds)} clips")

    # ----- headline counts: clips vs entities side by side -----
    headlines = [
        ("pedestrian",          "agent.type = ped"),
        ("vehicle",             "agent.type = vehicle"),
        ("VRU (ped/cyclist)",   "agent.type = vru"),
        ("cyclist",             "agent.type = cyclist"),
        ("stop sign",           "obj.type = stop_sign"),
        ("yield sign",          "obj.type = yield_sign"),
        ("crosswalk env",       "env.type = crosswalk"),
        ("intersection env",    "env.type = intersection"),
    ]
    print(f"\n{'category':<22} {'clips':>7} {'entities':>10}")
    print("-" * 42)
    for label, query in headlines:
        print(f"{label:<22} {ds.count(query):>7} {entity_count(ds, query):>10}")

    # ----- distributions: per-type entity counts vs per-type clip counts -----
    # group_by returns clip counts; entities_by_attr returns entity counts.
    # Both perspectives are useful: clip counts show coverage, entity counts
    # show prevalence (a single clip can contain many of the same thing).
    print_bar_chart(
        "agent-type distribution (clips per type)",
        ds.group_by(
            "agent.type = vehicle or agent.type = vru or agent.type = animal",
            key="agent.type",
        ),
    )
    print_bar_chart(
        "agent-type distribution (entity counts)",
        entities_by_attr(
            ds,
            "agent.type = vehicle or agent.type = vru or agent.type = animal",
            attr="type",
        ),
    )

    print_bar_chart(
        "ego-action distribution (clips per action type)",
        ds.group_by(
            "ego.action.type = drive or ego.action.type = stop or ego.action.type = decel "
            "or ego.action.type = yield or ego.action.type = turn_left "
            "or ego.action.type = turn_right or ego.action.type = change_lane",
            key="ego.action.type",
        ),
    )

    print_bar_chart(
        "environment-type distribution (clips)",
        ds.group_by("env.type = road or env.type = intersection or env.type = crosswalk "
                    "or env.type = sidewalk or env.type = roundabout or env.type = cycle_lane "
                    "or env.type = tunnel",
                    key="env.type"),
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

    n_jaywalk_clips = ds.count("agent(type = ped, action(jaywalk = true))")
    n_jaywalk_entities = entity_count(ds, "agent(type = ped, action(jaywalk = true))")
    print(f"  jaywalking pedestrian: {n_jaywalk_clips} clips, "
          f"{n_jaywalk_entities} pedestrian instances")


if __name__ == "__main__":
    main()
