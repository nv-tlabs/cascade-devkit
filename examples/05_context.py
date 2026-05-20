# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context windows — what else was happening during a match.

Every `Match` carries a `(clip_id, entity, interval)` tuple. Given a
match, `dataset.context_for(match)` snapshots every entity in the
clip whose annotated time range overlaps the match's interval —
visible agents, ego actions, environments, conditions, traffic-light
states, and persistent traffic objects.

    CASCADE_AV_DATASET_ROOT=/path/to/json_annotations \\
        uv run python examples/05_context.py
"""

from __future__ import annotations

import os
from pathlib import Path

from cascade_av.dataset import CascadeDataset


def main() -> None:
    try:
        dataset_root = Path(os.environ["CASCADE_AV_DATASET_ROOT"])
    except KeyError:
        raise SystemExit("set CASCADE_AV_DATASET_ROOT to the directory of JSON annotations")

    ds = CascadeDataset(dataset_root)

    # Pick a query that pairs entities via a temporal operator — `while`
    # tightens each match's interval to the intersection, so the window
    # is the actual moment both held.
    query = "agent.type = ped while ego.action = decel"
    matches = ds.find(query)

    # `while` returns the cross product of left × right matches whose
    # intervals intersect — one match per (pedestrian, ego-action) pair.
    # The same `(clip_id, window)` can appear multiple times when several
    # distinct pairs collapse to the same interval. For a context tour
    # that's noise, so dedupe by (clip_id, interval).
    seen: set[tuple[str, float, float]] = set()
    unique = []
    for m in matches.matches:
        if m.interval is None:
            continue
        key = (m.clip_id, m.interval.start, m.interval.end)
        if key in seen:
            continue
        seen.add(key)
        unique.append(m)
    print(f"{len(matches)} matches ({len(unique)} unique by clip+window) for: {query}\n")

    for m in unique[:3]:
        ctx = ds.context_for(m)
        t0, t1 = m.interval.start, m.interval.end
        print(f"=== clip {ctx.clip_id}   window {t0:.2f}–{t1:.2f}s ===")

        if ctx.agents:
            print("  agents visible in window:")
            for a in ctx.agents:
                actions = [ax.action_type for ax in a.actions] or ["(no active actions)"]
                print(f"    - {a.agent.type:<28} ({a.agent.id})  actions: {actions}")
        else:
            print("  agents visible in window: (none)")

        ego_types = [ea.type for ea in ctx.ego_actions]
        print(f"  ego actions:      {ego_types or '(none)'}")

        env_types = [e.type for e in ctx.environments]
        print(f"  environments:     {env_types or '(none)'}")

        cond_types = [c.type for c in ctx.conditions]
        print(f"  conditions:       {cond_types or '(none)'}")

        light_summary = [
            f"{ls.state.color}{f'/{ls.state.type}' if ls.state.type else ''}"
            for ls in ctx.light_states
        ]
        print(f"  light states:     {light_summary or '(none)'}")

        obj_types = [o.type for o in ctx.traffic_objects]
        print(f"  traffic objects:  {obj_types or '(none)'}")
        print()


if __name__ == "__main__":
    main()
