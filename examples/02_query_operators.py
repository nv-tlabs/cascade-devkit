# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tour of every operator in the query DSL.

Each section runs one query and prints the clip count, so you can see
what each operator does on the real corpus.

    CASCADE_AV_DATASET_ROOT=/path/to/json_annotations \\
        uv run python examples/02_query_operators.py
"""

from __future__ import annotations

import os
from pathlib import Path

from cascade_av.dataset import CascadeDataset


def section(title: str) -> None:
    print(f"\n# {title}")


def show(ds: CascadeDataset, query: str) -> None:
    n = ds.count(query)
    print(f"  {n:>4}  {query}")


def main() -> None:
    try:
        dataset_root = Path(os.environ["CASCADE_AV_DATASET_ROOT"])
    except KeyError:
        raise SystemExit("set CASCADE_AV_DATASET_ROOT to the directory of JSON annotations")

    ds = CascadeDataset(dataset_root)

    section("Attribute predicate (entity.attribute = value)")
    show(ds, "agent.type = ped")
    show(ds, "ego.action = decel")
    show(ds, "light.color = red")
    show(ds, "env.lanes >= 2")

    section("Hierarchical aliases (vehicle ⊃ car/truck/bus/...)")
    show(ds, "agent.type = vehicle")
    show(ds, "agent.type = vru")
    show(ds, "agent.type = cyclist")
    show(ds, "env.type = intersection")

    section("Set membership: 'in (a, b, c)' is OR")
    show(ds, "ego.action in (stop, yield, decel)")
    show(ds, "agent.type in (ped, cyclist, animal)")

    section("Same-entity coupling — agent(...) groups constraints on ONE agent")
    show(ds, "agent(type = vehicle, pos = front)")
    show(ds, "agent(type = ped, pos = left)")
    # Compare: free-floating predicates may match DIFFERENT agents.
    show(ds, "agent.type = vehicle and agent.pos = front")

    section("Boolean operators")
    show(ds, "agent.type = ped and env.type = crosswalk")
    show(ds, "ego.action = stop or ego.action = yield")
    show(ds, "agent.type = ped and not env.type = crosswalk")

    section("Action-type suffixes (schema 2.0.0 single source of truth)")
    # Flags are encoded as parenthesized suffixes on `action_type` —
    # `(jaywalk)`, `(erratic)`, `(jaywalk, erratic)`. Match them with
    # literal-string equality / `in (...)` on `action.type`.
    show(
        ds,
        "agent(type = ped, action.type in ("
        '"oxd:Walk (jaywalk)", "oxd:Walk (jaywalk, erratic)", '
        '"oxd:Run (jaywalk)", "oxd:Run (jaywalk, erratic)"))',
    )
    show(
        ds,
        "agent.action.type in ("
        '"oxd:Walk (erratic)", "oxd:Walk (jaywalk, erratic)", '
        '"oxd:Run (erratic)", "oxd:Run (jaywalk, erratic)")',
    )

    section("Temporal: 'A while B' — co-occurring intervals")
    show(ds, "agent.type = ped while ego.action = decel")
    show(ds, "light.color = red while ego.action = stop")

    section("Temporal: 'A then(K) B' — B starts during A or within K seconds after")
    show(ds, "light.color = yellow then(3) ego.action = stop")
    show(ds, "light.color = green then ego.action = drive")  # K defaults to 0

    section("Relational: 'A because_of B' — A's because_of edge points to B")
    # `because_of` walks the schema's `action.because_of` list. The list holds
    # the IDs of *actions, states, properties, objects* that caused this
    # action — never agents directly. So `because_of agent.type = X` always
    # returns 0; point at the agent's action instead, or at a signal state
    # / traffic object. The RHS can be any parenthesized expression.
    show(ds, 'ego.action = decel because_of agent.action.type in ("oxd:Walk", "oxd:Run")')
    show(ds, "ego.action = stop because_of light.color = red")
    show(ds, "ego.action = stop because_of obj.type = stop_sign")
    show(
        ds,
        "ego.action = yield because_of "
        "(agent.action.type = fst:DrivingInLane or agent.action.type = oxd:ChangeLane)",
    )

    section("Window scoping: 'within W: E' — restrict E's time to W's intervals")
    show(ds, "within light.color = red: not ego.action = stop")
    show(ds, "within env.type = crosswalk: agent.type = ped")

    section("Light flags (annotator-tagged temporal correlations)")
    show(ds, "light(color = yellow, ego_in_on_yellow = true)")
    show(ds, "light(color = yellow, could_have_cleared = true)")

    section("Clip-level attributes")
    show(ds, "clip.eventful = true")

    section("Agent group size — `agent.amount`")
    show(ds, "agent.amount = single")
    show(ds, "agent.amount = group")
    show(ds, "agent(type = ped, amount = group)")

    section("Signaling details on `agent.prop` / `ego.prop`")
    show(ds, "agent.prop.source = flashing_light")
    show(ds, "agent.prop.source in (flashing_light, holding_sign, other)")
    show(ds, "agent.prop.not_facing_ego = false")

    section("Containment flags — `lane_edge`, `near_lane`, `illegal_lane`")
    show(ds, "agent.illegal_lane = true")
    show(ds, "agent.lane_edge = true")
    show(ds, "obj.lane_edge = true")
    show(ds, "agent(type = car, illegal_lane = true)")

    section("Causal-link lists — `ego.influenced_by`, action `link_to` / `action_target`")
    # `ego.influenced_by` is populated on 72 % of clips; the IDs are
    # UUIDs, so demonstrate via the in-set membership idiom only.
    show(ds, 'ego.influenced_by in ("00000000-0000-0000-0000-000000000000")')
    show(ds, 'agent.action.link_to = "00000000-0000-0000-0000-000000000000"')

    section("Clip description — `clip.brief_description`")
    # Equality only; the exact string per clip is unique, so this is
    # most useful for spot-checks rather than aggregation.
    show(ds, 'clip.brief_description = "(no such description)"')

    section("Newly-aliased 'Other' / 'Vehicle' values")
    show(ds, "agent.type = other")
    show(ds, "agent.type = generic_vehicle")
    show(ds, "env.type = other")
    show(ds, "agent.action.type = other")

    section("Condition → environment lookup — `cond.env_*`")
    show(ds, "cond.env_type = road")
    show(ds, "cond.type = construction and cond.env_type = road")
    show(ds, "cond.env_one_way = true")
    show(ds, "cond.env_lanes > 2")


if __name__ == "__main__":
    main()
