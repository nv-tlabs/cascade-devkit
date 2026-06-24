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

    section("Ego-relative direction — `agent.dir`")
    # Direction aliases: `same`, `opposite`, `perpendicular_lr`,
    # `perpendicular_rl`, plus the parent `perpendicular` (the union
    # of the two perpendicular leaves).
    show(ds, "agent.dir = same")
    show(ds, "agent.dir = opposite")
    show(ds, "agent.dir = perpendicular")
    # Oncoming vehicle — the natural way to ask "is there an oncoming
    # car?" for unprotected-turn analysis.
    show(ds, "agent(type = vehicle, dir = opposite)")

    section("Per-interval pose — `pos_any` / `dir_any`")
    # `pos` / `dir` sample at the agent's visibility-window MIDPOINT
    # (one deterministic value per agent). For agents whose relative
    # pose changes mid-window — e.g. a vehicle that approaches in
    # front and ends up perpendicular as it crosses the intersection
    # — the midpoint reading misses the transient. `pos_any` /
    # `dir_any` are `kind="list"` and match if ANY pose interval
    # carried the value.
    #
    # The delta between the two flavors on this corpus shows how
    # often pose changes during visibility:
    show(ds, "agent.dir = perpendicular")  # midpoint-sampled
    show(ds, "agent.dir_any = perpendicular")  # any interval
    # Composite: vehicles that were either in front OR crossing at
    # some point during their visibility.
    show(
        ds,
        "agent(type = vehicle, pos_any in (front, perpendicular_lr, perpendicular_rl))",
    )

    section("Boolean operators")
    show(ds, "agent.type = ped and ego.action in (stop, yield, decel)")
    show(ds, "ego.action = stop or ego.action = yield")
    show(ds, "agent.type = ped and not ego.action = drive")

    section("Action-type suffixes (schema 2.0.0 single source of truth)")
    # Flags are encoded as parenthesized suffixes on `action_type` —
    # `(jaywalk)`, `(erratic)`, `(jaywalk, erratic)`. Match them with
    # literal-string equality / `in (...)` on `action.type`.
    #
    # Post-0.6.1, `Jaywalk` is also a standalone action (alias
    # `jaywalk_action`); include it alongside the combined-suffix forms.
    # The `oxd:Walk (jaywalk)` / `oxd:Run (jaywalk)` suffix variants are
    # deprecated but still match, so a complete jaywalk query enumerates
    # both the standalone action and the combined forms.
    show(
        ds,
        "agent(type = ped, action.type in ("
        '"Jaywalk", '
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

    section("Relational: 'A influenced_by B' — A's Influence side-channel points to B")
    # `influenced_by` walks the schema's `Influence.influencers` list, which is
    # distinct from action-rooted `because_of`: it reads "this entity *was
    # under the influence of* B", regardless of a specific action. The LHS is a
    # bare entity (`ego` / `agent(...)`); the match interval is the influence
    # window, so it composes with `while` / `then`. A LightStates RHS is scoped
    # to that window (a head cycling green→red matches `light.color = red` only
    # while the influence window overlaps the red phase).
    show(ds, "ego influenced_by light.color = red")
    show(ds, "agent(type = vehicle) influenced_by light.color = red")
    show(ds, "ego influenced_by (obj.type = stop_sign or light.color = red)")
    show(ds, "ego influenced_by light.color = red and ego.action = stop")

    section("Window scoping: 'within W: E' — restrict E's time to W's intervals")
    show(ds, "within light.color = red: not ego.action = stop")
    show(ds, "within env.type = road: agent.type = ped")

    section("Light flags (annotator-tagged temporal correlations)")
    show(ds, "light(color = yellow, ego_in_on_yellow = true)")
    show(ds, "light(color = yellow, could_have_cleared = true)")

    section("Clip-level attributes")
    show(ds, "clip.eventful = true")
    # `clip.eventful_reason` (post-0.6.1) is now queryable alongside the
    # boolean `clip.eventful`; values: ego_adapts, special_env,
    # agent_adapts, other.
    show(ds, "clip.eventful_reason = ego_adapts")

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
