# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Causal queries — the `because_of` operator, deep dive.

`because_of` is the DSL's distinguishing feature: it walks the schema's
`action.because_of` edge — a list of *cause IDs* on every `AgentAction`
and `EgoAction`. The corpus carries real human-annotated causality; this
example showcases what you can ask of it.

Three things to remember:

1. **`because_of` LHS must yield actions.** `ego.action = X because_of …`
   or `agent.action.type = Y because_of …`. The engine filters anything
   else.

2. **`because_of` RHS can yield anything with an `id` field that
   resolves into the LHS's `because_of` list — actions, signal states,
   properties, traffic objects, environments, traffic lights.** What it
   does NOT match is `agent.type = X` directly — agents themselves are
   never causal targets in the schema; their *actions* are.

3. **RHS supports any parenthesized expression.** Compound boolean
   forms (`(A or B)`, `(A and B)`, `not X`), set membership
   (`X in (a, b, c)`), and cross-modal compounds (mixing signal states
   with traffic objects) all parse and evaluate cleanly.

In the 04 corpus, `because_of` is populated on:

    298 ego.actions    pointing at: AgentAction (257), TrafficObject (58),
                                    SignalState (52), AgentProperty (40),
                                    Environment (2), TrafficLight (1),
                                    EgoProperty (1)
    146 agent.actions  pointing at: AgentAction (67), SignalState (60),
                                    EgoAction (15), AgentProperty (15),
                                    TrafficObject (5), TrafficLight (1),
                                    EgoProperty (1)

So roughly 1 in 3 ego actions and 1 in 13 agent actions are causally
annotated. Run this example end-to-end to see what queries surface.

    CASCADE_AV_DATASET_ROOT=/path/to/json_annotations \\
        uv run python examples/08_causal_queries.py
"""

from __future__ import annotations

import os
from pathlib import Path

from cascade_av.dataset import CascadeDataset


def section(title: str) -> None:
    print(f"\n# {title}")


def show(ds: CascadeDataset, query: str, gloss: str = "") -> None:
    n = ds.count(query)
    if gloss:
        print(f"  {n:>4}  {query}")
        print(f"        ↳ {gloss}")
    else:
        print(f"  {n:>4}  {query}")


def main() -> None:
    try:
        dataset_root = Path(os.environ["CASCADE_AV_DATASET_ROOT"])
    except KeyError:
        raise SystemExit("set CASCADE_AV_DATASET_ROOT to the directory of JSON annotations")
    ds = CascadeDataset(dataset_root)

    # =========================================================================
    section("1) Atomic causes — single-edge causality (one cause type per query)")
    # =========================================================================
    # Each query asks: "find clips where ego/agent action X was caused
    # specifically by a Y", where Y is a single, atomic predicate.

    section("1a) Caused by an agent's ACTION")
    show(ds,
         'ego.action = decel because_of agent.action.type in ("oxd:Walk", "oxd:Run")',
         "ego decelerates because of a pedestrian walking or running")
    show(ds,
         "ego.action = decel because_of agent.action.type = fst:DrivingInLane",
         "ego decelerates because another vehicle was driving in its lane")
    show(ds,
         "ego.action = decel because_of agent.action.type = oxd:ChangeLane",
         "ego decelerates because a vehicle was changing lane")
    show(ds,
         "ego.action = decel because_of "
         "agent.action.type in (oxd:MakeARightTurn, oxd:MakeALeftTurn)",
         "ego decelerates because of a turning vehicle")

    section("1b) Caused by a SIGNAL STATE (traffic light)")
    show(ds, "ego.action = stop because_of light.color = red",
         "ego stops because the light is red — the prototypical causal pattern")

    section("1c) Caused by a TRAFFIC OBJECT (sign, cone, …)")
    show(ds, "ego.action = stop because_of obj.type = stop_sign",
         "stop sign causes ego stop")
    show(ds, "ego.action = yield because_of obj.type = yield_sign",
         "yield sign causes ego yield")
    show(ds, "ego.action = decel because_of obj.type in (cone, barrier, debris)",
         "debris causes ego deceleration")

    # =========================================================================
    section("2) Compound RHS — multiple cause types in one query")
    # =========================================================================
    # The RHS of `because_of` accepts any parenthesized expression. This
    # is where the operator earns its keep — you can ask cross-modal
    # questions that no single field on the schema captures directly.

    section("2a) `in (...)` — set membership (cleanest form for many alternatives)")
    show(ds,
         "ego.action = decel because_of agent.action.type in (fst:Stop, fst:DrivingInLane)",
         "ego decels because of any stationary OR in-lane vehicle action")
    show(ds,
         'ego.action = decel because_of '
         'agent.action.type in ("oxd:Walk", "oxd:Run", oxd:Bike)',
         "ego decels because of any VRU motion type")

    section("2b) `(A or B)` — boolean union (works with mixed modalities)")
    show(ds,
         "ego.action = decel because_of (light.color = red or light.color = yellow)",
         "ego decels because the light is red OR yellow (defensive on amber)")
    show(ds,
         "ego.action = stop because_of (obj.type = stop_sign or light.color = red)",
         "ego stops because of a stop sign OR a red light — same defensive behavior, "
         "different cause modality")
    show(ds,
         "ego.action = stop because_of "
         "(light.color = red or obj.type = stop_sign or obj.type = yield_sign)",
         "ego stops at ANY positive traffic control — full survey of stop-causing controls")

    # =========================================================================
    section("3) Compound LHS — multiple effects from the same cause")
    # =========================================================================
    # The LHS can also be a parenthesized expression. This is useful when
    # you want to group similar effects (stop + decel = "defensive
    # braking") and ask what causes that pattern as a whole.

    show(ds,
         "(ego.action = stop or ego.action = decel) because_of light.color = red",
         "the full defensive-braking response to a red light")
    show(ds,
         "(ego.action = stop or ego.action = decel) because_of "
         "(light.color = red or obj.type = stop_sign)",
         "defensive braking caused by ANY of red light / stop sign — compound LHS + RHS")

    # =========================================================================
    section("4) Causality combined with entity predicates")
    # =========================================================================
    # `because_of` doesn't know about the broader scene. You add scene
    # context by AND-ing entity predicates into the LHS.

    show(ds,
         'ego.action = decel because_of agent.action.type in ("oxd:Walk", "oxd:Run") '
         "and agent.type = ped and env.type = crosswalk",
         "ego decels because of a walking/running pedestrian, AND a pedestrian "
         "is present, AND the environment is a crosswalk — three-way enrichment")

    # =========================================================================
    section("5) Causality scoped to a time window")
    # =========================================================================
    # `within` restricts the LHS's match time to a window defined by
    # another query. Compose with `because_of` for "causal facts during
    # X" — e.g. "during a red light, who stops because of red?"

    show(ds,
         "within light.color = red: ego.action = stop because_of light.color = red",
         "during red light, ego stops because of red (self-consistent causal annotation)")
    show(ds,
         "within agent.type = ped: ego.action = decel because_of "
         'agent.action.type in ("oxd:Walk", "oxd:Run", fst:Stop, fst:DrivingInLane)',
         "during a pedestrian-present window, ego decels because of any agent action")

    # =========================================================================
    section("6) Worth knowing — patterns that DON'T parse / don't match")
    # =========================================================================
    # These are the dead-ends, called out so you don't waste time on them.

    print("  -  `because_of agent.type = X` returns 0 — agents are never causal")
    print("     targets directly; the schema records their *actions* as causes.")
    print("     Use: `because_of agent.action.type = Y` instead.")
    print()
    print("  -  `because_of agent(type = vehicle, action.type = Y)` returns 0 —")
    print("     the engine resolves the right side by entity id, so the parent")
    print("     `agent(...)` clause's Agent.id doesn't appear in any action's")
    print("     because_of list. Use: `because_of agent.action.type = Y` and")
    print("     add the vehicle predicate as a separate `and` clause.")


if __name__ == "__main__":
    main()
