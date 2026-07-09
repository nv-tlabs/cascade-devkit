# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the `notebooks/` Jupyter files from this script.

Run after editing to regenerate:

    uv run --group notebooks python scripts/build_notebooks.py

Each notebook reads the dataset root from the `CASCADE_AV_DATASET_ROOT`
environment variable — export it before launching Jupyter:

    export CASCADE_AV_DATASET_ROOT=/path/to/json_annotations
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def save(cells: list[nbf.NotebookNode], path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (cascade_av)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Shared styling
# ---------------------------------------------------------------------------

STYLING = """
        # --- shared notebook styling -----------------------------------------------
        import matplotlib.pyplot as plt
        import pandas as pd

        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "figure.dpi": 110,
            "savefig.dpi": 110,
        })

        pd.options.display.max_colwidth = 110
        pd.options.display.width = 140
        pd.options.display.float_format = "{:.2f}".format
"""


# ---------------------------------------------------------------------------
# 01 — Quickstart
# ---------------------------------------------------------------------------


def build_quickstart() -> None:
    cells = [
        md("""
        # Quickstart — `cascade_av`

        This notebook introduces the CASCADE DevKit. You will
        load the corpus, inspect a single clip's annotation, and run
        your first queries in the embedded DSL.
        """),
        md("""
        ## About the dataset

        The **CASCADE dataset** adds causal and spatio-temporal action
        annotations on top of NVIDIA's *Physical AI AV Dataset*. Each
        clip is a short front-facing driving video paired with a JSON
        annotation bundle (schema `2.0.0`). The DevKit parses those
        bundles into a typed Pydantic tree and exposes a small query
        language for searching the corpus.

        ### Annotation scope

        Every clip is annotated with the following entity categories:

        | entity | what it captures |
        |---|---|
        | **Agents** | non-ego actors — vehicles, pedestrians, cyclists, animals, officers — with type, ego-relative position, visibility intervals, and per-interval actions |
        | **Ego vehicle** | actions taken by the recording car (drive, decel, stop, turn, change-lane, …) and the entities that motivated them |
        | **Environments** | road, intersection, crosswalk, sidewalk, roundabout, cycle-lane, tunnel, … with lane counts and one-way flags |
        | **Conditions** | weather, lighting, construction, occlusion |
        | **Traffic lights** | signal-head colors and state transitions, plus annotator-tagged flags such as `could_have_cleared` and `ego_in_on_yellow` |
        | **Traffic objects** | stop signs, yield signs, cones, debris, barriers |

        ### What makes it *causal*

        The schema includes a `because_of` edge on every action: each
        action can name the entities that caused it. "Ego decelerates
        *because of* a pedestrian" is not reconstructed from
        co-occurrence — it is annotated explicitly. The query DSL
        surfaces this directly through the `because_of` operator.

        ### Schema at a glance

        Each clip's annotation is a single JSON file with four
        top-level keys:

        - `video` — clip metadata (fps, duration, `clip_id`).
        - `annotation` — typed entity arrays
          (`agents[]`, `ego_vehicle`, `environments[]`, `conditions[]`,
          `traffic_lights[]`, `traffic_objects[]`).
        - `provenance` — annotator metadata.
        - `schema_version` — currently `2.0.0`.

        The corpus shipped with this DevKit contains 376 annotation
        files covering roughly 300 distinct clips.

        ### What this notebook covers

        1. Load the corpus and inspect a single clip's annotation.
        2. Run your first DSL queries against the whole corpus.
        3. Read back the `MatchSet` result type.
        4. Scope the same DSL to a single clip.

        > Set the `CASCADE_AV_DATASET_ROOT` environment variable to the
        > directory containing the JSON annotations before running the
        > setup cell below.
        """),
        md("## Setup"),
        code(
            """
        import os
        from pathlib import Path

        from cascade_av.dataset import CascadeDataset
        from cascade_av.io import load_dir
        from cascade_av.query import find_on_bundle
        """
            + STYLING
        ),
        code("""
        dataset_root = Path(os.environ["CASCADE_AV_DATASET_ROOT"])
        ds = CascadeDataset(dataset_root)
        print(f"corpus loaded — {len(ds)} clips")
        """),
        md("""
        ## 1. Inspect one clip's annotation

        `CascadeDataset.get_sequence` downloads the clip video lazily by
        default. To keep this tour offline-friendly we parse the JSON
        annotations directly via the local `io.load_dir` helper.
        """),
        code("""
        bundles = load_dir(dataset_root)
        b = bundles[0]

        summary = pd.DataFrame([
            ("clip_id",        b.video.clip_id),
            ("schema_version", b.schema_version),
            ("fps",            b.video.fps),
            ("duration_s",     b.video.duration_s),
            ("agents",         len(b.annotation.agents)),
            ("ego actions",    len(b.annotation.ego_vehicle.actions)),
            ("environments",   len(b.annotation.environments)),
            ("traffic lights", len(b.annotation.traffic_lights)),
            ("traffic objects", len(b.annotation.traffic_objects)),
        ], columns=["field", "value"])
        summary
        """),
        md("""
        ## 2. Run a DSL query against the whole corpus

        `CascadeDataset.count(query)` returns the number of clips with
        at least one match. Queries are plain strings in the DSL — the
        full grammar is covered in `02_query_dsl_tour.ipynb`.
        """),
        code("""
        queries = [
            "agent.type = ped",
            "agent.type = vehicle",
            "agent.type = ped and ego.action in (stop, yield, decel)",
            "ego.action = stop and light.color = red",
        ]
        pd.DataFrame(
            [(q, ds.count(q)) for q in queries],
            columns=["query", "matching clips"],
        )
        """),
        md("""
        ## 3. `find()` returns the full `MatchSet`

        Every match is a `(clip_id, entity, interval)` tuple. The
        `.clips()`, `.entities()`, and `.intervals()` projections pull
        each column individually.
        """),
        code("""
        matches = ds.find("agent.type = ped and ego.action in (stop, yield, decel)")
        clips = sorted(set(matches.clips()))

        print(f"{len(matches)} matches across {len(clips)} distinct clips")
        pd.DataFrame({"clip_id": clips[:5]})
        """),
        md("""
        ## 4. The same DSL on a single clip

        `find_on_bundle` runs a query against one parsed bundle and
        returns a `MatchSet` scoped to that clip only — useful when you
        already have a bundle in hand.
        """),
        code("""
        single = find_on_bundle(b, "agent.type = vehicle")
        print(f"{len(single)} vehicle matches in clip {b.video.clip_id}")
        pd.DataFrame({"entity": [str(e) for e in single.entities()[:5]]})
        """),
        md("""
        ## 5. What else was happening?

        Each `Match` carries `(clip_id, entity, interval)`. Given that
        interval, `ds.context_for(match)` returns a `ContextWindow`
        snapshot of every entity in the clip whose annotated time
        range overlaps the match: visible agents (with the actions
        they were doing at the time), ego actions, environments,
        conditions, traffic-light states, and persistent traffic
        objects.

        Temporal operators like `while` tighten each match's interval
        to the intersection, so the context window is the actual
        moment both conditions held.
        """),
        code("""
        matches = ds.find("agent.type = ped while ego.action = decel")
        m = matches.matches[0]
        ctx = ds.context_for(m)

        pd.DataFrame([
            ("clip_id",        ctx.clip_id),
            ("window",         f"{m.interval.start:.2f}–{m.interval.end:.2f}s"),
            ("agents",         len(ctx.agents)),
            ("ego actions",    [ea.type for ea in ctx.ego_actions]),
            ("environments",   [e.type for e in ctx.environments]),
            ("light states",   len(ctx.light_states)),
            ("traffic objects", [o.type for o in ctx.traffic_objects]),
        ], columns=["field", "value"])
        """),
        md("""
        Drill into the agents — each `AgentInWindow` carries the agent
        itself, its visibility window clipped to the match, and the
        actions of that agent that were active in the window.
        """),
        code("""
        pd.DataFrame([
            {
                "id":         a.agent.id,
                "type":       a.agent.type,
                "visible":    f"{a.visibility.start:.2f}–{a.visibility.end:.2f}s",
                "actions":    [ax.action_type for ax in a.actions],
            }
            for a in ctx.agents
        ])
        """),
    ]
    save(cells, NOTEBOOKS_DIR / "01_quickstart.ipynb")


# ---------------------------------------------------------------------------
# 02 — Query DSL tour
# ---------------------------------------------------------------------------


def build_dsl_tour() -> None:
    cells = [
        md("""
        # Query DSL — operator tour

        This notebook walks through every operator in the query DSL
        against the live corpus. Each section runs a small batch of
        queries and tabulates the matching clip counts so you can see
        what each operator does.

        > New to the DevKit? Start with `01_quickstart.ipynb` for an
        > introduction to the dataset and the DSL's place in it.

        ### Grammar in one screen

        | form | meaning |
        |---|---|
        | `<entity>.<attr> <cmp> <value>` | basic attribute predicate |
        | `<entity>(<expr>, …)` | same-entity grouping |
        | `and`, `or`, `not` | boolean composition |
        | `A while B` | A and B with intersecting intervals |
        | `A then(K) B` | B starts during A or within K seconds after |
        | `A because_of B` | A's `because_of` edge points at a B |
        | `A influenced_by B` | A's `Influence` side-channel points at a B |
        | `within W: E` | restrict E's time window to W's intervals |

        The full specification lives in `docs/user/query_language.md`.
        """),
        md("## Setup"),
        code(
            """
        import os
        from pathlib import Path

        from cascade_av.dataset import CascadeDataset
        """
            + STYLING
        ),
        code("""
        ds = CascadeDataset(Path(os.environ["CASCADE_AV_DATASET_ROOT"]))

        def run(queries: list[str]) -> pd.DataFrame:
            \"\"\"Run a batch of DSL queries and tabulate clip counts.\"\"\"
            return pd.DataFrame(
                [(q, ds.count(q)) for q in queries],
                columns=["query", "matching clips"],
            )
        """),
        md("""
        ## Attribute predicates

        The atomic predicate is `entity.attribute <cmp> value`. The
        comparison is `=` for equality and `<`, `<=`, `>`, `>=` for
        ordered attributes.
        """),
        code("""
        run([
            "agent.type = ped",
            "ego.action = decel",
            "light.color = red",
            "env.lanes >= 2",
        ])
        """),
        md("""
        ## Hierarchical aliases

        Several attribute values are *parent aliases* that expand to a
        union of children: `vehicle` (car, truck, bus, …),
        `vru` (pedestrian, cyclist), `intersection` (4-way, T-junction,
        roundabout, …), `turn` (left, right), `change_lane`
        (change-left, change-right).
        """),
        code("""
        run([
            "agent.type = vehicle",
            "agent.type = vru",
            "agent.type = cyclist",
            "env.type = intersection",
        ])
        """),
        md("""
        ## Set membership

        `attr in (a, b, c)` is sugar for an OR over the listed values.
        """),
        code("""
        run([
            "ego.action in (stop, yield, decel)",
            "agent.type in (ped, cyclist, animal)",
        ])
        """),
        md("""
        ## Same-entity coupling

        Inside an `agent(…)` clause, every constraint applies to the
        **same** agent. Free-floating predicates may bind to different
        agents — note how the count changes.
        """),
        code("""
        run([
            "agent(type = vehicle, pos = front)",
            "agent.type = vehicle and agent.pos = front",
        ])
        """),
        md("""
        ## Ego-relative direction — `agent.dir`

        Direction aliases: `same`, `opposite`, `perpendicular_lr`,
        `perpendicular_rl`, and the parent `perpendicular` (union of
        the two perpendicular leaves). Like `agent.pos`, `agent.dir`
        samples at the agent's visibility-window midpoint.
        """),
        code("""
        run([
            "agent.dir = same",
            "agent.dir = opposite",
            "agent.dir = perpendicular",
            # Oncoming vehicle — the natural way to ask "is there an
            # oncoming car?" for unprotected-turn analysis.
            "agent(type = vehicle, dir = opposite)",
        ])
        """),
        md("""
        ## Per-interval pose — `pos_any` / `dir_any`

        `agent.pos` / `agent.dir` sample at the **midpoint** of the
        agent's visibility window (one deterministic value per
        agent). For agents whose pose changes mid-window — e.g. a
        vehicle that approaches in front and ends up perpendicular as
        it crosses an intersection — the midpoint misses the
        transient. `pos_any` / `dir_any` are `kind="list"` and match
        if **any** `EgoRelativePose` interval carried the value.

        The delta between the two flavors shows how often pose
        changes during visibility:
        """),
        code("""
        # Midpoint vs interval — same query, different cardinality.
        run([
            "agent.dir = perpendicular",      # midpoint
            "agent.dir_any = perpendicular",  # any interval
        ])
        """),
        code("""
        # Composite: vehicles that were either in front OR crossing
        # at some point during their visibility.
        run([
            "agent(type = vehicle, "
            "pos_any in (front, perpendicular_lr, perpendicular_rl))",
        ])
        """),
        md("""
        ## Boolean operators

        Standard `and`, `or`, `not` over any sub-expression.
        """),
        code("""
        run([
            "agent.type = ped and ego.action in (stop, yield, decel)",
            "ego.action = stop or ego.action = yield",
            "agent.type = ped and not ego.action = drive",
        ])
        """),
        md("""
        ## Action flags

        Action types in the schema 2.0.0 corpus encode flags as
        parenthesized suffixes on the `action_type` string itself
        (`oxd:Walk (jaywalk)`). The legacy flag-field syntax
        `action(jaywalk = true)` no longer parses — match the
        suffix variants directly via literal-string equality or
        `in (...)`.

        Post-0.6.1, `Jaywalk` is also a standalone action (alias
        `jaywalk_action`); the combined-suffix forms are deprecated but
        still match, so a complete jaywalk query enumerates both.
        """),
        code("""
        run([
            "agent(type = ped, action.type in ("
            '"Jaywalk", '
            '"oxd:Walk (jaywalk)", "oxd:Walk (jaywalk, erratic)", '
            '"oxd:Run (jaywalk)", "oxd:Run (jaywalk, erratic)"))',
            "agent.action.type in ("
            '"oxd:Walk (erratic)", "oxd:Walk (jaywalk, erratic)", '
            '"oxd:Run (erratic)", "oxd:Run (jaywalk, erratic)")',
        ])
        """),
        md("""
        ## Temporal — `while`

        `A while B` keeps only pairs of matches whose intervals
        intersect.
        """),
        code("""
        run([
            "agent.type = ped while ego.action = decel",
            "light.color = red while ego.action = stop",
        ])
        """),
        md("""
        ## Temporal — `then(K)`

        `A then(K) B` keeps pairs where B starts during A or within
        K seconds after A ends. Bare `then` (no parens) defaults to
        K = 0 — touch or overlap only.
        """),
        code("""
        run([
            "light.color = yellow then(3) ego.action = stop",
            "light.color = green then ego.action = drive",
        ])
        """),
        md("""
        ## Relational — `because_of`

        Follows the schema's `because_of` causal edge. Both halves are
        emitted as match tuples so you can inspect cause and effect.
        """),
        code("""
        run([
            # `because_of` holds the IDs of *actions / states / objects*
            # that caused an action — never agent IDs directly. Point at
            # the agent's action, a signal state, or a traffic object.
            "ego.action = stop because_of light.color = red",
            "ego.action in (stop, yield, decel) because_of "
            'agent.action.type in ("oxd:Walk", "oxd:Stand")',
        ])
        """),
        md("""
        ## Relational — `influenced_by`

        Walks the schema's `Influence.influencers` side-channel (distinct
        from action-rooted `because_of`): "this entity *was under the
        influence of* B". The LHS is a bare entity (`ego` / `agent(...)`)
        and the match interval is the influence window, so it composes
        with `while` / `then`. A `light.color` RHS is scoped to that window.
        """),
        code("""
        run([
            "ego influenced_by light.color = red",
            "agent(type = vehicle) influenced_by light.color = red",
            "ego influenced_by (obj.type = stop_sign or light.color = red)",
        ])
        """),
        md("""
        ## Scoping — `within W: E`

        Restricts `E`'s evaluation window to `W`'s intervals. Inside,
        `not E` means *E does not happen during W* — useful for
        negative scenarios like "during red, ego never stops."
        """),
        code("""
        run([
            "within light.color = red: not ego.action = stop",
            "within env.type = road: agent.type = ped",
        ])
        """),
        md("""
        ## Annotator-tagged flags

        Some attributes are not derived from kinematics but tagged by
        the annotator (e.g. *"the ego could have safely cleared this
        yellow"*). They compose with the same DSL.
        """),
        code("""
        run([
            "light(color = yellow, ego_in_on_yellow = true)",
            "light(color = yellow, could_have_cleared = true)",
        ])
        """),
        md("""
        ## Agent group size — `agent.amount`

        Lone agent vs row/group vs traffic-density tier. Aliases:
        `single`, `row` (alias `group`), `light_traffic`,
        `medium_traffic`, `heavy_traffic`, plus the parents
        `multiple` (everything but `single`) and `traffic` (the
        three density tiers).
        """),
        code("""
        run([
            "agent.amount = single",
            "agent.amount = group",
            "agent(type = ped, amount = group)",
        ])
        """),
        md("""
        ## Signaling details on `agent.prop` / `ego.prop`

        Properties of type `signal` carry a `signaling_details`
        sub-object exposing the source modality (flashing light,
        hand gesture, holding sign, other) and a `not_facing_ego`
        boolean. Non-signal properties resolve to `None` so the
        predicates fall through cleanly.
        """),
        code("""
        run([
            "agent.prop.source = flashing_light",
            "agent.prop.source in (flashing_light, holding_sign, other)",
            "agent.prop.not_facing_ego = false",
            "agent(prop.source = flashing_light)",
        ])
        """),
        md("""
        ## Containment flags — `lane_edge`, `near_lane`, `illegal_lane`

        Boolean predicates that match existentially across the
        entity's containment records. Exposed on both `agent` and
        `obj`. `lane_edge` is true when any containment carries
        `edge ∈ {"left","right"}`; `near_lane` and `illegal_lane`
        mirror the boolean flags directly.
        """),
        code("""
        run([
            "agent.illegal_lane = true",
            "agent.lane_edge = true",
            "obj.lane_edge = true",
            "agent(type = car, illegal_lane = true)",
        ])
        """),
        md("""
        ## Clip-level attributes — `clip.eventful` / `clip.eventful_reason`

        The boolean `clip.eventful` flags clips worth a closer look.
        Post-0.6.1, `clip.eventful_reason` is queryable alongside it —
        values: `ego_adapts`, `special_env`, `agent_adapts`, `other`.
        """),
        code("""
        run([
            "clip.eventful = true",
            "clip.eventful_reason = ego_adapts",
        ])
        """),
        md("""
        ## Newly-aliased `"Other"` / `"Vehicle"` values

        Four families that previously required the schema-literal
        escape hatch (`agent.type = "Other"`) now resolve `other`
        as a friendly alias. The annotator's generic-vehicle
        fallback `"Vehicle"` (used when an agent is a vehicle that
        doesn't match any specific subtype) is folded into the
        `vehicle` parent alias and also reachable as the leaf
        `generic_vehicle`.
        """),
        code("""
        run([
            "agent.type = other",
            "agent.type = generic_vehicle",
            "env.type = other",
            "agent.action.type = other",
        ])
        """),
        md("""
        ## Condition → environment lookup — `cond.env_*`

        Each `Condition` carries an `env_id` pointing at one of the
        bundle's `Environment` records. The DSL has no cross-entity
        join operator today, so the four most-useful environment
        fields are exposed as *denormalized* attributes on `cond`:
        `cond.env_type` (reuses the `env_type` alias family),
        `cond.env_lanes`, `cond.env_one_way`, and `cond.env_id`
        (raw passthrough). A condition with an unresolvable
        `env_id` falls through to `None` for the derived fields, so
        predicates compose without an explicit guard.
        """),
        code("""
        run([
            "cond.env_type = road",
            "cond.type = construction and cond.env_type = road",
            "cond.env_one_way = true",
            "cond.env_lanes > 2",
        ])
        """),
    ]
    save(cells, NOTEBOOKS_DIR / "02_query_dsl_tour.ipynb")


# ---------------------------------------------------------------------------
# 03 — Statistics with charts
# ---------------------------------------------------------------------------


def build_statistics() -> None:
    cells = [
        md("""
        # Dataset statistics

        Count and group-by queries with matplotlib charts. Use this
        notebook to get a feel for how the corpus is distributed across
        agent types, ego actions, and environments before you go
        looking for specific scenarios.

        > See `01_quickstart.ipynb` for an introduction to the dataset
        > and `02_query_dsl_tour.ipynb` for the query language itself.
        """),
        md("## Setup"),
        code(
            """
        import os
        from collections import Counter
        from pathlib import Path

        from cascade_av.dataset import CascadeDataset
        """
            + STYLING
        ),
        code("""
        ds = CascadeDataset(Path(os.environ["CASCADE_AV_DATASET_ROOT"]))
        print(f"corpus loaded — {len(ds)} clips")

        def entity_count(query: str) -> int:
            \"\"\"Number of matching *entities* across the corpus (not clips).\"\"\"
            return len(ds.find(query).matches)

        def entities_by_attr(query: str, attr: str) -> dict:
            \"\"\"Bucket matching entities by a top-level attribute (e.g. ``type``).

            Like ``ds.group_by`` but counts entities rather than clips.
            \"\"\"
            out = Counter()
            for m in ds.find(query).matches:
                val = getattr(m.entity, attr, None)
                if isinstance(val, list):
                    for v in val:
                        out[v] += 1
                else:
                    out[val] += 1
            return dict(out)
        """),
        md("""
        ## Clips vs entities

        The query API exposes two complementary perspectives:

        - **`ds.count(query)`** — the number of *clips* with ≥1 match.
          Tells you about *coverage*: "how many clips even contain
          this thing?"
        - **`len(ds.find(query).matches)`** — the number of *matching
          entities*. Tells you about *prevalence*: "how often does
          this thing occur?" A single clip can contribute many
          entities (six pedestrians, four vehicles, …).

        Both are useful — coverage tells you how varied the corpus
        is; prevalence tells you the actual instance counts.
        """),
        md("## Headline counts"),
        code("""
        primitives = [
            ("pedestrian",              "agent.type = ped"),
            ("vehicle",                 "agent.type = vehicle"),
            ("VRU (ped/cyclist)",       "agent.type = vru"),
            ("cyclist",                 "agent.type = cyclist"),
            ("stop sign",               "obj.type = stop_sign"),
            ("yield sign",              "obj.type = yield_sign"),
            ("road environment",        "env.type = road"),
            ("other environment",       "env.type = other"),
            ("multi-lane environment",  "env.lanes >= 2"),
        ]
        headline = pd.DataFrame(
            [(label, ds.count(q), entity_count(q)) for label, q in primitives],
            columns=["category", "clips", "entities"],
        ).sort_values("entities", ascending=False, ignore_index=True)
        headline
        """),
        md("""
        ## Distribution of agent types

        `group_by(query, key)` buckets matches by the value of `key`
        and returns **clip counts**. Pair it with
        `entities_by_attr(query, attr)` to see entity counts in the
        same buckets.
        """),
        code("""
        agent_query = "agent.type = vehicle or agent.type = vru or agent.type = animal"

        clips_by_type    = ds.group_by(agent_query, key="agent.type")
        entities_by_type = entities_by_attr(agent_query, attr="type")

        agent_df = pd.DataFrame({
            "clips":    pd.Series(clips_by_type),
            "entities": pd.Series(entities_by_type),
        }).fillna(0).astype(int)
        agent_df = agent_df.sort_values("entities", ascending=True)

        fig, ax = plt.subplots(figsize=(9, 0.4 * len(agent_df) + 1.5))
        y = range(len(agent_df))
        ax.barh([i + 0.2 for i in y], agent_df["clips"],    height=0.4,
                color="#3b6ea5", label="clips")
        ax.barh([i - 0.2 for i in y], agent_df["entities"], height=0.4,
                color="#8aa6c8", label="entities")
        ax.set_yticks(list(y))
        ax.set_yticklabels(agent_df.index)
        ax.set_xlabel("count")
        ax.set_title("Agent types — clip coverage vs entity prevalence")
        ax.legend(loc="lower right", frameon=False)
        plt.tight_layout()
        plt.show()

        agent_df
        """),
        md("## Distribution of ego actions"),
        code("""
        ego_dist = ds.group_by(
            "ego.action.type = drive or ego.action.type = stop or ego.action.type = decel "
            "or ego.action.type = yield or ego.action.type = turn_left "
            "or ego.action.type = turn_right or ego.action.type = change_lane",
            key="ego.action.type",
        )
        s = pd.Series(ego_dist, name="clips").sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 0.45 * max(len(s), 4) + 1.5))
        s.plot(kind="barh", ax=ax, color="#4f7d4a")
        ax.set_xlabel("clips")
        ax.set_ylabel("")
        ax.set_title("Clips per ego-action type")
        for i, v in enumerate(s.values):
            ax.text(v + max(s.values) * 0.01, i, str(v), va="center", fontsize=9)
        plt.tight_layout()
        plt.show()
        """),
        md("## Distribution of environment types"),
        code("""
        env_dist = ds.group_by(
            "env.type = road or env.type = intersection or env.type = crosswalk "
            "or env.type = sidewalk or env.type = roundabout or env.type = cycle_lane "
            "or env.type = tunnel",
            key="env.type",
        )
        s = pd.Series(env_dist, name="clips").sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 0.45 * max(len(s), 4) + 1.5))
        s.plot(kind="barh", ax=ax, color="#8a5a3b")
        ax.set_xlabel("clips")
        ax.set_ylabel("")
        ax.set_title("Clips per environment type")
        for i, v in enumerate(s.values):
            ax.text(v + max(s.values) * 0.01, i, str(v), va="center", fontsize=9)
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## Composite questions

        DSL queries compose, so you can pose questions that combine
        primitives — for example, "of the clips that contain a red
        light, what fraction also contain an ego stop?"
        """),
        code("""
        n_red               = ds.count("light.color = red")
        n_red_stop          = ds.count("light.color = red and ego.action = stop")
        n_yellow_clear      = ds.count("light(color = yellow, could_have_cleared = true)")
        # Post-0.6.1, `Jaywalk` is a standalone action (alias
        # `jaywalk_action`); the combined-suffix forms are deprecated but
        # still match, so enumerate the standalone action alongside them.
        _jaywalk_q = (
            'agent(type = ped, action.type in ('
            '"Jaywalk", '
            '"oxd:Walk (jaywalk)", "oxd:Walk (jaywalk, erratic)", '
            '"oxd:Run (jaywalk)", "oxd:Run (jaywalk, erratic)"))'
        )
        n_jaywalk_clips     = ds.count(_jaywalk_q)
        n_jaywalk_entities  = entity_count(_jaywalk_q)

        composite = pd.DataFrame([
            ("ego stops at red light",            n_red_stop,         n_red, n_red_stop / max(n_red, 1)),
            ("yellow ego could have cleared",     n_yellow_clear,     None,  None),
            ("jaywalking pedestrian (clips)",     n_jaywalk_clips,    None,  None),
            ("jaywalking pedestrian (entities)",  n_jaywalk_entities, None,  None),
        ], columns=["question", "matching clips/entities", "denominator", "ratio"])
        composite.style.format({"ratio": "{:.0%}"}, na_rep="—")
        """),
    ]
    save(cells, NOTEBOOKS_DIR / "03_statistics.ipynb")


# ---------------------------------------------------------------------------
# 04 — Scenario catalog
# ---------------------------------------------------------------------------


def build_scenarios() -> None:
    cells = [
        md("""
        # Scenario catalog

        Twenty representative driving scenarios encoded as DSL queries
        and run against the corpus. The output is a ranked table of
        candidate clip counts — clips that satisfy the necessary
        conditions for that scenario.

        > These queries are *necessary-condition filters*. They reject
        > clearly-not-this-scenario candidates; a downstream classifier
        > (VLM, human reviewer, heuristic) would disambiguate the
        > survivors.

        > See `01_quickstart.ipynb` for an introduction to the dataset
        > and `02_query_dsl_tour.ipynb` for the DSL grammar.
        """),
        md("## Setup"),
        code(
            """
        import os
        from dataclasses import dataclass
        from pathlib import Path

        from cascade_av.dataset import CascadeDataset
        """
            + STYLING
        ),
        code("""
        ds = CascadeDataset(Path(os.environ["CASCADE_AV_DATASET_ROOT"]))

        @dataclass(frozen=True)
        class Scenario:
            id: str
            title: str
            query: str
        """),
        md("""
        ## The catalog

        Twenty scenarios spanning pedestrian interactions, traffic
        signals, signage, lane changes, junctions, and explicit
        causal queries.
        """),
        code("""
        SCENARIOS = [
            Scenario("1",  "Ego drives on while a pedestrian is present (no yield)",
                     "agent.type = ped and ego.action = drive"),
            Scenario("2",  "Ego yields/stops/brakes for a pedestrian",
                     "agent.type = ped and ego.action in (stop, yield, decel)"),
            Scenario("8",  "Pedestrian jaywalks; ego brakes / yields",
                     # Post-0.6.1, `Jaywalk` is a standalone action
                     # (alias jaywalk_action); the combined-suffix forms
                     # are deprecated but still match, so enumerate both.
                     'agent(type = ped, action.type in ('
                     '"Jaywalk", '
                     '"oxd:Walk (jaywalk)", "oxd:Walk (jaywalk, erratic)", '
                     '"oxd:Run (jaywalk)", "oxd:Run (jaywalk, erratic)"'
                     ')) and ego.action in (stop, yield, decel)'),
            Scenario("9",  "Pedestrian present while ego is turning",
                     "agent.type = ped and ego.action in (turn_left, turn_right)"),
            Scenario("12", "Cyclist mid-block in ego's path; ego defensive",
                     "agent.type = cyclist and ego.action in (stop, yield, decel)"),
            Scenario("14", "Ego stops at a red traffic light (nominal)",
                     "light.color = red and ego.action = stop"),
            Scenario("16", "Ego proceeds through a green light",
                     "light.color = green and ego.action = drive"),
            Scenario("20", "Ego already in intersection when yellow begins, proceeds",
                     "light(color = yellow, ego_in_on_yellow = true) and ego.action in (drive, enter, creep)"),
            Scenario("21", "Ego approaches intersection, yellow appears, ego stops",
                     "light(color = yellow, on_ego_path = true) and ego.action = stop"),
            Scenario("23", "Ego stops on yellow it could have cleared safely",
                     "light(color = yellow, could_have_cleared = true) and ego.action = stop"),
            Scenario("aws-basic", "Ego stops at a stop sign",
                     "obj.type = stop_sign and ego.action = stop"),
            Scenario("yield-basic", "Ego yields at a yield sign",
                     "obj.type = yield_sign and ego.action = yield"),
            Scenario("76", "Stopped vehicle ahead; ego nudges or changes lane",
                     "agent(type = vehicle, action(type in (stop, not_move))) "
                     "and ego.action in (nudge, change_lane_left, change_lane_right)"),
            Scenario("lane-multi", "Multi-lane road, ego changes lane",
                     "env(type = road, lanes >= 2) and ego.action in (change_lane_left, change_lane_right)"),
            Scenario("emergency-vehicle", "Emergency / hazard vehicle (flashing lights) present; ego brakes",
                     "agent(type = vehicle, prop.source = flashing_light) "
                     "and ego.action in (stop, yield, decel)"),
            Scenario("junction-turn", "Ego turns at a junction",
                     "env.type = road and ego.action in (turn_left, turn_right)"),
            Scenario("vehicle-cut-in", "Vehicle cuts in / changes lane ahead; ego brakes",
                     "agent(type = vehicle, action(type in (change_lane, change_lane_left, change_lane_right))) "
                     "and ego.action in (decel, stop, nudge)"),
            Scenario("stop-because-red", "Ego stops BECAUSE OF a red light (causal edge)",
                     "ego.action = stop because_of light.color = red"),
            Scenario("yellow-then-stop", "Yellow light followed by ego stop within 3s",
                     "light.color = yellow then(3) ego.action = stop"),
            Scenario("red-without-stop", "During red light, ego never stops (potential violation)",
                     "within light.color = red: not ego.action = stop"),
        ]
        print(f"{len(SCENARIOS)} scenarios catalogued")
        """),
        md("""
        ## Run them all and tabulate

        Sorted by candidate count, descending. The `example` column
        shows one matching `clip_id` per scenario — drop it into the
        annotation tool or `find_on_bundle` to inspect.
        """),
        code("""
        rows = []
        for s in SCENARIOS:
            try:
                matches = ds.find(s.query)
                clips = sorted(set(matches.clips()))
                rows.append({
                    "id":         s.id,
                    "title":      s.title,
                    "candidates": len(clips),
                    "example":    clips[0] if clips else "",
                })
            except Exception as e:
                rows.append({
                    "id":         s.id,
                    "title":      s.title,
                    "candidates": -1,
                    "example":    f"ERROR: {e}",
                })

        results = pd.DataFrame(rows).sort_values("candidates", ascending=False, ignore_index=True)
        results
        """),
        md("""
        ## Visualise the distribution

        A quick bar chart of candidate counts makes it obvious which
        scenarios are well-represented in the corpus and which are
        scarce.
        """),
        code("""
        plot_df = results[results["candidates"] >= 0].sort_values("candidates", ascending=True)

        fig, ax = plt.subplots(figsize=(9, 0.4 * len(plot_df) + 1.5))
        ax.barh(plot_df["title"], plot_df["candidates"], color="#3b6ea5")
        ax.set_xlabel("candidate clips")
        ax.set_title("Scenario coverage in the corpus")
        for i, v in enumerate(plot_df["candidates"].values):
            ax.text(v + max(plot_df["candidates"].max(), 1) * 0.01, i,
                    str(v), va="center", fontsize=9)
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## Dig into one scenario

        Pick a scenario, list every matching clip. Swap the `target`
        query for any expression you want to drill into.
        """),
        code("""
        target = (
            'agent(type = ped, action.type in ('
            '"Jaywalk", '
            '"oxd:Walk (jaywalk)", "oxd:Walk (jaywalk, erratic)", '
            '"oxd:Run (jaywalk)", "oxd:Run (jaywalk, erratic)"'
            ')) and ego.action in (stop, yield, decel)'
        )
        clips = sorted(set(ds.find(target).clips()))

        print(f"{len(clips)} clips match: {target}\\n")
        pd.DataFrame({"clip_id": clips})
        """),
    ]
    save(cells, NOTEBOOKS_DIR / "04_scenario_catalog.ipynb")


# ---------------------------------------------------------------------------
# 05 — Video inspection: query → match → download → view frames
# ---------------------------------------------------------------------------


def build_video_inspection() -> None:
    cells = [
        md("""
        # Inspecting matches in video

        The DSL tells you *where* in the corpus a scenario happens.
        This notebook closes the loop: given a match, pull the
        original video from the Physical AI AV Dataset on HuggingFace,
        decode frames over the match's interval, and view them
        alongside the `ContextWindow` snapshot.

        Pipeline:

        1. Run a DSL query and pick a match.
        2. Prefetch the clip's video and egomotion from HF with
           `ds.download_clips`.
        3. Build a `Sequence` and decode frames sampled across the
           match interval.
        4. Show what else was annotated during the window.

        > Requires the optional `hf` extra (`uv sync --extra hf`) plus
        > the notebook tooling (`--group notebooks`). The first
        > download pulls a few hundred MB per clip and can take
        > 5–15 seconds depending on your link.
        """),
        md("## Setup"),
        code(
            """
        import os
        from pathlib import Path

        import numpy as np

        from cascade_av.dataset import CascadeDataset
        """
            + STYLING
        ),
        code("""
        ds = CascadeDataset(Path(os.environ["CASCADE_AV_DATASET_ROOT"]))
        print(f"corpus loaded — {len(ds)} clips")
        """),
        md("""
        ## 1. Find a match

        Pick a scenario with a tight temporal window. `while` and
        `then` produce matches whose interval is the actual moment
        both conditions held — ideal for video inspection.
        """),
        code("""
        query = "agent.type = ped while ego.action = decel"
        matches = ds.find(query)
        print(f"{len(matches)} matches for: {query}")

        m = matches.matches[0]
        print(f"chosen match: clip {m.clip_id}  "
              f"window {m.interval.start:.2f}–{m.interval.end:.2f}s")
        """),
        md("""
        ## 2. Prefetch the clip from HuggingFace

        `ds.download_clips` accepts an iterable of clip ids and pulls
        the canonical camera plus egomotion (the minimum needed to
        build a `Sequence`). Files are cached locally — re-running
        is a no-op.
        """),
        code("""
        ds.download_clips([m.clip_id])
        """),
        md("""
        ## 3. Decode frames across the match window

        `seq.video.decode_images_from_timestamps` takes microsecond
        timestamps as an `int64` array. We sample uniformly across
        the match interval.
        """),
        code("""
        seq = ds.get_sequence(m.clip_id)
        print(f"clip duration: {seq.duration_s:.2f}s   fps: {seq.fps}")

        n_frames = 6
        t_seconds = np.linspace(m.interval.start, m.interval.end, n_frames)
        t_us = (t_seconds * 1_000_000).astype(np.int64)
        images, _ = seq.video.decode_images_from_timestamps(t_us)

        fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
        for ax, img, t in zip(axes.ravel(), images, t_seconds):
            ax.imshow(img)
            ax.set_title(f"t = {t:.2f}s", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.suptitle(
            f"{query}  —  clip {m.clip_id[:8]}…  window "
            f"{m.interval.start:.2f}–{m.interval.end:.2f}s",
            fontsize=12, fontweight="semibold",
        )
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## 4. What else was happening?

        Pair the frames with `ds.context_for(m)` — the same window
        seen through the annotation lens.
        """),
        code("""
        ctx = ds.context_for(m)

        pd.DataFrame([
            ("clip_id",        ctx.clip_id),
            ("window",         f"{ctx.interval.start:.2f}–{ctx.interval.end:.2f}s"),
            ("agents",         len(ctx.agents)),
            ("ego actions",    [ea.type for ea in ctx.ego_actions]),
            ("environments",   [e.type for e in ctx.environments]),
            ("light states",   len(ctx.light_states)),
            ("traffic objects", [o.type for o in ctx.traffic_objects]),
        ], columns=["field", "value"])
        """),
        code("""
        pd.DataFrame([
            {
                "id":         a.agent.id,
                "type":       a.agent.type,
                "visible":    f"{a.visibility.start:.2f}–{a.visibility.end:.2f}s",
                "actions":    [ax.action_type for ax in a.actions],
            }
            for a in ctx.agents
        ])
        """),
        md("""
        ## Inspect another scenario

        Swap the query, re-run the cells above. The flow is the same
        for any match: find → download → decode → contextualise.
        """),
    ]
    save(cells, NOTEBOOKS_DIR / "05_video_inspection.ipynb")


def build_visualize() -> None:
    cells = [
        md("""
        # Visualizing matches — frames, timelines, and the clip player

        The DSL tells you *where* in the corpus a scenario happens.
        `05_video_inspection.ipynb` closes the loop with bare
        matplotlib frames. This notebook layers on the interactive
        `cascade_av.viz` surface — the clip player widget that
        scrubs a video alongside its annotation timeline, plus
        headless helpers for reports and figures.

        Pipeline:

        1. Set up — load the corpus, pick a clip.
        2. `seq.visualize()` — full-clip scrubbable widget.
        3. `seq.visualize(t=..., static=True)` — single decoded frame
           as a `PIL.Image` (great for static reports).
        4. `viz.render_timeline(seq)` — static Plotly timeline figure.
        5. `mode="paper_figure"` — compose up to three selected frames
           above per-entity-filtered tracks for reports and papers.
        6. `matches.visualize()` — fan out a query result into a
           carousel of mini-players, one per match.
        7. `families=[...]` — render only a subset of the annotation
           families (just the actions, just the conditions, etc.).

        > Requires the optional `viz` extra (`uv sync --extra viz`)
        > and the `hf` extra (`uv sync --extra hf`) for the video
        > download. The first download per clip pulls a few hundred MB
        > and can take 5-15 s.
        """),
        md("## Setup"),
        code(
            """
        import os
        from pathlib import Path

        from cascade_av import viz
        from cascade_av.dataset import CascadeDataset
        """
            + STYLING
        ),
        code("""
        ds = CascadeDataset(
            Path(os.environ["CASCADE_AV_DATASET_ROOT"]),
            confirm_download_threshold_gb=1000.0,  # auto-confirm; notebook context
        )
        print(f"corpus loaded — {len(ds)} clips")
        """),
        md("""
        ## 1. Pick a clip with something interesting in it

        `while` tightens each match's interval to the intersection,
        so the highlight band is the actual moment both conditions
        held — exactly what we want to focus the player on.
        """),
        code("""
        query = "agent.type = ped while ego.action = decel"
        matches = ds.find(query)
        m = matches.matches[0]
        print(f"{len(matches)} matches for: {query}")
        print(f"chosen: clip {m.clip_id}  window {m.interval.start:.2f}-{m.interval.end:.2f}s")

        # Pull the clip's video + egomotion (cached on subsequent runs).
        ds.download_clips([m.clip_id])
        seq = ds.get_sequence(m.clip_id)
        print(f"clip duration: {seq.duration_s:.2f}s   fps: {seq.fps}")
        """),
        md("""
        ## 2. `seq.visualize()` — the full-clip scrubbable widget

        `seq.visualize()` builds a `ClipPlayer`: a Plotly `FigureWidget`
        with the decoded video frame on top, the annotation timeline
        below, and an `ipywidgets.FloatSlider` + `Play` button to drive
        the playhead. No arguments → the whole clip.
        """),
        code("""
        seq.visualize()
        """),
        md("""
        ## 3. Single frame as a `PIL.Image`

        Passing `static=True` to `seq.visualize(t=...)` skips the
        widget machinery and returns a plain `PIL.Image` — handy
        when you want a single decoded frame to save, embed, or
        post-process. Equivalent to `viz.render_frame(seq, t)`.
        """),
        code("""
        frame = seq.visualize(t=(m.interval.start + m.interval.end) / 2, static=True)
        print(f"decoded frame: {frame.size[0]}x{frame.size[1]}  mode={frame.mode}")
        frame
        """),
        md("""
        ## 4. Headless timeline figure

        `viz.render_timeline(seq, highlight=...)` returns a static
        `plotly.graph_objects.Figure` with one row per track group
        (Ego / Agents / Traffic Lights / Objects / Environments) and
        causal relationship arrows such as `because_of` overlaid. The
        `highlight` band marks a window — pass `m.interval` to focus
        attention on the match.
        """),
        code("""
        fig = viz.render_timeline(seq, highlight=(m.interval.start, m.interval.end))
        fig
        """),
        md("""
        ## 5. `mode="paper_figure"` — selected frames + selected tracks

        Publication figures often need a few exact moments from the
        clip above one shared annotation timeline. Paper mode accepts
        zero to three explicit timestamps and sorts the frames
        chronologically from left to right. Duplicate timestamps remain
        separate frames.

        `track_visibility` can switch a whole kind or individual
        top-level entities by stable annotation ID. Here we hide up to
        two Agent tracks while leaving every omitted entity visible.
        Their child rows and any causal arrows connected to them
        disappear too. The direct equivalent is
        `viz.render_paper_figure(seq, ...)`.

        Unlike compact interactive timelines, paper figures keep the
        complete name on every visible box, including narrow boxes; no
        label is ellipsized or relegated to a hover-only tooltip.
        """),
        code("""
        # Use the match start, midpoint, and end, snapped to actual video
        # frame timestamps so the figure is exactly reproducible.
        requested_timestamps = [
            float(m.interval.start),
            float((m.interval.start + m.interval.end) / 2),
            float(m.interval.end),
        ]
        video_timestamps_us = seq.video.timestamps
        paper_timestamps = [
            float(
                video_timestamps_us[
                    abs(video_timestamps_us - int(round(t * 1_000_000))).argmin()
                ]
            ) / 1_000_000
            for t in requested_timestamps
        ]

        # Select by stable Agent.id, not by `_track_index` (a reusable
        # visual lane). Ignore blank or duplicate IDs because those are
        # intentionally ambiguous selectors.
        all_agent_ids = [agent.id for agent in seq.annotation.annotation.agents]
        hidden_agent_ids = [
            agent_id
            for agent_id in all_agent_ids[-2:]
            if agent_id and all_agent_ids.count(agent_id) == 1
        ]
        track_visibility = {
            "agent": {agent_id: False for agent_id in hidden_agent_ids}
        }

        paper_figure = seq.visualize(
            mode="paper_figure",
            timestamps=paper_timestamps,
            highlight=(m.interval.start, m.interval.end),
            track_visibility=track_visibility,
            width=1200,
        )
        print("frame timestamps:", [f"{t:.3f}s" for t in paper_timestamps])
        print(f"hidden Agent tracks: {len(hidden_agent_ids)}")
        paper_figure
        """),
        md("""
        ## 6. `matches.visualize()` — carousel of mini-players

        A whole `MatchSet` can be turned into a carousel of
        `ClipPlayer`s, one per match. By default the carousel:

        - Pads each match's interval by 1 s on either side (`pad=1.0`).
        - Caps the carousel at 8 matches (`limit=8`) with a "showing N
          of M" notice if the result set is bigger.
        - Stacks the players vertically (`layout="stack"`); pass
          `layout="grid"` to switch to a multi-column grid.

        Each player opens to its match window with the raw match
        interval painted as the highlight band.
        """),
        code("""
        # `unique_clips=True` makes `limit=3` mean "three *distinct*
        # clips" — without it, three matches inside the same clip would
        # render that clip three times. The default is False because
        # match-by-match is occasionally what you want; the notebook
        # opts in because surveying distinct clips is the common case.
        # Pre-download the same three clips so widget construction
        # doesn't block on HF.
        first_three_clips = matches.clips()[:3]
        ds.download_clips(first_three_clips)

        matches.visualize(limit=3, unique_clips=True)
        """),
        md("""
        ## 7. `families=[...]` — render only selected annotation families

        Every visualize entry point (`seq.visualize`, `render_timeline`,
        `matches.visualize`) accepts a `families` whitelist. Pass a list
        of family leaves — `"action"`, `"condition"`, `"containment"`,
        `"property"`, `"pose"`, `"state"`,
        `"signal_head"`, `"env_control"`, `"physical_containment"` —
        and the timeline collapses to just those rows.

        `Influence` records remain queryable but are intentionally omitted
        from DevKit timeline rows; action-rooted `because_of` relationships
        are represented as arrows instead. The former
        `families=["influence"]` selector remains accepted as a compatibility
        no-op.

        Parent entity headers ("Env Track 1", "Agent Track 2", ...) are
        NOT named in the whitelist. They auto-render for any entity
        whose sub-rows survive the filter; entities with zero
        surviving sub-rows drop entirely (no orphan headers). This is
        the same composition rule used by `track_groups`, `entity_kinds`,
        `agent_ids`, and the clip-local `track_visibility` switches:
        every supplied segment filter ANDs together.
        """),
        code("""
        # Just the action rows across the clip — ego actions and any
        # agent actions, with their entity headers as the only context.
        viz.render_timeline(
            seq,
            highlight=(m.interval.start, m.interval.end),
            families=["action"],
        )
        """),
        code("""
        # Compose with `matches.visualize` — every mini-player in the
        # carousel renders the same family subset. `unique_clips=True`
        # keeps "limit=3" honest as three distinct clips even when
        # multiple matches happen to land in the same clip.
        matches.visualize(
            limit=3, families=["action", "condition"], unique_clips=True
        )
        """),
    ]
    save(cells, NOTEBOOKS_DIR / "06_visualize.ipynb")


def main() -> None:
    build_quickstart()
    build_dsl_tour()
    build_statistics()
    build_scenarios()
    build_video_inspection()
    build_visualize()


if __name__ == "__main__":
    main()
