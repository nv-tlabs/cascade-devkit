"""Generate the `notebooks/` Jupyter files from this script.

Run after editing to regenerate:

    uv run --group notebooks python scripts/build_notebooks.py

Each notebook reads the dataset root from the `CAUSAL_AV_DATASET_ROOT`
environment variable — export it before launching Jupyter:

    export CAUSAL_AV_DATASET_ROOT=/path/to/json_annotations
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
        "display_name": "Python 3 (causal_ai_av)",
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
        # Quickstart — `causal_ai_av`

        This notebook introduces the AV Causal Dataset DevKit. You will
        load the corpus, inspect a single clip's annotation, and run
        your first queries in the embedded DSL.
        """),
        md("""
        ## About the dataset

        The **AV Causal Dataset** adds causal and spatio-temporal action
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

        > Set the `CAUSAL_AV_DATASET_ROOT` environment variable to the
        > directory containing the JSON annotations before running the
        > setup cell below.
        """),
        md("## Setup"),
        code("""
        import os
        from pathlib import Path

        from causal_ai_av.dataset import CausalAVDataset
        from causal_ai_av.io import load_dir
        from causal_ai_av.query import find_on_bundle
        """ + STYLING),
        code("""
        dataset_root = Path(os.environ["CAUSAL_AV_DATASET_ROOT"])
        ds = CausalAVDataset(dataset_root)
        print(f"corpus loaded — {len(ds)} clips")
        """),
        md("""
        ## 1. Inspect one clip's annotation

        `CausalAVDataset.get_sequence` downloads the clip video lazily by
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

        `CausalAVDataset.count(query)` returns the number of clips with
        at least one match. Queries are plain strings in the DSL — the
        full grammar is covered in `02_query_dsl_tour.ipynb`.
        """),
        code("""
        queries = [
            "agent.type = ped",
            "agent.type = vehicle",
            "agent.type = ped and env.type = crosswalk",
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
        matches = ds.find("agent.type = ped and env.type = crosswalk")
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
        ## Where to next

        - `02_query_dsl_tour.ipynb` — every DSL operator with examples.
        - `03_statistics.ipynb` — count / group-by aggregations with charts.
        - `04_scenario_catalog.ipynb` — twenty representative driving
          scenarios encoded as DSL queries.
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
        | `within W: E` | restrict E's time window to W's intervals |

        The full specification lives in `docs/query_language.md`.
        """),
        md("## Setup"),
        code("""
        import os
        from pathlib import Path

        from causal_ai_av.dataset import CausalAVDataset
        """ + STYLING),
        code("""
        ds = CausalAVDataset(Path(os.environ["CAUSAL_AV_DATASET_ROOT"]))

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
        ## Boolean operators

        Standard `and`, `or`, `not` over any sub-expression.
        """),
        code("""
        run([
            "agent.type = ped and env.type = crosswalk",
            "ego.action = stop or ego.action = yield",
            "agent.type = ped and not env.type = crosswalk",
        ])
        """),
        md("""
        ## Action flags

        Action types in the corpus encode flags as parenthesized
        suffixes (`oxd:Walk (jaywalk)`). Flag attributes match either
        the schema flag field **or** the suffix token.
        """),
        code("""
        run([
            "agent(type = ped, action(jaywalk = true))",
            "agent(action(erratic = true))",
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
            "ego.action = decel because_of agent.type = ped",
            "ego.action = drive because_of agent.type = ped",
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
            "within env.type = crosswalk: agent.type = ped",
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
        code("""
        import os
        from pathlib import Path

        from causal_ai_av.dataset import CausalAVDataset
        """ + STYLING),
        code("""
        ds = CausalAVDataset(Path(os.environ["CAUSAL_AV_DATASET_ROOT"]))
        print(f"corpus loaded — {len(ds)} clips")
        """),
        md("""
        ## Headline counts

        Single-query counts for common driving primitives. `count()`
        returns the number of *clips* with at least one match, not the
        number of matching entities.
        """),
        code("""
        rows = [
            ("pedestrian",              ds.count("agent.type = ped")),
            ("vehicle",                 ds.count("agent.type = vehicle")),
            ("VRU (ped/cyclist)",       ds.count("agent.type = vru")),
            ("cyclist",                 ds.count("agent.type = cyclist")),
            ("stop sign",               ds.count("obj.type = stop_sign")),
            ("traffic light (red)",     ds.count("light.color = red")),
            ("crosswalk environment",   ds.count("env.type = crosswalk")),
            ("intersection environment", ds.count("env.type = intersection")),
            ("roundabout environment",  ds.count("env.type = roundabout")),
            ("multi-lane road",         ds.count("env(type = road, lanes >= 2)")),
            ("ego decelerates",         ds.count("ego.action = decel")),
            ("ego stops at red",        ds.count("light.color = red and ego.action = stop")),
        ]
        headline = pd.DataFrame(rows, columns=["query", "clips"]).sort_values(
            "clips", ascending=False, ignore_index=True
        )
        headline
        """),
        md("""
        ## Distribution of agent types

        `group_by(query, key)` runs the query and buckets matches by
        the value of `key`. Values returned are clip counts (one per
        distinct clip with a matching entity).
        """),
        code("""
        agent_dist = ds.group_by(
            "agent.type = vehicle or agent.type = vru or agent.type = animal",
            key="agent.type",
        )
        s = pd.Series(agent_dist, name="clips").sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 0.45 * max(len(s), 4) + 1.5))
        s.plot(kind="barh", ax=ax, color="#3b6ea5")
        ax.set_xlabel("clips")
        ax.set_ylabel("")
        ax.set_title("Clips per agent type")
        for i, v in enumerate(s.values):
            ax.text(v + max(s.values) * 0.01, i, str(v), va="center", fontsize=9)
        plt.tight_layout()
        plt.show()
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
        n_red          = ds.count("light.color = red")
        n_red_stop     = ds.count("light.color = red and ego.action = stop")
        n_yellow_clear = ds.count("light(color = yellow, could_have_cleared = true)")
        n_jaywalk      = ds.count("agent(type = ped, action(jaywalk = true))")

        composite = pd.DataFrame([
            ("ego stops at red light", n_red_stop, n_red, n_red_stop / max(n_red, 1)),
            ("yellow ego could have safely cleared", n_yellow_clear, None, None),
            ("jaywalking pedestrian present", n_jaywalk, None, None),
        ], columns=["question", "matching clips", "denominator", "ratio"])
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
        code("""
        import os
        from dataclasses import dataclass
        from pathlib import Path

        from causal_ai_av.dataset import CausalAVDataset
        """ + STYLING),
        code("""
        ds = CausalAVDataset(Path(os.environ["CAUSAL_AV_DATASET_ROOT"]))

        @dataclass(frozen=True)
        class Scenario:
            id: str
            title: str
            query: str
        """),
        md("""
        ## The catalog

        Twenty scenarios spanning pedestrian interactions, traffic
        signals, signage, lane changes, roundabouts, and explicit
        causal queries.
        """),
        code("""
        SCENARIOS = [
            Scenario("1",  "Ego drives through crosswalk while pedestrian present",
                     "agent.type = ped and env.type = crosswalk and ego.action = drive"),
            Scenario("2",  "Ego yields/stops at crosswalk for a pedestrian",
                     "agent.type = ped and env.type = crosswalk and ego.action in (stop, yield, decel)"),
            Scenario("8",  "Pedestrian jaywalks; ego brakes / yields",
                     "agent(type = ped, action(jaywalk = true)) and ego.action in (stop, yield, decel)"),
            Scenario("9",  "Pedestrian crosses while ego is turning",
                     "agent.type = ped and env.type = crosswalk and ego.action in (turn_left, turn_right)"),
            Scenario("12", "Cyclist mid-block in ego's path; ego defensive",
                     "agent.type = cyclist and ego.action in (stop, yield, decel)"),
            Scenario("14", "Ego stops at a red traffic light (nominal)",
                     "light.color = red and ego.action = stop and env.type = intersection"),
            Scenario("16", "Signal blackout — ego treats as all-way stop",
                     "light.state = off and ego.action = stop and env.type = intersection"),
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
            Scenario("76", "Vehicle stopped in front of ego, ego nudges or changes lane",
                     "agent(type = vehicle, pos = front, action(type in (stop, not_move))) "
                     "and ego.action in (nudge, change_lane_left, change_lane_right)"),
            Scenario("lane-multi", "Multi-lane road, ego changes lane",
                     "env(type = road, lanes >= 2) and ego.action in (change_lane_left, change_lane_right)"),
            Scenario("officer-stop", "Officer signaling stop; ego stops",
                     "agent(type = officer, signaling = stop) and ego.action = stop"),
            Scenario("144", "Ego transits roundabout without yielding",
                     "env.type = roundabout and ego.action = drive"),
            Scenario("142", "Vehicle cuts in at roundabout exit; ego brakes",
                     "env.type = roundabout "
                     "and agent(type = vehicle, action(type in (change_lane, change_lane_left, change_lane_right))) "
                     "and ego.action in (decel, stop, nudge)"),
            Scenario("decel-because-ped", "Ego decelerates BECAUSE OF a pedestrian (causal edge)",
                     "ego.action = decel because_of agent.type = ped"),
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
        target = "agent(type = ped, action(jaywalk = true)) and ego.action in (stop, yield, decel)"
        clips = sorted(set(ds.find(target).clips()))

        print(f"{len(clips)} clips match: {target}\\n")
        pd.DataFrame({"clip_id": clips})
        """),
    ]
    save(cells, NOTEBOOKS_DIR / "04_scenario_catalog.ipynb")


def main() -> None:
    build_quickstart()
    build_dsl_tour()
    build_statistics()
    build_scenarios()


if __name__ == "__main__":
    main()
