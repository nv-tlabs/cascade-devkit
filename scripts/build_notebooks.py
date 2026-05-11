"""Generate the `notebooks/` Jupyter files from this script.

Run after editing to regenerate:

    uv run --group notebooks python scripts/build_notebooks.py

Each notebook starts with a `pip` cell hint and assumes the corpus
lives at `/home/horde/01_json_annotations`.
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
# 01 — Quickstart
# ---------------------------------------------------------------------------

def build_quickstart() -> None:
    cells = [
        md("""
        # Quickstart — `causal_ai_av`

        End-to-end tour of the DevKit:

        1. Load the corpus.
        2. Inspect a single clip's annotation.
        3. Run a query in the DSL.
        4. Read the `MatchSet` back.
        """),
        code("""
        from pathlib import Path
        from causal_ai_av.dataset import CausalAVDataset

        CORPUS = Path("/home/horde/01_json_annotations")
        ds = CausalAVDataset(CORPUS)
        print(f"{len(ds)} clips")
        """),
        md("""
        ## 1. Inspect one clip's annotation

        `get_sequence` downloads the clip video lazily by default. We
        bypass it here and read the parsed annotation directly via the
        local IO module — quick and offline-friendly.
        """),
        code("""
        from causal_ai_av.io import load_dir
        bundles = load_dir(CORPUS)
        b = bundles[0]
        print("clip_id:", b.video.clip_id)
        print("schema:", b.schema_version)
        print("agents:", len(b.annotation.agents))
        print("ego actions:", len(b.annotation.ego_vehicle.actions))
        print("environments:", len(b.annotation.environments))
        """),
        md("## 2. Run a DSL query against the whole corpus"),
        code("""
        ds.count("agent.type = ped")
        """),
        code("""
        ds.count("agent.type = ped and env.type = crosswalk")
        """),
        md("""
        ## 3. `find()` returns the full `MatchSet`

        Every match is `(clip_id, entity, interval)`. The `.clips()` /
        `.entities()` / `.intervals()` projections pull each column.
        """),
        code("""
        matches = ds.find("agent.type = ped and env.type = crosswalk")
        print(f"{len(matches)} matches across {len(set(matches.clips()))} distinct clips")
        list(set(matches.clips()))[:3]
        """),
        md("""
        ## 4. A per-clip query

        The same DSL runs on a single `AnnotationBundle` via
        `find_on_bundle`. Returns a `MatchSet` scoped to that clip only.
        """),
        code("""
        from causal_ai_av.query import find_on_bundle
        single = find_on_bundle(b, "agent.type = vehicle")
        single.entities()[:3]
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

        Every operator in `meta/07_query_language.md` demonstrated against
        the real corpus. Each cell prints the clip count for that query.

        The DSL grammar:

        * `<entity>.<attr> <cmp> <value>` — basic predicate.
        * `<entity>(<expr>)` — same-entity grouping.
        * Boolean: `and`, `or`, `not`.
        * Temporal: `while`, `then(K)`.
        * Relational: `because_of`.
        * Scoping: `within W: E`.
        """),
        code("""
        from pathlib import Path
        from causal_ai_av.dataset import CausalAVDataset

        ds = CausalAVDataset(Path("/home/horde/01_json_annotations"))

        def show(q: str) -> None:
            print(f"{ds.count(q):>4}  {q}")
        """),
        md("## Attribute predicates"),
        code("""
        show("agent.type = ped")
        show("ego.action = decel")
        show("light.color = red")
        show("env.lanes >= 2")
        """),
        md("""
        ## Hierarchical aliases

        `vehicle`, `vru`, `intersection`, `turn`, `change_lane` are *parent*
        aliases — they expand to the union of their children.
        """),
        code("""
        show("agent.type = vehicle")
        show("agent.type = vru")
        show("agent.type = cyclist")
        show("env.type = intersection")
        """),
        md("## Set membership — `in (a, b, c)` is OR"),
        code("""
        show("ego.action in (stop, yield, decel)")
        show("agent.type in (ped, cyclist, animal)")
        """),
        md("""
        ## Same-entity coupling — `agent(...)`

        Inside an entity clause, the constraints apply to the **same** agent.
        Compare to free-floating predicates which may match different agents.
        """),
        code("""
        show("agent(type = vehicle, pos = front)")
        show("agent.type = vehicle and agent.pos = front")
        """),
        md("## Boolean operators"),
        code("""
        show("agent.type = ped and env.type = crosswalk")
        show("ego.action = stop or ego.action = yield")
        show("agent.type = ped and not env.type = crosswalk")
        """),
        md("""
        ## Action flags

        Action types in the corpus encode flags as parenthesized suffixes
        (`oxd:Walk (jaywalk)`). Flag attributes match either the schema
        flag field OR the suffix token.
        """),
        code("""
        show("agent(type = ped, action(jaywalk = true))")
        show("agent(action(erratic = true))")
        """),
        md("""
        ## Temporal — `while`

        Pairs of matches whose intervals intersect.
        """),
        code("""
        show("agent.type = ped while ego.action = decel")
        show("light.color = red while ego.action = stop")
        """),
        md("""
        ## Temporal — `then(K)`

        B starts during A or within K seconds after A ends. Default K=0
        means must touch or overlap.
        """),
        code("""
        show("light.color = yellow then(3) ego.action = stop")
        show("light.color = green then ego.action = drive")
        """),
        md("""
        ## Relational — `because_of`

        Follows the schema's `because_of` causal edge. Both halves are
        emitted as match tuples.
        """),
        code("""
        show("ego.action = decel because_of agent.type = ped")
        show("ego.action = drive because_of agent.type = ped")
        """),
        md("""
        ## Scoping — `within W: E`

        Restricts E's temporal window to W's intervals. Inside, `not E`
        means "E doesn't happen during W."
        """),
        code("""
        show("within light.color = red: not ego.action = stop")
        show("within env.type = crosswalk: agent.type = ped")
        """),
        md("## Annotator-tagged flags"),
        code("""
        show("light(color = yellow, ego_in_on_yellow = true)")
        show("light(color = yellow, could_have_cleared = true)")
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

        Count and group-by queries with matplotlib charts.
        """),
        code("""
        from pathlib import Path
        import matplotlib.pyplot as plt
        import pandas as pd

        from causal_ai_av.dataset import CausalAVDataset

        ds = CausalAVDataset(Path("/home/horde/01_json_annotations"))
        print(f"corpus has {len(ds)} clips")
        """),
        md("## Headline counts"),
        code("""
        rows = [
            ("pedestrian", ds.count("agent.type = ped")),
            ("vehicle", ds.count("agent.type = vehicle")),
            ("VRU", ds.count("agent.type = vru")),
            ("cyclist", ds.count("agent.type = cyclist")),
            ("stop sign", ds.count("obj.type = stop_sign")),
            ("traffic light (red)", ds.count("light.color = red")),
            ("crosswalk env", ds.count("env.type = crosswalk")),
            ("intersection env", ds.count("env.type = intersection")),
            ("roundabout env", ds.count("env.type = roundabout")),
            ("multi-lane road", ds.count("env(type = road, lanes >= 2)")),
            ("ego decel", ds.count("ego.action = decel")),
            ("ego stops at red", ds.count("light.color = red and ego.action = stop")),
        ]
        pd.DataFrame(rows, columns=["query", "clips"]).sort_values("clips", ascending=False)
        """),
        md("## Distribution of agent types"),
        code("""
        agent_dist = ds.group_by(
            "agent.type = vehicle or agent.type = vru or agent.type = animal",
            key="agent.type",
        )
        s = pd.Series(agent_dist).sort_values(ascending=True)
        ax = s.plot(kind="barh", figsize=(8, 6))
        ax.set_xlabel("clips")
        ax.set_title("Clips per agent type")
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
        s = pd.Series(ego_dist).sort_values(ascending=True)
        ax = s.plot(kind="barh", figsize=(8, 6))
        ax.set_xlabel("clips")
        ax.set_title("Clips per ego-action type")
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
        s = pd.Series(env_dist).sort_values(ascending=True)
        ax = s.plot(kind="barh", figsize=(8, 5))
        ax.set_xlabel("clips")
        ax.set_title("Clips per environment type")
        plt.tight_layout()
        plt.show()
        """),
        md("## Composite questions"),
        code("""
        n_red = ds.count("light.color = red")
        n_red_stop = ds.count("light.color = red and ego.action = stop")
        n_yellow_clear = ds.count("light(color = yellow, could_have_cleared = true)")
        n_jaywalk = ds.count("agent(type = ped, action(jaywalk = true))")

        composite = pd.DataFrame([
            ("ego stops at red", f"{n_red_stop}/{n_red}", f"{100*n_red_stop/max(n_red,1):.0f}%"),
            ("yellow ego could have cleared", n_yellow_clear, ""),
            ("jaywalking pedestrian", n_jaywalk, ""),
        ], columns=["question", "count", "ratio"])
        composite
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

        20 representative driving scenarios from
        `meta/scenarios_and_queries_reference.md` encoded as DSL queries.
        Each row shows the candidate-clip count and an example matching clip id.
        """),
        code("""
        from dataclasses import dataclass
        from pathlib import Path
        import pandas as pd

        from causal_ai_av.dataset import CausalAVDataset

        ds = CausalAVDataset(Path("/home/horde/01_json_annotations"))

        @dataclass(frozen=True)
        class Scenario:
            id: str
            title: str
            query: str
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
        len(SCENARIOS)
        """),
        md("## Run them all and tabulate"),
        code("""
        rows = []
        for s in SCENARIOS:
            try:
                matches = ds.find(s.query)
                clips = list(set(matches.clips()))
                rows.append({
                    "id": s.id,
                    "title": s.title,
                    "candidates": len(clips),
                    "example": clips[0] if clips else "",
                    "query": s.query,
                })
            except Exception as e:
                rows.append({"id": s.id, "title": s.title, "candidates": -1,
                             "example": f"ERROR: {e}", "query": s.query})

        df = pd.DataFrame(rows).sort_values("candidates", ascending=False)
        df[["id", "title", "candidates", "example"]]
        """),
        md("""
        ## Dig into one

        Pick a scenario, list every matching clip.
        """),
        code("""
        target = "agent(type = ped, action(jaywalk = true)) and ego.action in (stop, yield, decel)"
        clips = sorted(set(ds.find(target).clips()))
        print(f"{len(clips)} clips for: {target}")
        for c in clips[:10]:
            print(" ", c)
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
