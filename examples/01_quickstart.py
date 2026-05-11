"""Smallest possible end-to-end usage.

Load the local corpus, run one query, print the count.

    uv run python examples/01_quickstart.py
"""

from __future__ import annotations

from pathlib import Path

from causal_ai_av.dataset import CausalAVDataset

CORPUS = Path("/home/horde/01_json_annotations")


def main() -> None:
    ds = CausalAVDataset(CORPUS)

    # A query is a plain string in the query DSL.
    n = ds.count("agent.type = ped")
    print(f"clips with at least one pedestrian: {n}")

    # `find()` returns the full MatchSet — clips, entities, intervals.
    matches = ds.find("agent.type = ped and env.type = crosswalk")
    print(f"pedestrian × crosswalk: {len(matches)} matches across {len(set(matches.clips()))} clips")
    for clip_id in list(set(matches.clips()))[:3]:
        print(f"  e.g. {clip_id}")


if __name__ == "__main__":
    main()
