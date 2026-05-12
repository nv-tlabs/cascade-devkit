# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smallest possible end-to-end usage.

Load the local corpus, run one query, print the count.

    CAUSAL_AV_DATASET_ROOT=/path/to/json_annotations \\
        uv run python examples/01_quickstart.py
"""

from __future__ import annotations

import os
from pathlib import Path

from causal_ai_av.dataset import CausalAVDataset


def main() -> None:
    try:
        dataset_root = Path(os.environ["CAUSAL_AV_DATASET_ROOT"])
    except KeyError:
        raise SystemExit("set CAUSAL_AV_DATASET_ROOT to the directory of JSON annotations")

    ds = CausalAVDataset(dataset_root)

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
