# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sensor data beyond the canonical front camera.

The Physical AI AV Dataset ships every clip with a 7-camera rig, a
roof-mounted 360° LiDAR, 19 radars, egomotion, and calibration. The
DevKit's `download_clips(..., features=...)` lets you opt into any
subset.

This example:

1. Lists what's available per clip.
2. Runs a query and picks a match.
3. Shows how to fetch each sensor group via the shortcuts on
   `ds.features.{CAMERA,LIDAR,RADAR}.ALL`.
4. Actually pulls one extra camera for the matched clip and decodes
   a frame from it.

Heads-up: the parent downloads at *chunk* granularity (many clips per
chunk), so fetching even one extra sensor for one clip can pull a few
GB. By default this example **only prints the recipes** — set
`SENSOR_DEMO_DOWNLOAD=1` to actually pull the extra camera and decode
a frame.

    # catalog + recipes only (no downloads)
    CAUSAL_AV_DATASET_ROOT=/path/to/json_annotations \\
        uv run python examples/06_sensor_data.py

    # opt in to the actual download (~several GB)
    CAUSAL_AV_DATASET_ROOT=/path/to/json_annotations \\
        SENSOR_DEMO_DOWNLOAD=1 \\
        uv run python examples/06_sensor_data.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from causal_ai_av.dataset import CausalAVDataset


def main() -> None:
    try:
        dataset_root = Path(os.environ["CAUSAL_AV_DATASET_ROOT"])
    except KeyError:
        raise SystemExit("set CAUSAL_AV_DATASET_ROOT to the directory of JSON annotations")

    # Auto-confirm larger downloads — fine for a scripted example.
    ds = CausalAVDataset(dataset_root, confirm_download_threshold_gb=1000.0)

    # 1) What's available per clip
    print("# sensor catalog (per clip)")
    print(f"  cameras ({len(ds.features.CAMERA.ALL)}):")
    for c in sorted(ds.features.CAMERA.ALL):
        print(f"    {c}")
    print(f"  lidar ({len(ds.features.LIDAR.ALL)}): {sorted(ds.features.LIDAR.ALL)}")
    print(f"  radars ({len(ds.features.RADAR.ALL)}) — names like:")
    for r in sorted(ds.features.RADAR.ALL)[:5]:
        print(f"    {r}")
    print(f"    … and {len(ds.features.RADAR.ALL) - 5} more")
    print()

    # 2) Find a match worth inspecting
    query = "agent.type = ped while ego.action = decel"
    matches = ds.find(query)
    m = matches.matches[0]
    clip_id = m.clip_id
    print(f"# chosen match\n  clip {clip_id}  window {m.interval.start:.2f}–{m.interval.end:.2f}s\n")

    # 3) The download API — recipes for each sensor group
    print("# download recipes")
    print("  one sister camera:")
    print(f'    ds.download_clips(["{clip_id}"], features=["camera_rear_tele_30fov"])')
    print("  every camera:")
    print('    ds.download_clips([clip_id], features=ds.features.CAMERA.ALL)')
    print("  LiDAR:")
    print('    ds.download_clips([clip_id], features=ds.features.LIDAR.ALL)')
    print("  every radar:")
    print('    ds.download_clips([clip_id], features=ds.features.RADAR.ALL)')
    print("  everything:")
    print('    ds.download_clips([clip_id], features=ds.features.ALL)')
    print()

    # 4) Optionally pull one extra camera and decode a frame from it
    if os.environ.get("SENSOR_DEMO_DOWNLOAD") != "1":
        print("# skipping actual download (set SENSOR_DEMO_DOWNLOAD=1 to run it)")
        return

    extra_camera = "camera_rear_tele_30fov"
    print(f"# fetching {extra_camera} for one clip (chunk-granularity download)")
    ds.download_clips([clip_id], features=[extra_camera])

    seq = ds.get_sequence(clip_id)
    rear = seq.cameras[extra_camera]
    t_mid_us = int((m.interval.start + m.interval.end) / 2 * 1_000_000)
    images, _ = rear.decode_images_from_timestamps(np.array([t_mid_us], dtype=np.int64))
    frame = images[0]
    print(f"  decoded frame from {extra_camera}: shape={frame.shape}, dtype={frame.dtype}")


if __name__ == "__main__":
    main()
