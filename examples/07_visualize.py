# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless visualization — single frame, timeline, and paper figure.

The interactive `seq.visualize()` widget lives in a Jupyter cell; this
example exercises the same `cascade_av.viz` surface from a plain
Python script, no notebook required. Useful for batch report
generation, doc figures, and CI smoke tests.

Pipeline:

1. Load the corpus and pick a match worth looking at.
2. Prefetch the clip's video and egomotion from HuggingFace so
   `seq.video` resolves locally.
3. Decode a single frame at the match midpoint with `viz.render_frame`
   (returns a `PIL.Image`).
4. Build a static timeline figure with `viz.render_timeline` (returns a
   `plotly.graph_objects.Figure`), painting the match interval as the
   yellow highlight band.
5. Compose up to three selected video frames above filtered tracks with
   `viz.render_paper_figure`.
6. Save the artifacts under `examples/_out/visualize/`.

    # full pipeline — pulls one clip from HuggingFace, ~few hundred MB
    CASCADE_AV_DATASET_ROOT=/path/to/json_annotations \\
        uv run --extra viz python examples/07_visualize.py

The Plotly figures are saved as HTML by default. Pass
`VIZ_DEMO_KALEIDO=1` to also write PNGs via Plotly's `kaleido` exporter
(extra install: `uv add kaleido`).
"""

from __future__ import annotations

import os
from pathlib import Path

from cascade_av import viz
from cascade_av.dataset import CascadeDataset


# Output directory mirrors the example's number so multiple examples
# can each carve out their own corner of `_out/`.
OUT_DIR = Path(__file__).resolve().parent / "_out" / "visualize"


def main() -> None:
    try:
        dataset_root = Path(os.environ["CASCADE_AV_DATASET_ROOT"])
    except KeyError:
        raise SystemExit(
            "set CASCADE_AV_DATASET_ROOT to the directory of JSON annotations"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Auto-confirm chunk-granularity downloads — fine for a scripted
    # example. See `examples/06_sensor_data.py` for the heads-up about
    # chunk sizes.
    ds = CascadeDataset(dataset_root, confirm_download_threshold_gb=1000.0)

    # 1. Find a match worth looking at. `while` tightens each match's
    # interval to the intersection — exactly the moment both held —
    # which makes for a focused highlight band.
    query = "agent.type = ped while ego.action = decel"
    matches = ds.find(query)
    if not matches:
        raise SystemExit(f"no matches for: {query!r}")
    m = matches.matches[0]
    t0, t1 = m.interval.start, m.interval.end
    print("# chosen match")
    print(f"  clip {m.clip_id}")
    print(f"  window {t0:.2f}-{t1:.2f}s")
    print()

    # 2. Pull the clip's video + egomotion if not already cached.
    print("# fetching canonical-camera video + egomotion from HuggingFace")
    ds.download_clips([m.clip_id])

    # 3. Single frame at the match midpoint.
    seq = ds.get_sequence(m.clip_id)
    t_mid = (t0 + t1) / 2.0
    frame = viz.render_frame(seq, t_mid)
    frame_path = OUT_DIR / f"{m.clip_id}_t{t_mid:.2f}s.png"
    frame.save(frame_path)
    print(f"# wrote frame: {frame_path}  ({frame.size[0]}x{frame.size[1]})")

    # 4. Timeline figure with the match interval painted as a highlight.
    fig = viz.render_timeline(seq, highlight=(t0, t1))
    html_path = OUT_DIR / f"{m.clip_id}_timeline.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"# wrote timeline: {html_path}")

    # 5. Publication figure with up to three explicitly selected frames.
    # Select actual frame timestamps so the strict paper API never needs
    # to clamp a caller's request. The whole Environment group is hidden
    # here to demonstrate a track switch without assuming a corpus-specific
    # entity ID.
    video_timestamps = seq.video.timestamps
    frame_indexes = list(
        dict.fromkeys((0, len(video_timestamps) // 2, len(video_timestamps) - 1))
    )
    paper_timestamps = [
        float(video_timestamps[index]) / 1_000_000 for index in frame_indexes
    ]
    paper_fig = viz.render_paper_figure(
        seq,
        timestamps=paper_timestamps,
        highlight=(t0, t1),
        track_visibility={"env": False},
    )
    paper_html_path = OUT_DIR / f"{m.clip_id}_paper_figure.html"
    paper_fig.write_html(str(paper_html_path), include_plotlyjs="cdn")
    print(f"# wrote paper figure: {paper_html_path}")

    # 6. Optional PNG export via kaleido — extra opt-in.
    if os.environ.get("VIZ_DEMO_KALEIDO") == "1":
        png_path = OUT_DIR / f"{m.clip_id}_timeline.png"
        paper_png_path = OUT_DIR / f"{m.clip_id}_paper_figure.png"
        try:
            fig.write_image(str(png_path))
            print(f"# wrote timeline PNG: {png_path}")
            paper_fig.write_image(str(paper_png_path))
            print(f"# wrote paper figure PNG: {paper_png_path}")
        except Exception as exc:  # pragma: no cover — depends on optional dep
            print(f"# kaleido export skipped: {exc}")


if __name__ == "__main__":
    main()
