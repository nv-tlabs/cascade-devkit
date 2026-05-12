# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`causal_ai_av.viz` — the DevKit's visualization surface.

The PR-1 skeleton exposes:

- `Segment` / `annotation_to_segments` — the Python port of the annotator's
  `TimelineSegment[]` shape (see `meta/10_visualization_api_plan.md`).
- `entity_color` — the annotator's `--color-entity-*` palette as hex strings.
- `render_frame` — headless single-frame decode → `PIL.Image`.

Later PRs add `render_timeline` (Plotly figure), `ClipPlayer`
(`FigureWidget` scrubbable widget), and the `MatchSet.visualize()` carousel.
On-frame overlays (boxes, keypoints) are deferred to v1.
"""

from causal_ai_av.viz.colors import entity_color
from causal_ai_av.viz.render import render_frame, render_timeline
from causal_ai_av.viz.segments import Segment, annotation_to_segments

__all__ = [
    "Segment",
    "annotation_to_segments",
    "entity_color",
    "render_frame",
    "render_timeline",
]
