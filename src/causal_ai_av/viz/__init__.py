# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`causal_ai_av.viz` — the DevKit's visualization surface.

Public exports:

- `Segment` / `annotation_to_segments` — the Python port of the annotator's
  `TimelineSegment[]` shape (see `meta/10_visualization_api_plan.md`).
- `entity_color` — the annotator's `--color-entity-*` palette as hex strings.
- `render_frame` — headless single-frame decode → `PIL.Image`.
- `render_timeline` — static Plotly Figure for a clip's annotation timeline.
- `ClipPlayer` — interactive Plotly + ipywidgets scrubber pairing a video
  frame with the timeline below it.

Still on the roadmap: `MatchSet.visualize()` carousel (PR-5), the
`Sequence.visualize()` wiring (PR-6), and on-frame overlays (deferred to v1).
"""

from causal_ai_av.viz.colors import entity_color
from causal_ai_av.viz.render import render_frame, render_timeline
from causal_ai_av.viz.segments import Segment, annotation_to_segments
from causal_ai_av.viz.widget import ClipPlayer

__all__ = [
    "ClipPlayer",
    "Segment",
    "annotation_to_segments",
    "entity_color",
    "render_frame",
    "render_timeline",
]
