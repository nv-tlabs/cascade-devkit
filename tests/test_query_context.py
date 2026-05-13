# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `query.context.context_at` / `CascadeDataset.context_for`."""

from __future__ import annotations

from pathlib import Path

import pytest

from cascade_av.dataset import CascadeDataset
from cascade_av.query import ContextWindow, Interval, context_at

CORPUS = Path("/home/horde/01_json_annotations")


@pytest.fixture
def patched_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub out the parent PhysicalAIAV interface — context tests don't need video."""
    from cascade_av import dataset as ds_mod

    class _StubParent:
        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr(ds_mod, "PhysicalAIAVDatasetInterface", _StubParent)


def test_context_at_overlaps_window(patched_parent: None) -> None:
    ds = CascadeDataset(CORPUS)
    # Pick a clip with a real "ped while ego decel" moment.
    matches = ds.find("agent.type = ped while ego.action = decel")
    assert matches, "expected at least one match across the corpus"

    m = matches.matches[0]
    ctx = ds.context_for(m)

    assert isinstance(ctx, ContextWindow)
    assert ctx.clip_id == m.clip_id
    assert ctx.interval == m.interval

    # Every returned agent's clipped visibility must lie within the window.
    for a in ctx.agents:
        assert a.visibility.start >= m.interval.start - 1e-6
        assert a.visibility.end <= m.interval.end + 1e-6

    # Every ego action's [start, end] must overlap the window.
    for ea in ctx.ego_actions:
        iv = Interval.from_strings(ea.start_timestamp, ea.end_timestamp)
        assert iv is not None and iv.overlaps(m.interval)


def test_context_at_directly_on_bundle(patched_parent: None) -> None:
    ds = CascadeDataset(CORPUS)
    # Reach into the dataset for a bundle; we want to exercise the
    # bundle-level entry point without a Match.
    clip_id = next(iter(ds._by_clip))
    _path, _batch, bundle = ds._by_clip[clip_id]

    window = Interval(0.0, bundle.video.duration_s)
    ctx = context_at(bundle, window)

    assert ctx.clip_id == clip_id
    # Whole-clip context: every agent with any visibility should appear.
    visible_agents = [a for a in bundle.annotation.agents
                      if a.visibility_start_timestamp and a.visibility_end_timestamp]
    assert len(ctx.agents) == len(visible_agents)
