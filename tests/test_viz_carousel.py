# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the `MatchSet.visualize()` carousel (PR-5).

Covers the public surface of `cascade_av.viz.carousel.
build_matchset_carousel` plus the thin `MatchSet.visualize()` delegator
on the query engine side. No real video decode — every `Sequence`
gets a `FakeVideoReader` from `tests/conftest.py`.
"""

from __future__ import annotations

import weakref
from typing import Any

import ipywidgets
import pytest

from cascade_av.dataset import Sequence
from cascade_av.query.engine import Match, MatchSet
from cascade_av.query.time import Interval
from cascade_av.spec import (
    AnnotationBundle,
    SilAvAnnotation,
    VideoMeta,
)
from cascade_av.viz import ClipPlayer, build_matchset_carousel
from tests.conftest import FakeVideoReader


# ---------------------------------------------------------------------------
# Fixtures — minimal bundle + a fake dataset that hands out Sequences
# with mocked video readers, no real decode.
# ---------------------------------------------------------------------------


def _make_minimal_bundle(clip_id: str, duration_s: float = 10.0) -> AnnotationBundle:
    """Smallest valid bundle — empty `SilAvAnnotation`, named clip.

    The carousel doesn't care about annotation content; it just walks
    matches and asks each `Sequence` for its `duration_s`. The timeline
    painter inside `ClipPlayer` handles an empty annotation cleanly.
    """
    return AnnotationBundle(
        schema_version="2.0.0",
        video=VideoMeta(clip_id=clip_id, duration_s=duration_s),
        annotation=SilAvAnnotation(),
    )


def _seq_with_fake_video(
    clip_id: str = "clip_a", duration_s: float = 10.0
) -> Sequence:
    bundle = _make_minimal_bundle(clip_id, duration_s)
    seq = Sequence.from_annotation(bundle)
    # `Sequence.video` is a property reading from `_cameras`; install a
    # dict that satisfies the contract without hitting a real parent.
    reader = FakeVideoReader()
    seq._cameras = {seq.annotation_camera: reader}  # type: ignore[assignment]
    return seq


class _FakeDataset:
    """Stand-in for `CascadeDataset` — just answers `get_sequence`.

    The carousel only needs `.get_sequence(clip_id)`; the rest of the
    `CascadeDataset` surface is irrelevant here. Holding the Sequence
    by clip_id ensures the same instance is returned on repeat lookups
    (matches `MatchSet.sequences()`'s dedup behavior).
    """

    def __init__(self, sequences: dict[str, Sequence]) -> None:
        self._seqs = sequences

    def get_sequence(self, clip_id: str) -> Sequence:
        return self._seqs[clip_id]


def _make_matchset(
    matches: list[Match], dataset: _FakeDataset | None
) -> MatchSet:
    """Build a `MatchSet` with the same weakref shape `find_on_dataset` uses."""
    ref = weakref.ref(dataset) if dataset is not None else None
    return MatchSet(tuple(matches), _dataset=ref)


# ---------------------------------------------------------------------------
# 1. Empty MatchSet → HTML widget, no players constructed.
# ---------------------------------------------------------------------------


def test_empty_matchset_returns_html_widget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Sentinel: spy on ClipPlayer construction to confirm nothing was
    # built. If the empty short-circuit fails the spy would fire.
    construction_calls = []
    real_init = ClipPlayer.__init__

    def _spy_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        construction_calls.append(1)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(ClipPlayer, "__init__", _spy_init)

    ms = _make_matchset([], dataset=None)
    out = build_matchset_carousel(ms)
    assert isinstance(out, ipywidgets.HTML)
    assert "No matches" in out.value
    assert construction_calls == []


# ---------------------------------------------------------------------------
# 2. Smoke (stack): 3 matches across 2 clips → VBox with 3 labeled children.
# ---------------------------------------------------------------------------


def test_stack_layout_smoke() -> None:
    seq_a = _seq_with_fake_video("clip_a")
    seq_b = _seq_with_fake_video("clip_b")
    ds = _FakeDataset({"clip_a": seq_a, "clip_b": seq_b})

    matches = [
        Match("clip_a", "agent_0", Interval(1.0, 2.0)),
        Match("clip_a", "agent_1", Interval(3.0, 4.0)),
        Match("clip_b", "agent_2", Interval(5.0, 6.0)),
    ]
    ms = _make_matchset(matches, dataset=ds)

    out = build_matchset_carousel(ms)
    assert isinstance(out, ipywidgets.VBox)
    assert len(out.children) == 3
    # Each child is a `VBox([HTML(label), player.widget])`.
    for child in out.children:
        assert isinstance(child, ipywidgets.VBox)
        assert len(child.children) == 2
        assert isinstance(child.children[0], ipywidgets.HTML)
        # Player widget is the VBox composite from ClipPlayer.
        assert isinstance(child.children[1], ipywidgets.VBox)


# ---------------------------------------------------------------------------
# 3. Smoke (grid): same matches, layout="grid", cols=2 → GridBox.
# ---------------------------------------------------------------------------


def test_grid_layout_smoke() -> None:
    seq_a = _seq_with_fake_video("clip_a")
    seq_b = _seq_with_fake_video("clip_b")
    ds = _FakeDataset({"clip_a": seq_a, "clip_b": seq_b})

    matches = [
        Match("clip_a", "agent_0", Interval(1.0, 2.0)),
        Match("clip_a", "agent_1", Interval(3.0, 4.0)),
        Match("clip_b", "agent_2", Interval(5.0, 6.0)),
    ]
    ms = _make_matchset(matches, dataset=ds)

    out = build_matchset_carousel(ms, layout="grid", cols=2)
    assert isinstance(out, ipywidgets.GridBox)
    assert out.layout.grid_template_columns == "repeat(2, 1fr)"
    assert len(out.children) == 3


# ---------------------------------------------------------------------------
# 4. Limit truncation: 10 matches, limit=5 → 5 players + notice header.
# ---------------------------------------------------------------------------


def test_limit_truncates_and_emits_notice() -> None:
    seq = _seq_with_fake_video("clip_a")
    ds = _FakeDataset({"clip_a": seq})

    # 10 matches all in the same clip; the carousel should render 5
    # and prepend a notice.
    matches = [
        Match("clip_a", f"agent_{i}", Interval(float(i), float(i) + 0.5))
        for i in range(10)
    ]
    ms = _make_matchset(matches, dataset=ds)

    out = build_matchset_carousel(ms, limit=5)
    # Truncation wraps the body in an outer VBox: [notice, body].
    assert isinstance(out, ipywidgets.VBox)
    assert len(out.children) == 2
    notice = out.children[0]
    assert isinstance(notice, ipywidgets.HTML)
    assert "5 of 10" in notice.value
    body = out.children[1]
    assert isinstance(body, ipywidgets.VBox)
    assert len(body.children) == 5


# ---------------------------------------------------------------------------
# 5. Pad widens the player window; highlight tracks the raw interval.
# ---------------------------------------------------------------------------


def test_pad_widens_window_and_highlight_uses_raw_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seq = _seq_with_fake_video("clip_a", duration_s=10.0)
    ds = _FakeDataset({"clip_a": seq})

    matches = [Match("clip_a", "agent_0", Interval(3.0, 4.0))]
    ms = _make_matchset(matches, dataset=ds)

    captured: list[dict] = []
    real_init = ClipPlayer.__init__

    def _capture_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(dict(kwargs))
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(ClipPlayer, "__init__", _capture_init)

    build_matchset_carousel(ms, pad=2.0)

    assert len(captured) == 1
    kwargs = captured[0]
    assert kwargs["t_start"] == 1.0  # max(0, 3.0 - 2.0)
    assert kwargs["t_end"] == 6.0  # min(10.0, 4.0 + 2.0)
    assert kwargs["highlight"] == (3.0, 4.0)


def test_pad_clamps_to_clip_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`max(0, t0 - pad)` and `min(duration, t1 + pad)` both clamp."""
    seq = _seq_with_fake_video("clip_a", duration_s=5.0)
    ds = _FakeDataset({"clip_a": seq})

    # Match right at the start and end of the clip; large pad would
    # overshoot both bounds.
    matches = [Match("clip_a", "agent_0", Interval(0.5, 4.5))]
    ms = _make_matchset(matches, dataset=ds)

    captured: list[dict] = []
    real_init = ClipPlayer.__init__

    def _capture_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(dict(kwargs))
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(ClipPlayer, "__init__", _capture_init)

    build_matchset_carousel(ms, pad=10.0)

    assert captured[0]["t_start"] == 0.0
    assert captured[0]["t_end"] == 5.0


# ---------------------------------------------------------------------------
# 6. No dataset attached → sequences() raises, error propagates.
# ---------------------------------------------------------------------------


def test_no_dataset_raises_runtime_error() -> None:
    matches = [Match("clip_a", "agent_0", Interval(1.0, 2.0))]
    ms = _make_matchset(matches, dataset=None)
    # The error originates in MatchSet.sequences(); the carousel does
    # not catch it.
    with pytest.raises(RuntimeError, match="no dataset back-reference"):
        build_matchset_carousel(ms)


# ---------------------------------------------------------------------------
# 7. Two matches in the same clip get two distinct ClipPlayer instances.
# ---------------------------------------------------------------------------


def test_same_clip_matches_each_get_own_player(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seq = _seq_with_fake_video("clip_a")
    ds = _FakeDataset({"clip_a": seq})

    matches = [
        Match("clip_a", "agent_0", Interval(1.0, 2.0)),
        Match("clip_a", "agent_1", Interval(5.0, 6.0)),
    ]
    ms = _make_matchset(matches, dataset=ds)

    captured: list[dict] = []
    real_init = ClipPlayer.__init__

    def _capture_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(dict(kwargs))
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(ClipPlayer, "__init__", _capture_init)

    out = build_matchset_carousel(ms)

    # Two players built — one per match, even though both share clip_a.
    assert len(captured) == 2
    # Distinct highlight windows.
    assert captured[0]["highlight"] == (1.0, 2.0)
    assert captured[1]["highlight"] == (5.0, 6.0)
    # The VBox container has both as labeled children.
    assert isinstance(out, ipywidgets.VBox)
    assert len(out.children) == 2


# ---------------------------------------------------------------------------
# 8. MatchSet.visualize delegates to build_matchset_carousel with kwargs.
# ---------------------------------------------------------------------------


def test_families_kwarg_forwarded_to_each_clip_player(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`build_matchset_carousel(ms, families=[...])` forwards the
    whitelist to every per-match `ClipPlayer`. Monkeypatching the
    `ClipPlayer` constructor lets us capture the kwargs each
    instantiation receives — every call should see the same family
    list."""
    captured: list[dict[str, Any]] = []

    class _SpyClipPlayer:
        def __init__(self, _seq: Any, **kwargs: Any) -> None:  # noqa: D401
            captured.append(kwargs)
            self.widget = ipywidgets.VBox([])  # minimum interface

    monkeypatch.setattr(
        "cascade_av.viz.carousel.ClipPlayer", _SpyClipPlayer
    )

    seq_a = _seq_with_fake_video("clip_a")
    ds = _FakeDataset({"clip_a": seq_a})
    matches = [
        Match("clip_a", "agent_0", Interval(1.0, 2.0)),
        Match("clip_a", "agent_1", Interval(3.0, 4.0)),
    ]
    ms = _make_matchset(matches, dataset=ds)

    build_matchset_carousel(ms, families=["action", "condition"])

    assert len(captured) == 2, (
        f"expected one ClipPlayer per match, got {len(captured)}"
    )
    for kw in captured:
        assert kw.get("families") == ["action", "condition"], (
            f"families kwarg not forwarded; got {kw.get('families')!r}"
        )


def test_matchset_visualize_delegates_with_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    seen: list[tuple] = []

    def _fake_builder(ms, **kwargs):  # type: ignore[no-untyped-def]
        seen.append((ms, kwargs))
        return sentinel

    monkeypatch.setattr(
        "cascade_av.viz.carousel.build_matchset_carousel", _fake_builder
    )

    matches = [Match("clip_a", "agent_0", Interval(1.0, 2.0))]
    ms = _make_matchset(matches, dataset=None)

    result = ms.visualize(layout="grid", cols=4, limit=2)
    assert result is sentinel
    assert len(seen) == 1
    passed_ms, passed_kwargs = seen[0]
    assert passed_ms is ms
    assert passed_kwargs == {"layout": "grid", "cols": 4, "limit": 2}


# ---------------------------------------------------------------------------
# 9. Pad-clamp guard at the end-of-clip boundary — Bug B from the
# viz-e2e smoke session. A match whose interval ends at exactly
# `seq.duration_s` with `pad=0` previously collapsed to
# `t_end == t_start == duration` and tripped `ClipPlayer`'s
# `t_end > t_start` invariant. The fix expands `t_start` downward at
# the end-of-clip boundary.
# ---------------------------------------------------------------------------


def test_pad_clamp_at_end_of_clip_does_not_crash() -> None:
    duration = 10.0
    seq = _seq_with_fake_video("clip_a", duration_s=duration)
    ds = _FakeDataset({"clip_a": seq})

    # Match interval butts up against `duration_s` and `pad=0`. Before
    # the fix this produced `t_start == t_end == duration`.
    matches = [Match("clip_a", "agent_0", Interval(duration, duration))]
    ms = _make_matchset(matches, dataset=ds)

    out = build_matchset_carousel(ms, pad=0.0)
    assert isinstance(out, ipywidgets.VBox)
    assert len(out.children) == 1
    # Inner VBox is `[label, player.widget]`.
    inner = out.children[0]
    # The player widget itself is the second child of the inner VBox;
    # the ClipPlayer lives in `player.widget` which is the same VBox.
    # We can verify `t_end > t_start` by reaching back into the
    # constructed ClipPlayer through a capture below — but the simpler
    # assertion is that the function returned without raising and the
    # constructed widget tree is well-formed.
    assert isinstance(inner, ipywidgets.VBox)
    assert len(inner.children) == 2


def test_pad_clamp_at_end_of_clip_keeps_window_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The constructed `ClipPlayer` has `t_end > t_start` (Bug B)."""
    duration = 10.0
    seq = _seq_with_fake_video("clip_a", duration_s=duration)
    ds = _FakeDataset({"clip_a": seq})

    matches = [Match("clip_a", "agent_0", Interval(duration, duration))]
    ms = _make_matchset(matches, dataset=ds)

    captured: list[dict] = []
    real_init = ClipPlayer.__init__

    def _capture_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(dict(kwargs))
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(ClipPlayer, "__init__", _capture_init)

    build_matchset_carousel(ms, pad=0.0, fps=8.0)

    assert len(captured) == 1
    t_start = captured[0]["t_start"]
    t_end = captured[0]["t_end"]
    assert t_end > t_start
    # `t_end` stayed pinned to duration; `t_start` backed off by one
    # tick (1/fps == 0.125s at fps=8).
    assert t_end == pytest.approx(duration)
    assert t_start == pytest.approx(duration - 1.0 / 8.0)


# ---------------------------------------------------------------------------
# 10. None-interval match → full clip window. Previously the carousel
# resolved to `(0.0, 0.0)` for matches whose interval was None,
# collapsing the player. Per the docstring, the player should cover
# the whole clip in that case.
# ---------------------------------------------------------------------------


def test_none_interval_match_uses_full_clip_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duration = 7.5
    seq = _seq_with_fake_video("clip_a", duration_s=duration)
    ds = _FakeDataset({"clip_a": seq})

    matches = [Match("clip_a", "agent_0", interval=None)]
    ms = _make_matchset(matches, dataset=ds)

    captured: list[dict] = []
    real_init = ClipPlayer.__init__

    def _capture_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(dict(kwargs))
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(ClipPlayer, "__init__", _capture_init)

    build_matchset_carousel(ms, pad=0.0)

    assert len(captured) == 1
    assert captured[0]["t_start"] == pytest.approx(0.0)
    assert captured[0]["t_end"] == pytest.approx(duration)


# ---------------------------------------------------------------------------
# 11. Invalid `layout` string → ValueError (instead of silent fall-through
# to VBox). The `Literal["stack", "grid"]` annotation is advisory at
# runtime; a typo'd string would previously render a stack with no
# warning.
# ---------------------------------------------------------------------------


def test_invalid_layout_raises_value_error() -> None:
    seq = _seq_with_fake_video("clip_a")
    ds = _FakeDataset({"clip_a": seq})

    matches = [Match("clip_a", "agent_0", Interval(1.0, 2.0))]
    ms = _make_matchset(matches, dataset=ds)

    with pytest.raises(ValueError, match="layout must be 'stack' or 'grid'"):
        build_matchset_carousel(ms, layout="carousel")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Per-match label uses short entity summary, not Pydantic repr
# ---------------------------------------------------------------------------


def _collect_label_html(box: ipywidgets.Widget) -> list[str]:
    """Walk a VBox/GridBox and pull out every ipywidgets.HTML's value.

    The carousel emits one `[HTML(label), VBox(player)]` row per match;
    we extract the label HTML strings for assertion.
    """
    out: list[str] = []
    for child in getattr(box, "children", ()) or ():
        if isinstance(child, ipywidgets.HTML):
            out.append(child.value)
        else:
            out.extend(_collect_label_html(child))
    return out


def test_match_label_uses_short_entity_summary_not_pydantic_repr() -> None:
    """Per-match label HTML must be a tight `ClassName(id, kind)`
    string — NOT Pydantic's default repr (which dumps every field
    and floods the Jupyter cell with thousands of lines per match
    on a 3-match carousel).
    """
    from cascade_av.spec import Agent

    seq = _seq_with_fake_video("clip_a", duration_s=10.0)
    ds = _FakeDataset({"clip_a": seq})
    agent = Agent(
        id="Agent3",
        type="oxd:Pedestrian",
        visibility_start_timestamp="0:0.0",
        visibility_end_timestamp="0:5.0",
    )
    match = Match("clip_a", agent, Interval(1.0, 2.0))
    ms = _make_matchset([match], dataset=ds)
    box = build_matchset_carousel(ms)
    labels = _collect_label_html(box)
    assert labels, "expected at least one label HTML in the carousel"
    label = labels[0]
    # Tight format: ClassName(id, kind)
    assert "Agent(Agent3, oxd:Pedestrian)" in label, (
        f"expected short label, got {label!r}"
    )
    # Negative pin: the Pydantic repr fields must NOT leak in.
    for forbidden in (
        "visibility_start_timestamp",
        "visibility_end_timestamp",
        "actions=[",
        "properties=[",
        "bounding_boxes=",
    ):
        assert forbidden not in label, (
            f"Pydantic repr field {forbidden!r} leaked into label: "
            f"{label[:300]!r}"
        )


def test_short_entity_label_per_type() -> None:
    """Spot-check the short-label format across the common entity
    types the carousel might receive."""
    from cascade_av.spec import Agent, AgentAction, Condition, Environment

    from cascade_av.viz.carousel import _short_entity_label

    a = Agent(
        id="Agent3",
        type="oxd:Pedestrian",
        visibility_start_timestamp="0:0.0",
        visibility_end_timestamp="0:5.0",
    )
    assert _short_entity_label(a) == "Agent(Agent3, oxd:Pedestrian)"

    aa = AgentAction(
        id="AgentAction4",
        action_type="oxd:Walk",
        start_timestamp="0:0.0",
        end_timestamp="0:5.0",
    )
    assert _short_entity_label(aa) == "AgentAction(AgentAction4, oxd:Walk)"

    env = Environment(
        id="Env3",
        type="fst:Road",
        start_timestamp="0:0.0",
        end_timestamp="0:10.0",
    )
    assert _short_entity_label(env) == "Environment(Env3, fst:Road)"

    cond = Condition(
        id="Cond1",
        env_id="Env3",
        type=["Construction Zone"],
        start_timestamp="0:0.0",
        end_timestamp="0:5.0",
    )
    # `Condition.type` is `list[str]`; helper picks the first item.
    assert _short_entity_label(cond) == "Condition(Cond1, Construction Zone)"

    # Bare string (legacy) still produces something usable.
    assert _short_entity_label("agent_0") == "str"
    # None case.
    assert _short_entity_label(None) == "&mdash;"


# ---------------------------------------------------------------------------
# 12. unique_clips=True — dedup matches by clip_id so the carousel renders
# one player per distinct clip, with the union of intervals as the
# playback window and a "N matches" label when the clip has more than one.
# Locks the fix for the per-match default rendering the same clip three
# times when a single clip contained three matches.
# ---------------------------------------------------------------------------


def test_unique_clips_dedups_by_clip_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seq_a = _seq_with_fake_video("clip_a")
    seq_b = _seq_with_fake_video("clip_b")
    ds = _FakeDataset({"clip_a": seq_a, "clip_b": seq_b})

    # Three matches in clip_a, one in clip_b. Default behaviour would
    # render four players; `unique_clips=True` collapses to two.
    matches = [
        Match("clip_a", "agent_0", Interval(1.0, 2.0)),
        Match("clip_a", "agent_1", Interval(3.0, 4.0)),
        Match("clip_a", "agent_2", Interval(5.0, 6.0)),
        Match("clip_b", "agent_0", Interval(2.0, 3.0)),
    ]
    ms = _make_matchset(matches, dataset=ds)

    captured: list[dict[str, Any]] = []

    class _SpyClipPlayer:
        def __init__(self, _seq: Any, **kwargs: Any) -> None:
            captured.append(kwargs)
            self.widget = ipywidgets.VBox([])

    monkeypatch.setattr(
        "cascade_av.viz.carousel.ClipPlayer", _SpyClipPlayer
    )

    out = build_matchset_carousel(ms, unique_clips=True)

    assert len(captured) == 2, (
        f"expected one player per unique clip, got {len(captured)}"
    )
    assert isinstance(out, ipywidgets.VBox)
    assert len(out.children) == 2


def test_unique_clips_unions_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seq = _seq_with_fake_video("clip_a", duration_s=20.0)
    ds = _FakeDataset({"clip_a": seq})

    # Two matches far apart; union should span 1.0 → 11.0.
    matches = [
        Match("clip_a", "agent_0", Interval(1.0, 2.0)),
        Match("clip_a", "agent_1", Interval(10.0, 11.0)),
    ]
    ms = _make_matchset(matches, dataset=ds)

    captured: list[dict[str, Any]] = []

    class _SpyClipPlayer:
        def __init__(self, _seq: Any, **kwargs: Any) -> None:
            captured.append(kwargs)
            self.widget = ipywidgets.VBox([])

    monkeypatch.setattr(
        "cascade_av.viz.carousel.ClipPlayer", _SpyClipPlayer
    )

    build_matchset_carousel(ms, pad=0.5, unique_clips=True)

    assert len(captured) == 1
    kw = captured[0]
    # Highlight = union of intervals (no pad).
    assert kw["highlight"] == (1.0, 11.0)
    # Playback window = union + pad on each side, clamped to clip.
    assert kw["t_start"] == 0.5  # max(0, 1.0 - 0.5)
    assert kw["t_end"] == 11.5  # min(20.0, 11.0 + 0.5)


def test_unique_clips_limit_counts_clips_not_matches() -> None:
    """`limit=2` with `unique_clips=True` caps at 2 distinct clips even
    when the matchset contains many more matches. Without the flag the
    same matchset would have rendered the first 2 matches (both in
    clip_a), giving the same clip twice. Locks the surprise from
    notebook 06 where `matches.visualize(limit=3, ...)` showed the same
    clip three times."""
    seqs = {f"clip_{i}": _seq_with_fake_video(f"clip_{i}") for i in range(5)}
    ds = _FakeDataset(seqs)

    # Five matches in clip_0 + one each in clip_1..clip_4. Default
    # mode would render five clip_0 players; unique_clips caps at
    # distinct clips.
    matches = [Match("clip_0", f"a_{i}", Interval(float(i), float(i) + 0.5))
               for i in range(5)]
    matches += [Match(f"clip_{i}", "a_0", Interval(1.0, 2.0))
                for i in range(1, 5)]
    ms = _make_matchset(matches, dataset=ds)

    out = build_matchset_carousel(ms, limit=2, unique_clips=True)

    # Truncation wraps the body in [notice, body].
    assert isinstance(out, ipywidgets.VBox)
    assert len(out.children) == 2
    notice = out.children[0]
    assert isinstance(notice, ipywidgets.HTML)
    # Notice copy switches "matches" → "clips" so the user knows which
    # axis the limit was counted along.
    assert "of 5 clips" in notice.value, notice.value
    body = out.children[1]
    assert isinstance(body, ipywidgets.VBox)
    assert len(body.children) == 2


def test_unique_clips_label_shows_match_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seq = _seq_with_fake_video("clip_a")
    ds = _FakeDataset({"clip_a": seq})

    matches = [
        Match("clip_a", "agent_0", Interval(1.0, 2.0)),
        Match("clip_a", "agent_1", Interval(3.0, 4.0)),
        Match("clip_a", "agent_2", Interval(5.0, 6.0)),
    ]
    ms = _make_matchset(matches, dataset=ds)

    out = build_matchset_carousel(ms, unique_clips=True)

    # Single child (one clip), with label HTML showing "3 matches".
    assert isinstance(out, ipywidgets.VBox)
    assert len(out.children) == 1
    cell = out.children[0]
    label_html = cell.children[0]
    assert isinstance(label_html, ipywidgets.HTML)
    assert "3 matches" in label_html.value, label_html.value
    # Entity-led format is suppressed when the group has > 1 match.
    assert "agent_0" not in label_html.value


def test_unique_clips_single_match_clip_keeps_entity_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clip with exactly one match still gets the entity-led label
    even with `unique_clips=True`. The "N matches" copy only kicks in
    when the dedup actually collapsed something."""
    seq_a = _seq_with_fake_video("clip_a")
    seq_b = _seq_with_fake_video("clip_b")
    ds = _FakeDataset({"clip_a": seq_a, "clip_b": seq_b})

    # One match per clip — no collapse.
    matches = [
        Match("clip_a", "agent_42", Interval(1.0, 2.0)),
        Match("clip_b", "agent_99", Interval(3.0, 4.0)),
    ]
    ms = _make_matchset(matches, dataset=ds)

    out = build_matchset_carousel(ms, unique_clips=True)

    assert isinstance(out, ipywidgets.VBox)
    assert len(out.children) == 2
    for cell in out.children:
        label_html = cell.children[0]
        # No "N matches" wording for single-match clips.
        assert "matches" not in label_html.value, label_html.value
