# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`build_matchset_carousel` — fan out a `MatchSet` into one player per match.

PR-5 of the visualization API rollout. The carousel is the consumer-facing
entry point for `matches.visualize()` — it walks the `MatchSet`, resolves
each match to a `Sequence` via the dataset back-reference, and constructs
a `ClipPlayer` whose playback window is the match interval padded by
`pad` seconds and clamped to the clip duration. The yellow highlight
band on the timeline tracks the *raw* match interval (not the padded
window) so the user can see exactly where the matched event sits inside
the surrounding context.

Layouts:

- `stack` — vertical `ipywidgets.VBox` of `[label, player]` boxes.
- `grid`  — `ipywidgets.GridBox` with `repeat(cols, 1fr)` columns.

The carousel is a free function rather than a method on `MatchSet`
because `MatchSet` lives in `causal_ai_av.query.engine` and a top-level
import of `viz` from `query/engine.py` would create a cycle
(`query → viz → dataset → query`). `MatchSet.visualize()` is a two-line
delegator that imports this function lazily inside the method body.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from causal_ai_av.viz.widget import ClipPlayer

if TYPE_CHECKING:  # pragma: no cover — typing only
    import ipywidgets

    from causal_ai_av.query.engine import MatchSet


def build_matchset_carousel(
    matchset: "MatchSet",
    *,
    layout: Literal["stack", "grid"] = "stack",
    cols: int = 3,
    limit: int = 8,
    pad: float = 1.0,
    fps: float = 8.0,
    arrows: dict[str, bool] | None = None,
    families: list[str] | None = None,
    unique_clips: bool = False,
) -> "ipywidgets.Widget":
    """Build a carousel of `ClipPlayer` widgets.

    By default the carousel renders **one player per match** — if the
    same clip backs three matches, you see three players with that
    clip's video. Set `unique_clips=True` to render **one player per
    unique clip** instead, with the player's playback window covering
    the union of all match intervals in that clip and its label showing
    the match count.

    Args:
        matchset: a `MatchSet` whose `.sequences()` resolves the dataset
            back-reference. A bundle-level `MatchSet` (produced by
            `find_on_bundle` without `dataset=`) raises `RuntimeError`
            from `.sequences()`; the error propagates uncaught.
        layout: `"stack"` for a vertical `VBox`, `"grid"` for a
            `GridBox` with `cols` columns.
        cols: number of columns when `layout="grid"`. Ignored otherwise.
        limit: cap the number of players. With the default
            `unique_clips=False`, `limit` caps the number of matches
            rendered. With `unique_clips=True`, `limit` caps the number
            of distinct clips rendered. In both cases a small notice is
            prepended if the cap clipped the result.
        pad: seconds of padding to apply on either side of each match's
            interval when building the player's playback window. Clamped
            to `[0, sequence.duration_s]`. With `unique_clips=True` the
            pad is applied around the union of all match intervals in
            the clip, not each individually.
        fps: scrub rate forwarded to each `ClipPlayer`.
        arrows: per-family arrow-on/off toggles forwarded to each
            `ClipPlayer`. See `_paint_timeline_onto` for the keys.
        families: optional whitelist of family-leaf names (e.g.
            `["action"]`, `["condition", "containment"]`) forwarded to
            each `ClipPlayer`. Parent entity headers auto-render for
            entities whose sub-rows survive; entities with no
            surviving sub-rows drop completely. `None` = all
            families. See `render_timeline` for the recognized leaf
            names.
        unique_clips: when `True`, dedup matches by `clip_id` — exactly
            one player per distinct clip, with playback window
            `[min(t0)-pad, max(t1)+pad]` covering every match in the
            clip and the highlight band spanning the union. The
            per-player label shows `"N matches"` when the clip has more
            than one. Use this when you want to *survey distinct clips*
            (e.g., `matches.visualize(limit=3, unique_clips=True)` for
            three distinct clips); leave the default `False` when you
            want to see every match individually.

    Returns:
        an `ipywidgets.Widget` — `HTML` if `matchset` is empty,
        `VBox` for `layout="stack"`, or `GridBox` for `layout="grid"`.
        The `VBox` / `GridBox` contains one labeled child per
        rendered unit (match or clip), plus a leading truncation notice
        if `limit` clipped the result.

    Raises:
        RuntimeError: if `matchset` has no dataset back-reference
            (e.g., produced by `find_on_bundle` without `dataset=`).
            The error is raised by `MatchSet.sequences()` and
            propagates uncaught — a bundle-level MatchSet simply
            cannot be visualized without knowing how to fetch video.
    """
    import ipywidgets  # noqa: PLC0415 — lazy import keeps the dep optional

    # Empty MatchSet: short-circuit before touching `.sequences()`. The
    # bundle-level "no dataset back-reference" RuntimeError only fires
    # when the iterator is actually advanced, but a totally empty set
    # has nothing to render either way; an HTML placeholder is friendlier
    # than an empty box.
    if len(matchset) == 0:
        return ipywidgets.HTML("<em>No matches.</em>")

    # A "unit" is what becomes one player: a clip_id, the matches that
    # contribute to it, and the resolved Sequence. With `unique_clips`
    # off the unit is (clip_id, [match], seq) — one per match. With it
    # on we group by clip_id and union the intervals so each clip shows
    # up exactly once. `matchset.sequences()` raises `RuntimeError` if
    # `_dataset` is None; we deliberately do not catch — the carousel
    # can't be built without a way to fetch each clip's video.
    if unique_clips:
        groups: dict[str, tuple[list[Any], Any]] = {}
        for match, seq in matchset.sequences():
            entry = groups.get(match.clip_id)
            if entry is None:
                groups[match.clip_id] = ([match], seq)
            else:
                entry[0].append(match)
        units: list[tuple[str, list[Any], Any]] = [
            (clip_id, ms, seq) for clip_id, (ms, seq) in groups.items()
        ]
        total = len(units)
        unit_kind = "clips"
    else:
        units = [
            (match.clip_id, [match], seq)
            for match, seq in matchset.sequences()
        ]
        total = len(units)
        unit_kind = "matches"

    truncated = total > limit

    children: list[Any] = []
    for i, (clip_id, group, seq) in enumerate(units):
        if i >= limit:
            break

        # Union the per-match intervals — for `unique_clips=False` the
        # group has exactly one element so the union collapses to that
        # single interval (no behaviour change for the default path).
        intervals = [_interval_seconds(m, seq) for m in group]
        t0_raw = min(t0 for t0, _ in intervals)
        t1_raw = max(t1 for _, t1 in intervals)

        duration = float(seq.duration_s) if seq.duration_s else 0.0
        t_start = max(0.0, t0_raw - pad)
        t_end = min(duration, t1_raw + pad)
        # `ClipPlayer` requires `t_end > t_start`; if pad clamping
        # collapsed the window, expand to at least one fps-tick. The
        # original PR-25 implementation always pushed `t_end` upward,
        # which silently failed at the end-of-clip boundary
        # (`t_start == duration` clamped the expansion right back).
        # Now we push `t_start` downward at the end-of-clip boundary
        # and `t_end` upward otherwise. If the clip is shorter than a
        # single fps tick, neither direction has room — surface a
        # clear `ValueError` instead of constructing a degenerate
        # player.
        if t_end <= t_start:
            tick = 1.0 / max(fps, 1.0)
            if duration < tick:
                raise ValueError(
                    f"clip duration ({duration:.6f}s) is shorter than one "
                    f"frame at fps={fps}; cannot build a non-degenerate "
                    f"playback window for match {clip_id}"
                )
            if t_end >= duration:
                # End-of-clip boundary: back `t_start` off by one tick.
                t_start = max(0.0, t_start - tick)
            else:
                t_end = t_start + tick

        player = ClipPlayer(
            seq,
            t_start=t_start,
            t_end=t_end,
            fps=fps,
            highlight=(t0_raw, t1_raw),
            arrows=arrows,
            families=families,
        )

        # Single match in the group → original entity-led label. Multi
        # match group (only possible with `unique_clips=True`) → swap
        # the entity for a count so the reader knows why the highlight
        # spans more than one match interval.
        if len(group) == 1:
            label = (
                f"<b>{clip_id}</b> &middot; "
                f"{_short_entity_label(group[0].entity)} &middot; "
                f"[{t0_raw:.2f}, {t1_raw:.2f}]s"
            )
        else:
            label = (
                f"<b>{clip_id}</b> &middot; "
                f"{len(group)} matches &middot; "
                f"[{t0_raw:.2f}, {t1_raw:.2f}]s"
            )
        children.append(
            ipywidgets.VBox([ipywidgets.HTML(label), player.widget])
        )

    if layout == "grid":
        body: Any = ipywidgets.GridBox(
            children=children,
            layout=ipywidgets.Layout(
                grid_template_columns=f"repeat({cols}, 1fr)"
            ),
        )
    elif layout == "stack":
        body = ipywidgets.VBox(children=children)
    else:
        # The `Literal["stack", "grid"]` type hint is advisory at
        # runtime only; a typo'd string would previously fall through
        # to a `VBox` silently. Raise instead so the caller knows
        # their `layout=` argument was ignored.
        raise ValueError(
            f"layout must be 'stack' or 'grid', got {layout!r}"
        )

    if truncated:
        notice = ipywidgets.HTML(
            f"<em>Showing {limit} of {total} {unit_kind}.</em>"
        )
        return ipywidgets.VBox([notice, body])
    return body


def _short_entity_label(entity: Any) -> str:
    """One-line summary of the matched entity for the carousel label.

    `Match.entity` is typed `object` — at runtime it's one of the
    spec Pydantic models (`Agent`, `EgoVehicle`, `Environment`,
    `TrafficObject`, `TrafficLight`) or one of their sub-items
    (`AgentAction`, `EgoAction`, `Containment`, `Condition`,
    `LightStates`, `AgentProperty`, etc.). Pydantic's default
    `__str__` / `__repr__` dumps every field, which floods the
    Jupyter cell output for a 3-match carousel with thousands of
    lines of state. Here we walk the common attribute set
    (`id`, then a type-disambiguating field) and return a tight
    `ClassName(id, kind)` string. Falls back to the bare class
    name when no recognized attributes are present, so an unknown
    entity type still renders something readable.
    """
    if entity is None:
        return "&mdash;"
    cls_name = type(entity).__name__
    # `id` is the spec's stable handle; almost every entity has one.
    entity_id = getattr(entity, "id", None)
    # The disambiguating "what kind" attribute varies by class:
    #   Agent / TrafficObject / Environment → `type`
    #   AgentAction / EgoAction              → `action_type`
    #   AgentProperty / EgoProperty          → `property_type`
    #   LightStates                          → `color` or `type`
    #   Condition                            → `type` (list)
    # Try them in order and pick the first non-empty one.
    kind: str | None = None
    for attr in ("action_type", "property_type", "type", "color"):
        value = getattr(entity, attr, None)
        if value is None:
            continue
        # `Condition.type` is `list[str]`; stringify the first item.
        if isinstance(value, list):
            value = value[0] if value else None
        if value:
            kind = str(value)
            break
    parts = [p for p in (entity_id, kind) if p]
    if not parts:
        return cls_name
    return f"{cls_name}({', '.join(parts)})"


def _interval_seconds(match: Any, seq: Any) -> tuple[float, float]:
    """Pull `(start, end)` seconds out of a `Match.interval`.

    `Match.interval` is an `Interval` dataclass with `.start` / `.end`
    attributes; it may also be `None` for matches that don't carry a
    time window (e.g., bundle-level `not` matches that span the whole
    clip). When it's `None` we return the full clip span
    `(0.0, seq.duration_s)` so the resulting player covers the whole
    clip — the caller has already chosen to visualize this match.
    Previously this returned `(0.0, 0.0)`, which collapsed the player
    window and tripped `ClipPlayer`'s `t_end > t_start` invariant.
    """
    iv = match.interval
    if iv is None:
        duration = float(seq.duration_s) if seq.duration_s else 0.0
        return 0.0, duration
    return float(iv.start), float(iv.end)


__all__ = ["build_matchset_carousel"]
