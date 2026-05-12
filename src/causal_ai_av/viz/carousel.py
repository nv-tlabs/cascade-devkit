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
) -> "ipywidgets.Widget":
    """Build a carousel of `ClipPlayer` widgets — one per match.

    Args:
        matchset: a `MatchSet` whose `.sequences()` resolves the dataset
            back-reference. A bundle-level `MatchSet` (produced by
            `find_on_bundle` without `dataset=`) raises `RuntimeError`
            from `.sequences()`; the error propagates uncaught.
        layout: `"stack"` for a vertical `VBox`, `"grid"` for a
            `GridBox` with `cols` columns.
        cols: number of columns when `layout="grid"`. Ignored otherwise.
        limit: cap the number of players. If the `MatchSet` has more
            matches than `limit`, the first `limit` are rendered and a
            small notice is prepended to the carousel.
        pad: seconds of padding to apply on either side of each match's
            interval when building the player's playback window. Clamped
            to `[0, sequence.duration_s]`.
        fps: scrub rate forwarded to each `ClipPlayer`.
        arrows: per-family arrow-on/off toggles forwarded to each
            `ClipPlayer`. See `_paint_timeline_onto` for the keys.

    Returns:
        an `ipywidgets.Widget` — `HTML` if `matchset` is empty,
        `VBox` for `layout="stack"`, or `GridBox` for `layout="grid"`.
        The `VBox` / `GridBox` contains one labeled child per match
        (a `VBox([HTML(label), player.widget])`), plus a leading
        truncation notice if `limit` clipped the result.

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

    total = len(matchset)
    truncated = total > limit

    children: list[Any] = []
    # `matchset.sequences()` raises `RuntimeError` if `_dataset` is
    # None; we deliberately do not catch — the carousel can't be built
    # without a way to fetch each clip's video.
    for i, (match, seq) in enumerate(matchset.sequences()):
        if i >= limit:
            break

        t0_raw, t1_raw = _interval_seconds(match)
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
                    f"playback window for match {match.clip_id}"
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
        )

        label = (
            f"<b>{match.clip_id}</b> &middot; {match.entity!s} &middot; "
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
            f"<em>Showing {limit} of {total} matches.</em>"
        )
        return ipywidgets.VBox([notice, body])
    return body


def _interval_seconds(match: Any) -> tuple[float, float]:
    """Pull `(start, end)` seconds out of a `Match.interval`.

    `Match.interval` is an `Interval` dataclass with `.start` / `.end`
    attributes; it may also be `None` for matches that don't carry a
    time window (e.g., bundle-level `not` matches that span the whole
    clip). When it's None the player gets the full clip duration —
    the caller has already chosen to visualize this match.
    """
    iv = match.interval
    if iv is None:
        return 0.0, 0.0
    return float(iv.start), float(iv.end)


__all__ = ["build_matchset_carousel"]
