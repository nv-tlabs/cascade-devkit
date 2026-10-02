# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time helpers for the corpus's `M:S.D` and bare-second timestamps."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Anchored `M:S[.D]` or bare-second pattern. Strict by design — signs,
# whitespace, and leading or trailing garbage are rejected, never silently
# stripped. Historically the viz layer carried its own loose `re.search`
# variant; that divergence is gone (see `parse_timestamp_or` below for the
# TS-parity, default-on-failure entry point that callers like `viz.segments`
# use).
_TS_RE = re.compile(r"(?:(\d+):)?(\d+(?:\.\d+)?)")


def parse_timestamp(ts: str | None) -> float | None:
    """Parse `M:S[.D]` or bare seconds. Return `None` on empty / invalid.

    Strict: the whole string must match a non-negative supported form. Use
    `parse_timestamp_or` if you want a fallback value instead of `None`
    (e.g. when laying out a timeline that must always produce numeric
    coordinates).
    """
    if not ts:
        return None
    m = _TS_RE.fullmatch(ts)
    if not m:
        return None
    minutes = int(m.group(1)) if m.group(1) is not None else 0
    return minutes * 60 + float(m.group(2))


def parse_timestamp_or(ts: str | None, default: float = 0.0) -> float:
    """Parse a supported timestamp, returning `default` when unparseable.

    Same anchored regex and arithmetic as `parse_timestamp` — there is one
    parser; this entry point just substitutes `default` where the strict
    variant returns `None`. Use this at the boundary where a numeric
    coordinate is required (timeline layout, plotter inputs) and a stray
    bad-input would corrupt downstream math silently if we crashed only at
    render-time. Use `parse_timestamp` everywhere a failure means "no
    interval here" and you want the caller to decide.
    """
    parsed = parse_timestamp(ts)
    return default if parsed is None else parsed


def format_timestamp(seconds: float, *, decimals: int = 1) -> str:
    """Format a second-count back to `M:S.D` (default 1 decimal place)."""
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m}:{s:.{decimals}f}"


@dataclass(frozen=True, slots=True)
class Interval:
    """A closed time interval in seconds: `[start, end]`."""

    start: float
    end: float

    @classmethod
    def from_strings(cls, start_ts: str | None, end_ts: str | None) -> Interval | None:
        """Build from timestamp strings; return `None` if either is unparseable."""
        a = parse_timestamp(start_ts)
        b = parse_timestamp(end_ts)
        if a is None or b is None:
            return None
        return cls(a, b)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def contains(self, t: float) -> bool:
        """True if `t` lies inside `[start, end]` (closed)."""
        return self.start <= t <= self.end

    def overlaps(self, other: Interval) -> bool:
        """True if this interval and `other` share at least one instant."""
        return self.start <= other.end and other.start <= self.end

    def to_strings(self, *, decimals: int = 1) -> tuple[str, str]:
        return format_timestamp(self.start, decimals=decimals), format_timestamp(
            self.end, decimals=decimals
        )
