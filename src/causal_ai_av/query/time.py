# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time helpers for the corpus's `M:S.D` timestamp strings."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TS_RE = re.compile(r"^(\d+):(\d+(?:\.\d+)?)$")


def parse_timestamp(ts: str | None) -> float | None:
    """Parse a `M:S.D` timestamp to seconds. Returns `None` on empty / invalid."""
    if not ts:
        return None
    m = _TS_RE.match(ts)
    if not m:
        return None
    return int(m.group(1)) * 60 + float(m.group(2))


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
        """Build from `M:S.D` strings; returns `None` if either is unparseable."""
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
