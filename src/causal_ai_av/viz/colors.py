# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entity-kind + family color palettes — Python mirror of the annotator's CSS tokens.

The annotator's dark palette is defined as `--color-entity-*` CSS custom
properties in `tools/annotator/web/src/index.css`. To keep the headless and
widget renderers visually identical to the annotator, this module returns the
exact same hex strings.

`entity_color(kind)` returns the BASE color for one of the five entity kinds
(`env`, `ego`, `object`, `agent`, `light`) — used for the "parent" / main
segment bar on each category's parent row.

`family_color(category, family)` returns the per-family BAR fill the
annotator paints on each sub-row stripe. The annotator's
`Timeline.tsx:990-1068` hard-codes a `subColor` per subtrack family; we
mirror those exact hex values here so the headless renderer and the
annotator agree visually on every sub-row. When a `(category, family)` pair
has no family-specific color (typically the parent row itself), we fall
back to `entity_color(category_kind)`.
"""

from __future__ import annotations

# Verbatim from tools/annotator/web/src/index.css lines 53-58.
_ENTITY_PALETTE: dict[str, str] = {
    "env": "#22c55e",
    "ego": "#3b82f6",
    "object": "#f59e0b",
    "agent": "#a855f7",
    "light": "#ef4444",
}


def entity_color(kind: str) -> str:
    """Return the `#RRGGBB` color for an entity `kind`.

    Recognized kinds: `"env"`, `"ego"`, `"object"`, `"agent"`, `"light"`.
    Mirrors the annotator's `--color-entity-*` CSS variables verbatim, so the
    headless renderer and the Plotly widget share one palette.

    Raises:
        KeyError: if `kind` is not one of the five known entity kinds.
    """
    try:
        return _ENTITY_PALETTE[kind]
    except KeyError as e:
        raise KeyError(
            f"unknown entity kind {kind!r}; expected one of "
            f"{sorted(_ENTITY_PALETTE)}"
        ) from e


# Per-family bar fills.
#
# Verbatim ports of `subColor` assignments in
# `tools/annotator/web/src/components/Timeline.tsx:990-1068`. Each
# `(Category, family)` key maps to the hex string the annotator paints
# onto that subtrack's segment bar. Families not listed here (notably
# `parent`) inherit `entity_color(category_kind)` via `family_color`'s
# fallback path.
#
# Category names match the annotator's display labels (the user-facing
# strings the screenshot shows): "Environments", "Traffic Lights",
# "Objects", "Agents", "Ego".
#
# Notes:
#   - Traffic Lights · state in the annotator picks a hex based on each
#     individual state's `color` attribute (green / yellow / red /
#     other). The simple port settles on the annotator's default
#     `#dc2626` (red); per-state tinting can land in a follow-up if it
#     proves visually load-bearing.
#   - The arrow palette in `viz.timeline._ARROW_COLORS` overlaps two
#     family hexes by design (`#14b8a6` = link_to / Agents·pose;
#     `#f97316` = action_target / influence / signal_head). The
#     annotator has the same overlap. The base `entity_color` palette
#     stays disjoint from arrows so the load-bearing
#     `test_arrow_palette_disjoint_from_entity_palette` invariant is
#     preserved.
_FAMILY_PALETTE: dict[tuple[str, str], str] = {
    # Environments
    ("Environments", "condition"): "#06b6d4",
    # Ego
    ("Ego", "containment"): "#22c55e",
    ("Ego", "influence"): "#f97316",
    ("Ego", "action"): "#3b82f6",
    ("Ego", "property"): "#93c5fd",
    # Agents
    ("Agents", "containment"): "#22c55e",
    ("Agents", "pose"): "#14b8a6",
    ("Agents", "influence"): "#f97316",
    ("Agents", "action"): "#c084fc",
    ("Agents", "property"): "#d8b4fe",
    # Objects
    ("Objects", "containment"): "#22c55e",
    ("Objects", "state"): "#d97706",
    # Traffic Lights
    ("Traffic Lights", "physical_containment"): "#22c55e",
    ("Traffic Lights", "signal_head"): "#f97316",
    ("Traffic Lights", "env_control"): "#22c55e",
    ("Traffic Lights", "state"): "#dc2626",
}

# Map category label → entity_color kind. Used by the `family_color`
# fallback so an unrecognized family for a known category still returns
# a sensible bar fill (the entity base color).
_CATEGORY_KIND: dict[str, str] = {
    "Environments": "env",
    "Ego": "ego",
    "Objects": "object",
    "Agents": "agent",
    "Traffic Lights": "light",
}


def family_color(category: str, family: str) -> str:
    """Return the `#RRGGBB` fill for a segment in `(category, family)`.

    Walks the per-family annotator palette ported from
    `Timeline.tsx:990-1068`. Falls back to `entity_color(category_kind)`
    when the family has no family-specific color (parent rows, or any
    future family not yet covered by the palette table) — preserves
    backward compat for callers that don't yet know about families.

    Args:
        category: one of `"Environments"`, `"Traffic Lights"`,
            `"Objects"`, `"Agents"`, `"Ego"`. Unknown categories raise
            `KeyError`.
        family: one of the `SegmentFamily` literals (`"parent"`,
            `"action"`, ...). Unknown families fall through to the
            entity-base fallback rather than raising — segments with
            new families still render with a sensible color.

    Raises:
        KeyError: if `category` is not one of the five known category
            labels.
    """
    palette_hit = _FAMILY_PALETTE.get((category, family))
    if palette_hit is not None:
        return palette_hit
    # Fallback: entity base color. Raise on unknown categories so
    # callers can't silently mis-spell a label and get the agent purple.
    kind = _CATEGORY_KIND.get(category)
    if kind is None:
        raise KeyError(
            f"unknown category {category!r}; expected one of "
            f"{sorted(_CATEGORY_KIND)}"
        )
    return entity_color(kind)


__all__ = ["entity_color", "family_color"]
