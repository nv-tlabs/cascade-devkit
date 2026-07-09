# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless Plotly timeline renderer — the static `render_timeline` Figure.

This is the v0 of the timeline view from `meta/10_visualization_api_plan.md`.
It consumes a `Sequence`'s `AnnotationBundle`, flattens it through PR-1's
`annotation_to_segments`, and emits a `plotly.graph_objects.Figure` showing
each segment as a colored rectangle in one of five track groups, ordered
top-to-bottom as **Ego**, **Agents**, **Traffic Lights**, **Objects**,
**Environments**. The group contents mirror the annotator's `Timeline.tsx`
model, while the DevKit owns this ego-first publication reading order. The
per-lane lattice is intentionally squashed here, since the headless view
doesn't have to be pixel-perfect with the annotator and the same shape works
inside the interactive widget.

Segment colors follow PR-1's `entity_color()` palette so the headless view
and the annotator share one set of hex strings. Causal relationships
(`because_of`, `link_to`, `containment`, `action_target`) are drawn as
upward-curving bezier `path` shapes between segment centers, mirroring
`drawBecauseOfCurve` in `tools/annotator/web/src/components/Timeline.tsx`.
The schema's `Influence` side-channel remains available to queries and segment
consumers but is intentionally omitted from rendered rows and arrows. Each
rendered arrow family has its own color:

- `because_of`     → rose      (`#f43f5e`) — same hue family as the
                                              annotator's "BECAUSE" pill
                                              (the TS code uses `#b45309`;
                                              we pick a brighter rose so it
                                              reads clearly on the
                                              `plotly_white` template).
- `link_to`        → teal      (`#14b8a6`)
- `containment`    → green     (`#10b981`)
- `action_target`  → orange    (`#f97316`)

These are chosen to be high-contrast against `plotly_white` while
staying distinguishable from each other. **Invariant:** the arrow palette must
stay disjoint from `entity_color()`'s row-fill palette — an arrow that
shares a hex with the row it crosses would visually disappear. A guard
test in `tests/test_viz_timeline.py` pins this.

`highlight=(t0, t1)` paints a translucent yellow vertical band on top of
the rows — useful when callers want to draw attention to a particular
query window. `arrows={...}` toggles arrow families individually; the
defaults are all-on. The Figure is fully static — the widget-driving
variant lands in PR-4.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from cascade_av.viz.colors import entity_color, family_color
from cascade_av.viz.segments import Segment, annotation_to_segments, assign_lanes

if TYPE_CHECKING:  # pragma: no cover — typing only
    import plotly.graph_objects as go

    from cascade_av.dataset import Sequence
    from cascade_av.spec import AnnotationBundle


# Five categories, top → bottom. This is the DevKit visualization's
# canonical reading order: the controlled actor first, then other dynamic
# actors, infrastructure, objects, and finally scene context. The annotator
# UI owns its layout independently and may use a different order. The y-axis
# is built
# band-by-band; one band per populated
# `(category, entity_idx, family, sub_row_idx, sh_idx)` tuple. See
# `_BAND_ORDER_BY_CATEGORY` for the family ordering inside each
# category. Category labels match the annotator's display strings.
_CATEGORIES: tuple[str, ...] = (
    "Ego",
    "Agents",
    "Traffic Lights",
    "Objects",
    "Environments",
)

# Per-entity switches accepted by every timeline-backed renderer. Top-level
# keys use the same kind vocabulary as ``entity_kinds``. A bool toggles the
# whole kind; a nested mapping toggles individual top-level annotation
# entities by their stable ``id``. Omitted kinds and IDs remain visible.
# Ego is a singleton addressed by the synthetic ID ``"ego"``.
TrackVisibility = Mapping[str, bool | Mapping[str, bool]]
_InlineLabelMode = Literal["compact", "full"]
_TRACK_KINDS: frozenset[str] = frozenset(
    {"ego", "agent", "light", "object", "env"}
)

# Per-category family ordering. Mirrors the annotator's row layout in
# `Timeline.tsx:230-275`. For Traffic Lights, signal_head / env_control
# / state are repeated per signal head — the per-head families are
# pulled out of this list into `_PER_SH_FAMILIES`.
#
# Sources (`tools/annotator/web/src/components/Timeline.tsx:230-275`):
#   - Environments:   parent → condition
#   - Traffic Lights: parent → physical_containment → (per sh:
#                     signal_head → env_control → state)
#   - Objects:        parent → containment → state
#   - Agents:         parent → containment → pose → action → property
#   - Ego:            containment → action → property
#                     (singleton — no parent bar; track id is the
#                     synthetic "ego_act")
_BAND_ORDER_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "Environments": ("parent", "condition"),
    "Traffic Lights": (
        "parent",
        "physical_containment",
        # per-signal-head bands handled separately
        "signal_head",
        "env_control",
        "state",
    ),
    "Objects": ("parent", "containment", "state"),
    "Agents": ("parent", "containment", "pose", "action", "property"),
    # Ego is a singleton — no `_track_index`, no per-entity name — but it
    # still emits a `parent` band so the families filter retains an "Ego"
    # header above whichever sub-rows survive (without the parent band,
    # `families=["action"]` would render the ego action sub-row labelled
    # "actions" with no indication that it belongs to Ego).
    "Ego": ("parent", "containment", "action", "property"),
}

# Recognized family-leaf names accepted by the `families=[...]` filter
# kwarg. Derived from `_BAND_ORDER_BY_CATEGORY` with `"parent"` removed,
# plus the removed `"influence"` leaf as a compatibility no-op. Parent rows
# are NOT named in the whitelist; they auto-render for entities whose
# sub-rows survive the filter.
_RECOGNIZED_FAMILIES: frozenset[str] = frozenset(
    {
        f
        for families in _BAND_ORDER_BY_CATEGORY.values()
        for f in families
        if f != "parent"
    }
    | {"influence"}
)

# Families that, in Traffic Lights, repeat per signal head and need an
# `sh_idx` discriminator. The painter walks signal heads in ascending
# index and emits one block per head.
_PER_SH_FAMILIES: tuple[str, ...] = ("signal_head", "env_control", "state")

# Families whose sub-row count is data-driven (each segment carries a
# `_X_track_index` field in `meta`; the number of sub-rows is
# `max(_X_track_index) + 1`). Other families paint as a single 1-row
# band. Maps family → meta key.
_FAMILY_SUB_ROW_META_KEY: dict[str, str] = {
    "condition": "_cond_track_index",
    "containment": "_cont_track_index",
    "physical_containment": "_cont_track_index",
}

# Per-family tick-label leaf text. Mirrors the annotator's terse
# row-label text (`Timeline.tsx:230-275`): conditions/actions/
# properties/states pluralized; TL control / signal head
# substituted; env_control labeled "containment" under each signal
# head (the annotator's actual on-screen leaf).
_FAMILY_LABEL: dict[str, str] = {
    "condition": "conditions",
    "action": "actions",
    "property": "properties",
    "state": "states",
    "physical_containment": "TL control",
    "signal_head": "signal head",
    "env_control": "containment",
}

# Display name for each category's per-entity track row. Mirrors the
# annotator's `buildTrackList` (`timeline-utils.ts:354-372`). Ego is a
# singleton with no per-entity tag; absent from this map.
_ENTITY_TRACK_PREFIX: dict[str, str] = {
    "Environments": "Env Track",
    "Traffic Lights": "Light Track",
    "Objects": "Object Track",
    "Agents": "Agent Track",
}

# `BandKey` — `(category, entity_idx, family, sub_row_idx, sh_idx)`.
#
#   - category:     one of `_CATEGORIES`.
#   - entity_idx:   0-based entity ordinal within the category, parsed
#                   from each segment's `track_id` (e.g.
#                   `"agent_0"` → 0). Ego always uses ordinal 0.
#   - family:       one of the rendered `SegmentFamily` literals.
#   - sub_row_idx:  0-based sub-row index *within this family for this
#                   entity*. Data-driven from each segment's
#                   `_X_track_index` meta key (see
#                   `_FAMILY_SUB_ROW_META_KEY`). Families without a
#                   sub-row meta key always use 0.
#   - sh_idx:       Lights' per-signal-head discriminator; None
#                   elsewhere.
#
# The tuple is sorted lexicographically by the painter so the on-
# screen reading is category → entity → family → sub_row → SH —
# matching the annotator's nested visual hierarchy.
BandKey = tuple[str, int, str, int, "int | None"]

# Arrow-family colors. See module docstring for the choice rationale.
# Invariant: values here must be disjoint from `entity_color()`'s row
# fills — guarded by `test_arrow_palette_disjoint_from_entity_palette`.
_ARROW_COLORS: dict[str, str] = {
    "because_of": "#f43f5e",
    "link_to": "#14b8a6",
    "containment": "#10b981",
    "action_target": "#f97316",
}

# Default arrow toggles — every family on. Callers override via the
# `arrows` kwarg; any key omitted from the user override falls back to
# this default.
_DEFAULT_ARROWS: dict[str, bool] = {name: True for name in _ARROW_COLORS}


def _track_id_to_category(track_id: str) -> str | None:
    """Map a `Segment.track_id` to one of the five category labels.

    Returns `None` for unknown prefixes — those segments get dropped from
    the figure rather than crashing.
    """
    if track_id.startswith("env_"):
        return "Environments"
    if track_id.startswith("light_"):
        return "Traffic Lights"
    if track_id.startswith("obj_"):
        return "Objects"
    if track_id.startswith("agent_"):
        return "Agents"
    if track_id == "ego_act":
        return "Ego"
    return None


# Back-compat alias for legacy call sites and tests. New code should
# prefer `_track_id_to_category`.
_track_id_to_group = _track_id_to_category


def _track_id_to_entity_idx(track_id: str) -> int:
    """Parse the trailing 0-based entity ordinal from a `Segment.track_id`.

    `"agent_0"` → 0, `"agent_3"` → 3, `"env_2"` → 2, `"ego_act"` → 0.
    Defaults to 0 on malformed input so a missing `_track_index` field
    on an upstream entity still groups onto a single block rather than
    crashing the figure.
    """
    if track_id == "ego_act":
        return 0
    if "_" not in track_id:
        return 0
    tail = track_id.rsplit("_", 1)[1]
    try:
        return int(tail)
    except ValueError:
        return 0


def _segment_kind(track_id: str) -> str | None:
    """Map a track id to the `entity_color()` palette key."""
    category = _track_id_to_category(track_id)
    if category is None:
        return None
    return {
        "Environments": "env",
        "Traffic Lights": "light",
        "Objects": "object",
        "Agents": "agent",
        "Ego": "ego",
    }[category]


def _track_entity_ids(bundle: "AnnotationBundle") -> dict[str, tuple[str, ...]]:
    """Return selectable top-level entity IDs grouped by timeline kind."""
    ann = bundle.annotation
    return {
        "ego": ("ego",),
        "agent": tuple(entity.id for entity in ann.agents if entity.id),
        "light": tuple(entity.id for entity in ann.traffic_lights if entity.id),
        "object": tuple(entity.id for entity in ann.traffic_objects if entity.id),
        "env": tuple(entity.id for entity in ann.environments if entity.id),
    }


def _resolve_track_visibility(
    bundle: "AnnotationBundle",
    track_visibility: TrackVisibility | None,
) -> tuple[dict[str, bool], dict[str, dict[str, bool]]]:
    """Validate and normalize grouped kind/entity visibility switches.

    ``None`` and omitted entries mean visible. A top-level bool switches an
    entire kind, while a nested mapping switches individual stable entity IDs.
    Validation is deliberately strict: silently accepting a misspelled ID in a
    publication figure would leave the wrong track visible.
    """
    if track_visibility is None:
        return {}, {}
    if not isinstance(track_visibility, Mapping):
        raise TypeError(
            "track_visibility must be a mapping from entity kind to a bool "
            "or an ID-to-bool mapping"
        )

    non_string_kinds = [
        kind for kind in track_visibility if not isinstance(kind, str)
    ]
    if non_string_kinds:
        raise TypeError(
            "track_visibility keys must be entity-kind strings; got "
            f"{non_string_kinds!r}"
        )
    unknown_kinds = set(track_visibility) - _TRACK_KINDS
    if unknown_kinds:
        raise ValueError(
            f"unknown track_visibility kinds: {sorted(unknown_kinds)!r}. "
            f"Recognized: {sorted(_TRACK_KINDS)!r}"
        )

    entity_ids = _track_entity_ids(bundle)
    known_ids = {kind: set(ids) for kind, ids in entity_ids.items()}
    ambiguous_ids = {
        kind: {
            entity_id
            for entity_id, count in Counter(ids).items()
            if count > 1
        }
        for kind, ids in entity_ids.items()
    }
    kind_switches: dict[str, bool] = {}
    entity_switches: dict[str, dict[str, bool]] = {}
    for kind, setting in track_visibility.items():
        if isinstance(setting, bool):
            kind_switches[kind] = setting
            continue
        if not isinstance(setting, Mapping):
            raise TypeError(
                f"track_visibility[{kind!r}] must be a bool or an "
                "entity-ID-to-bool mapping"
            )

        non_string_ids = [
            entity_id for entity_id in setting if not isinstance(entity_id, str)
        ]
        if non_string_ids:
            raise TypeError(
                f"track_visibility[{kind!r}] keys must be entity-ID strings; "
                f"got {non_string_ids!r}"
            )
        unknown_ids = set(setting) - known_ids[kind]
        if unknown_ids:
            raise ValueError(
                f"unknown {kind!r} entity IDs in track_visibility: "
                f"{sorted(unknown_ids)!r}. Known: {sorted(known_ids[kind])!r}"
            )
        ambiguous = set(setting) & ambiguous_ids[kind]
        if ambiguous:
            raise ValueError(
                f"ambiguous {kind!r} entity IDs in track_visibility: "
                f"{sorted(ambiguous)!r}; each ID occurs more than once"
            )
        resolved: dict[str, bool] = {}
        for entity_id, visible in setting.items():
            if not isinstance(visible, bool):
                raise TypeError(
                    f"track_visibility[{kind!r}][{entity_id!r}] must be bool"
                )
            resolved[entity_id] = visible
        entity_switches[kind] = resolved

    return kind_switches, entity_switches


def _segment_track_entity_id(
    seg: Segment,
    bundle: "AnnotationBundle",
    kind: str,
) -> str | None:
    """Resolve a segment to the stable ID of its top-level track owner.

    Sub-rows inherit their owner: conditions resolve through ``env_id``;
    signal-head/state rows resolve to their TrafficLight; every Agent/Object
    child carries its parent's model index in ``Segment.meta``. Ego uses the
    synthetic singleton ID ``"ego"``.
    """
    if kind == "ego":
        return "ego"

    meta = seg.meta or {}
    ann = bundle.annotation
    if kind == "agent":
        entities = ann.agents
        index = meta.get("_agentIndex")
    elif kind == "light":
        entities = ann.traffic_lights
        index = meta.get("_lightIndex")
    elif kind == "object":
        entities = ann.traffic_objects
        index = meta.get("_objIndex")
    elif kind == "env":
        env_index = meta.get("_envIndex")
        if isinstance(env_index, int) and 0 <= env_index < len(ann.environments):
            return ann.environments[env_index].id or None
        condition_index = meta.get("_condIndex")
        if (
            isinstance(condition_index, int)
            and 0 <= condition_index < len(ann.conditions)
        ):
            return ann.conditions[condition_index].env_id or None
        return None
    else:
        return None

    if not isinstance(index, int) or not 0 <= index < len(entities):
        return None
    return entities[index].id or None


def _segment_center_x(seg: Segment) -> float:
    return (seg.t0 + seg.t1) / 2.0


# Band / lane geometry.
#
# Each populated band gets one integer "band row" of unit height. A band
# row at index `r` covers `[r - _ROW_HALF_HEIGHT, r + _ROW_HALF_HEIGHT]`,
# with `lane_count` lanes inside (each lane a stripe of height
# `2 * _ROW_HALF_HEIGHT / lane_count`). Lane 0 sits at the "top" of the
# row (which renders at the bottom of the row in data coordinates
# because `yaxis.range` is reversed).
_ROW_HALF_HEIGHT: float = 0.4


def _lane_band(band_row: int, lane: int, lane_count: int) -> tuple[float, float]:
    """Return `(y0, y1)` for a segment painted at `lane` inside `band_row`."""
    total = 2.0 * _ROW_HALF_HEIGHT
    lane_h = total / max(lane_count, 1)
    y0 = band_row - _ROW_HALF_HEIGHT + lane * lane_h
    y1 = y0 + lane_h
    return y0, y1


def _segment_top_y(band_row: int, lane: int = 0, lane_count: int = 1) -> float:
    """y-coordinate of a segment's top edge (used as arrow anchor).

    With the yaxis range reversed (top → bottom), "top" in data
    coordinates is the *larger* y value of the lane band — that's the
    edge that visually faces away from the rows below.
    """
    _y0, y1 = _lane_band(band_row, lane, lane_count)
    return y1


def _family_sub_row_idx(seg: Segment) -> int:
    """Extract the sub-row index this segment occupies within its
    family band.

    Walks `_FAMILY_SUB_ROW_META_KEY` to find the matching `_X_track_
    index` meta field. Defaults to 0 if the field is missing or
    non-int — keeps the painter stable when upstream entities don't
    populate the field (well-formed annotations always do).
    """
    meta_key = _FAMILY_SUB_ROW_META_KEY.get(seg.family)
    if meta_key is None or not seg.meta:
        return 0
    value = seg.meta.get(meta_key)
    if isinstance(value, int) and value >= 0:
        return value
    return 0


def _family_sh_idx(seg: Segment) -> int | None:
    """Return the signal-head ordinal for Lights per-head families;
    `None` for any other (category, family) combination.

    The per-SH discriminator only applies to Traffic Lights — `"state"`
    appears in both `Objects` and `Traffic Lights`, but only the Lights
    case wants a per-head split.
    """
    if seg.family not in _PER_SH_FAMILIES:
        return None
    if _track_id_to_category(seg.track_id) != "Traffic Lights":
        return None
    if not seg.meta:
        return 0
    value = seg.meta.get("_sh_index")
    return value if isinstance(value, int) else 0


def _segment_band_key(seg: Segment) -> BandKey | None:
    """Map a segment to its
    `(category, entity_idx, family, sub_row_idx, sh_idx)` band.

    Returns `None` for segments whose `track_id` doesn't resolve to a
    known category — they get dropped from the figure rather than
    crashing.
    """
    category = _track_id_to_category(seg.track_id)
    if category is None:
        return None
    entity_idx = _track_id_to_entity_idx(seg.track_id)
    family = seg.family
    sub_row_idx = _family_sub_row_idx(seg)
    sh_idx = _family_sh_idx(seg)
    return (category, entity_idx, family, sub_row_idx, sh_idx)


def _band_label_text(key: BandKey) -> str:
    """Plain text for a band's y-tick label (no color wrapper).

      - Parent row:                            "<Entity Track Name>"
        (e.g. "Env Track 1", "Agent Track 2", "Ego")
      - Sub-row, family head (sub_row_idx == 0):   "<leaf>"
        (e.g. "conditions", "actions", "containment")
      - Sub-row, trailing (sub_row_idx > 0):        ""  (no label)
      - Lights per-SH family head:             "sh<i> · <leaf>"
        (kept — a single Light can host multiple signal heads, so
        the sh tag is the only disambiguator within the block)
      - Lights per-SH trailing sub-row:              ""
      - Ego: singleton — its parent row reads just "Ego" (no
        per-track ordinal), sub-rows are the bare leaf names.
    """
    category, entity_idx, family, sub_row_idx, sh_idx = key
    if family == "parent":
        # The parent bar labels its own row. Ego is a singleton, so it
        # reads as just "Ego" — no `_ENTITY_TRACK_PREFIX` lookup and no
        # `entity_idx + 1` ordinal. Every other category uses the
        # "Prefix N" convention.
        if category == "Ego":
            return "Ego"
        track_prefix = _ENTITY_TRACK_PREFIX.get(category, f"{category} Track")
        return f"{track_prefix} {entity_idx + 1}"
    if sub_row_idx > 0:
        # Trailing sub-rows render blank — the annotator's labeling
        # rule keeps the family name on the FIRST sub-row only.
        return ""
    leaf = _FAMILY_LABEL.get(family, family)
    return f"sh{sh_idx} · {leaf}" if sh_idx is not None else leaf


# Non-breaking space used to pad tick labels for left-alignment.
# Regular spaces are collapsed by SVG text rendering; non-breaking
# spaces are not. With the y-axis tick text anchored at its right
# edge (the SVG default), trailing nbsp's push the visible text
# leftward into the label margin — giving a left-aligned look that
# matches the annotator's row-label layout.
_LABEL_PAD: str = " "

# Extra padding chars added past the longest plain label width.
# Without this cushion, when a parent row is the widest label
# (e.g., "Object Track 1" tying with a sub-row's pad-to width),
# centering math gives the parent 0 trailing nbsp's — visually
# it right-aligns against the axis. With +4, every parent gets at
# least 2 trailing nbsp's (visibly dents inward = clearly
# centered), and every sub-row pads at least +4 nbsp's past the
# longest label (visibly extends left = clearly left-aligned).
# The two row types now read as distinct alignments at a glance.
_PAD_COLUMN_EXTRA: int = 4


def _band_label_html(key: BandKey, pad_to: int) -> str:
    """Render a band's tick label as colored HTML with per-row-type
    alignment padding.

    Tick text is wrapped in `<span style="color:#...">` so each tick
    carries the category cue via color (Plotly's SVG renderer
    interprets HTML inside `ticktext`). The right edge of the span
    sits at the y-axis tick anchor; trailing non-breaking spaces
    push the visible text leftward by the amount of padding.

    Alignment rule:
      - **Parent rows** (entity headers like "Env Track 1",
        "Agent Track 2") → **centered**: pad with `(pad_to - len)
        // 2` trailing nbsp's so the visible text sits in the
        middle of the longest-label's column.
      - **Sub-rows** (family heads like "containment", "actions")
        → **left-aligned**: pad with the full `pad_to - len`
        trailing nbsp's so the visible text starts at the same
        left edge as every other sub-row.

    This mirrors the annotator's row-label hierarchy: an entity
    header is the visual centerpiece of its block, its sub-rows
    indent under it along a shared left edge.
    """
    category, _entity_idx, family, _sub_row_idx, _sh_idx = key
    text = _band_label_text(key)
    if not text:
        return ""
    deficit = max(0, pad_to - len(text))
    if family == "parent":
        # Centering: pad with half the deficit so the visible text
        # sits in the middle of the longest-label column. Floor
        # division biases the visible text very slightly leftward
        # when the deficit is odd — imperceptible at typical label
        # lengths.
        pad_count = deficit // 2
        # Parent rows are the section heads for each entity block
        # ("Env Track 1", "Agent Track 2", ...) — bold them so the
        # block boundary reads at a glance. Sub-rows stay regular
        # weight so the hierarchy is visually obvious.
        inner = f"<b>{text}</b>{_LABEL_PAD * pad_count}"
    else:
        # Left-aligning: full deficit pushes the visible text to
        # the same left column as every other sub-row.
        pad_count = deficit
        inner = f"{text}{_LABEL_PAD * pad_count}"
    color = _category_label_color(category)
    return f'<span style="color:{color}">{inner}</span>'


# Back-compat alias. Older call sites (and tests) may import
# `_band_label` directly. The new two-stage API is
# `_band_label_text` + `_band_label_html`; this alias produces an
# unpadded label so out-of-tree callers keep working.
def _band_label(key: BandKey) -> str:
    text = _band_label_text(key)
    if not text:
        return ""
    category = key[0]
    color = _category_label_color(category)
    if key[2] == "parent":
        return f'<span style="color:{color}"><b>{text}</b></span>'
    return f'<span style="color:{color}">{text}</span>'


def _populated_bands(segments: list[Segment]) -> list[BandKey]:
    """Return the ordered list of populated bands, top → bottom.

    Y-axis hierarchy: **Category → Entity (track) → Family →
    Sub-row → (Lights only) SH index**. Each band corresponds to one
    `(category, entity_idx, family, sub_row_idx, sh_idx)` tuple that
    has at least one segment; empty tuples drop entirely.

    Sub-row counts are data-driven from each segment's
    `_X_track_index` meta field — an Agent with three containments
    at `_cont_track_index = 0, 1, 2` produces three distinct sub-rows
    in its containment family band. Families without a sub-row meta
    key always collapse to a single sub-row.

    Reading order:
      - Categories in `_CATEGORIES` order (Ego → Agents → Traffic
        Lights → Objects → Environments).
      - Inside each category, entity ordinals ascending — each ordinal
        is a separate per-entity block.
      - Inside each entity block, families in `_BAND_ORDER_BY_
        CATEGORY[category]` order, each with `0..N-1` sub-rows.
      - For Lights, per-SH families (signal_head / env_control /
        state) repeat per signal head AFTER the non-per-head families
        (parent / physical_containment).
    """
    # Bucket segments by full BandKey, and remember which entity
    # blocks / SH indices exist so the canonical ordering can prune
    # empty bands cleanly.
    populated: set[BandKey] = set()
    entities_per_cat: dict[str, set[int]] = {c: set() for c in _CATEGORIES}
    sh_indices_per_entity: dict[tuple[str, int], set[int]] = {}
    for seg in segments:
        key = _segment_band_key(seg)
        if key is None:
            continue
        populated.add(key)
        category, entity_idx, _family, _sub_row, sh_idx = key
        entities_per_cat[category].add(entity_idx)
        if category == "Traffic Lights" and sh_idx is not None:
            sh_indices_per_entity.setdefault(
                (category, entity_idx), set()
            ).add(sh_idx)

    def _max_sub_row(
        category: str, entity_idx: int, family: str, sh_idx: int | None
    ) -> int:
        """The maximum sub_row_idx seen for this entity/family/SH —
        determines how many sub-rows the band group has."""
        n = -1
        for key in populated:
            if (
                key[0] == category
                and key[1] == entity_idx
                and key[2] == family
                and key[4] == sh_idx
            ):
                if key[3] > n:
                    n = key[3]
        return n

    ordered: list[BandKey] = []
    for category in _CATEGORIES:
        families = _BAND_ORDER_BY_CATEGORY.get(category, ())
        for entity_idx in sorted(entities_per_cat[category]):
            if category == "Traffic Lights":
                # Non-per-head families first (parent, phys_cont),
                # then per-head bands per signal head.
                for family in families:
                    if family in _PER_SH_FAMILIES:
                        continue
                    n = _max_sub_row(category, entity_idx, family, None)
                    for sub_row_idx in range(n + 1):
                        key = (category, entity_idx, family, sub_row_idx, None)
                        if key in populated:
                            ordered.append(key)
                sh_indices = sorted(
                    sh_indices_per_entity.get((category, entity_idx), set())
                )
                for sh_idx in sh_indices:
                    for family in _PER_SH_FAMILIES:
                        if family not in families:
                            continue
                        n = _max_sub_row(category, entity_idx, family, sh_idx)
                        for sub_row_idx in range(n + 1):
                            key = (
                                category, entity_idx, family,
                                sub_row_idx, sh_idx,
                            )
                            if key in populated:
                                ordered.append(key)
            else:
                for family in families:
                    n = _max_sub_row(category, entity_idx, family, None)
                    for sub_row_idx in range(n + 1):
                        key = (category, entity_idx, family, sub_row_idx, None)
                        if key in populated:
                            ordered.append(key)
    return ordered


# Inline-label font size, in points. Tuned for the 24px-per-lane
# layout from `render_timeline`'s adaptive height — at that lane
# height a `size=8` label reads cleanly without clipping vertically
# against the rectangle's top/bottom edge. Bumping this back up is
# the right knob if a caller forces a taller layout via the `height`
# kwarg.
_INLINE_LABEL_FONT_SIZE: int = 8

# Maximum characters for an inline segment label. Anything longer gets
# truncated with an ellipsis; the full label always lives in the hover
# tooltip. Tightened from 11 → 8 chars to keep labels from spilling out
# of narrow boxes once the family-band layout multiplies the band count.
_INLINE_LABEL_MAX_CHARS: int = 8

# Minimum segment width *as a fraction of the clip duration* for an
# inline label to render. The fraction-based threshold replaces the old
# absolute `_INLINE_LABEL_MIN_WIDTH_S = 0.5` so the same segment scales
# correctly across clip lengths: a 1.0s segment in a 20s clip is 5% of
# duration (suppressed); the same segment in a 5s clip is 20% (shown).
# 6% lands at the empirical "label legibly fits" knee for the
# 24px-per-lane band height at the default Jupyter cell width.
_INLINE_LABEL_MIN_WIDTH_FRAC: float = 0.06


def _truncate_label(label: str, *, max_chars: int = _INLINE_LABEL_MAX_CHARS) -> str:
    """Truncate `label` to `max_chars`, appending `…` when shortened."""
    if len(label) <= max_chars:
        return label
    return label[: max(0, max_chars - 1)].rstrip() + "…"


def _xref_to_axis(xref: str) -> str:
    """Translate a shape's `xref` (`"x"`, `"x2"`, …) to a trace's `xaxis`.

    Plotly uses the same string for both — `"x"` for the primary
    subplot, `"x2"` for the second, etc. The function is a no-op alias
    but keeps the caller's intent explicit: shape `xref` and trace
    `xaxis` are semantically distinct fields even though they share a
    representation.
    """
    return xref


def _yref_to_axis(yref: str) -> str:
    """Translate a shape's `yref` to a trace's `yaxis`."""
    return yref


def _bezier_path(x0: float, y0: float, x1: float, y1: float) -> str:
    """Build an SVG quadratic-bezier path mirroring `drawBecauseOfCurve`.

    The annotator draws `M x1 y1 Q midX ctrlY x2 y2` where `ctrlY` is the
    minimum of the two endpoints minus 30px of canvas. We mimic the
    *shape* in data coordinates — the control point sits one full row
    above the higher of the two endpoints, so the arc reads as an upward
    curve regardless of source/target ordering.
    """
    mid_x = (x0 + x1) / 2.0
    # y grows downward in the annotator's canvas but upward in Plotly's
    # default `yaxis`. Either way, the visual is the same: the curve
    # bulges *away* from the rows. With our top-of-row anchors at
    # `row + 0.4` and a y-axis that grows upward, we lift the control
    # point *above* both endpoints by adding 0.8 to the max y.
    ctrl_y = max(y0, y1) + 0.8
    return f"M {x0} {y0} Q {mid_x} {ctrl_y} {x1} {y1}"


def _segments_by_entity_id(
    bundle: "AnnotationBundle",
    segments: list[Segment],
) -> dict[str, Segment]:
    """Build a stable annotation-ID → rendered-segment lookup table.

    ``Segment.id`` values are synthetic and include a nonce, while causal
    fields store stable schema IDs. Walk every segment-backed model in the
    same order as :class:`~cascade_av.query.IdIndex` and pair its stable ID
    with the emitted segment. Later duplicate IDs overwrite earlier ones, so
    visualization and query resolution agree. Visible Traffic-Light
    containments, which the canonical index does not yet cover, supplement
    that surface without overriding canonical IDs. This covers nested action,
    state, property, containment, condition, and signal-head targets in
    addition to top-level entities.
    """
    from cascade_av.query.index import KINDS, IdIndex

    ann = bundle.annotation
    by_id: dict[str, Segment] = {}
    canonical_index = IdIndex(bundle)
    canonical_ids = {
        subject.id
        for kind in KINDS
        for subject in canonical_index.of_kind(kind)
    }
    real_ids: set[str] = set()

    def _add(
        stable_id: str,
        *,
        id_starts_with: str,
        meta_key: str | None = None,
        meta_val: Any = None,
        supplemental: bool = False,
    ) -> None:
        if not stable_id:
            return
        real_ids.add(stable_id)
        if supplemental and stable_id in canonical_ids:
            return
        segment = _find_seg(
            segments,
            id_starts_with=id_starts_with,
            meta_key=meta_key,
            meta_val=meta_val,
        )
        if segment is None:
            # Preserve last-writer semantics even when the winning entity has
            # no rendered segment (notably a removed Influence record).
            by_id.pop(stable_id, None)
        else:
            by_id[stable_id] = segment

    # IdIndex installs the synthetic Ego anchor before all real IDs. A real
    # entity using the reserved "Ego" ID therefore follows its documented
    # last-writer behavior even though such an annotation is ambiguous.
    ego_parent = _find_seg(segments, id_starts_with="ego_parent_")
    if ego_parent is not None:
        by_id["Ego"] = ego_parent

    for env_index, env in enumerate(ann.environments):
        _add(env.id, id_starts_with=f"env_{env_index}_")
    for condition_index, condition in enumerate(ann.conditions):
        _add(condition.id, id_starts_with=f"cond_{condition_index}_")

    for object_index, obj in enumerate(ann.traffic_objects):
        _add(obj.id, id_starts_with=f"obj_{object_index}_")
        for containment_index, containment in enumerate(obj.containment):
            _add(
                containment.id,
                id_starts_with=(
                    f"obj_cont_{object_index}_{containment_index}_"
                ),
            )
        for state_index, state in enumerate(obj.state_sequence):
            _add(
                state.id,
                id_starts_with=f"obj_state_{object_index}_{state_index}_",
            )

    for light_index, light in enumerate(ann.traffic_lights):
        _add(light.id, id_starts_with=f"light_{light_index}_")
        for containment_index, containment in enumerate(light.containment):
            _add(
                containment.id,
                id_starts_with=(
                    f"light_phys_cont_{light_index}_{containment_index}_"
                ),
                supplemental=True,
            )
        for head_index, head in enumerate(light.signal_heads):
            _add(
                head.id,
                id_starts_with=f"light_sh_{light_index}_{head_index}_",
            )
            for state_index, state in enumerate(head.state_sequence):
                _add(
                    state.id,
                    id_starts_with=(
                        f"light_state_{light_index}_{head_index}_{state_index}_"
                    ),
                )
            for containment_index, containment in enumerate(
                head.env_controlled
            ):
                _add(
                    containment.id,
                    id_starts_with=(
                        f"light_cont_{light_index}_{head_index}_"
                        f"{containment_index}_"
                    ),
                    supplemental=True,
                )

    for action_index, action in enumerate(ann.ego_vehicle.actions):
        _add(
            action.id,
            id_starts_with="ego_act_",
            meta_key="_egoActIndex",
            meta_val=action_index,
        )
    for property_index, prop in enumerate(ann.ego_vehicle.properties):
        _add(prop.id, id_starts_with=f"ego_prop_{property_index}_")
    for containment_index, containment in enumerate(
        ann.ego_vehicle.containment
    ):
        _add(
            containment.id,
            id_starts_with=f"ego_cont_{containment_index}_",
        )
    for influence_index, influence in enumerate(ann.ego_vehicle.influenced_by):
        _add(
            influence.id,
            id_starts_with=f"ego_infl_{influence_index}_",
        )

    for agent_index, agent in enumerate(ann.agents):
        _add(agent.id, id_starts_with=f"agent_{agent_index}_")
        for action_index, action in enumerate(agent.actions):
            _add(
                action.id,
                id_starts_with=(
                    f"agent_action_{agent_index}_{action_index}_"
                ),
            )
        for property_index, prop in enumerate(agent.properties):
            _add(
                prop.id,
                id_starts_with=f"agent_prop_{agent_index}_{property_index}_",
            )
        for containment_index, containment in enumerate(agent.containment):
            _add(
                containment.id,
                id_starts_with=(
                    f"agent_cont_{agent_index}_{containment_index}_"
                ),
            )
        for influence_index, influence in enumerate(agent.influenced_by):
            _add(
                influence.id,
                id_starts_with=(
                    f"agent_infl_{agent_index}_{influence_index}_"
                ),
            )

    # Older fixtures/exporters used lowercase "ego" for the synthetic anchor.
    # Keep that alias only when no real stable ID claimed it.
    if ego_parent is not None and "ego" not in real_ids:
        by_id.setdefault("ego", ego_parent)
    return by_id


# Per-entity background band tints. Mirrors the annotator's
# "soft background spanning an entity's rows" pattern: a quiet
# *neutral* white tint that clusters every row belonging to one
# entity, with adjacent blocks alternating opacities so neighboring
# entities stay visually separable. The category cue moves to the
# colored tick label (`_band_label`'s HTML span), not the band —
# so the band stays soft/neutral and doesn't compete with the
# per-family bar palette.
_ENTITY_BLOCK_TINT_EVEN: float = 0.05
_ENTITY_BLOCK_TINT_ODD: float = 0.10

# Paper-x extent for the entity-block band's left edge. The band
# uses `xref="paper"` and walks *into the y-tick-label margin* so it
# clusters the label region along with the bar region (the
# annotator's bigger-soft-background trick). Calibrated to our
# `render_timeline` margin (`l=80`) at moderate figure widths
# (700-900px wide); -0.18 cushions narrower Jupyter cells.
_ENTITY_BLOCK_PAPER_X_LEFT: float = -0.18
_ENTITY_BLOCK_PAPER_X_RIGHT: float = 1.0

# Map category label → `entity_color()` kind. Used by the
# entity-block tinter's fallback path AND by the colored-tick-label
# wrapper in `_band_label`. Mirrors the private `_CATEGORY_KIND`
# table in `colors.py` so the timeline module doesn't reach into a
# sibling module's privates.
_CATEGORY_KIND: dict[str, str] = {
    "Environments": "env",
    "Traffic Lights": "light",
    "Objects": "object",
    "Agents": "agent",
    "Ego": "ego",
}


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert `"#RRGGBB"` to `"rgba(r, g, b, a)"` for translucent fills.

    Plotly accepts both hex and rgba in `fillcolor`; using rgba lets us
    tune opacity without baking pre-darkened hex variants into the
    palette table.
    """
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _entity_blocks(
    band_keys: list[BandKey],
) -> list[tuple[str, int, int, int]]:
    """Return `[(category, entity_idx, first_row, last_row), ...]` for
    each contiguous run of bands sharing the same `(category,
    entity_idx)`. `band_keys` is expected in painter order (from
    `_populated_bands`), so the rows for a given entity are always
    contiguous.

    Ego collapses to a single block at entity_idx 0.
    """
    blocks: list[tuple[str, int, int, int]] = []
    if not band_keys:
        return blocks
    current_cat: str | None = None
    current_idx: int = -1
    start_row: int = 0
    for row, key in enumerate(band_keys):
        cat, entity_idx, _family, _sub_row, _sh = key
        if cat != current_cat or entity_idx != current_idx:
            if current_cat is not None:
                blocks.append((current_cat, current_idx, start_row, row - 1))
            current_cat = cat
            current_idx = entity_idx
            start_row = row
    assert current_cat is not None  # band_keys non-empty guarded above
    blocks.append((current_cat, current_idx, start_row, len(band_keys) - 1))
    return blocks


def _entity_block_fill(_category: str, entity_idx: int) -> str:
    """Translucent fillcolor for a `(category, entity_idx)` block.

    Soft neutral *dark* tint — paints `#0f172a` (Tailwind slate-900)
    at low alpha so it reads as a faint gray on the `plotly_white`
    background. The category cue is carried by the colored tick
    label, not the band. Alternating opacities
    (`_ENTITY_BLOCK_TINT_EVEN` vs `_ENTITY_BLOCK_TINT_ODD`) per-entity
    keep adjacent blocks visually distinct, like the annotator's
    soft-row-background pattern.
    """
    alpha = (
        _ENTITY_BLOCK_TINT_EVEN
        if entity_idx % 2 == 0
        else _ENTITY_BLOCK_TINT_ODD
    )
    return _hex_to_rgba("#0f172a", alpha)


def _category_label_color(category: str) -> str:
    """`#RRGGBB` hex for tick labels belonging to `category`.

    Wraps `entity_color()` with the category → kind lookup. Used by
    `_band_label` to color each tick text per its block's category —
    the visual category cue that complements the per-family bar
    palette in the timeline.
    """
    kind = _CATEGORY_KIND.get(category)
    if kind is None:
        return "#ffffff"
    return entity_color(kind)


@dataclass(frozen=True)
class PaintResult:
    """What `_paint_timeline_onto` reports back to the caller.

    Attributes:
        bands: ordered list of populated `(group, family, sh_idx)`
            band keys, top → bottom on the y-axis. Empty bands were
            already dropped.
        band_lane_counts: per-band lane count after greedy lane
            assignment. Used to size the figure proportionally to the
            deepest stack per band.
        y_range: `(y_min, y_max)` data-coordinate range of the painted
            timeline. Callers (the widget's playhead line in
            particular) read this back so their overlay shapes span
            the full band stack, not the legacy 5-row integer grid.
        total_lanes: convenience aggregate — `sum(band_lane_counts)`.
            Falls back to a sentinel >0 when no bands are populated so
            the adaptive-height path still produces a non-zero figure.
    """

    bands: tuple[BandKey, ...]
    band_lane_counts: dict[BandKey, int]
    y_range: tuple[float, float]
    total_lanes: int


def _paint_timeline_onto(
    fig: "go.Figure",
    seq: "Sequence",
    *,
    xref: str = "x",
    yref: str = "y",
    xaxis_key: str = "xaxis",
    yaxis_key: str = "yaxis",
    highlight: tuple[float, float] | None = None,
    arrows: dict[str, bool] | None = None,
    entity_kinds: list[str] | None = None,
    agent_ids: list[str] | None = None,
    track_groups: list[str] | None = None,
    families: list[str] | None = None,
    track_visibility: TrackVisibility | None = None,
    show_inline_labels: bool = True,
    inline_label_mode: _InlineLabelMode = "compact",
) -> PaintResult:
    """Append timeline shapes for `seq` onto `fig`, on the given axes.

    Factored out of `render_timeline` so the PR-4 widget can paint the
    same timeline onto the bottom subplot of a `FigureWidget` without
    rebuilding the rasterization logic from scratch. Mutates `fig` in
    place — appends shapes to `fig.layout.shapes`, label annotations to
    `fig.layout.annotations`, and invisible hover-scatter traces to
    `fig.data`, then updates the `xaxis_key` / `yaxis_key` axis
    settings.

    Args:
        fig: the Plotly Figure (or FigureWidget) to paint onto.
        seq: source `Sequence`.
        xref, yref: axis references for the painted shapes
            (`"x"`/`"y"` for the primary subplot, `"x2"`/`"y2"` for the
            second subplot in a `make_subplots(rows=2, ...)` figure,
            and so on).
        xaxis_key, yaxis_key: layout keys for the axis configuration
            (`"xaxis"`/`"yaxis"` or `"xaxis2"`/`"yaxis2"`, etc.).
        highlight: optional `(t0, t1)` in seconds — translucent yellow
            band drawn behind the rows.
        arrows: optional family-on/off toggle. Recognized keys are
            `"because_of"`, `"link_to"`, `"containment"`, and
            `"action_target"`. Missing keys default to `True`. Unknown keys
            are ignored.
        entity_kinds: optional whitelist of segment kinds to draw.
            Recognized values are `"env"`, `"light"`, `"object"`,
            `"agent"`, `"ego"`. `None` = all kinds.
        agent_ids: optional whitelist of `Agent.id` values. Only
            restricts segments whose kind is `"agent"`; segments of
            other kinds are untouched. `None` = all agents.
        track_groups: optional whitelist of category rows to render.
            Recognized values are the canonical category labels
            (`"Environments"`, `"Traffic Lights"`, `"Objects"`,
            `"Agents"`, `"Ego"`); the legacy short names (`"Env"`,
            `"Lights"`) are accepted as aliases for backward compat
            with PR-37 callers. Rows not in the whitelist drop their
            tick labels too, so the y-axis collapses to the visible
            categories. `None` = all five categories.
        families: optional whitelist of family names to render. The
            recognized values are the per-category family leaves —
            `"condition"`, `"containment"`, `"physical_containment"`,
            `"signal_head"`, `"env_control"`, `"state"`, `"pose"`,
            `"action"`, `"property"`. The removed `"influence"` leaf is
            accepted as a compatibility no-op. Parent rows
            (the entity-track headers like "Env Track 1" / "Agent
            Track 2") are NOT named in this whitelist — they auto-
            render for any entity that has at least one surviving
            non-parent segment, so the family filter never produces
            orphan headers. Entities with zero surviving sub-rows
            drop completely (their parent disappears too).
            `None` = all families. Unrecognized leaf names raise
            `ValueError` listing the recognized set — silent
            empty-timeline output on a typo (e.g. plural
            `"actions"` instead of `"action"`) is worse than a
            clear error.
        track_visibility: optional grouped visibility switches. Keys are
            ``"ego"``, ``"agent"``, ``"light"``, ``"object"``, and
            ``"env"``. A bool toggles the whole kind; a nested mapping
            toggles stable top-level entity IDs. For example,
            ``{"agent": {"agent_4": False}}`` hides that Agent and all
            of its sub-rows. Omitted switches default to visible.
        show_inline_labels: when False, suppress every inline label
            annotation; hover tooltips still fire. Defaults to True
            (the historical behaviour). Callers wanting a maximally
            compact timeline (very short clips, dashboard cards)
            pass `False` so the rectangles stay clean.
        inline_label_mode: private renderer policy. ``"compact"`` keeps the
            interactive timeline's width threshold and eight-character
            ellipsis. ``"full"`` emits the complete label for every visible
            segment and anchors edge labels toward the plot interior. The
            latter is used by publication figures, where hover-only text is
            not an acceptable fallback.

    Returns:
        A `PaintResult` describing the painted band stack — band keys
        in y-axis order, per-band lane counts, the y-coordinate range,
        and the total lane count. Callers use it to size the figure
        proportionally to the deepest stack per band (see
        `render_timeline` and `widget.ClipPlayer`'s adaptive-height
        path). The figure itself is also mutated in place.

    Filter semantics:
        - The four whitelists and visibility switches AND together: a
          segment is drawn only if its group is in `track_groups`, its
          kind is in
          `entity_kinds`, (for agent kinds) its agent id is in
          `agent_ids`, and its family is in `families` (or it is a
          `"parent"` segment for an entity whose sub-rows survived).
        - Arrow filtering follows segment filtering: an arrow whose
          source or target segment was filtered out is dropped.
    """
    import plotly.graph_objects as go
    bundle = seq.annotation
    duration = float(seq.duration_s) if seq.duration_s else 0.0
    axis_end = duration if duration > 0 else 1.0

    # Resolve the arrow-family toggles up front.
    enabled_arrows = dict(_DEFAULT_ARROWS)
    if arrows:
        for k, v in arrows.items():
            if k in enabled_arrows:
                enabled_arrows[k] = bool(v)

    # Resolve filter whitelists. `None` means "all", which we mirror by
    # building a set populated with every legal value.
    allowed_kinds: set[str] = (
        {"env", "light", "object", "agent", "ego"}
        if entity_kinds is None
        else set(entity_kinds)
    )
    # `track_groups` accepts both the legacy group names ("Env",
    # "Lights") and the new category labels ("Environments",
    # "Traffic Lights") so PR-37 callers keep working.
    _LEGACY_GROUP_ALIASES: dict[str, str] = {
        "Env": "Environments",
        "Lights": "Traffic Lights",
    }
    allowed_groups: set[str] = (
        set(_CATEGORIES) if track_groups is None
        else {_LEGACY_GROUP_ALIASES.get(g, g) for g in track_groups}
    )
    allowed_agent_ids: set[str] | None = (
        None if agent_ids is None else set(agent_ids)
    )
    kind_visibility, entity_visibility = _resolve_track_visibility(
        bundle, track_visibility
    )
    # `families` is a flat whitelist of family-leaf names (sub-rows).
    # Parents are never named here — they survive automatically for
    # any entity whose sub-rows pass the filter (see two-pass logic
    # below). `None` = no family filter.
    #
    # The y-tick labels show families in pluralized form ("actions",
    # "conditions", ...) but the kwarg takes the singular leaf name
    # ("action", "condition", ...). It's an easy mistake to copy the
    # visible label back into the kwarg, so unrecognized leaves raise
    # `ValueError` with the recognized list — silent empty-timeline
    # output is worse than a clear error.
    if families is not None:
        unknown = set(families) - _RECOGNIZED_FAMILIES
        if unknown:
            if "parent" in unknown:
                raise ValueError(
                    "'parent' is not a valid families filter entry — "
                    "parent rows auto-render for entities whose "
                    "sub-rows survive the filter. Recognized families: "
                    f"{sorted(_RECOGNIZED_FAMILIES)!r}"
                )
            raise ValueError(
                f"unknown family leaves: {sorted(unknown)!r}. "
                f"Recognized: {sorted(_RECOGNIZED_FAMILIES)!r}"
            )
    allowed_families: set[str] | None = (
        None if families is None else set(families)
    )

    shapes: list[dict[str, Any]] = []

    # Flatten the bundle via the public semantic segment adapter, then apply
    # the DevKit renderer's presentation policy. ``Influence`` remains in the
    # schema, query API, and ``annotation_to_segments`` output, but it no
    # longer consumes timeline rows or produces arrows. Empty bundles still
    # emit a shapes-free axis layout.
    segments = [
        segment
        for segment in annotation_to_segments(bundle)
        if segment.family != "influence"
    ]
    # ``annotation_to_segments`` creates the synthetic Ego parent whenever
    # Ego has any child, including Influence. If Influence was the only child,
    # the renderer-side removal above would otherwise leave a blank Ego row.
    if not any(
        segment.track_id == "ego_act" and segment.family != "parent"
        for segment in segments
    ):
        segments = [
            segment
            for segment in segments
            if not (
                segment.track_id == "ego_act" and segment.family == "parent"
            )
        ]

    # ---------------------------------------------------------------
    # Segment-level filter. Used both for the rectangle/label loop
    # and for the arrow-endpoint check.
    # ---------------------------------------------------------------
    ann_for_filter = bundle.annotation
    agent_id_by_index: dict[int, str] = {
        i: a.id for i, a in enumerate(ann_for_filter.agents) if a.id
    }

    def _segment_passes_non_family_filters(seg: Segment) -> bool:
        """Group / kind / agent-id checks. Used by both the family
        pre-pass (to find entities with surviving sub-rows) and the
        full `_segment_allowed` predicate."""
        group = _track_id_to_group(seg.track_id)
        if group is None or group not in allowed_groups:
            return False
        kind = _segment_kind(seg.track_id)
        if kind is None or kind not in allowed_kinds:
            return False
        if not kind_visibility.get(kind, True):
            return False
        owner_id = _segment_track_entity_id(seg, bundle, kind)
        if owner_id is not None and not entity_visibility.get(kind, {}).get(
            owner_id, True
        ):
            return False
        if kind == "agent" and allowed_agent_ids is not None:
            ai = (seg.meta or {}).get("_agentIndex")
            if not isinstance(ai, int):
                return False
            agent_id = agent_id_by_index.get(ai)
            if agent_id is None or agent_id not in allowed_agent_ids:
                return False
        return True

    # Pre-compute the set of `(category, entity_idx)` pairs that have
    # at least one surviving NON-parent segment under the family
    # filter. A parent row only renders if its entity's sub-rows
    # survive — otherwise we'd leave orphan section heads pointing
    # at nothing.
    surviving_entities: set[tuple[str, int]] = set()
    if allowed_families is not None:
        for s in segments:
            if s.family == "parent":
                continue
            if s.family not in allowed_families:
                continue
            if not _segment_passes_non_family_filters(s):
                continue
            cat = _track_id_to_category(s.track_id)
            if cat is None:
                continue
            ent_idx = _track_id_to_entity_idx(s.track_id)
            surviving_entities.add((cat, ent_idx))

    def _segment_allowed(seg: Segment) -> bool:
        if not _segment_passes_non_family_filters(seg):
            return False
        if allowed_families is None:
            return True
        if seg.family == "parent":
            cat = _track_id_to_category(seg.track_id)
            if cat is None:
                return False
            ent_idx = _track_id_to_entity_idx(seg.track_id)
            return (cat, ent_idx) in surviving_entities
        return seg.family in allowed_families

    # Cache the predicate result so arrow lookups can short-circuit
    # quickly when the source / target segment was filtered out.
    allowed_ids: set[str] = {s.id for s in segments if _segment_allowed(s)}

    # ---------------------------------------------------------------
    # 1a. Decide the populated bands BEFORE the highlight band so the
    #     yellow rect spans the actual y-range. Filter segments first
    #     so dropped families don't reserve y-axis space.
    # ---------------------------------------------------------------
    visible_segments = [s for s in segments if s.id in allowed_ids]
    band_keys: list[BandKey] = _populated_bands(visible_segments)
    band_row: dict[BandKey, int] = {key: i for i, key in enumerate(band_keys)}

    # ---------------------------------------------------------------
    # 1pre. Per-entity background bands. One translucent rect per
    #     `(category, entity_idx)` block, spanning all of that block's
    #     band rows.
    #
    #     `xref="paper"` (not the subplot's data x) so the band extends
    #     across the y-tick-label margin AND the bar region — that
    #     gives the annotator's "soft background spanning labels and
    #     boxes" look. y is still data coords so each band lines up
    #     with its block's row range.
    #
    #     Y-extent is inset to `[first_row - _ROW_HALF_HEIGHT,
    #     last_row + _ROW_HALF_HEIGHT]` (i.e. ±0.4 instead of ±0.5).
    #     The lane bands inside each row already span exactly
    #     ±_ROW_HALF_HEIGHT, so this tightens the entity block to
    #     hug the painted segments AND leaves a 0.2-row dark gap
    #     between adjacent blocks — the "visual break" between
    #     entities the annotator uses.
    #
    #     Painted FIRST (so it sits below the highlight band, segment
    #     rectangles, and arrows) with `layer: "below"`.
    # ---------------------------------------------------------------
    # Pick the right paper xref for the current subplot. Plotly
    # treats `"paper"` as figure-relative for the primary subplot;
    # for stacked subplots (the widget), each subplot's `xref`
    # already maps to its own axis. The entity-block intentionally
    # uses paper so it reaches into the y-tick-label margin — the
    # widget's bottom-subplot timeline uses `yref="y2"` but the same
    # `xref="paper"` still anchors the band to the figure's left
    # edge (which IS the label margin for the timeline subplot).
    for category, entity_idx, first_row, last_row in _entity_blocks(band_keys):
        shapes.append(
            {
                "type": "rect",
                "x0": _ENTITY_BLOCK_PAPER_X_LEFT,
                "x1": _ENTITY_BLOCK_PAPER_X_RIGHT,
                "y0": first_row - _ROW_HALF_HEIGHT,
                "y1": last_row + _ROW_HALF_HEIGHT,
                "xref": "paper",
                "yref": yref,
                "fillcolor": _entity_block_fill(category, entity_idx),
                "opacity": 1.0,
                "line": {"width": 0},
                "layer": "below",
                "name": f"entity_block:{category}:{entity_idx}",
            }
        )

    # ---------------------------------------------------------------
    # 1. Optional highlight band — drawn AFTER the entity blocks so it
    #    sits *on top of* them visually (the highlight should be
    #    recognizable as such, not absorbed by the row tint). Still
    #    `layer: "below"` so the segment rectangles remain on top.
    # ---------------------------------------------------------------
    if highlight is not None:
        h0, h1 = float(highlight[0]), float(highlight[1])
        if h0 > h1:
            h0, h1 = h1, h0
        highlight_y0 = -0.5
        highlight_y1 = (len(band_keys) - 0.5) if band_keys else 0.5
        shapes.append(
            {
                "type": "rect",
                "x0": h0,
                "x1": h1,
                "y0": highlight_y0,
                "y1": highlight_y1,
                "xref": xref,
                "yref": yref,
                "fillcolor": "yellow",
                "opacity": 0.15,
                "line": {"width": 0},
                "layer": "below",
                # Tag for tests / inspection. Plotly preserves unknown
                # keys at the top level on the Shape object, but to
                # stay strict we keep `name` (an officially supported
                # field) as the marker.
                "name": "highlight",
            }
        )

    # ---------------------------------------------------------------
    # 2. One rectangle per segment, stacked into sub-lanes inside its
    #    family band.
    #
    #    `assign_lanes` (the port of the annotator's `assignTracks`)
    #    still runs per `track_id`, but we scope the lane-grid to each
    #    `(group, family, sh_idx)` band so neighboring families don't
    #    inflate each other's vertical budget. Each band's lane count
    #    is the deepest stack among its own segments.
    # ---------------------------------------------------------------
    paintable: list[tuple[Segment, BandKey, str]] = []
    for seg in visible_segments:
        key = _segment_band_key(seg)
        if key is None:
            continue
        kind = _segment_kind(seg.track_id)
        if kind is None:
            continue
        paintable.append((seg, key, kind))

    # Lane assignment is scoped per band — the annotator's behavior is
    # that each labeled stripe ("containment", "actions", ...) owns its
    # own lane grid. Run `assign_lanes` once per band rather than once
    # globally so a deep stack in one band doesn't push other bands'
    # segments to higher lane indices.
    seg_lane: dict[str, int] = {}
    band_lane_count: dict[BandKey, int] = {key: 1 for key in band_keys}
    by_band: dict[BandKey, list[Segment]] = {}
    for seg, key, _ in paintable:
        by_band.setdefault(key, []).append(seg)
    for key, band_segments in by_band.items():
        sub_lanes = assign_lanes(band_segments)
        seg_lane.update(sub_lanes)
        if sub_lanes:
            band_lane_count[key] = max(band_lane_count[key], max(sub_lanes.values()) + 1)

    # Cache `(band_key, lane) -> (y0, y1)` for the segment loop and the
    # arrow-anchor loop. The band_row is its index in the populated
    # band list — empty bands were already dropped, so the y-axis
    # collapses to populated bands only.
    def _band_for(key: BandKey, lane: int) -> tuple[float, float]:
        return _lane_band(band_row[key], lane, band_lane_count[key])

    # Collect inline annotation entries (text drawn on top of each
    # rectangle) and hover-overlay traces (invisible scatter markers at
    # segment centres carrying a tooltip with the full label, window,
    # and track id). Plotly shapes don't support hover natively; an
    # overlay trace is the standard idiom.
    annotations: list[dict[str, Any]] = []
    hover_traces: list[go.Scatter] = []
    xa = _xref_to_axis(xref)
    ya = _yref_to_axis(yref)

    # Cache the seg-id → band key so the arrow loop can recover the
    # right band row even when the segment isn't in `paintable` (e.g.,
    # a containment arrow whose target is an Env parent on a different
    # band).
    seg_band: dict[str, BandKey] = {seg.id: key for seg, key, _ in paintable}

    for seg, key, _kind in paintable:
        lane = seg_lane.get(seg.id, 0)
        y0, y1 = _band_for(key, lane)
        # Per-family fill: each subtrack family has its own hue (see
        # `family_color`'s palette table, ported verbatim from the
        # annotator's `Timeline.tsx:990-1068`). Parent rows fall back
        # to the entity base color via `family_color`'s `entity_color`
        # fallback path.
        category = key[0]
        fill = family_color(category, seg.family)
        shapes.append(
            {
                "type": "rect",
                "x0": seg.t0,
                "x1": seg.t1,
                "y0": y0,
                "y1": y1,
                "xref": xref,
                "yref": yref,
                "fillcolor": fill,
                "opacity": 0.85,
                "line": {
                    "width": 1.5 if seg.illegal else 0.5,
                    "color": "#fbbf24" if seg.illegal else "#0f172a",
                },
                "layer": "above",
                "name": f"segment:{seg.id}",
            }
        )

        # Inline label. Compact interactive timelines suppress narrow boxes
        # and truncate text; publication figures keep every complete label
        # because static exports have no hover fallback. Full labels use an
        # inward-facing anchor so boxes at either time-axis edge do not send
        # their text outside the figure canvas.
        seg_width = max(seg.t1 - seg.t0, 0.0)
        min_inline_width = (
            _INLINE_LABEL_MIN_WIDTH_FRAC * duration if duration > 0 else 0.0
        )
        segment_t0 = min(seg.t0, seg.t1)
        segment_t1 = max(seg.t0, seg.t1)
        segment_intersects_axis = segment_t1 >= 0.0 and segment_t0 <= axis_end
        label_width_allowed = (
            inline_label_mode == "full" or seg_width >= min_inline_width
        )
        label_axis_allowed = (
            inline_label_mode != "full" or segment_intersects_axis
        )
        if show_inline_labels and label_width_allowed and label_axis_allowed:
            label_x = (seg.t0 + seg.t1) / 2.0
            label_text = _truncate_label(seg.label)
            full_label_layout: dict[str, Any] = {}
            if inline_label_mode == "full":
                label_text = seg.label
                visible_t0 = min(max(segment_t0, 0.0), axis_end)
                visible_t1 = min(max(segment_t1, 0.0), axis_end)
                if (visible_t0 + visible_t1) / 2.0 <= axis_end / 2.0:
                    label_x = visible_t0
                    full_label_layout = {
                        "xanchor": "left",
                        "xshift": 2,
                        "align": "left",
                    }
                else:
                    label_x = visible_t1
                    full_label_layout = {
                        "xanchor": "right",
                        "xshift": -2,
                        "align": "right",
                    }
            annotations.append(
                {
                    "x": label_x,
                    "y": (y0 + y1) / 2.0,
                    "xref": xref,
                    "yref": yref,
                    "text": label_text,
                    "showarrow": False,
                    "font": {"size": _INLINE_LABEL_FONT_SIZE, "color": "#0f172a"},
                    "name": f"label:{seg.id}",
                    **full_label_layout,
                }
            )

        # Hover overlay — invisible scatter marker at the segment
        # midpoint. `opacity=0` keeps it from drawing, `size=20` makes
        # the hover hitbox generous, and `hovertext` carries the full
        # tooltip body (label / window / track id).
        hover_traces.append(
            go.Scatter(
                x=[(seg.t0 + seg.t1) / 2.0],
                y=[(y0 + y1) / 2.0],
                xaxis=xa,
                yaxis=ya,
                mode="markers",
                marker={"size": 20, "opacity": 0, "color": fill},
                hoverinfo="text",
                hovertext=(
                    f"{seg.label}<br>"
                    f"[{seg.t0:.2f}s – {seg.t1:.2f}s]<br>"
                    f"track: {seg.track_id}"
                ),
                showlegend=False,
                name=f"hover:segment:{seg.id}",
            )
        )

    # ---------------------------------------------------------------
    # 3. Arrows. For each family that's enabled, walk the relevant
    #    segments and emit a bezier `path` shape. The annotator
    #    resolves arrow endpoints against time-overlapping segments;
    #    we keep that simpler here — we look up the by-id entity
    #    segment, and if not found, drop the arrow rather than
    #    crash.
    # ---------------------------------------------------------------
    by_id = _segments_by_entity_id(bundle, segments)
    ann = bundle.annotation

    def _add_arrow(
        src: Segment | None,
        tgt: Segment | None,
        family: str,
    ) -> None:
        if src is None or tgt is None:
            return
        # Drop the arrow if either endpoint was filtered out (per
        # `entity_kinds` / `agent_ids` / `track_groups`). Symmetric:
        # both ends must be present for the bezier to make sense.
        if src.id not in allowed_ids or tgt.id not in allowed_ids:
            return
        src_key = seg_band.get(src.id)
        tgt_key = seg_band.get(tgt.id)
        if src_key is None or tgt_key is None:
            return
        x0 = _segment_center_x(src)
        x1 = _segment_center_x(tgt)
        # Anchor each end of the bezier to its segment's band+lane so
        # multi-lane stacks don't collapse arrows onto the same edge.
        # With family bands, the band_row is its index in the populated
        # band list — the arrow's y follows the segment to its NEW band
        # rather than the legacy whole-group row center.
        src_lane = seg_lane.get(src.id, 0)
        tgt_lane = seg_lane.get(tgt.id, 0)
        y0 = _segment_top_y(band_row[src_key], src_lane, band_lane_count[src_key])
        y1 = _segment_top_y(band_row[tgt_key], tgt_lane, band_lane_count[tgt_key])
        shapes.append(
            {
                "type": "path",
                "path": _bezier_path(x0, y0, x1, y1),
                "xref": xref,
                "yref": yref,
                "line": {"color": _ARROW_COLORS[family], "width": 2},
                "fillcolor": "rgba(0,0,0,0)",
                "layer": "above",
                "name": f"arrow:{family}:{src.id}->{tgt.id}",
            }
        )

        # Arrowhead at the target end. Plotly's SVG `path` shape doesn't
        # render arrowheads, so we add a rotated triangle marker via a
        # `go.Scatter` trace. The angle is computed from the bezier's
        # tangent at t=1, which for a quadratic curve through control
        # point `(mid_x, ctrl_y)` is `(x1 - mid_x, y1 - ctrl_y)` in
        # data coords. With the reversed y-axis, flip dy when feeding
        # `atan2` so the triangle points along the on-screen direction.
        import math

        mid_x = (x0 + x1) / 2.0
        ctrl_y = max(y0, y1) + 0.8
        dx = x1 - mid_x
        dy = y1 - ctrl_y
        # Convert tangent vector to a Plotly `marker.angle` (degrees
        # clockwise from "north", which is +y in screen space).
        # Screen-y points "up" in Plotly's default but the timeline
        # uses a reversed yaxis, so the on-screen y direction is `-dy`
        # in data coords.
        screen_dy = -dy
        angle_rad = math.atan2(dx, screen_dy)
        angle_deg = math.degrees(angle_rad)
        hover_traces.append(
            go.Scatter(
                x=[x1],
                y=[y1],
                xaxis=xa,
                yaxis=ya,
                mode="markers",
                marker={
                    "symbol": "triangle-up",
                    "size": 10,
                    "angle": angle_deg,
                    "color": _ARROW_COLORS[family],
                    # Tailwind slate-900 outline — pops off any
                    # same-hue target row (containment-green over
                    # Env, action-target orange over Agents, etc.) on
                    # the `plotly_white` background. Family color
                    # stays as the fill so the head is still
                    # identifiable by hue.
                    "line": {"color": "#0f172a", "width": 1.5},
                },
                hoverinfo="text",
                hovertext=(
                    f"{family}<br>"
                    f"{src.label}<br>"
                    f"  → {tgt.label}"
                ),
                showlegend=False,
                name=f"arrowhead:{family}:{src.id}->{tgt.id}",
            )
        )

        # Mid-arc hover hotspot — invisible marker at the bezier
        # midpoint so the bezier line itself is hoverable end-to-end
        # rather than only at the arrowhead.
        bezier_mid_x = 0.25 * x0 + 0.5 * mid_x + 0.25 * x1
        bezier_mid_y = 0.25 * y0 + 0.5 * ctrl_y + 0.25 * y1
        hover_traces.append(
            go.Scatter(
                x=[bezier_mid_x],
                y=[bezier_mid_y],
                xaxis=xa,
                yaxis=ya,
                mode="markers",
                marker={"size": 18, "opacity": 0, "color": _ARROW_COLORS[family]},
                hoverinfo="text",
                hovertext=(
                    f"{family}<br>"
                    f"{src.label}<br>"
                    f"  → {tgt.label}"
                ),
                showlegend=False,
                name=f"hover:arrow:{family}:{src.id}->{tgt.id}",
            )
        )

    # 3a. because_of — read straight off the Segment dataclass.
    if enabled_arrows["because_of"]:
        for seg in segments:
            for target_id in seg.because_of:
                _add_arrow(seg, by_id.get(target_id), "because_of")

    # 3b. link_to — walk the bundle's actions and (signaling) properties.
    if enabled_arrows["link_to"]:
        # Ego actions.
        for i, act in enumerate(ann.ego_vehicle.actions):
            src = _find_seg(
                segments,
                id_starts_with="ego_act_",
                meta_key="_egoActIndex",
                meta_val=i,
            )
            for target_id in act.link_to:
                _add_arrow(src, by_id.get(target_id), "link_to")
        # Agent actions.
        for ai, agent in enumerate(ann.agents):
            for act_idx, act in enumerate(agent.actions):
                src = _find_seg(
                    segments,
                    id_starts_with=f"agent_action_{ai}_{act_idx}_",
                )
                for target_id in act.link_to:
                    _add_arrow(src, by_id.get(target_id), "link_to")
        # Ego signaling properties.
        for pi, prop in enumerate(ann.ego_vehicle.properties):
            sd = prop.signaling_details
            if sd is None:
                continue
            src = _find_seg(segments, id_starts_with=f"ego_prop_{pi}_")
            for target_id in sd.link_to:
                _add_arrow(src, by_id.get(target_id), "link_to")
        # Agent signaling properties.
        for ai, agent in enumerate(ann.agents):
            for pi, prop in enumerate(agent.properties):
                sd = prop.signaling_details
                if sd is None:
                    continue
                src = _find_seg(segments, id_starts_with=f"agent_prop_{ai}_{pi}_")
                for target_id in sd.link_to:
                    _add_arrow(src, by_id.get(target_id), "link_to")

    # 3c. containment — the annotator draws an arrow from a containment
    #     subtrack to its parent environment.
    if enabled_arrows["containment"]:
        # Ego containment.
        for ci, cont in enumerate(ann.ego_vehicle.containment):
            src = _find_seg(segments, id_starts_with=f"ego_cont_{ci}_")
            _add_arrow(src, by_id.get(cont.env_id), "containment")
        # Agent containment.
        for ai, agent in enumerate(ann.agents):
            for ci, cont in enumerate(agent.containment):
                src = _find_seg(segments, id_starts_with=f"agent_cont_{ai}_{ci}_")
                _add_arrow(src, by_id.get(cont.env_id), "containment")
        # Traffic-object containment.
        for oi, obj in enumerate(ann.traffic_objects):
            for ci, cont in enumerate(obj.containment):
                src = _find_seg(segments, id_starts_with=f"obj_cont_{oi}_{ci}_")
                _add_arrow(src, by_id.get(cont.env_id), "containment")
        # Traffic-light physical containment.
        for li, light in enumerate(ann.traffic_lights):
            for ci, cont in enumerate(light.containment):
                src = _find_seg(segments, id_starts_with=f"light_phys_cont_{li}_{ci}_")
                _add_arrow(src, by_id.get(cont.env_id), "containment")
        # Signal-head env_controlled.
        for li, light in enumerate(ann.traffic_lights):
            for hi, sh in enumerate(light.signal_heads):
                for ci, cont in enumerate(sh.env_controlled):
                    src = _find_seg(
                        segments,
                        id_starts_with=f"light_cont_{li}_{hi}_{ci}_",
                    )
                    _add_arrow(src, by_id.get(cont.env_id), "containment")

    # 3d. action_target — actions only (not properties).
    if enabled_arrows["action_target"]:
        for i, act in enumerate(ann.ego_vehicle.actions):
            src = _find_seg(
                segments,
                id_starts_with="ego_act_",
                meta_key="_egoActIndex",
                meta_val=i,
            )
            for target_id in act.action_target:
                _add_arrow(src, by_id.get(target_id), "action_target")
        for ai, agent in enumerate(ann.agents):
            for act_idx, act in enumerate(agent.actions):
                src = _find_seg(
                    segments,
                    id_starts_with=f"agent_action_{ai}_{act_idx}_",
                )
                for target_id in act.action_target:
                    _add_arrow(src, by_id.get(target_id), "action_target")

    # ---------------------------------------------------------------
    # 4. Axis layout — one tick per populated band, labeled
    #    "Group · family" (or "Group · sh{i} · family" for per-head
    #    light bands). Figure-wide settings (template / height /
    #    margins) are the caller's responsibility so this helper
    #    composes inside a multi-subplot figure without overwriting
    #    the host's chrome.
    #
    #    Append the new shapes after any shapes already on `fig` so
    #    callers can pre-paint playheads or other overlays without
    #    losing them.
    # ---------------------------------------------------------------
    if band_keys:
        ticks_vals = [band_row[k] for k in band_keys]
        # Per-row alignment via trailing non-breaking-space padding.
        # SVG `<text>` anchors at the end (y-axis line), so trailing
        # nbsp's push the visible portion leftward.
        #
        # `pad_to` is intentionally WIDER than the longest plain
        # label by `_PAD_COLUMN_EXTRA` chars so:
        #   - The longest parent still gets some trailing nbsp's
        #     (its visible right edge dents inward from the axis,
        #     reading as "centered" rather than "right-aligned").
        #   - Sub-rows pad even further past the longest label, so
        #     their visible left edge sits clearly left of any
        #     parent's left edge — reading as "left-aligned".
        # Without this extra width, the longest parent (when a
        # parent happens to be the widest label) would tie with the
        # axis line and visually right-align.
        plain_texts = [_band_label_text(k) for k in band_keys]
        max_label = max((len(t) for t in plain_texts if t), default=0)
        pad_to = max_label + _PAD_COLUMN_EXTRA if max_label > 0 else 0
        ticks_text = [_band_label_html(k, pad_to=pad_to) for k in band_keys]
        y_min = -0.5
        y_max = len(band_keys) - 0.5
    else:
        # No populated bands — caller filtered everything out, or the
        # bundle has no entities. Keep something sane so Plotly still
        # renders a frame; tests pin both the zero-band and empty-bundle
        # paths.
        ticks_vals = []
        ticks_text = []
        y_min = -0.5
        y_max = 0.5

    existing_shapes = list(fig.layout.shapes or ())
    existing_annotations = list(fig.layout.annotations or ())
    fig.update_layout(
        shapes=existing_shapes + shapes,
        annotations=existing_annotations + annotations,
        **{
            xaxis_key: {
                "title": "Time (s)",
                "range": [0, axis_end],
                "showgrid": False,
                "zeroline": False,
            },
            yaxis_key: {
                "tickmode": "array",
                "tickvals": ticks_vals,
                "ticktext": ticks_text,
                # Monospace fontstack — the nbsp-based left-alignment
                # in `_band_label_html` only produces visually aligned
                # columns if every character has the same advance
                # width. Plotly's default `Open Sans` is proportional,
                # so two labels with the same total char count land
                # at different pixel widths (a string of 'l's is far
                # narrower than a string of 'M's). Switching to
                # monospace makes char-count → pixel-width linear.
                "tickfont": {
                    "size": 10,
                    "family": (
                        "ui-monospace, SFMono-Regular, "
                        "Menlo, Consolas, monospace"
                    ),
                },
                "range": [y_max, y_min],  # top → bottom
                "showgrid": False,
                "zeroline": False,
            },
        },
    )

    # Append the hover-overlay traces last so any pre-existing traces
    # (e.g. the widget's video-frame Image at index 0) keep their
    # positions in `fig.data`.
    for trace in hover_traces:
        fig.add_trace(trace)

    total_lanes = sum(band_lane_count.values()) if band_lane_count else 0
    return PaintResult(
        bands=tuple(band_keys),
        band_lane_counts=dict(band_lane_count),
        y_range=(y_min, y_max),
        total_lanes=total_lanes,
    )


# Vertical pixels per lane band for the adaptive-height path. Drives
# both `render_timeline`'s `height` and `ClipPlayer`'s `row_heights`
# split. Tuned so a label rendered at `_INLINE_LABEL_FONT_SIZE` reads
# cleanly without clipping the row above/below.
_PX_PER_LANE: int = 24

# Minimum total timeline height in pixels. Even a sparse band stack
# needs some breathing room; this is the floor before band/lane
# stacking grows the figure further.
_MIN_TIMELINE_PX: int = 240


def _timeline_px_for(paint: "PaintResult | dict[str, int]") -> int:
    """Compute the adaptive timeline height in pixels.

    Accepts either the modern `PaintResult` or the legacy
    `dict[str, int]` per-group lane count (kept for backwards
    compatibility with any out-of-tree caller wired to the old return
    type). Internally everything sums to a total lane count, multiplies
    by `_PX_PER_LANE`, and clamps to `_MIN_TIMELINE_PX`.
    """
    if isinstance(paint, PaintResult):
        total_lanes = paint.total_lanes or 1
    else:
        total_lanes = sum(paint.values()) or 1
    return max(_MIN_TIMELINE_PX, total_lanes * _PX_PER_LANE)


def render_timeline(
    seq: "Sequence",
    *,
    highlight: tuple[float, float] | None = None,
    arrows: dict[str, bool] | None = None,
    entity_kinds: list[str] | None = None,
    agent_ids: list[str] | None = None,
    track_groups: list[str] | None = None,
    families: list[str] | None = None,
    track_visibility: TrackVisibility | None = None,
    height: int | None = None,
    show_inline_labels: bool = True,
) -> "go.Figure":
    """Return a Plotly Figure showing the clip's annotation timeline.

    Rows = track groups (Ego, Agents, Traffic Lights, Objects, Environments);
    each segment is a colored rectangle via `go.layout.Shape`.
    Causal arrows are drawn as bezier `path` shapes between segment
    centers. `highlight=(t0, t1)` paints a translucent vertical band.
    `arrows={"because_of": True, ...}` toggles arrow families;
    default all-on. The Figure is fully static (no traitlets); the
    widget-driving variant lives in `viz.widget.ClipPlayer` and reuses
    the same `_paint_timeline_onto` helper.

    Args:
        seq: a `Sequence` (typically from `CascadeDataset.get_sequence`
            or `Sequence.from_annotation`). Only `seq.annotation` and
            `seq.duration_s` are read — no video / parent dataset
            access, so this is safe in any environment.
        highlight: optional `(t0, t1)` in seconds. When set, a single
            translucent yellow vrect is drawn behind the rows.
        arrows: optional family-on/off toggle. Recognized keys are
            `"because_of"`, `"link_to"`, `"containment"`, and
            `"action_target"`. Missing keys default to `True`. Unknown keys
            are ignored.
        entity_kinds: optional kind whitelist (`"env"`, `"light"`,
            `"object"`, `"agent"`, `"ego"`). `None` = all kinds.
        agent_ids: optional `Agent.id` whitelist. Restricts only the
            `"agent"` kind. `None` = all agents.
        track_groups: optional category whitelist (`"Environments"`,
            `"Traffic Lights"`, `"Objects"`, `"Agents"`, `"Ego"`); the
            legacy short names `"Env"` and `"Lights"` are accepted as
            aliases. Rows not in the whitelist drop from the y-axis
            layout. `None` = all categories.
        families: optional whitelist of family leaves to render
            (`"condition"`, `"containment"`, `"physical_containment"`,
            `"signal_head"`, `"env_control"`, `"state"`, `"pose"`,
            `"action"`, `"property"`). The removed `"influence"` leaf is
            accepted as a compatibility no-op. Parent entity
            headers auto-render for any entity whose sub-rows survive
            — entities with zero surviving sub-rows drop completely.
            `None` = all families. Unrecognized leaves (e.g. the
            plural form `"actions"` accidentally copied from a tick
            label) raise `ValueError`.
        track_visibility: optional grouped switches for whole kinds or
            individual stable entity IDs. For example,
            ``{"agent": {"agent_4": False, "agent_5": False}}`` hides
            those two Agent tracks while leaving unspecified entities on.
            Child rows inherit their top-level entity's switch, and arrows
            with a hidden endpoint are omitted.
        height: optional explicit pixel height. `None` (default) means
            adaptive — the height tracks the deepest sub-lane stack so
            a busy clip gets a taller timeline while a sparse one
            stays compact. Pass an int to pin a specific value; useful
            when embedding the figure in a fixed-size dashboard cell.
        show_inline_labels: when False, suppress every inline label
            annotation; hover tooltips still fire. Defaults to True
            (the historical behaviour). Useful when callers want a
            maximally clean timeline for screenshots / dense clips.

    Returns:
        A `plotly.graph_objects.Figure`. The visible rectangles, arrows,
        and inline labels live in `layout.shapes` + `layout.annotations`
        (so the widget can replace just the playhead line without
        touching the rest of the figure). The Figure also carries one
        invisible `go.Scatter` trace per segment/arrow purely to
        provide hover tooltips — Plotly shapes don't support hover
        natively.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    paint = _paint_timeline_onto(
        fig,
        seq,
        xref="x",
        yref="y",
        xaxis_key="xaxis",
        yaxis_key="yaxis",
        highlight=highlight,
        arrows=arrows,
        entity_kinds=entity_kinds,
        agent_ids=agent_ids,
        track_groups=track_groups,
        families=families,
        track_visibility=track_visibility,
        show_inline_labels=show_inline_labels,
    )
    resolved_height = (
        int(height) if height is not None else _timeline_px_for(paint)
    )
    fig.update_layout(
        template="plotly_white",
        height=resolved_height,
        margin={"l": 80, "r": 20, "t": 20, "b": 40},
    )
    return fig


def _find_seg(
    segments: list[Segment],
    *,
    id_starts_with: str,
    meta_key: str | None = None,
    meta_val: Any = None,
) -> Segment | None:
    """Find the first segment whose id starts with `id_starts_with`.

    For the ego-action case we also disambiguate by `meta[_egoActIndex]`
    because all ego-action segment ids share the `ego_act_` prefix (the
    nonce is the only differentiator — see `annotation_to_segments`).
    """
    for seg in segments:
        if not seg.id.startswith(id_starts_with):
            continue
        if meta_key is not None:
            if not seg.meta or seg.meta.get(meta_key) != meta_val:
                continue
        return seg
    return None


__all__ = ["TrackVisibility", "render_timeline"]
