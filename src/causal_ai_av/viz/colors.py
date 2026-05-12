# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entity-kind color palette — Python mirror of the annotator's CSS tokens.

The annotator's dark palette is defined as `--color-entity-*` CSS custom
properties in `tools/annotator/web/src/index.css`. To keep the headless and
widget renderers visually identical to the annotator, this module returns the
exact same hex strings.

Only the five base entity kinds are exposed here — `env`, `ego`, `object`,
`agent`, `light`. The annotator defines additional auxiliary tokens (`signal`,
`link`, `contained`, `influence`, `target`, `agent-2`) that are reserved for
arrow / relation rendering in later PRs.
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


__all__ = ["entity_color"]
