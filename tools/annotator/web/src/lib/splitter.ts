// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * Pure splitter math for the video/timeline divider. Kept independent of the
 * DOM so the drag math is unit-testable without spinning up a browser.
 *
 * The drag tracks the cursor by *delta* from mousedown — never by absolute
 * cursor position — so the splitter follows the cursor exactly regardless of
 * where in the hit-bar the user grabbed it, and regardless of any siblings
 * (e.g. the LockBar) sitting above the resizable region.
 */

export interface SplitterDragState {
  /** Cursor clientY at mousedown. */
  startY: number
  /** Video-pane height as a percentage of the container at mousedown. */
  startPct: number
  /** Container height in pixels at mousedown (captured once, treated as stable for the drag). */
  containerHeight: number
}

export interface SplitterBounds {
  /** Minimum video-pane percentage (inclusive). */
  minPct: number
  /** Maximum video-pane percentage (inclusive). */
  maxPct: number
}

/**
 * Given the mousedown snapshot and the current cursor position, return the
 * clamped video-pane percentage. Returns `startPct` unchanged when the
 * container height is non-positive (defensive guard for the unmounted case).
 */
export function nextVideoPct(
  state: SplitterDragState,
  currentY: number,
  bounds: SplitterBounds,
): number {
  if (state.containerHeight <= 0) return state.startPct
  const raw = state.startPct + ((currentY - state.startY) / state.containerHeight) * 100
  return Math.max(bounds.minPct, Math.min(bounds.maxPct, raw))
}
