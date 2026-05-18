// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * Pure track-layout primitives extracted from `components/Timeline.tsx`.
 *
 * The canvas painter and every hit-test in the timeline must agree on
 * where each track row lives on the Y axis. Historically the painter
 * used `getTrackY()` while four duplicated row-find loops used their own
 * `cumY = HEADER_HEIGHT` initialisation, missing the
 * `CATEGORY_HEADER_HEIGHT` slot reserved for the first group header pill
 * — which meant clicks in the bottom 24 px of every track row landed in
 * the next track's hit range and the last subtrack of each track was
 * effectively unclickable.
 *
 * Collecting the layout in one module keeps the painter and the hit-test
 * in lockstep and makes the geometry directly unit-testable.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const TRACK_HEIGHT = 32
export const ENV_TRACK_HEIGHT = 50
export const EGO_TRACK_HEIGHT = 42
export const EGO_SUBTRACK_BASE = 24
export const AGENT_TRACK_HEIGHT = 84
export const HEADER_HEIGHT = 36
export const SCRUBBER_LANE_HEIGHT = 16
export const LABEL_PANEL_WIDTH = 140
export const COND_SUBTRACK_HEIGHT = 18
export const ENV_MAIN_HEIGHT = 30
export const AGENT_SUBTRACK_BASE = ENV_MAIN_HEIGHT
export const OBJ_CONT_ROW_Y = ENV_MAIN_HEIGHT
export const SH_GAP = 4
export const COLLAPSED_GROUP_HEIGHT = 24
/**
 * Height reserved above the first non-collapsed group's first track for
 * the category header pill ("AGENTS", "OBJECTS", ...). The same amount
 * is folded into the inter-group `getTrackGap()` for every subsequent
 * group transition.
 */
export const CATEGORY_HEADER_HEIGHT = 24
export const GROUP_GAP = 6
export const TRACK_GAP = 4
export const BOTTOM_PADDING = 40

// ---------------------------------------------------------------------------
// Object-track row layout
// ---------------------------------------------------------------------------

/** Y of the object state row for a given containment-row count (min 1). */
export const getObjStateRowY = (contRows: number) =>
  OBJ_CONT_ROW_Y + Math.max(1, contRows) * COND_SUBTRACK_HEIGHT

/** Total height of an object track for a given containment-row count. */
export const getObjTrackHeight = (contRows: number) =>
  getObjStateRowY(contRows) + COND_SUBTRACK_HEIGHT

export const OBJ_LIGHT_TRACK_HEIGHT = getObjTrackHeight(1)

// ---------------------------------------------------------------------------
// Per-track row item types
// ---------------------------------------------------------------------------

export interface TrackRow {
  group: string
  id: string
  _collapsed?: boolean
}

// ---------------------------------------------------------------------------
// Track height + inter-track gap
// ---------------------------------------------------------------------------

export function getTrackHeight(
  group: string,
  trackId?: string,
  lightSubRowCounts?: Map<string, number>,
  collapsed?: boolean,
): number {
  if (collapsed) return COLLAPSED_GROUP_HEIGHT
  if (group === 'Agents') {
    if (trackId && lightSubRowCounts) {
      const contRows = lightSubRowCounts.get(trackId) ?? 1
      const inflRows = lightSubRowCounts.get(`${trackId}_infl`) ?? 0
      const propRows = lightSubRowCounts.get(`${trackId}_prop`) ?? 0
      return AGENT_SUBTRACK_BASE +
        (contRows + inflRows + 1 + propRows) * COND_SUBTRACK_HEIGHT +
        COND_SUBTRACK_HEIGHT
    }
    return AGENT_TRACK_HEIGHT
  }
  if (group === 'Ego') {
    if (lightSubRowCounts) {
      const contRows = lightSubRowCounts.get('ego_act') ?? 1
      const inflRows = lightSubRowCounts.get('ego_act_infl') ?? 0
      const propRows = lightSubRowCounts.get('ego_act_prop') ?? 0
      return EGO_SUBTRACK_BASE +
        contRows * COND_SUBTRACK_HEIGHT +
        inflRows * COND_SUBTRACK_HEIGHT +
        COND_SUBTRACK_HEIGHT +
        propRows * COND_SUBTRACK_HEIGHT
    }
    return EGO_TRACK_HEIGHT
  }
  if (group === 'Environments') {
    if (trackId && lightSubRowCounts) {
      const condRows = lightSubRowCounts.get(`${trackId}_cond`) ?? 1
      return ENV_MAIN_HEIGHT + condRows * COND_SUBTRACK_HEIGHT + 2
    }
    return ENV_TRACK_HEIGHT
  }
  if (group === 'TrafficLights' && trackId && lightSubRowCounts) {
    const rows = lightSubRowCounts.get(trackId) ?? 2
    const shCount = lightSubRowCounts.get(`${trackId}_shcount`) ?? 0
    return ENV_MAIN_HEIGHT + rows * COND_SUBTRACK_HEIGHT + shCount * SH_GAP
  }
  if (group === 'Objects') {
    if (trackId && lightSubRowCounts) {
      const contRows = lightSubRowCounts.get(trackId) ?? 1
      return getObjTrackHeight(contRows)
    }
    return OBJ_LIGHT_TRACK_HEIGHT
  }
  if (group === 'TrafficLights') return OBJ_LIGHT_TRACK_HEIGHT
  return TRACK_HEIGHT
}

/**
 * Gap *after* row `i`. Returns:
 *   - 0 for the last row
 *   - `GROUP_GAP + CATEGORY_HEADER_HEIGHT` between groups when the next
 *     track is non-collapsed (the gap also hosts the next group's header)
 *   - `GROUP_GAP` between groups when the next track is collapsed
 *   - `TRACK_GAP` for adjacent same-group non-collapsed tracks
 *   - 0 within a collapsed run
 */
export function getTrackGap(trackList: TrackRow[], i: number): number {
  if (i + 1 >= trackList.length) return 0
  if (trackList[i].group !== trackList[i + 1].group) {
    return GROUP_GAP + (trackList[i + 1]._collapsed ? 0 : CATEGORY_HEADER_HEIGHT)
  }
  if (!trackList[i]._collapsed && trackList[i].group === trackList[i + 1].group) return TRACK_GAP
  return 0
}

// ---------------------------------------------------------------------------
// Initial track Y + row finder
// ---------------------------------------------------------------------------

/**
 * Y at which the first track row starts. Includes the
 * `CATEGORY_HEADER_HEIGHT` slot when the first group is non-collapsed —
 * this is the slot the painter reserves for the first group's header
 * pill, and the row-finder must reserve the same slot or hit-test will
 * drift up by 24 px relative to the canvas.
 *
 * Exported for tests; outside callers should prefer `getTrackY` /
 * `findTrackAtY` so they cannot diverge from each other.
 */
export function _initialTrackY(trackList: TrackRow[]): number {
  let y = HEADER_HEIGHT
  if (trackList.length > 0 && !trackList[0]._collapsed) y += CATEGORY_HEADER_HEIGHT
  return y
}

/** Y of the top of row `rowIdx`. Used by the canvas painter. */
export function getTrackY(
  trackList: TrackRow[],
  rowIdx: number,
  lightLaneCounts?: Map<string, number>,
): number {
  let y = _initialTrackY(trackList)
  for (let i = 0; i < rowIdx; i++) {
    y += getTrackHeight(trackList[i].group, trackList[i].id, lightLaneCounts, trackList[i]._collapsed)
    y += getTrackGap(trackList, i)
  }
  return y
}

/**
 * Return the row index that contains canvas Y `my`, or `-1` if `my` is
 * above the first track, below the last, or in an inter-track gap.
 *
 * Always agrees with `getTrackY` by construction: both start from
 * `_initialTrackY` and walk the same `getTrackHeight` + `getTrackGap`
 * progression.
 */
export function findTrackAtY(
  my: number,
  trackList: TrackRow[],
  lightLaneCounts?: Map<string, number>,
): number {
  let cumY = _initialTrackY(trackList)
  for (let i = 0; i < trackList.length; i++) {
    const h = getTrackHeight(trackList[i].group, trackList[i].id, lightLaneCounts, trackList[i]._collapsed)
    if (my >= cumY && my < cumY + h) return i
    cumY += h + getTrackGap(trackList, i)
  }
  return -1
}
