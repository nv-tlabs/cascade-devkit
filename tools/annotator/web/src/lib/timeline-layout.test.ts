// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { describe, expect, it } from 'vitest'

import {
  CATEGORY_HEADER_HEIGHT,
  COND_SUBTRACK_HEIGHT,
  ENV_MAIN_HEIGHT,
  HEADER_HEIGHT,
  TRACK_GAP,
  _initialTrackY,
  findTrackAtY,
  getTrackGap,
  getTrackHeight,
  getTrackY,
} from './timeline-layout'
import type { TrackRow } from './timeline-layout'

// ---------------------------------------------------------------------------
// _initialTrackY
// ---------------------------------------------------------------------------

describe('_initialTrackY', () => {
  it('reserves the category-header slot above the first non-collapsed track', () => {
    const list: TrackRow[] = [{ group: 'Environments', id: 'env_0' }]
    expect(_initialTrackY(list)).toBe(HEADER_HEIGHT + CATEGORY_HEADER_HEIGHT)
  })

  it('omits the category-header slot when the first track is collapsed', () => {
    const list: TrackRow[] = [{ group: 'Environments', id: 'env_0', _collapsed: true }]
    expect(_initialTrackY(list)).toBe(HEADER_HEIGHT)
  })

  it('falls back to header-only on an empty list', () => {
    expect(_initialTrackY([])).toBe(HEADER_HEIGHT)
  })
})

// ---------------------------------------------------------------------------
// getTrackY and findTrackAtY must agree by construction.
//
// This is the bug-regression contract: prior to #PR the painter's
// `getTrackY` reserved `CATEGORY_HEADER_HEIGHT` for the first group
// header but the four row-find loops did not, causing every track's hit
// range to drift 24 px above the painted bars and the visible last
// subtrack of each track to be unclickable.
// ---------------------------------------------------------------------------

describe('getTrackY vs findTrackAtY', () => {
  const oneEnvList: TrackRow[] = [{ group: 'Environments', id: 'env_0' }]
  const oneEnvLanes = new Map<string, number>([['env_0_cond', 1]])

  it('returns the same Y for the painter and for the hit-test on a single env track', () => {
    const y0 = getTrackY(oneEnvList, 0, oneEnvLanes)
    expect(y0).toBe(_initialTrackY(oneEnvList))
    expect(findTrackAtY(y0, oneEnvList, oneEnvLanes)).toBe(0)
  })

  it('finds the right row at every interior pixel of every track', () => {
    const list: TrackRow[] = [
      { group: 'Environments', id: 'env_0' },
      { group: 'Environments', id: 'env_1' },
      { group: 'Objects', id: 'obj_0' },
      { group: 'Agents', id: 'agent_0' },
    ]
    const lanes = new Map<string, number>([
      ['env_0_cond', 1],
      ['env_1_cond', 2],
      ['obj_0', 1],
      ['agent_0', 1],
      ['agent_0_infl', 1],
      ['agent_0_prop', 1],
    ])
    for (let rowIdx = 0; rowIdx < list.length; rowIdx++) {
      const top = getTrackY(list, rowIdx, lanes)
      const h = getTrackHeight(list[rowIdx].group, list[rowIdx].id, lanes, list[rowIdx]._collapsed)
      // Top, middle, and the very last interior pixel must all resolve
      // to this row.
      expect(findTrackAtY(top, list, lanes)).toBe(rowIdx)
      expect(findTrackAtY(top + Math.floor(h / 2), list, lanes)).toBe(rowIdx)
      expect(findTrackAtY(top + h - 1, list, lanes)).toBe(rowIdx)
    }
  })

  it('finds the last subtrack pixel of an Agents track (regression for off-by-24)', () => {
    const list: TrackRow[] = [
      { group: 'Environments', id: 'env_0' },
      { group: 'Agents', id: 'agent_0' },
    ]
    // Two property rows on the agent so its bottom region carries
    // visible content that pre-fix would not have been clickable.
    const lanes = new Map<string, number>([
      ['env_0_cond', 1],
      ['agent_0', 1],
      ['agent_0_infl', 1],
      ['agent_0_prop', 2],
    ])
    const agentTop = getTrackY(list, 1, lanes)
    const agentHeight = getTrackHeight('Agents', 'agent_0', lanes, false)
    // The last property row sits in the bottom COND_SUBTRACK_HEIGHT px
    // of the agent track — these are the pixels that were unclickable
    // before the fix.
    const lastSubtrackTop = agentTop + agentHeight - COND_SUBTRACK_HEIGHT
    for (let dy = 0; dy < COND_SUBTRACK_HEIGHT; dy++) {
      expect(findTrackAtY(lastSubtrackTop + dy, list, lanes)).toBe(1)
    }
  })

  it('returns -1 above the first track', () => {
    const list: TrackRow[] = [{ group: 'Environments', id: 'env_0' }]
    expect(findTrackAtY(0, list, undefined)).toBe(-1)
    expect(findTrackAtY(HEADER_HEIGHT - 1, list, undefined)).toBe(-1)
    expect(findTrackAtY(HEADER_HEIGHT + CATEGORY_HEADER_HEIGHT - 1, list, undefined)).toBe(-1)
  })

  it('returns -1 below the last track', () => {
    const list: TrackRow[] = [{ group: 'Environments', id: 'env_0' }]
    const lanes = new Map<string, number>([['env_0_cond', 1]])
    const lastY = getTrackY(list, 0, lanes) + getTrackHeight('Environments', 'env_0', lanes)
    expect(findTrackAtY(lastY, list, lanes)).toBe(-1)
    expect(findTrackAtY(lastY + 1000, list, lanes)).toBe(-1)
  })

  it('returns -1 inside the inter-group gap (category header pill area)', () => {
    const list: TrackRow[] = [
      { group: 'Environments', id: 'env_0' },
      { group: 'Agents', id: 'agent_0' },
    ]
    const lanes = new Map<string, number>([
      ['env_0_cond', 1],
      ['agent_0', 1],
      ['agent_0_infl', 1],
      ['agent_0_prop', 1],
    ])
    const env0Top = getTrackY(list, 0, lanes)
    const env0Bottom = env0Top + getTrackHeight('Environments', 'env_0', lanes)
    const agent0Top = getTrackY(list, 1, lanes)
    // The gap between env_0 and agent_0 includes
    // GROUP_GAP + CATEGORY_HEADER_HEIGHT. Pixels inside the gap belong
    // to neither track.
    expect(env0Bottom).toBeLessThan(agent0Top)
    expect(findTrackAtY(env0Bottom, list, lanes)).toBe(-1)
    expect(findTrackAtY(agent0Top - 1, list, lanes)).toBe(-1)
  })
})

// ---------------------------------------------------------------------------
// getTrackGap
// ---------------------------------------------------------------------------

describe('getTrackGap', () => {
  it('returns TRACK_GAP between two non-collapsed same-group tracks', () => {
    const list: TrackRow[] = [
      { group: 'Environments', id: 'env_0' },
      { group: 'Environments', id: 'env_1' },
    ]
    expect(getTrackGap(list, 0)).toBe(TRACK_GAP)
  })

  it('adds CATEGORY_HEADER_HEIGHT to the inter-group gap when next group is non-collapsed', () => {
    const list: TrackRow[] = [
      { group: 'Environments', id: 'env_0' },
      { group: 'Agents', id: 'agent_0' },
    ]
    const g = getTrackGap(list, 0)
    expect(g).toBeGreaterThanOrEqual(CATEGORY_HEADER_HEIGHT)
  })

  it('omits CATEGORY_HEADER_HEIGHT when next group is collapsed', () => {
    const list: TrackRow[] = [
      { group: 'Environments', id: 'env_0' },
      { group: 'Agents', id: 'agent_0', _collapsed: true },
    ]
    const g = getTrackGap(list, 0)
    expect(g).toBeLessThan(CATEGORY_HEADER_HEIGHT)
  })

  it('returns 0 after the last track', () => {
    const list: TrackRow[] = [{ group: 'Environments', id: 'env_0' }]
    expect(getTrackGap(list, 0)).toBe(0)
  })
})

// ---------------------------------------------------------------------------
// getTrackHeight — minimal smoke; this function is exercised indirectly
// by every test above, but locking a few invariants prevents regressions
// in the heights that drive the row-find.
// ---------------------------------------------------------------------------

describe('getTrackHeight', () => {
  it('returns the collapsed sentinel regardless of group when collapsed=true', () => {
    expect(getTrackHeight('Agents', 'agent_0', undefined, true)).toBeGreaterThan(0)
    expect(getTrackHeight('Environments', 'env_0', undefined, true))
      .toBe(getTrackHeight('Agents', 'agent_0', undefined, true))
  })

  it('grows with the property-row count on an Agents track', () => {
    const lanes1 = new Map<string, number>([['agent_0', 1], ['agent_0_infl', 1], ['agent_0_prop', 1]])
    const lanes3 = new Map<string, number>([['agent_0', 1], ['agent_0_infl', 1], ['agent_0_prop', 3]])
    expect(getTrackHeight('Agents', 'agent_0', lanes3))
      .toBeGreaterThan(getTrackHeight('Agents', 'agent_0', lanes1))
    // Two extra rows == exactly 2 * COND_SUBTRACK_HEIGHT
    expect(
      getTrackHeight('Agents', 'agent_0', lanes3) - getTrackHeight('Agents', 'agent_0', lanes1)
    ).toBe(2 * COND_SUBTRACK_HEIGHT)
  })

  it('Environments track has at least the main bar plus one condition row', () => {
    const lanes = new Map<string, number>([['env_0_cond', 1]])
    expect(getTrackHeight('Environments', 'env_0', lanes))
      .toBeGreaterThanOrEqual(ENV_MAIN_HEIGHT + COND_SUBTRACK_HEIGHT)
  })
})
