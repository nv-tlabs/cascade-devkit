// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect } from 'vitest'
import { nextVideoPct, type SplitterDragState, type SplitterBounds } from './splitter'

const BOUNDS: SplitterBounds = { minPct: 15, maxPct: 70 }

function state(startY: number, startPct: number, containerHeight = 1000): SplitterDragState {
  return { startY, startPct, containerHeight }
}

describe('nextVideoPct', () => {
  it('returns startPct unchanged when cursor has not moved', () => {
    expect(nextVideoPct(state(300, 38), 300, BOUNDS)).toBe(38)
  })

  it('drag down by 100 px on a 1000 px container shifts pct by +10', () => {
    expect(nextVideoPct(state(300, 38), 400, BOUNDS)).toBeCloseTo(48, 5)
  })

  it('drag up by 100 px on a 1000 px container shifts pct by -10', () => {
    expect(nextVideoPct(state(300, 38), 200, BOUNDS)).toBeCloseTo(28, 5)
  })

  it('clamps to maxPct when the cursor is dragged past the bottom bound', () => {
    expect(nextVideoPct(state(300, 38), 9999, BOUNDS)).toBe(70)
  })

  it('clamps to minPct when the cursor is dragged past the top bound', () => {
    expect(nextVideoPct(state(300, 38), -9999, BOUNDS)).toBe(15)
  })

  it('returns startPct unchanged when containerHeight is zero (defensive)', () => {
    expect(nextVideoPct(state(300, 38, 0), 999, BOUNDS)).toBe(38)
  })

  it('returns startPct unchanged when containerHeight is negative (defensive)', () => {
    expect(nextVideoPct(state(300, 38, -100), 999, BOUNDS)).toBe(38)
  })

  describe('regression: cursor offset within the hit-bar', () => {
    // The pre-fix bug used (clientY - rect.top) / rect.height. Grabbing the
    // splitter at offset +5 px from its top jumped the splitter by 5 px on
    // first drag. With the delta-based approach the splitter does not jump.
    it('grabbing anywhere on the splitter does not snap the splitter to the click point', () => {
      // Whatever the initial click offset within the hit-bar, drag-by-0 returns startPct.
      expect(nextVideoPct(state(305, 38), 305, BOUNDS)).toBe(38)
      expect(nextVideoPct(state(310, 38), 310, BOUNDS)).toBe(38)
    })

    it('drag delta is independent of LockBar height (no rect-top dependence)', () => {
      // Two drags with the same delta produce the same pct change, regardless
      // of where startY is anchored. The old formula made result depend on
      // the absolute clientY (which moves when the LockBar takes vertical space).
      const a = nextVideoPct(state(100, 38), 200, BOUNDS) // delta +100, +10%
      const b = nextVideoPct(state(500, 38), 600, BOUNDS) // delta +100, +10%
      expect(a).toBeCloseTo(b, 5)
      expect(a).toBeCloseTo(48, 5)
    })
  })

  it('honours custom bounds', () => {
    const custom: SplitterBounds = { minPct: 25, maxPct: 60 }
    expect(nextVideoPct(state(300, 30), -9999, custom)).toBe(25)
    expect(nextVideoPct(state(300, 30), 9999, custom)).toBe(60)
  })
})
