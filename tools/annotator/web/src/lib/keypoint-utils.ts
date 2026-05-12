// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import type { Keypoint } from './types'
import { formatTs, parseTs } from './timeline-utils'

// If an existing keypoint's timestamp is within this window of an upsert, replace it
// rather than inserting a duplicate. Matches the 0.1 s timestamp quantization.
const SNAP_SECONDS = 0.1

// Each annotated keypoint is rendered as a solid marker for this many seconds
// on either side of its timestamp, and this is also the outer visibility bound:
// nothing is drawn outside [first - EXACT_SNAP, last + EXACT_SNAP].
export const EXACT_SNAP_SECONDS = 0.15

export interface InterpolatedKeypoint {
  x: number
  y: number
  interpolated: boolean
}

export function interpolateKeypoint(
  kps: Keypoint[] | undefined,
  time: number,
): InterpolatedKeypoint | null {
  if (!kps || kps.length === 0) return null
  const first = parseTs(kps[0].timestamp)
  const last = parseTs(kps[kps.length - 1].timestamp)
  if (time < first - EXACT_SNAP_SECONDS || time > last + EXACT_SNAP_SECONDS) return null

  if (time <= first) return { x: kps[0].x, y: kps[0].y, interpolated: false }
  if (time >= last) {
    const tail = kps[kps.length - 1]
    return { x: tail.x, y: tail.y, interpolated: false }
  }

  // Strictly between first and last ⇒ kps.length >= 2.
  for (let i = 0; i < kps.length - 1; i++) {
    const ta = parseTs(kps[i].timestamp)
    const tb = parseTs(kps[i + 1].timestamp)
    if (time >= ta && time <= tb) {
      if (Math.abs(time - ta) < EXACT_SNAP_SECONDS) return { x: kps[i].x, y: kps[i].y, interpolated: false }
      if (Math.abs(time - tb) < EXACT_SNAP_SECONDS) return { x: kps[i + 1].x, y: kps[i + 1].y, interpolated: false }
      if (tb === ta) return { x: kps[i].x, y: kps[i].y, interpolated: false }
      const u = (time - ta) / (tb - ta)
      return {
        x: kps[i].x + u * (kps[i + 1].x - kps[i].x),
        y: kps[i].y + u * (kps[i + 1].y - kps[i].y),
        interpolated: true,
      }
    }
  }
  return null
}

/**
 * Insert or replace a keypoint at (x, y) for the given time.
 * If an existing keypoint is within SNAP_SECONDS of `time`, it is replaced.
 * Otherwise a new keypoint is inserted in sorted position.
 * Returns a new array (does not mutate input).
 */
export function upsertKeypoint(
  kps: Keypoint[] | undefined,
  time: number,
  x: number,
  y: number,
): Keypoint[] {
  const list = (kps || []).slice()
  const ts = formatTs(time)
  const newKp: Keypoint = { timestamp: ts, x, y }
  for (let i = 0; i < list.length; i++) {
    const ti = parseTs(list[i].timestamp)
    if (Math.abs(ti - time) <= SNAP_SECONDS) {
      list[i] = newKp
      return list
    }
  }
  let insertAt = list.length
  for (let i = 0; i < list.length; i++) {
    if (parseTs(list[i].timestamp) > time) { insertAt = i; break }
  }
  list.splice(insertAt, 0, newKp)
  return list
}

/**
 * Replace the x/y of the keypoint at the given index, preserving its timestamp.
 * Returns a new array.
 */
export function moveKeypointAt(
  kps: Keypoint[] | undefined,
  idx: number,
  x: number,
  y: number,
): Keypoint[] {
  const list = (kps || []).slice()
  if (idx < 0 || idx >= list.length) return list
  list[idx] = { ...list[idx], x, y }
  return list
}

export function removeKeypointAt(
  kps: Keypoint[] | undefined,
  idx: number,
): Keypoint[] {
  const list = (kps || []).slice()
  if (idx < 0 || idx >= list.length) return list
  list.splice(idx, 1)
  return list
}

/**
 * Drop keypoints that fall outside the [t0, t1] visibility window.
 * Used when a visibility window is trimmed.
 */
export function trimKeypointsToWindow(
  kps: Keypoint[] | undefined,
  t0: number,
  t1: number,
): Keypoint[] {
  if (!kps) return []
  return kps.filter(kp => {
    const t = parseTs(kp.timestamp)
    return t >= t0 - 1e-6 && t <= t1 + 1e-6
  })
}
