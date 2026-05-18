// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect } from 'vitest'
import { isInProgress, stageHeadline, stageDetail } from './video-status'
import type { VideoStatus } from './api'

const mk = (stage: string, message = ''): VideoStatus => ({ stage, message })

describe('isInProgress', () => {
  it('returns false for null (poller has not fired yet)', () => {
    expect(isInProgress(null)).toBe(false)
  })

  it('returns false for idle (request not started)', () => {
    expect(isInProgress(mk('idle'))).toBe(false)
  })

  it('returns false for ready (video bytes incoming)', () => {
    expect(isInProgress(mk('ready'))).toBe(false)
  })

  it('returns false for error (error panel takes over)', () => {
    expect(isInProgress(mk('error', 'boom'))).toBe(false)
  })

  it.each(['locating', 'downloading', 'extracting', 'probing', 'transcoding'])(
    'returns true for stage %s',
    (stage) => {
      expect(isInProgress(mk(stage))).toBe(true)
    },
  )

  it('returns false for an unrecognised stage', () => {
    // Future stages should not silently show the overlay until the frontend
    // knows about them. Default-deny.
    expect(isInProgress(mk('quantum-relocation'))).toBe(false)
  })
})

describe('stageHeadline', () => {
  it('maps every advertised in-progress stage to a non-empty label', () => {
    for (const stage of ['locating', 'downloading', 'extracting', 'probing', 'transcoding']) {
      const h = stageHeadline(stage)
      expect(typeof h).toBe('string')
      expect(h.length).toBeGreaterThan(0)
    }
  })

  it('falls back to a generic label for unknown stages', () => {
    expect(stageHeadline('quantum-relocation')).toBe('Preparing video…')
  })
})

describe('stageDetail', () => {
  it('returns the server message verbatim when present', () => {
    expect(stageDetail(mk('transcoding', 'HEVC → H.264'))).toBe('HEVC → H.264')
  })

  it('returns empty string when the server omits a message', () => {
    expect(stageDetail(mk('locating'))).toBe('')
  })
})
