// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * Pure presentation logic for the video-resolve progress overlay. Maps the
 * server's stage tokens to user-facing labels and decides whether the overlay
 * should be visible at all.
 *
 * The server emits these stage tokens (see `tools/annotator/src/annotator/server/video.py`):
 *   idle | locating | downloading | extracting | probing | transcoding | ready | error
 *
 * Kept DOM-free so the mapping is unit-testable under Node — no React, no fetch.
 */

import type { VideoStatus } from './api'

/** Stage tokens that warrant showing a progress overlay. `idle` and `ready`
 * are intentionally absent: idle means the request hasn't started yet (no
 * useful information for the user) and ready means the video element is
 * about to receive bytes. `error` is also absent because the
 * already-existing "Video unavailable" panel renders error state. */
const IN_PROGRESS_STAGES: ReadonlySet<string> = new Set([
  'locating',
  'downloading',
  'extracting',
  'probing',
  'transcoding',
])

/** Returns true when the overlay should be rendered. */
export function isInProgress(status: VideoStatus | null): boolean {
  return status !== null && IN_PROGRESS_STAGES.has(status.stage)
}

/** Headline copy shown in large text in the overlay. */
export function stageHeadline(stage: string): string {
  switch (stage) {
    case 'locating':
      return 'Locating clip…'
    case 'downloading':
      return 'Downloading clip…'
    case 'extracting':
      return 'Extracting clip…'
    case 'probing':
      return 'Inspecting video…'
    case 'transcoding':
      return 'Transcoding video…'
    default:
      // Defensive: unknown stages shouldn't reach the overlay, but if they do
      // we'd rather show *something* than crash. The poll keeps trying and
      // the next tick will probably carry a known stage.
      return 'Preparing video…'
  }
}

/** Secondary line (smaller, italicised) — the server-supplied free-text
 * detail, which may be empty for early stages. */
export function stageDetail(status: VideoStatus): string {
  return status.message ?? ''
}
