// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import type { AnnotationBundle } from './types'

const BASE = '/api'

export type ClipKind = 'annotated' | 'unlabelled'

export interface ClipEntry {
  clip_id: string
  kind: ClipKind
  has_video: boolean
  /**
   * Mirrors `AnnotationBundle.status` from the bundle's JSON. `null` for
   * unlabelled clips (no file on disk) and for legacy bundles that omit
   * the field. The sidebar paints "complete" green for `status === 'complete'`,
   * amber "in progress" for any other annotated state, and gray
   * "unlabelled" for the unlabelled kind.
   */
  status?: string | null
}

export interface HealthResponse {
  ok: boolean
  clip_count: number
  read_only: boolean
}

export class ApiError extends Error {
  status: number
  detail: unknown
  constructor(message: string, status: number, detail: unknown) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {}
  const opts: RequestInit = { method, headers }
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json'
    opts.body = JSON.stringify(body)
  }
  const res = await fetch(`${BASE}${path}`, opts)
  if (!res.ok) {
    let detail: unknown = res.statusText
    try {
      detail = await res.json()
    } catch {
      /* keep statusText */
    }
    let message = res.statusText
    if (typeof detail === 'object' && detail !== null) {
      const d = detail as Record<string, unknown>
      if (typeof d.error === 'string') message = d.error
      else if (typeof d.detail === 'string') message = d.detail
      else if (Array.isArray(d.errors)) message = `validation failed (${d.errors.length} error(s))`
    } else if (typeof detail === 'string') {
      message = detail
    }
    throw new ApiError(message, res.status, detail)
  }
  return res.json() as Promise<T>
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('GET', '/health')
}

export async function listClips(): Promise<ClipEntry[]> {
  return request<ClipEntry[]>('GET', '/clips')
}

export async function getBundle(clipId: string): Promise<AnnotationBundle> {
  // As of the 2.0.0 schema reboot the server keeps timeline-layout indices
  // inline on each entity (`_track_index` etc.) — no `ui/1.0` sidecar
  // round-trip is needed. The server still emits an `_extensions` envelope
  // for unrelated payloads; the rest of the frontend ignores it.
  return request<AnnotationBundle>(
    'GET',
    `/clips/${encodeURIComponent(clipId)}/annotations`,
  )
}

/** Save a bundle. The server returns `{saved_to, bundle}` — we return the
 *  echoed bundle so callers can keep using it directly. */
export async function saveBundle(clipId: string, bundle: AnnotationBundle): Promise<AnnotationBundle> {
  const res = await request<{ saved_to: string; bundle: AnnotationBundle }>(
    'PUT',
    `/clips/${encodeURIComponent(clipId)}/annotations`,
    bundle,
  )
  return res.bundle
}

/** Stage labels the server emits during a video resolve. The frontend treats
 * every non-{idle,ready,error} value as "in progress, show overlay." */
export interface VideoStatus {
  stage: string
  message: string
}

export async function getVideoStatus(clipId: string): Promise<VideoStatus> {
  return request<VideoStatus>('GET', `/clips/${encodeURIComponent(clipId)}/video/status`)
}
