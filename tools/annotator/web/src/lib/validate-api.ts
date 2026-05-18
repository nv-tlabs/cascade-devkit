// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import type { AnnotationBundle } from './types'
import { extractUiExtension } from './ui-extension'
import { ApiError } from './api'

/**
 * Wire shape for one validation finding, mirroring
 * `cascade_av.validate.Issue.to_dict`. The frontend renders the same
 * structure regardless of whether it came from `POST /validate` or from
 * the `422` response of `PUT /annotations` with `status="complete"`.
 */
export interface ValidateIssue {
  severity: 'error' | 'warning'
  entity_path: string
  entity_id: string | null
  field: string | null
  rule: string
  message: string
}

export interface ValidateResponse {
  ok: boolean
  issues: ValidateIssue[]
}

const BASE = '/api'

/**
 * Run the server-side rule set against a bundle without persisting it.
 *
 * Mirrors `saveBundle`: the bundle is packed into the ui/1.0 wire envelope
 * via `extractUiExtension` so the server sees the on-disk shape, not the
 * frontend's hydrated form. Throws `ApiError` on non-2xx responses so
 * callers can distinguish "endpoint unavailable" (server down, 503) from
 * "validation completed" (200 with `ok=false` and issues).
 */
export async function validateBundle(
  clipId: string,
  bundle: AnnotationBundle,
): Promise<ValidateResponse> {
  const wireOut = extractUiExtension(bundle)
  const res = await fetch(
    `${BASE}/clips/${encodeURIComponent(clipId)}/annotations/validate`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(wireOut),
    },
  )
  if (!res.ok) {
    let detail: unknown = res.statusText
    try {
      detail = await res.json()
    } catch {
      /* keep statusText */
    }
    const message = typeof detail === 'object' && detail !== null && 'error' in detail
      ? String((detail as { error: unknown }).error)
      : res.statusText
    throw new ApiError(message, res.status, detail)
  }
  return res.json() as Promise<ValidateResponse>
}
