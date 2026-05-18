// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Pure helpers for the IssuesPanel — no DOM, no React, no fetch. The
// rendering layer composes these into the visible panel and gates the
// "Mark complete" button. Vitest covers each function in isolation.
import type { ValidateIssue } from './validate-api'

export interface GroupedIssues {
  errors: ValidateIssue[]
  warnings: ValidateIssue[]
}

/**
 * Partition a flat issue list into errors-first / warnings-rest. Preserves
 * the server's intra-severity order (which is itself stable: hard rules in
 * `HARD_RULES` order, soft rules in `SOFT_RULES` order).
 */
export function groupIssuesBySeverity(issues: ValidateIssue[]): GroupedIssues {
  const errors: ValidateIssue[] = []
  const warnings: ValidateIssue[] = []
  for (const i of issues) {
    if (i.severity === 'error') errors.push(i)
    else warnings.push(i)
  }
  return { errors, warnings }
}

/**
 * Headline copy for the issues panel. Phrasing is severity-led:
 *   - any errors        → "N issue(s) block completion"
 *   - warnings only     → "N warning(s) — review before finishing"
 *   - clean             → "All clear"
 *
 * The leading number, when shown, is the count of the dominant severity;
 * the panel renders the full breakdown in a sub-line for clarity.
 */
export function issuesHeadline(errors: number, warnings: number): string {
  if (errors > 0) {
    return `${errors} issue${errors === 1 ? '' : 's'} block completion`
  }
  if (warnings > 0) {
    return `${warnings} warning${warnings === 1 ? '' : 's'} — review before finishing`
  }
  return 'All clear'
}

/**
 * The Mark-complete button is enabled iff no `severity="error"` issue is
 * present. Warnings never block. Mirrors the server-side gate in
 * `tools/annotator/src/annotator/server/app.py:put_annotations`.
 */
export function canMarkComplete(issues: ValidateIssue[]): boolean {
  for (const i of issues) {
    if (i.severity === 'error') return false
  }
  return true
}

/**
 * What the IssuesPanel renders at the top of the panel, given the current
 * validation state. Three concerns are merged in one place so the component
 * stays presentation-only:
 *
 *   - `transportError` — the last `POST /validate` failed at the transport
 *     layer (network down, 503, 5xx, JSON parse failure). When present, a
 *     banner appears above the issues list. The banner does NOT replace the
 *     issues list; if the user previously got a successful result and the
 *     next click fails the network, both the banner AND the prior issues are
 *     visible.
 *   - `issues` — the last successful validate result. `null` means the user
 *     hasn't run validate yet (or only run it once and it failed).
 *   - `errors` / `warnings` — derived from `issues`, surfaced separately so
 *     the panel can render the breakdown sub-line.
 *
 * Returns ``null`` when the panel should not render at all. The caller
 * mounts the component iff this returns non-null.
 */
export interface ValidationSummary {
  /** Free-text banner shown above the issues list, or `null` to hide. */
  transportBanner: string | null
  /** True when there's a prior successful result to display. */
  hasIssuesResult: boolean
  /** Grouped issues (empty arrays when `hasIssuesResult` is false). */
  errors: ValidateIssue[]
  warnings: ValidateIssue[]
  /** Headline copy for the prior-result section; `null` when none to show. */
  headline: string | null
}

export function composeValidationSummary(opts: {
  transportError: string | null
  issues: ValidateIssue[] | null
}): ValidationSummary | null {
  const { transportError, issues } = opts
  if (transportError === null && issues === null) return null
  const hasIssuesResult = issues !== null
  const { errors, warnings } = hasIssuesResult
    ? groupIssuesBySeverity(issues!)
    : { errors: [], warnings: [] }
  return {
    transportBanner: transportError,
    hasIssuesResult,
    errors,
    warnings,
    headline: hasIssuesResult ? issuesHeadline(errors.length, warnings.length) : null,
  }
}
