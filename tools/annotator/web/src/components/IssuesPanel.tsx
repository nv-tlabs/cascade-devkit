// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { AlertTriangle, XCircle, CheckCircle2, WifiOff } from 'lucide-react'
import type { ValidateIssue } from '../lib/validate-api'
import { composeValidationSummary } from '../lib/issues-presenter'

/**
 * Renders the validation result inline in the right panel after the user
 * clicks "Mark complete". Errors first, then warnings; each row gets a
 * severity icon and (when an `entity_id` is present) is clickable so the
 * parent can pan the timeline / right-panel selection to the offending
 * entity. Whole-clip issues (`entity_id === null`) render as non-clickable
 * rows. The parent owns the entity-id → segment lookup because the mapping
 * lives in `annotationToSegments` and we want this component to stay
 * presentation-only.
 *
 * `transportError`, when non-null, renders as a banner above the issues
 * list. The banner does NOT overwrite the issues list — when the user has
 * a prior successful validate result and the next request fails the
 * network, both the banner and the prior findings stay visible. This is
 * the #91 contract: transport-layer failures must not erase content the
 * user is actively working through.
 */
export interface IssuesPanelProps {
  /**
   * Last successful validate result. `null` means the user has not yet
   * received a successful response for this clip (only relevant when
   * `transportError` is set — otherwise the parent skips rendering).
   */
  issues: ValidateIssue[] | null
  /**
   * Free-text error message from the most recent failed validate request,
   * or `null` if the last request succeeded (or none has been made).
   */
  transportError?: string | null
  /**
   * Callback fired when the user clicks a clickable issue row. `field` is
   * the wire `Issue.field` value verbatim — the host (`RightPanel`) runs it
   * through `resolveFocus` to derive a `data-issue-target` and, after
   * selecting the parent entity, focuses the matching form control. We pass
   * the raw field instead of pre-resolving here so this component stays
   * presentation-only (no knowledge of the rule → control mapping).
   */
  onSelect?: (entityId: string, field?: string) => void
  /**
   * Optional human-readable name lookup. When set, the row title prefers
   * `nameForEntity(entity_id)` over the raw id; falls back to entity_path
   * for whole-clip issues.
   */
  nameForEntity?: (entityId: string) => string | null
}

function rowTitle(
  issue: ValidateIssue,
  nameForEntity?: (id: string) => string | null,
): string {
  if (issue.entity_id) {
    const named = nameForEntity?.(issue.entity_id)
    return named ?? issue.entity_id
  }
  return issue.entity_path
}

export function IssuesPanel({
  issues,
  transportError = null,
  onSelect,
  nameForEntity,
}: IssuesPanelProps) {
  const summary = composeValidationSummary({ transportError, issues })
  if (summary === null) return null
  const { transportBanner, hasIssuesResult, errors, warnings, headline } = summary
  const isClear = hasIssuesResult && errors.length === 0 && warnings.length === 0

  return (
    <div className="px-5 py-4 border-b border-border-subtle">
      <h3 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-text-secondary mb-3 flex items-center gap-1.5">
        Validation
      </h3>
      {transportBanner !== null && (
        <div
          role="alert"
          className="mb-3 flex items-start gap-2 px-2.5 py-2 rounded-md border bg-warning-bg border-warning/30 text-warning text-[11px]"
        >
          <WifiOff className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <div className="font-medium">Validation request failed</div>
            <div className="text-text-secondary whitespace-normal break-words">
              {transportBanner}
              {hasIssuesResult && (
                <span> — showing results from the last successful run.</span>
              )}
            </div>
          </div>
        </div>
      )}
      {hasIssuesResult && headline !== null && (
        <>
          <div className="mb-3 flex items-center gap-2 text-xs">
            {isClear ? (
              <CheckCircle2 className="w-4 h-4 text-success flex-shrink-0" />
            ) : errors.length > 0 ? (
              <XCircle className="w-4 h-4 text-danger flex-shrink-0" />
            ) : (
              <AlertTriangle className="w-4 h-4 text-warning flex-shrink-0" />
            )}
            <span className="font-medium text-text-primary">{headline}</span>
          </div>
          {!isClear && (
            <p className="mb-3 text-[10px] text-text-muted">
              {errors.length > 0 && (
                <span>
                  {errors.length} error{errors.length === 1 ? '' : 's'}
                </span>
              )}
              {errors.length > 0 && warnings.length > 0 && <span> · </span>}
              {warnings.length > 0 && (
                <span>
                  {warnings.length} warning{warnings.length === 1 ? '' : 's'}
                </span>
              )}
            </p>
          )}
          {!isClear && (
            <ul className="space-y-1.5">
              {[...errors, ...warnings].map((issue, idx) => {
                const title = rowTitle(issue, nameForEntity)
                const clickable = issue.entity_id !== null && onSelect != null
                const baseCls =
                  'w-full text-left px-2.5 py-2 rounded-md border text-[11px] flex gap-2 items-start'
                const interactionCls = clickable
                  ? 'hover:bg-surface-hover cursor-pointer'
                  : 'cursor-default'
                const severityCls = issue.severity === 'error'
                  ? 'bg-danger-bg border-danger/30 text-danger'
                  : 'bg-warning-bg border-warning/30 text-warning'
                const content = (
                  <>
                    {issue.severity === 'error' ? (
                      <XCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                    ) : (
                      <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="font-mono text-[10px] truncate text-text-primary">
                        {title}
                      </div>
                      <div className="text-text-secondary whitespace-normal break-words">
                        {issue.message}
                      </div>
                    </div>
                  </>
                )
                return (
                  <li key={`${issue.rule}-${idx}`}>
                    {clickable ? (
                      <button
                        type="button"
                        onClick={() => onSelect?.(issue.entity_id!, issue.field ?? undefined)}
                        className={`${baseCls} ${interactionCls} ${severityCls}`}
                      >
                        {content}
                      </button>
                    ) : (
                      <div className={`${baseCls} ${interactionCls} ${severityCls}`}>
                        {content}
                      </div>
                    )}
                  </li>
                )
              })}
            </ul>
          )}
        </>
      )}
    </div>
  )
}
