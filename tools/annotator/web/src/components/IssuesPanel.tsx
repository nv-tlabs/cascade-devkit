// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { AlertTriangle, XCircle, CheckCircle2 } from 'lucide-react'
import type { ValidateIssue } from '../lib/validate-api'
import { groupIssuesBySeverity, issuesHeadline } from '../lib/issues-presenter'

/**
 * Renders the validation result inline in the right panel after the user
 * clicks "Mark complete". Errors first, then warnings; each row gets a
 * severity icon and (when an `entity_id` is present) is clickable so the
 * parent can pan the timeline / right-panel selection to the offending
 * entity. Whole-clip issues (`entity_id === null`) render as non-clickable
 * rows. The parent owns the entity-id → segment lookup because the mapping
 * lives in `annotationToSegments` and we want this component to stay
 * presentation-only.
 */
export interface IssuesPanelProps {
  issues: ValidateIssue[]
  onSelect?: (entityId: string) => void
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

export function IssuesPanel({ issues, onSelect, nameForEntity }: IssuesPanelProps) {
  const { errors, warnings } = groupIssuesBySeverity(issues)
  const headline = issuesHeadline(errors.length, warnings.length)
  const isClear = errors.length === 0 && warnings.length === 0

  return (
    <div className="px-5 py-4 border-b border-border-subtle">
      <h3 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-text-secondary mb-3 flex items-center gap-1.5">
        Validation
      </h3>
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
                    onClick={() => onSelect?.(issue.entity_id!)}
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
    </div>
  )
}
