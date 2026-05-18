// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect } from 'vitest'
import {
  canMarkComplete,
  composeValidationSummary,
  groupIssuesBySeverity,
  issuesHeadline,
} from './issues-presenter'
import type { ValidateIssue } from './validate-api'

function err(rule: string, msg = 'x'): ValidateIssue {
  return {
    severity: 'error',
    entity_path: 'annotation.x',
    entity_id: null,
    field: null,
    rule,
    message: msg,
  }
}

function warn(rule: string, msg = 'x'): ValidateIssue {
  return {
    severity: 'warning',
    entity_path: 'annotation.x',
    entity_id: null,
    field: null,
    rule,
    message: msg,
  }
}

describe('groupIssuesBySeverity', () => {
  it('returns empty buckets for an empty list', () => {
    expect(groupIssuesBySeverity([])).toEqual({ errors: [], warnings: [] })
  })

  it('partitions errors and warnings preserving original order', () => {
    const issues = [err('a'), warn('b'), err('c'), warn('d')]
    const out = groupIssuesBySeverity(issues)
    expect(out.errors.map(i => i.rule)).toEqual(['a', 'c'])
    expect(out.warnings.map(i => i.rule)).toEqual(['b', 'd'])
  })

  it('handles errors only', () => {
    const out = groupIssuesBySeverity([err('a'), err('b')])
    expect(out.errors).toHaveLength(2)
    expect(out.warnings).toEqual([])
  })

  it('handles warnings only', () => {
    const out = groupIssuesBySeverity([warn('a'), warn('b')])
    expect(out.errors).toEqual([])
    expect(out.warnings).toHaveLength(2)
  })
})

describe('issuesHeadline', () => {
  it('reports "All clear" for zero / zero', () => {
    expect(issuesHeadline(0, 0)).toBe('All clear')
  })

  it('singularizes "1 issue blocks completion"', () => {
    expect(issuesHeadline(1, 0)).toBe('1 issue block completion')
  })

  it('pluralizes errors > 1', () => {
    expect(issuesHeadline(3, 0)).toBe('3 issues block completion')
  })

  it('reports warnings-only with singular/plural matching', () => {
    expect(issuesHeadline(0, 1)).toBe('1 warning — review before finishing')
    expect(issuesHeadline(0, 4)).toBe('4 warnings — review before finishing')
  })

  it('errors win when both are present', () => {
    // Mixed: warnings count is hidden in the headline; the sub-line shows the
    // breakdown. This keeps the dominant severity legible at a glance.
    expect(issuesHeadline(2, 5)).toBe('2 issues block completion')
  })
})

describe('canMarkComplete', () => {
  it('returns true for an empty list', () => {
    expect(canMarkComplete([])).toBe(true)
  })

  it('returns true when only warnings are present', () => {
    expect(canMarkComplete([warn('a'), warn('b')])).toBe(true)
  })

  it('returns false when at least one error is present', () => {
    expect(canMarkComplete([err('a')])).toBe(false)
  })

  it('returns false for a mixed list (warnings do not rescue errors)', () => {
    expect(canMarkComplete([warn('a'), err('b'), warn('c')])).toBe(false)
  })
})

describe('composeValidationSummary', () => {
  it('returns null when there is nothing to show', () => {
    expect(composeValidationSummary({ transportError: null, issues: null })).toBeNull()
  })

  it('returns a summary with no banner when only prior issues exist', () => {
    const out = composeValidationSummary({
      transportError: null,
      issues: [err('a'), warn('b')],
    })
    expect(out).not.toBeNull()
    expect(out!.transportBanner).toBeNull()
    expect(out!.hasIssuesResult).toBe(true)
    expect(out!.errors.map(i => i.rule)).toEqual(['a'])
    expect(out!.warnings.map(i => i.rule)).toEqual(['b'])
    expect(out!.headline).toBe('1 issue block completion')
  })

  it('renders banner without overwriting prior issues — the #91 regression guard', () => {
    // The user got issues from a previous successful validate, then the next
    // request failed at the transport layer. The banner must coexist with
    // the prior findings — not replace them.
    const out = composeValidationSummary({
      transportError: 'network unreachable',
      issues: [err('hard_rule_a'), warn('soft_rule_b')],
    })
    expect(out!.transportBanner).toBe('network unreachable')
    expect(out!.hasIssuesResult).toBe(true)
    expect(out!.errors).toHaveLength(1)
    expect(out!.warnings).toHaveLength(1)
    expect(out!.headline).toBe('1 issue block completion')
  })

  it('renders banner alone when there is no prior result', () => {
    // First Mark-complete click failed at the transport layer; there's no
    // prior result to display, just the banner.
    const out = composeValidationSummary({
      transportError: '503 Service Unavailable',
      issues: null,
    })
    expect(out!.transportBanner).toBe('503 Service Unavailable')
    expect(out!.hasIssuesResult).toBe(false)
    expect(out!.errors).toEqual([])
    expect(out!.warnings).toEqual([])
    expect(out!.headline).toBeNull()
  })

  it('renders an all-clear prior result alongside a transport banner', () => {
    // Edge case: the user got a clean validate, kept editing without fixing
    // anything (clean → clean), and the second request blew up at transport.
    // The banner shows, but the prior "All clear" headline stays so the user
    // can see the last good state.
    const out = composeValidationSummary({
      transportError: 'parse error',
      issues: [],
    })
    expect(out!.transportBanner).toBe('parse error')
    expect(out!.hasIssuesResult).toBe(true)
    expect(out!.headline).toBe('All clear')
  })
})
