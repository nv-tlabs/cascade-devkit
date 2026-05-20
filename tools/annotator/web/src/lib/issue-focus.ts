// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * Issue → right-panel focus resolver.
 *
 * Maps a `ValidateIssue` (rule + entity_id + field) into a focus target the
 * right-panel host can act on: `{entity_id, field}`. The host first calls
 * `selectPath(...)` to bring the entity's editor into view, then (after the
 * panel re-renders) looks up the matching `data-issue-target="<field>"`
 * control inside the panel and `.focus()` + `.scrollIntoView()`s it.
 *
 * ## Rule → control mapping
 *
 * Each row below documents which UI control the listed rule's `field` should
 * focus on. "Parent fallback" means the rule has no specific UI control to
 * focus (e.g. a clip-level rule, or the absence of a record); the host will
 * just select the parent entity and stop there.
 *
 * | Rule                              | Field on issue                  | UI target (data-issue-target value) |
 * |-----------------------------------|---------------------------------|-------------------------------------|
 * | environment_requires_type         | `type`                          | `type`                              |
 * | condition_requires_type           | `type`                          | `type`                              |
 * | agent_requires_type               | `type`                          | `type`                              |
 * | traffic_object_requires_type      | `type`                          | `type`                              |
 * | ego_has_at_least_one_action       | `actions`                       | (parent fallback — no UI control)   |
 * | id_references_resolve             | `because_of`                    | `because_of`                        |
 * | id_references_resolve             | `link_to`                       | `link_to`                           |
 * | id_references_resolve             | `action_target`                 | `action_target`                     |
 * | id_references_resolve             | `influencers`                   | `influencers`                       |
 * | id_references_resolve             | `influenced_agent_ids`          | (parent fallback)                   |
 * | id_references_have_value          | (same as id_references_resolve) | (same)                              |
 * | timestamps_in_video_range         | `start_timestamp`               | `start_timestamp`                   |
 * | timestamps_in_video_range         | `end_timestamp`                 | `end_timestamp`                     |
 * | timestamps_in_video_range         | `visibility_start_timestamp`    | `start_timestamp` (shared input)    |
 * | timestamps_in_video_range         | `visibility_end_timestamp`      | `end_timestamp` (shared input)      |
 * | timestamps_have_value             | (same)                          | (same)                              |
 *
 * The right-panel renders a single Start/End time pair for the selected
 * segment regardless of which underlying entity carries the timestamps, so
 * `visibility_*_timestamp` fields collapse onto the same `start_timestamp` /
 * `end_timestamp` controls. The mapping is therefore many-to-one.
 *
 * Anything not in the table above passes through unchanged — the host tries
 * `[data-issue-target="<field>"]`, and if no element matches it silently
 * falls back to "just selected the parent" (no focus). New rules can be
 * added by either tagging a new control with `data-issue-target="<x>"`
 * (and emitting `field="<x>"` from the backend rule), or by adding an
 * explicit rewrite below.
 */
import type { ValidateIssue } from './validate-api'

export interface FocusTarget {
  /** Entity id to select (e.g. an action id like "AA1", or a top-level entity like "A1"). */
  entity_id: string
  /**
   * Optional `data-issue-target` value to focus inside the right-panel form.
   * `undefined` means "just select the parent" — fall back behavior.
   */
  field?: string
}

/** Fields that the backend emits but which collapse onto the shared
 * Start/End time controls in the right panel. */
const VISIBILITY_TO_TIMESTAMP: Record<string, string> = {
  visibility_start_timestamp: 'start_timestamp',
  visibility_end_timestamp: 'end_timestamp',
}

/** Fields with no UI control to focus — fall back to parent-entity select. */
const NO_UI_CONTROL = new Set<string>([
  // clip-level: ego_has_at_least_one_action emits this against the ego
  // vehicle as a whole; there's no "missing record" input to focus on.
  'actions',
  // SignalHead.influenced_agent_ids isn't currently rendered as an editable
  // control in the signal-head subtrack — fall back to selecting the head.
  'influenced_agent_ids',
])

/**
 * Resolve an `Issue` to a `{entity_id, field?}` focus target.
 *
 * Returns `null` when the issue has no `entity_id` (clip-level findings —
 * the IssuesPanel renders these as non-clickable rows, so the host should
 * never reach the resolver for them; we still tolerate the case defensively).
 */
export function resolveFocus(issue: ValidateIssue): FocusTarget | null {
  if (!issue.entity_id) return null
  const target: FocusTarget = { entity_id: issue.entity_id }
  if (!issue.field) return target
  if (NO_UI_CONTROL.has(issue.field)) return target
  // Collapse visibility_* timestamps onto the shared start/end inputs.
  const mapped = VISIBILITY_TO_TIMESTAMP[issue.field] ?? issue.field
  target.field = mapped
  return target
}
