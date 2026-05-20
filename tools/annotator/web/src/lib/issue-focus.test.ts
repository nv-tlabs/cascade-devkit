// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect } from 'vitest'
import { resolveFocus } from './issue-focus'
import type { ValidateIssue } from './validate-api'

function issue(partial: Partial<ValidateIssue>): ValidateIssue {
  return {
    severity: 'warning',
    entity_path: 'annotation.x',
    entity_id: 'AA1',
    field: null,
    rule: 'id_references_have_value',
    message: 'x',
    ...partial,
  }
}

describe('resolveFocus', () => {
  it('returns null for clip-level issues (no entity_id)', () => {
    expect(
      resolveFocus(
        issue({ entity_id: null, rule: 'ego_has_at_least_one_action', field: 'actions' }),
      ),
    ).toBeNull()
  })

  it('passes through entity_id when there is no field', () => {
    expect(resolveFocus(issue({ entity_id: 'A1', field: null }))).toEqual({
      entity_id: 'A1',
    })
  })

  describe('id-reference rules', () => {
    it.each(['because_of', 'link_to', 'action_target', 'influencers'])(
      'maps field %s to the same data-issue-target',
      (field) => {
        expect(
          resolveFocus(issue({ rule: 'id_references_resolve', field })),
        ).toEqual({ entity_id: 'AA1', field })
        expect(
          resolveFocus(issue({ rule: 'id_references_have_value', field })),
        ).toEqual({ entity_id: 'AA1', field })
      },
    )

    it('falls back to parent for influenced_agent_ids (no editable control today)', () => {
      expect(
        resolveFocus(
          issue({
            rule: 'id_references_have_value',
            field: 'influenced_agent_ids',
            entity_id: 'SH1',
          }),
        ),
      ).toEqual({ entity_id: 'SH1' })
    })
  })

  describe('type rules', () => {
    it.each([
      'environment_requires_type',
      'condition_requires_type',
      'agent_requires_type',
      'traffic_object_requires_type',
    ])('maps field type to data-issue-target=type for %s', (rule) => {
      expect(resolveFocus(issue({ rule, field: 'type', entity_id: 'A1' }))).toEqual({
        entity_id: 'A1',
        field: 'type',
      })
    })
  })

  describe('clip-level ego rule', () => {
    it('falls back to parent for ego_has_at_least_one_action (no UI control)', () => {
      // Backend emits this with entity_id=null; defensive case here ensures
      // even a non-null id wouldn't resolve to a focus target.
      expect(
        resolveFocus(
          issue({
            rule: 'ego_has_at_least_one_action',
            field: 'actions',
            entity_id: 'Ego',
          }),
        ),
      ).toEqual({ entity_id: 'Ego' })
    })
  })

  describe('timestamp rules', () => {
    it.each(['start_timestamp', 'end_timestamp'])(
      'passes %s through unchanged',
      (field) => {
        expect(
          resolveFocus(issue({ rule: 'timestamps_in_video_range', field })),
        ).toEqual({ entity_id: 'AA1', field })
        expect(
          resolveFocus(issue({ rule: 'timestamps_have_value', field })),
        ).toEqual({ entity_id: 'AA1', field })
      },
    )

    it('collapses visibility_start_timestamp onto the shared start_timestamp control', () => {
      expect(
        resolveFocus(
          issue({
            rule: 'timestamps_in_video_range',
            field: 'visibility_start_timestamp',
            entity_id: 'A1',
          }),
        ),
      ).toEqual({ entity_id: 'A1', field: 'start_timestamp' })
    })

    it('collapses visibility_end_timestamp onto the shared end_timestamp control', () => {
      expect(
        resolveFocus(
          issue({
            rule: 'timestamps_have_value',
            field: 'visibility_end_timestamp',
            entity_id: 'A1',
          }),
        ),
      ).toEqual({ entity_id: 'A1', field: 'end_timestamp' })
    })
  })

  describe('unknown fields', () => {
    it('passes unknown field through unchanged (host will try to match a data-issue-target and fail silently)', () => {
      expect(
        resolveFocus(issue({ rule: 'some_future_rule', field: 'novel_field' })),
      ).toEqual({ entity_id: 'AA1', field: 'novel_field' })
    })
  })
})
