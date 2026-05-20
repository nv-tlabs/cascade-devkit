// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { afterEach, describe, expect, it, vi } from 'vitest'
import { saveBundle } from './api'
import type { AnnotationBundle } from './types'

/**
 * Regression guard for PR #121: `extractUiExtension` in the deleted
 * `lib/ui-extension.ts` was destructively stripping inline `_track_index`
 * (and friends) from outbound PUT bodies. The fix was to send the bundle
 * straight through. This test pins the new behavior: every inline
 * `_track_index` / `_cond_track_index` / `_state_track_index` /
 * `_influence_track_index` we set on the input bundle must survive into
 * the JSON body the network layer sends.
 *
 * If this test ever fails, somebody re-introduced the strip-and-repack
 * adapter — read PR #121's description before touching it.
 */

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}

function fullBundle(): AnnotationBundle {
  return {
    schema_version: '2.0.0',
    video: { clip_id: 'clip-001', fps: 30, duration_s: 10 },
    status: 'annotating',
    provenance: {},
    annotation: {
      brief_description: '',
      environments: [
        {
          id: 'E1',
          type: 'fst:Intersection',
          num_lanes: 2,
          start_timestamp: '0',
          end_timestamp: '10000000',
          _track_index: 7,
        },
      ],
      conditions: [
        {
          id: 'C1',
          env_id: 'E1',
          type: 'cnd:Rain',
          start_timestamp: '0',
          end_timestamp: '5000000',
          _track_index: 3,
          _cond_track_index: 4,
        },
      ],
      traffic_objects: [
        {
          id: 'O1',
          type: 'obj:Cone',
          visibility_start_timestamp: '0',
          visibility_end_timestamp: '10000000',
          state_sequence: [],
          containment: [
            {
              id: 'OC1',
              env_id: 'E1',
              start_timestamp: '0',
              end_timestamp: '10000000',
              _track_index: 11,
            },
          ],
          _track_index: 9,
        },
      ],
      traffic_lights: [
        {
          id: 'L1',
          visibility_start_timestamp: '0',
          visibility_end_timestamp: '10000000',
          signal_heads: [
            {
              id: 'SH1',
              start_timestamp: '0',
              end_timestamp: '10000000',
              state_sequence: [
                {
                  color: 'red',
                  start_timestamp: '0',
                  end_timestamp: '5000000',
                  _state_track_index: 2,
                },
              ],
            },
          ],
          _track_index: 13,
        },
      ],
      ego_vehicle: {
        actions: [],
        influenced_by: [
          {
            id: 'EI1',
            influencers: ['A1'],
            start_timestamp: '0',
            end_timestamp: '5000000',
            _influence_track_index: 5,
          },
        ],
      },
      agents: [
        {
          id: 'A1',
          amount: '1',
          type: 'oxd:Car',
          actions: [],
          _track_index: 17,
        },
      ],
    },
  }
}

describe('saveBundle PUT body', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('preserves every inline _track_index field on the outbound body', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(
        jsonResponse({ saved_to: '/tmp/clip-001.json', bundle: fullBundle() }),
      )

    const bundle = fullBundle()
    await saveBundle('clip-001', bundle)

    expect(fetchSpy).toHaveBeenCalledTimes(1)
    const [, init] = fetchSpy.mock.calls[0]
    expect(init?.method).toBe('PUT')
    expect(typeof init?.body).toBe('string')

    const sent = JSON.parse(init?.body as string) as AnnotationBundle
    const a = sent.annotation
    expect(a.environments[0]._track_index).toBe(7)
    expect(a.conditions[0]._track_index).toBe(3)
    expect(a.conditions[0]._cond_track_index).toBe(4)
    expect(a.traffic_objects[0]._track_index).toBe(9)
    expect(a.traffic_objects[0].containment?.[0]._track_index).toBe(11)
    expect(a.traffic_lights[0]._track_index).toBe(13)
    expect(a.traffic_lights[0].signal_heads[0].state_sequence[0]._state_track_index).toBe(2)
    expect(a.ego_vehicle.influenced_by?.[0]._influence_track_index).toBe(5)
    expect(a.agents[0]._track_index).toBe(17)
  })
})
