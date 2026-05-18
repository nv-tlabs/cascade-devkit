// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect } from 'vitest'
import {
  extractUiExtension,
  hydrateUiExtension,
  type WireBundle,
} from './ui-extension'
import type { AnnotationBundle } from './types'

function mkBundle(overrides: Partial<AnnotationBundle> = {}): AnnotationBundle {
  return {
    schema_version: '2.2.0',
    video: { clip_id: 'c', fps: 30, duration_s: 1 },
    annotation: {
      environments: [],
      conditions: [],
      traffic_objects: [],
      traffic_lights: [],
      ego_vehicle: { actions: [] },
      agents: [],
      brief_description: '',
    },
    ...overrides,
  } as AnnotationBundle
}

describe('hydrateUiExtension', () => {
  it('returns the bundle unchanged when no _extensions envelope is present', () => {
    const bundle = mkBundle({
      annotation: {
        environments: [{
          id: 'Env1', type: 'oxd:Road', num_lanes: 1,
          start_timestamp: '0:0.0', end_timestamp: '0:1.0',
        }],
        conditions: [], traffic_objects: [], traffic_lights: [],
        ego_vehicle: { actions: [] }, agents: [],
        brief_description: '',
      },
    }) as WireBundle
    const result = hydrateUiExtension(bundle)
    expect(result.annotation.environments[0]._track_index).toBeUndefined()
  })

  it('lifts track_index off the envelope onto the matching entity', () => {
    const bundle = mkBundle({
      annotation: {
        environments: [
          { id: 'Env1', type: 'oxd:Road', num_lanes: 1,
            start_timestamp: '0:0.0', end_timestamp: '0:1.0' },
          { id: 'Env2', type: 'oxd:Road', num_lanes: 2,
            start_timestamp: '0:1.0', end_timestamp: '0:2.0' },
        ],
        conditions: [], traffic_objects: [], traffic_lights: [],
        ego_vehicle: { actions: [] }, agents: [],
        brief_description: '',
      },
    }) as WireBundle
    bundle._extensions = {
      'ui/1.0': {
        track_index: { Env1: 0, Env2: 1 },
      },
    }
    const result = hydrateUiExtension(bundle)
    expect(result.annotation.environments[0]._track_index).toBe(0)
    expect(result.annotation.environments[1]._track_index).toBe(1)
    // The wire envelope is stripped from the runtime bundle after hydration.
    expect((result as WireBundle)._extensions).toBeUndefined()
  })

  it('hydrates nested-entity indices (Containment cont_track_index under Agent)', () => {
    const bundle = mkBundle({
      annotation: {
        environments: [], conditions: [], traffic_objects: [],
        traffic_lights: [],
        ego_vehicle: { actions: [] },
        agents: [{
          id: 'A1', amount: 'Single', type: 'oxd:Car',
          visibility_start_timestamp: '0:0.0',
          visibility_end_timestamp: '0:5.0',
          actions: [],
          containment: [
            { id: 'C1', env_id: 'env_0', start_timestamp: '0:0.0', end_timestamp: '0:1.0' },
            { id: 'C2', env_id: 'env_0', start_timestamp: '0:0.0', end_timestamp: '0:1.0' },
          ],
        }],
        brief_description: '',
      },
    }) as WireBundle
    bundle._extensions = {
      'ui/1.0': {
        cont_track_index: { C1: 0, C2: 1 },
      },
    }
    const result = hydrateUiExtension(bundle)
    // `_cont_track_index` is a runtime field attached by the wire-adapter; the
    // schema's `Containment` type intentionally doesn't declare it (2.2.0
    // moved layout indices to the sidecar). Cast to read it the same way the
    // adapter writes it.
    const containment = result.annotation.agents![0].containment! as unknown as
      Array<{ _cont_track_index?: number }>
    expect(containment[0]._cont_track_index).toBe(0)
    expect(containment[1]._cont_track_index).toBe(1)
  })

  it('skips ids that have no matching entity (no errors on stale keys)', () => {
    const bundle = mkBundle({
      annotation: {
        environments: [{ id: 'Env1', type: 'oxd:Road', num_lanes: 1,
          start_timestamp: '0:0.0', end_timestamp: '0:1.0' }],
        conditions: [], traffic_objects: [], traffic_lights: [],
        ego_vehicle: { actions: [] }, agents: [],
        brief_description: '',
      },
    }) as WireBundle
    bundle._extensions = {
      'ui/1.0': {
        track_index: { Env1: 2, GhostEntity: 99 },
      },
    }
    const result = hydrateUiExtension(bundle)
    expect(result.annotation.environments[0]._track_index).toBe(2)
    // The ghost entity simply has nowhere to land — no crash, no leak.
  })

  it('handles a missing ui/1.0 payload (other extensions present)', () => {
    const bundle = mkBundle() as WireBundle
    bundle._extensions = { 'bbox/1.0': { somekey: 'somevalue' } }
    const result = hydrateUiExtension(bundle)
    expect((result as WireBundle)._extensions).toBeUndefined()
  })
})

describe('extractUiExtension', () => {
  it('returns a wire bundle with indices in the ui/1.0 payload, stripped from entities in the wire copy', () => {
    const runtime = mkBundle({
      annotation: {
        environments: [{ id: 'Env1', type: 'oxd:Road', num_lanes: 1,
          start_timestamp: '0:0.0', end_timestamp: '0:1.0',
          _track_index: 3 }],
        conditions: [], traffic_objects: [], traffic_lights: [],
        ego_vehicle: { actions: [] }, agents: [],
        brief_description: '',
      },
    })
    const wire = extractUiExtension(runtime)
    expect(wire._extensions).toBeDefined()
    expect((wire._extensions!['ui/1.0'] as { track_index: Record<string, number> }).track_index)
      .toEqual({ Env1: 3 })
    // Wire copy has the inline field stripped …
    expect(wire.annotation.environments[0]._track_index).toBeUndefined()
    // … but the runtime bundle is not mutated.
    expect(runtime.annotation.environments[0]._track_index).toBe(3)
  })

  it('omits the _extensions envelope entirely when no indices are present', () => {
    const runtime = mkBundle({
      annotation: {
        environments: [{ id: 'Env1', type: 'oxd:Road', num_lanes: 1,
          start_timestamp: '0:0.0', end_timestamp: '0:1.0' }],
        conditions: [], traffic_objects: [], traffic_lights: [],
        ego_vehicle: { actions: [] }, agents: [],
        brief_description: '',
      },
    })
    const wire = extractUiExtension(runtime)
    expect(wire._extensions).toBeUndefined()
  })

  it('aggregates multiple index keys across the entity tree', () => {
    const runtime = mkBundle({
      annotation: {
        environments: [{ id: 'Env1', type: 'oxd:Road', num_lanes: 1,
          start_timestamp: '0:0.0', end_timestamp: '0:1.0', _track_index: 0 }],
        conditions: [{
          id: 'C1', env_id: 'Env1', type: ['Clear'] as unknown as string,
          start_timestamp: '0:0.0', end_timestamp: '0:1.0',
          _track_index: 0, _cond_track_index: 2,
        }],
        traffic_objects: [], traffic_lights: [],
        ego_vehicle: { actions: [] },
        agents: [{
          id: 'A1', amount: 'Single', type: 'oxd:Car',
          visibility_start_timestamp: '0:0.0',
          visibility_end_timestamp: '0:5.0',
          actions: [],
          properties: [{
            id: 'P1', property_type: 'Parked',
            start_timestamp: '0:0.0', end_timestamp: '0:1.0',
            // `_prop_track_index` is a runtime field; AgentProperty's TS type
            // doesn't declare it. Mirror the wire-adapter and attach via cast.
            _prop_track_index: 1,
          } as unknown as import('./types').AgentProperty],
          _track_index: 1,
        }],
        brief_description: '',
      },
    })
    const wire = extractUiExtension(runtime)
    const ui = wire._extensions!['ui/1.0'] as {
      track_index?: Record<string, number>
      cond_track_index?: Record<string, number>
      prop_track_index?: Record<string, number>
    }
    expect(ui.track_index).toEqual({ Env1: 0, C1: 0, A1: 1 })
    expect(ui.cond_track_index).toEqual({ C1: 2 })
    expect(ui.prop_track_index).toEqual({ P1: 1 })
  })

  it('stale-key sweep: deleted entities cannot orphan indices', () => {
    // Simulate the post-delete state — the agent that "had" track_index is
    // gone from the array, so the sidecar payload simply doesn't include it.
    const runtime = mkBundle({
      annotation: {
        environments: [], conditions: [], traffic_objects: [], traffic_lights: [],
        ego_vehicle: { actions: [] },
        agents: [{
          id: 'A2', amount: 'Single', type: 'oxd:Car',
          visibility_start_timestamp: '0:0.0',
          visibility_end_timestamp: '0:5.0',
          actions: [], _track_index: 0,
        }],
        brief_description: '',
      },
    })
    const wire = extractUiExtension(runtime)
    const ui = wire._extensions!['ui/1.0'] as { track_index?: Record<string, number> }
    // No A1 key — only the live A2 lands in the payload.
    expect(ui.track_index).toEqual({ A2: 0 })
    expect(ui.track_index!.A1).toBeUndefined()
  })

  it('round-trip: extract then hydrate restores the same in-memory shape', () => {
    const runtime = mkBundle({
      annotation: {
        environments: [
          { id: 'E1', type: 'oxd:Road', num_lanes: 1,
            start_timestamp: '0:0.0', end_timestamp: '0:1.0', _track_index: 0 },
          { id: 'E2', type: 'oxd:Road', num_lanes: 2,
            start_timestamp: '0:1.0', end_timestamp: '0:2.0', _track_index: 1 },
        ],
        conditions: [], traffic_objects: [], traffic_lights: [],
        ego_vehicle: { actions: [] }, agents: [],
        brief_description: '',
      },
    })
    const wire = extractUiExtension(runtime)
    const rehydrated = hydrateUiExtension(wire)
    expect(rehydrated.annotation.environments[0]._track_index).toBe(0)
    expect(rehydrated.annotation.environments[1]._track_index).toBe(1)
  })
})
