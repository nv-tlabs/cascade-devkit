// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * Wire-adapter for the ``ui/1.0`` schema extension.
 *
 * As of schema 2.2.0 the timeline-layout indices (`_track_index`,
 * `_cond_track_index`, etc.) live in the bundle's sidecar under
 * `_extensions["ui/1.0"]`, keyed by entity id. The annotator's runtime
 * model still carries them on each entity for ergonomics — the renderer
 * reads `env._track_index`, the drag handler writes `agent._track_index`,
 * nothing in the timeline pipeline needs to know about the sidecar.
 *
 * This module bridges the on-the-wire shape and the runtime shape:
 *
 *   GET  /annotations  →  server sends `bundle + _extensions`
 *                      →  `hydrateUiExtension` lifts payload values onto
 *                         each entity's `_track_index` etc. in place
 *                      →  rest of the frontend sees the pre-2.2.0 shape
 *
 *   PUT  /annotations  →  `extractUiExtension` walks every entity, pulls
 *                         the `_*_track_index` values back into the
 *                         payload, strips the fields off entities
 *                      →  client sends `bundle + _extensions`
 *
 * Stale-key invariant: extract walks the *live* entity tree, so deleted
 * entities cannot orphan indices in the sidecar — there is no entity
 * left to read the index off, so nothing gets emitted for that id.
 */

import type {
  Agent,
  AnnotationBundle,
  Containment,
  Influence,
  LightStates,
  SignalHead,
  SilAvAnnotation,
  TrafficLight,
  TrafficObject,
} from './types'

/** On-the-wire sidecar payload as the FastAPI server emits it. */
export interface UiExtensionPayload {
  track_index?: Record<string, number>
  cond_track_index?: Record<string, number>
  cont_track_index?: Record<string, number>
  state_track_index?: Record<string, number>
  influence_track_index?: Record<string, number>
  prop_track_index?: Record<string, number>
}

/** Map of all extension payloads served alongside a bundle. */
export type ExtensionEnvelope = Record<string, unknown>

/** Bundle shape on the wire — main bundle keys + ``_extensions`` envelope. */
export interface WireBundle extends AnnotationBundle {
  _extensions?: ExtensionEnvelope
}

type IndexKey =
  | 'track_index'
  | 'cond_track_index'
  | 'cont_track_index'
  | 'state_track_index'
  | 'influence_track_index'
  | 'prop_track_index'

const ALL_INDEX_KEYS: readonly IndexKey[] = [
  'track_index',
  'cond_track_index',
  'cont_track_index',
  'state_track_index',
  'influence_track_index',
  'prop_track_index',
] as const

interface EntityWithIndices {
  id?: string
  _track_index?: number
  _cond_track_index?: number
  _cont_track_index?: number
  _state_track_index?: number
  _influence_track_index?: number
  _prop_track_index?: number
}

/** Walk every entity in the annotation tree that may carry an index. */
function* walkEntities(ann: SilAvAnnotation | undefined): Generator<EntityWithIndices> {
  if (!ann) return
  for (const env of ann.environments || []) yield env as unknown as EntityWithIndices
  for (const cond of ann.conditions || []) yield cond as unknown as EntityWithIndices
  for (const obj of (ann.traffic_objects || []) as TrafficObject[]) {
    yield obj as unknown as EntityWithIndices
    for (const cont of (obj.containment || []) as Containment[]) {
      yield cont as unknown as EntityWithIndices
    }
  }
  for (const light of (ann.traffic_lights || []) as TrafficLight[]) {
    yield light as unknown as EntityWithIndices
    for (const cont of (light.containment || []) as Containment[]) {
      yield cont as unknown as EntityWithIndices
    }
    for (const head of (light.signal_heads || []) as SignalHead[]) {
      for (const cont of (head.env_controlled || []) as Containment[]) {
        yield cont as unknown as EntityWithIndices
      }
      for (const state of (head.state_sequence || []) as LightStates[]) {
        yield state as unknown as EntityWithIndices
      }
    }
  }
  if (ann.ego_vehicle) {
    for (const prop of ann.ego_vehicle.properties || []) {
      yield prop as unknown as EntityWithIndices
    }
    for (const cont of (ann.ego_vehicle.containment || []) as Containment[]) {
      yield cont as unknown as EntityWithIndices
    }
    for (const infl of (ann.ego_vehicle.influenced_by || []) as Influence[]) {
      yield infl as unknown as EntityWithIndices
    }
  }
  for (const agent of (ann.agents || []) as Agent[]) {
    yield agent as unknown as EntityWithIndices
    for (const prop of agent.properties || []) {
      yield prop as unknown as EntityWithIndices
    }
    for (const cont of (agent.containment || []) as Containment[]) {
      yield cont as unknown as EntityWithIndices
    }
    for (const infl of (agent.influenced_by || []) as Influence[]) {
      yield infl as unknown as EntityWithIndices
    }
  }
}

const INDEX_KEY_TO_FIELD: Record<IndexKey, keyof EntityWithIndices> = {
  track_index: '_track_index',
  cond_track_index: '_cond_track_index',
  cont_track_index: '_cont_track_index',
  state_track_index: '_state_track_index',
  influence_track_index: '_influence_track_index',
  prop_track_index: '_prop_track_index',
}

/**
 * Mutate ``bundle`` in place: lift indices from the ``ui/1.0`` payload onto
 * each entity's typed field. Idempotent; a second call with the same payload
 * is a no-op.
 */
export function hydrateUiExtension(bundle: WireBundle): AnnotationBundle {
  const env = (bundle._extensions ?? {}) as ExtensionEnvelope
  const payload = env['ui/1.0'] as UiExtensionPayload | undefined
  if (!payload) {
    // Drop the _extensions envelope from the runtime shape regardless.
    delete bundle._extensions
    return bundle
  }
  for (const entity of walkEntities(bundle.annotation)) {
    const id = entity.id
    if (!id) continue
    for (const key of ALL_INDEX_KEYS) {
      const map = payload[key]
      if (!map) continue
      const value = map[id]
      if (typeof value === 'number') {
        entity[INDEX_KEY_TO_FIELD[key]] = value
      }
    }
  }
  delete bundle._extensions
  return bundle
}

/**
 * Build the wire envelope from the runtime bundle: read every entity's
 * `_*_track_index` fields and pack them into the ``ui/1.0`` payload,
 * stripping the fields off the in-memory entities so the main JSON sent to
 * the server has the 2.2.0 shape (no inline indices).
 *
 * Returns a new wire bundle (does not mutate the input's array references —
 * the caller can keep using the runtime bundle if they wish; the per-entity
 * `_*_track_index` properties on the input object are removed in place,
 * which is the price of the strip).
 */
export function extractUiExtension(bundle: AnnotationBundle): WireBundle {
  const payload: UiExtensionPayload = {}
  for (const entity of walkEntities(bundle.annotation)) {
    const id = entity.id
    if (!id) continue
    for (const key of ALL_INDEX_KEYS) {
      const field = INDEX_KEY_TO_FIELD[key]
      const value = entity[field]
      if (typeof value !== 'number') continue
      let sub = payload[key]
      if (!sub) {
        sub = {}
        payload[key] = sub
      }
      sub[id] = value
    }
  }
  // The runtime bundle keeps the inline fields — they're needed for the
  // local rendering pipeline. We build the wire shape as a shallow copy with
  // the per-entity indices stripped out, leaving the runtime bundle intact.
  const wire = stripIndicesForWire(bundle)
  if (Object.keys(payload).length > 0) {
    wire._extensions = { ...(wire._extensions ?? {}), 'ui/1.0': payload }
  }
  return wire
}

/**
 * Return a deep-enough copy of ``bundle`` with every `_*_track_index` field
 * stripped off each entity. The annotation arrays are cloned to a depth that
 * lets us mutate without touching the in-memory bundle — the renderer holds
 * references to those entity objects and would re-render incorrectly if we
 * mutated them.
 *
 * Implementation note: ``structuredClone`` is the cheapest correct way; it's
 * been a stable browser API for a few years and ships in Node ≥ 17 for the
 * vitest harness too. Performance is fine for our bundle sizes (low hundreds
 * of entities at most).
 */
function stripIndicesForWire(bundle: AnnotationBundle): WireBundle {
  const wire = structuredClone(bundle) as WireBundle
  for (const entity of walkEntities(wire.annotation)) {
    delete entity._track_index
    delete entity._cond_track_index
    delete entity._cont_track_index
    delete entity._state_track_index
    delete entity._influence_track_index
    delete entity._prop_track_index
  }
  return wire
}
