// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import type { SilAvAnnotation, TimelineSegment, TrackId, AgentProperty, Condition } from './types'
import { envDisplayName, EGO_ACTION_DISPLAY_NAMES, AGENT_ACTION_DISPLAY_NAMES, AGENT_TYPE_DISPLAY_NAMES, TRAFFIC_OBJECT_DISPLAY_NAMES, CONDITION_DISPLAY_NAMES, PROPERTY_DISPLAY_NAMES } from './attribute-cycling'
import { nextStringId } from './string-id'
import { trimKeypointsToWindow } from './keypoint-utils'

export { nextStringId } from './string-id'

/**
 * Build the sorted list of free gaps on a track (between/around existing segments).
 */
function buildGaps(others: TimelineSegment[]): [number, number][] {
  const gaps: [number, number][] = []
  let prev = 0
  for (const o of others) {
    if (o.t0 > prev) gaps.push([prev, o.t0])
    prev = Math.max(prev, o.t1)
  }
  gaps.push([prev, Infinity])
  return gaps
}

/**
 * Prevent overlap: clamp a proposed [t0,t1] so it doesn't collide with
 * other segments on the same track (excluding segId).
 * Returns clamped {t0,t1}, or null for 'create' if no space.
 */
export function clampToAvoidOverlap(
  allSegments: TimelineSegment[],
  trackId: string,
  segId: string | null,
  t0: number,
  t1: number,
  mode: 'move' | 'left' | 'right' | 'create',
  isSubtrack?: boolean | 'agent_action' | 'agent_property' | 'agent_pose' | 'agent_cont' | 'obj_cont' | 'light_cont' | 'light_phys_cont' | 'signal_head' | 'ego_influence' | 'ego_property' | 'agent_influence' | 'ego_action',
  stateTrackIndex?: number,
  headIndex?: number,
): { t0: number; t1: number } | null {
  // When checking env/obj/light tracks, subtracks and main segments live on the same trackId
  // but must only collide with their own kind
  const others = allSegments
    .filter(s => {
      if (s.trackId !== trackId || s.id === segId) return false
      if (trackId.startsWith('env_')) {
        const sIsCond = !!(s.meta as Record<string, unknown>)?._isCondSubtrack
        return sIsCond === !!isSubtrack
      }
      if (trackId.startsWith('obj_')) {
        const sIsState = !!(s.meta as Record<string, unknown>)?._isObjStateSubtrack
        const sIsCont = !!(s.meta as Record<string, unknown>)?._isObjContSubtrack
        if (!isSubtrack) return !sIsState && !sIsCont
        if (isSubtrack === 'obj_cont') return sIsCont
        return sIsState
      }
      if (trackId.startsWith('light_')) {
        const sm = s.meta as Record<string, unknown>
        const sIsSignalHead = !!sm?._isSignalHeadSubtrack
        const sIsState = !!sm?._isLightStateSubtrack
        const sIsCont = !!sm?._isLightContSubtrack
        const sIsPhysCont = !!sm?._isLightPhysContSubtrack
        if (!isSubtrack) return !sIsSignalHead && !sIsState && !sIsCont && !sIsPhysCont
        if (isSubtrack === 'light_phys_cont') {
          // Physical containment only collides with other phys cont on same light
          if (!sIsPhysCont) return false
          if (headIndex != null) return (sm._lightIndex as number) === headIndex
          return true
        }
        if (isSubtrack === 'signal_head') {
          // Signal heads only collide with other signal heads on same light
          if (!sIsSignalHead) return false
          if (headIndex != null) return (sm._lightIndex as number) === (s.meta as Record<string, unknown>)?._lightIndex
          return true
        }
        if (isSubtrack === 'light_cont') {
          // env_controlled only collide within same head
          if (!sIsCont) return false
          if (headIndex != null) return (sm._headIndex as number) === headIndex
          return true
        }
        // Light states: filter by head and lane
        if (!sIsState) return false
        if (headIndex != null && (sm._headIndex as number) !== headIndex) return false
        if (stateTrackIndex != null) {
          const sLane = sm?._state_track_index as number ?? 0
          return sLane === stateTrackIndex
        }
        return true
      }
      if (trackId === 'ego_act') {
        const sIsEgoCont = !!(s.meta as Record<string, unknown>)?._isEgoContSubtrack
        const sIsEgoInfluence = !!(s.meta as Record<string, unknown>)?._isEgoInfluenceSubtrack
        const sIsEgoAction = !!(s.meta as Record<string, unknown>)?._isEgoActionSubtrack
        const sIsEgoProperty = !!(s.meta as Record<string, unknown>)?._isEgoPropertySubtrack
        const sIsEgoMain = !!(s.meta as Record<string, unknown>)?._isEgoMain
        if (isSubtrack === 'ego_influence') return sIsEgoInfluence
        if (isSubtrack === 'ego_action') return sIsEgoAction
        if (isSubtrack === 'ego_property') return sIsEgoProperty
        if (!isSubtrack) return !sIsEgoCont && !sIsEgoInfluence && !sIsEgoAction && !sIsEgoProperty && !sIsEgoMain
        return sIsEgoCont
      }
      if (trackId.startsWith('agent_')) {
        const sIsAction = !!(s.meta as Record<string, unknown>)?._isAgentActionSubtrack
        const sIsPose = !!(s.meta as Record<string, unknown>)?._isAgentPoseSubtrack
        const sIsCont = !!(s.meta as Record<string, unknown>)?._isAgentContSubtrack
        const sIsInfluence = !!(s.meta as Record<string, unknown>)?._isAgentInfluenceSubtrack
        const sIsProperty = !!(s.meta as Record<string, unknown>)?._isAgentPropertySubtrack
        if (!isSubtrack) return !sIsAction && !sIsPose && !sIsCont && !sIsInfluence && !sIsProperty
        if (isSubtrack === 'agent_action') return sIsAction
        if (isSubtrack === 'agent_property') return sIsProperty
        if (isSubtrack === 'agent_pose') return sIsPose
        if (isSubtrack === 'agent_cont') return sIsCont
        if (isSubtrack === 'agent_influence') return sIsInfluence
        return sIsAction || sIsPose || sIsCont || sIsInfluence || sIsProperty
      }
      return true
    })
    .sort((a, b) => a.t0 - b.t0)

  if (others.length === 0) return { t0, t1 }

  const segLen = t1 - t0

  if (mode === 'create') {
    // Check if proposed range is clear
    let overlaps = false
    for (const o of others) {
      if (t0 < o.t1 && t1 > o.t0) { overlaps = true; break }
    }
    if (overlaps) {
      // Find the gap containing the center and shrink to fit
      const center = (t0 + t1) / 2
      const gaps = buildGaps(others)
      for (const [gl, gr] of gaps) {
        if (center >= gl && center < gr) {
          const gapLen = gr - gl
          if (gapLen < 0.2) return null // too small
          return { t0: gl, t1: gr }
        }
      }
      return null
    }
    return { t0, t1 }
  }

  const gaps = buildGaps(others)

  if (mode === 'left') {
    for (const [gl, gr] of gaps) {
      if (t1 > gl && t1 <= gr) {
        return { t0: Math.max(t0, gl), t1 }
      }
    }
    return { t0, t1 }
  }

  if (mode === 'right') {
    for (const [gl, gr] of gaps) {
      if (t0 >= gl && t0 < gr) {
        return { t0, t1: Math.min(t1, gr) }
      }
    }
    return { t0, t1 }
  }

  // mode === 'move'
  const center = (t0 + t1) / 2
  for (const [gl, gr] of gaps) {
    if (center >= gl && center < gr) {
      if (gr - gl < segLen) return null
      let newT0 = Math.max(t0, gl)
      let newT1 = newT0 + segLen
      if (newT1 > gr) { newT1 = gr; newT0 = newT1 - segLen }
      return { t0: newT0, t1: newT1 }
    }
  }

  let bestGap: [number, number] | null = null
  let bestDist = Infinity
  for (const [gl, gr] of gaps) {
    if (gr - gl < segLen) continue
    const dist = Math.min(Math.abs(center - gl), Math.abs(center - gr))
    if (dist < bestDist) { bestDist = dist; bestGap = [gl, gr] }
  }
  if (!bestGap) return null

  let newT0 = Math.max(t0, bestGap[0])
  let newT1 = newT0 + segLen
  if (newT1 > bestGap[1]) { newT1 = bestGap[1]; newT0 = newT1 - segLen }
  return { t0: newT0, t1: newT1 }
}

/**
 * Greedy interval-coloring: assign _track_index to items so no two
 * overlapping items share the same track.  Returns the number of tracks used.
 */
function assignTracks(items: { start: number; end: number; item: Record<string, unknown> }[], key = '_track_index'): number {
  // Single pass in start-time order. For each item: if it already has an index
  // and that lane is free, keep it (stability). Otherwise assign to the lowest
  // free lane. This avoids the two-pass bug where a later-starting item with a
  // persisted index claims a lane before an earlier-starting new item.
  const sorted = [...items].sort((a, b) => a.start - b.start || a.end - b.end)
  const trackEnds: number[] = []
  for (const { start, end, item } of sorted) {
    const existing = item[key]
    if (typeof existing === 'number' && existing >= 0) {
      while (trackEnds.length <= existing) trackEnds.push(0)
      if (trackEnds[existing] <= start) {
        trackEnds[existing] = end
        continue
      }
    }
    // Assign to lowest free lane
    let placed = false
    for (let t = 0; t < trackEnds.length; t++) {
      if (trackEnds[t] <= start) { item[key] = t; trackEnds[t] = end; placed = true; break }
    }
    if (!placed) { item[key] = trackEnds.length; trackEnds.push(end) }
  }
  // Only trim empty lanes from the end — never shuffle mid-track gaps
  let maxLane = 0
  for (const { item } of items) maxLane = Math.max(maxLane, (item[key] as number) + 1)
  return maxLane
}

/**
 * Auto-assign _track_index on all entity groups so overlapping items
 * land on different tracks.  Mutates the annotation in-place.
 */
export function autoAssignOverlappingTracks(ann: SilAvAnnotation): void {
  if (!ann) return

  const p = (ts: string) => {
    if (!ts) return 0
    const m = ts.match(/(\d+):(\d+(?:\.\d+)?)/)
    if (m) return parseInt(m[1]) * 60 + parseFloat(m[2])
    return parseFloat(ts) || 0
  }

  if (ann.environments?.length) {
    assignTracks(ann.environments.map(e => ({ start: p(e.start_timestamp), end: p(e.end_timestamp), item: e as unknown as Record<string, unknown> })))
  }

  // Conditions don't need independent track assignment — they nest under env tracks

  // Ego action containment doesn't need independent track assignment — they nest under the ego_act track

  if ((ann.traffic_objects || []).length) {
    assignTracks(ann.traffic_objects.map(o => ({ start: p(o.visibility_start_timestamp), end: p(o.visibility_end_timestamp), item: o as unknown as Record<string, unknown> })))
  }

  if ((ann.traffic_lights || []).length) {
    assignTracks(ann.traffic_lights.map(l => ({ start: p(l.visibility_start_timestamp), end: p(l.visibility_end_timestamp), item: l as unknown as Record<string, unknown> })))
  }

  // Per-signal-head state lane assignment for multi-lane LightState subtracks
  for (const light of ann.traffic_lights || []) {
    for (const sh of light.signal_heads || []) {
      if ((sh.state_sequence || []).length > 0) {
        const laneCount = assignTracks(
          sh.state_sequence.map(st => ({
            start: p(st.start_timestamp), end: p(st.end_timestamp),
            item: st as unknown as Record<string, unknown>,
          })),
          '_state_track_index'
        )
        ;(sh as unknown as Record<string, unknown>)._state_lane_count = Math.max(1, laneCount)
      } else {
        ;(sh as unknown as Record<string, unknown>)._state_lane_count = 1
      }
    }
  }

  if (ann.agents?.length) {
    const agentSpans = ann.agents.map(a => {
      let start: number, end: number
      if (a.visibility_start_timestamp && a.visibility_end_timestamp) {
        start = p(a.visibility_start_timestamp)
        end = p(a.visibility_end_timestamp)
      } else {
        start = Infinity; end = 0
        for (const act of a.actions || []) { start = Math.min(start, p(act.start_timestamp)); end = Math.max(end, p(act.end_timestamp)) }
        for (const sp of a.ego_relative_pose || []) { start = Math.min(start, p(sp.start_timestamp)); end = Math.max(end, p(sp.end_timestamp)) }
        if (!isFinite(start)) start = 0
      }
      return { start, end, item: a as unknown as Record<string, unknown> }
    })
    assignTracks(agentSpans)
  }
}

export function formatTs(sec: number): string {
  const m = Math.floor(sec / 60)
  const s = sec % 60
  return `${m}:${s.toFixed(1)}`
}

export function parseTs(ts: string): number {
  if (!ts) return 0
  const m = ts.match(/(\d+):(\d+(?:\.\d+)?)/)
  if (m) return parseInt(m[1]) * 60 + parseFloat(m[2])
  return parseFloat(ts) || 0
}

export interface TrackConfig {
  id: TrackId
  name: string
  color: string
  group: string
  _collapsed?: boolean
}

export type DynamicGroup = 'Environments' | 'Objects' | 'TrafficLights' | 'Agents' | 'EgoContainment'

/**
 * Resolve a `--color-entity-*` CSS variable from <html>. Returns the trimmed
 * value or an empty string when unavailable (SSR / tests). Callers should
 * fall back to a sensible default when the result is empty.
 *
 * Reading CSS vars on each render is cheap (a single property lookup), but
 * callers are expected to memoize on `effectiveTheme` so the values flip
 * atomically when the theme changes.
 */
export function readEntityCssVar(name: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return v || fallback
}

const ENTITY_DEFAULTS = {
  env: '#22c55e',
  object: '#f59e0b',
  agent: '#a855f7',
  light: '#ef4444',
  ego: '#3b82f6',
}

export function buildTrackList(
  _ann: SilAvAnnotation | undefined,
  envTrackCount: number,
  objectTrackCount: number,
  agentTrackCount: number,
  _egoContTrackCount: number = 1,
  lightTrackCount: number = 1
): TrackConfig[] {
  const envColor = readEntityCssVar('--color-entity-env', ENTITY_DEFAULTS.env)
  const objColor = readEntityCssVar('--color-entity-object', ENTITY_DEFAULTS.object)
  const lightColor = readEntityCssVar('--color-entity-light', ENTITY_DEFAULTS.light)
  const agentColor = readEntityCssVar('--color-entity-agent', ENTITY_DEFAULTS.agent)
  const egoColor = readEntityCssVar('--color-entity-ego', ENTITY_DEFAULTS.ego)

  const envTracks: TrackConfig[] = []
  for (let i = 0; i < envTrackCount; i++) {
    envTracks.push({ id: `env_${i}`, name: `Env Track ${i + 1}`, color: envColor, group: 'Environments' })
  }

  const objTracks: TrackConfig[] = []
  for (let i = 0; i < objectTrackCount; i++) {
    objTracks.push({ id: `obj_${i}`, name: `Object Track ${i + 1}`, color: objColor, group: 'Objects' })
  }

  const lightTracks: TrackConfig[] = []
  for (let i = 0; i < lightTrackCount; i++) {
    lightTracks.push({ id: `light_${i}`, name: `Light Track ${i + 1}`, color: lightColor, group: 'TrafficLights' })
  }

  const agentTracks: TrackConfig[] = []
  for (let i = 0; i < agentTrackCount; i++) {
    agentTracks.push({ id: `agent_${i}`, name: `Agent Track ${i + 1}`, color: agentColor, group: 'Agents' })
  }

  const egoTrack: TrackConfig = { id: 'ego_act', name: 'Ego Actions', color: egoColor, group: 'Ego' }
  return [...envTracks, ...lightTracks, ...objTracks, ...agentTracks, egoTrack]
}

// TRACK_CONFIG is referenced as a const elsewhere — keep a single default
// ego-action entry. Consumers that need theme-reactive colors should call
// buildTrackList() inside a useMemo keyed on `effectiveTheme`.
export const TRACK_CONFIG: TrackConfig[] = [
  { id: 'ego_act', name: 'Ego Actions', color: ENTITY_DEFAULTS.ego, group: 'Ego' },
]

/** Migrate legacy ego action types to new enum values (in-place). */
function migrateEgoActions(ann: SilAvAnnotation): void {
  for (const act of ann.ego_vehicle?.actions || []) {
    const legacy = act as unknown as Record<string, unknown>
    if (act.type === 'oxd:ChangeLane') {
      act.type = legacy.change_where === 'Right' ? 'oxd:ChangeLane (right)' : 'oxd:ChangeLane (left)'
      delete legacy.change_where
    }
    if (act.type === 'oxd:MakeALeftTurn' || act.type === 'oxd:MakeARightTurn' || act.type === 'fst:MakeAUTurn' || act.type === 'Other oxd:MakeATurn') {
      const suffix = legacy.turn_protected ? ' (protected)' : ' (unprotected)'
      act.type = act.type + suffix
      delete legacy.turn_protected
    }
  }
}

/** Migrate legacy agent action types to new enum values (in-place). */
function migrateAgentActions(ann: SilAvAnnotation): void {
  for (const agent of ann.agents || []) {
    const existingPropIds = new Set((agent.properties || []).map(p => p.id).filter(Boolean) as string[])
    const addProp = (property_type: string, start_timestamp: string, end_timestamp: string) => {
      const alreadyHas = (agent.properties || []).some(
        p => p.property_type === property_type && p.start_timestamp === start_timestamp && p.end_timestamp === end_timestamp
      )
      if (alreadyHas) return
      const id = nextStringId('AgentProperty', [...existingPropIds])
      existingPropIds.add(id)
      agent.properties = agent.properties || []
      agent.properties.push({ id, property_type, start_timestamp, end_timestamp })
    }

    for (const act of agent.actions || []) {
      const at = act.action_type
      // Decompose compound Stop variants → base type + create properties directly
      if (at === 'oxd:Stop (emergency situation)') {
        act.action_type = 'oxd:Stop'; addProp('Emergency', act.start_timestamp, act.end_timestamp)
      } else if (at === 'oxd:Stop (double-parked)') {
        act.action_type = 'oxd:Stop'; addProp('Double Parked', act.start_timestamp, act.end_timestamp)
      } else if (at === 'oxd:Stop (emergency situation, double-parked)') {
        act.action_type = 'oxd:Stop'; addProp('Emergency', act.start_timestamp, act.end_timestamp); addProp('Double Parked', act.start_timestamp, act.end_timestamp)
      }
      // Decompose compound NotMove variants
      if (at === 'oxd:NotMove (emergency situation)') {
        act.action_type = 'oxd:NotMove'; addProp('Emergency', act.start_timestamp, act.end_timestamp)
      } else if (at === 'oxd:NotMove (double-parked)') {
        act.action_type = 'oxd:NotMove'; addProp('Double Parked', act.start_timestamp, act.end_timestamp)
      } else if (at === 'oxd:NotMove (emergency situation, double-parked)') {
        act.action_type = 'oxd:NotMove'; addProp('Emergency', act.start_timestamp, act.end_timestamp); addProp('Double Parked', act.start_timestamp, act.end_timestamp)
      }
      // Decompose compound Park variant
      if (at === 'fst:Park (double-parked)') {
        act.action_type = 'fst:Park'; addProp('Double Parked', act.start_timestamp, act.end_timestamp)
      }
      // Walk/Run variants
      if (at === 'oxd:Walk' || at === 'oxd:Run') {
        const parts: string[] = []
        if (act.jaywalk_flag) parts.push('jaywalk')
        if (act.erratic_flag) parts.push('erratic')
        if (parts.length > 0) {
          act.action_type = `${at} (${parts.join(', ')})`
          delete act.jaywalk_flag; delete act.erratic_flag
        }
      }
      // Decompose compound Stand variant
      if (at === 'oxd:Stand (emergency situation)') {
        act.action_type = 'oxd:Stand'; addProp('Emergency', act.start_timestamp, act.end_timestamp)
      }
      // Nudge variants
      if (at === 'fst:Nudge') {
        if (act.nudge_magnitude === 'Out-of-Lane') {
          act.action_type = act.ego_lane_flag ? 'fst:Nudge (out of lane: into ego lane)' : 'fst:Nudge (out of lane: not into ego lane)'
        } else {
          act.action_type = 'fst:Nudge (in lane)'
        }
        delete act.nudge_magnitude; delete act.ego_lane_flag
      }
      // Overtake variants
      if (at === 'oxd:Overtake') {
        act.action_type = act.ego_lane_flag ? 'oxd:Overtake (using ego lane)' : 'oxd:Overtake (not using ego lane)'
        delete act.ego_lane_flag
      }
      // ChangeLane
      if (at === 'oxd:ChangeLane') {
        act.action_type = act.change_where === 'Right' ? 'oxd:ChangeLane (right)' : 'oxd:ChangeLane (left)'
        delete act.change_where
      }
      // Turns
      if (at === 'oxd:MakeALeftTurn' || at === 'oxd:MakeARightTurn' || at === 'fst:MakeAUTurn') {
        const suffix = act.turn_protected ? ' (protected)' : ' (unprotected)'
        act.action_type = at + suffix
        delete act.turn_protected
      }
    }
  }
}

/** Migrate legacy Environment.lanes_obscured_or_unmarked flag into a
 * Condition of the new "Lanes obscured / unmarked" type. */
function migrateLanesObscuredToCondition(ann: SilAvAnnotation): void {
  const envs = ann.environments || []
  if (envs.length === 0) return
  ann.conditions = ann.conditions || []
  const existingIds = ann.conditions.map(c => c.id).filter(Boolean)
  for (const env of envs) {
    const e = env as unknown as Record<string, unknown>
    if (e.lanes_obscured_or_unmarked === true) {
      const newId = nextStringId('Condition', existingIds)
      existingIds.push(newId)
      ann.conditions.push({
        id: newId,
        env_id: env.id,
        type: 'Lanes obscured / unmarked',
        start_timestamp: env.start_timestamp,
        end_timestamp: env.end_timestamp,
      })
    }
    delete e.lanes_obscured_or_unmarked
  }
}

/** Migrate legacy condition types: array → single string (in-place).
 * Conditions with multiple types are split into separate conditions sharing
 * env_id and timestamps; `assignTracks` later places them on parallel rows. */
function migrateConditionTypes(ann: SilAvAnnotation): void {
  const conds = ann.conditions || []
  if (conds.length === 0) return
  const out: Condition[] = []
  const existingIds = conds.map(c => c.id).filter(Boolean)
  for (const cond of conds) {
    const raw = (cond as unknown as Record<string, unknown>).type
    const types: string[] = Array.isArray(raw)
      ? (raw as string[]).filter(t => t && t !== 'Clear')
      : (typeof raw === 'string' && raw && raw !== 'Clear' ? [raw] : [])
    // Drop conditions that become empty after removing 'Clear' — absence of
    // a condition now implicitly means clear.
    if (types.length === 0 && raw !== undefined && (raw === 'Clear' || (Array.isArray(raw) && (raw as string[]).every(t => t === 'Clear' || !t)))) {
      continue
    }
    if (types.length === 0) {
      cond.type = ''
      out.push(cond)
    } else {
      cond.type = types[0]
      out.push(cond)
      for (let i = 1; i < types.length; i++) {
        const newId = nextStringId('Condition', existingIds)
        existingIds.push(newId)
        const clone: Condition = { ...cond, id: newId, type: types[i] }
        delete (clone as unknown as Record<string, unknown>)._cond_track_index
        out.push(clone)
      }
    }
  }
  ann.conditions = out
}

/** Assign IDs to AgentActions and LightStates that lack them (in-place). */
function migrateSubEntityIds(ann: SilAvAnnotation): void {
  // AgentAction IDs
  const existingActionIds = new Set<string>()
  for (const agent of ann.agents || []) {
    for (const act of agent.actions || []) {
      const a = act as unknown as Record<string, unknown>
      if (a.id) existingActionIds.add(a.id as string)
    }
  }
  for (const agent of ann.agents || []) {
    for (const act of agent.actions || []) {
      const a = act as unknown as Record<string, unknown>
      if (!a.id) {
        const newId = nextStringId('AgentAction', [...existingActionIds])
        a.id = newId
        existingActionIds.add(newId)
      }
    }
  }
  // EgoAction IDs
  const existingEgoActionIds = new Set<string>()
  for (const act of ann.ego_vehicle?.actions || []) {
    if (act.id) existingEgoActionIds.add(act.id)
  }
  for (const act of ann.ego_vehicle?.actions || []) {
    if (!act.id) {
      const newId = nextStringId('EgoAction', [...existingEgoActionIds])
      act.id = newId
      existingEgoActionIds.add(newId)
    }
  }
  // SignalHead IDs
  const existingSignalHeadIds = new Set<string>()
  for (const light of ann.traffic_lights || []) {
    for (const sh of light.signal_heads || []) {
      if (sh.id) existingSignalHeadIds.add(sh.id)
    }
  }
  for (const light of ann.traffic_lights || []) {
    for (const sh of light.signal_heads || []) {
      if (!sh.id) {
        const newId = nextStringId('SignalHead', [...existingSignalHeadIds])
        sh.id = newId
        existingSignalHeadIds.add(newId)
      }
    }
  }
  // LightStates IDs
  const existingLightStateIds = new Set<string>()
  for (const light of ann.traffic_lights || []) {
    for (const sh of light.signal_heads || []) {
      for (const st of sh.state_sequence || []) {
        const s = st as unknown as Record<string, unknown>
        if (s.id) existingLightStateIds.add(s.id as string)
      }
    }
  }
  for (const light of ann.traffic_lights || []) {
    for (const sh of light.signal_heads || []) {
      for (const st of sh.state_sequence || []) {
        const s = st as unknown as Record<string, unknown>
        if (!s.id) {
          const newId = nextStringId('LightState', [...existingLightStateIds])
          s.id = newId
          existingLightStateIds.add(newId)
        }
      }
    }
  }
  // EgoContainment IDs (vehicle-level)
  const existingContIds = new Set<string>()
  for (const c of ann.ego_vehicle?.containment || []) {
    if (c.id) existingContIds.add(c.id)
  }
  for (const c of ann.ego_vehicle?.containment || []) {
    if (!c.id) {
      const newId = nextStringId('EgoContainment', [...existingContIds])
      c.id = newId
      existingContIds.add(newId)
    }
  }
  // AgentContainment IDs
  for (const agent of ann.agents || []) {
    for (const c of agent.containment || []) {
      if (c.id) existingContIds.add(c.id)
    }
  }
  for (const agent of ann.agents || []) {
    for (const c of agent.containment || []) {
      if (!c.id) {
        const newId = nextStringId('AgentContainment', [...existingContIds])
        c.id = newId
        existingContIds.add(newId)
      }
    }
  }
  // LightContainment IDs (TrafficLight-level)
  for (const light of ann.traffic_lights || []) {
    for (const c of light.containment || []) {
      if (c.id) existingContIds.add(c.id)
    }
  }
  for (const light of ann.traffic_lights || []) {
    for (const c of light.containment || []) {
      if (!c.id) {
        const newId = nextStringId('LightContainment', [...existingContIds])
        c.id = newId
        existingContIds.add(newId)
      }
    }
  }
  // SignalHead env_controlled IDs
  for (const light of ann.traffic_lights || []) {
    for (const sh of light.signal_heads || []) {
      for (const c of sh.env_controlled || []) {
        if (c.id) existingContIds.add(c.id)
      }
    }
  }
  for (const light of ann.traffic_lights || []) {
    for (const sh of light.signal_heads || []) {
      for (const c of sh.env_controlled || []) {
        if (!c.id) {
          const newId = nextStringId('LightContainment', [...existingContIds])
          c.id = newId
          existingContIds.add(newId)
        }
      }
    }
  }
}

/** Migrate signaling_details.target_agent_ids → link_to (in-place). */
/** Lift ego containment and influenced_by from per-action to ego_vehicle level. Mutates in place. */
function migrateEgoContainmentToVehicleLevel(ann: SilAvAnnotation): void {
  const ego = ann.ego_vehicle
  if (!ego) return
  // If ego_vehicle already has top-level containment or influenced_by, nothing to do
  if ((ego.containment && ego.containment.length > 0) || (ego.influenced_by && ego.influenced_by.length > 0)) return
  // Check if any action still has the old nested fields
  const hasLegacy = (ego.actions || []).some(a => {
    const la = a as unknown as Record<string, unknown>
    return (la.containment as unknown[] | undefined)?.length || (la.influenced_by as unknown[] | undefined)?.length
  })
  if (!hasLegacy) return
  ego.containment = ego.containment || []
  ego.influenced_by = ego.influenced_by || []
  for (const act of ego.actions || []) {
    const la = act as unknown as Record<string, unknown>
    for (const cont of (la.containment as import('./types').Containment[] | undefined) || []) {
      ego.containment.push(cont)
    }
    for (const infl of (la.influenced_by as import('./types').Influence[] | undefined) || []) {
      ego.influenced_by.push(infl)
    }
    delete la.containment
    delete la.influenced_by
  }
}

function migrateSignalingLinkTo(ann: SilAvAnnotation): void {
  for (const agent of ann.agents || []) {
    for (const act of agent.actions || []) {
      const sd = act.signaling_details as Record<string, unknown> | undefined
      if (sd && sd.target_agent_ids && !sd.link_to) {
        sd.link_to = sd.target_agent_ids
        delete sd.target_agent_ids
      }
    }
  }
}

/** Migrate old TrafficLight flat state_sequence/containment into signal_heads. Mutates in place. */
function migrateTrafficLightSignalHeads(ann: SilAvAnnotation): void {
  for (const light of ann.traffic_lights || []) {
    const legacy = light as unknown as Record<string, unknown>
    const oldStates = legacy.state_sequence as import('./types').LightStates[] | undefined
    if (oldStates && oldStates.length > 0 && (!light.signal_heads || light.signal_heads.length === 0)) {
      // Migrate: create one signal head per light with all the old states
      const oldCont = (legacy.containment as import('./types').Containment[] | undefined) || []
      light.signal_heads = [{
        id: '',
        start_timestamp: light.visibility_start_timestamp,
        end_timestamp: light.visibility_end_timestamp,
        state_sequence: oldStates,
        env_controlled: oldCont,
        influenced_agent_ids: (oldStates[0] as unknown as Record<string, unknown>)?.influenced_agent_ids as string[] || [],
        affects_ego: (oldStates[0] as unknown as Record<string, unknown>)?.affects_ego as string | undefined,
      }]
      delete legacy.state_sequence
      // Keep containment on TrafficLight for physical location but clear it from migration
    }
    if (!light.signal_heads) light.signal_heads = []
  }
}

/** Migrate legacy Signals actions and boolean flags on actions to properties subtrack. Mutates in place. */
function migrateActionFlagsToProperties(ann: SilAvAnnotation): void {
  // Collect existing property IDs to avoid duplicates
  const existingPropIds = new Set<string>()
  for (const agent of ann.agents || []) {
    for (const p of agent.properties || []) if (p.id) existingPropIds.add(p.id)
  }
  for (const p of ann.ego_vehicle?.properties || []) if (p.id) existingPropIds.add(p.id)

  const nextPropId = (prefix: string): string => {
    return nextStringId(prefix, [...existingPropIds])
  }

  // Migrate agent actions
  for (const agent of ann.agents || []) {
    // Skip if agent already has properties (already migrated)
    if (agent.properties && agent.properties.length > 0) continue
    const newProps: AgentProperty[] = []

    const actionsToRemove: number[] = []
    for (let i = 0; i < (agent.actions || []).length; i++) {
      const act = agent.actions[i]

      // Migrate 'Signals' action → Signal property
      if (act.action_type === 'Signals') {
        const propId = nextPropId('AgentProperty')
        existingPropIds.add(propId)
        newProps.push({
          id: propId,
          property_type: 'Signal',
          start_timestamp: act.start_timestamp,
          end_timestamp: act.end_timestamp,
          signaling_details: act.signaling_details,
        })
        actionsToRemove.push(i)
        continue
      }

      // Migrate is_aggressive_or_cut_in flag → Aggressive property
      if (act.is_aggressive_or_cut_in) {
        const propId = nextPropId('AgentProperty')
        existingPropIds.add(propId)
        newProps.push({
          id: propId,
          property_type: 'Aggressive',
          start_timestamp: act.start_timestamp,
          end_timestamp: act.end_timestamp,
        })
        delete act.is_aggressive_or_cut_in
      }

      // Migrate erratic_flag → Erratic property
      if (act.erratic_flag) {
        const propId = nextPropId('AgentProperty')
        existingPropIds.add(propId)
        newProps.push({
          id: propId,
          property_type: 'Erratic',
          start_timestamp: act.start_timestamp,
          end_timestamp: act.end_timestamp,
        })
        delete act.erratic_flag
      }
    }

    // Remove migrated Signals actions (reverse order to preserve indices)
    for (let i = actionsToRemove.length - 1; i >= 0; i--) {
      agent.actions.splice(actionsToRemove[i], 1)
    }

    if (newProps.length > 0) {
      agent.properties = newProps
    }
  }

  // Migrate ego actions
  if (ann.ego_vehicle && !(ann.ego_vehicle.properties && ann.ego_vehicle.properties.length > 0)) {
    const newProps: AgentProperty[] = []
    for (const act of ann.ego_vehicle.actions || []) {
      if (act.is_aggressive_or_cut_in) {
        const propId = nextPropId('EgoProperty')
        existingPropIds.add(propId)
        newProps.push({
          id: propId,
          property_type: 'Aggressive',
          start_timestamp: act.start_timestamp,
          end_timestamp: act.end_timestamp,
        })
        delete act.is_aggressive_or_cut_in
      }
    }
    if (newProps.length > 0) {
      ann.ego_vehicle.properties = newProps
    }
  }
}

export function annotationToSegments(ann: SilAvAnnotation | undefined): TimelineSegment[] {
  if (!ann) return []
  // Run migrations on first access
  migrateEgoActions(ann)
  migrateAgentActions(ann)
  migrateLanesObscuredToCondition(ann)
  migrateConditionTypes(ann)
  migrateTrafficLightSignalHeads(ann)
  migrateSubEntityIds(ann)
  migrateSignalingLinkTo(ann)
  migrateEgoContainmentToVehicleLevel(ann)
  migrateActionFlagsToProperties(ann)

  // --- Assign all sub-entity track indices BEFORE building segments ---
  // (so segment meta gets the fresh indices via spread)

  // Signal-head state lane indices
  {
    const p = (ts: string) => {
      if (!ts) return 0
      const m = ts.match(/(\d+):(\d+(?:\.\d+)?)/)
      if (m) return parseInt(m[1]) * 60 + parseFloat(m[2])
      return parseFloat(ts) || 0
    }
    for (const light of ann.traffic_lights || []) {
      for (const sh of light.signal_heads || []) {
        if ((sh.state_sequence || []).length > 0) {
          const laneCount = assignTracks(
            sh.state_sequence.map(st => ({
              start: p(st.start_timestamp), end: p(st.end_timestamp),
              item: st as unknown as Record<string, unknown>,
            })),
            '_state_track_index'
          )
          ;(sh as unknown as Record<string, unknown>)._state_lane_count = Math.max(1, laneCount)
        } else {
          ;(sh as unknown as Record<string, unknown>)._state_lane_count = 1
        }
      }
    }
  }
  // Condition track indices (per parent environment's physical track)
  {
    const envIdToTrackIdx = new Map<string, number>()
    for (const env of ann.environments || []) {
      envIdToTrackIdx.set(env.id, env._track_index ?? 0)
    }
    const perTrack = new Map<number, { start: number; end: number; item: Record<string, unknown> }[]>()
    for (const cond of ann.conditions || []) {
      const trackIdx = envIdToTrackIdx.get(cond.env_id) ?? 0
      const items = perTrack.get(trackIdx) || []
      items.push({ start: parseTs(cond.start_timestamp), end: parseTs(cond.end_timestamp), item: cond as unknown as Record<string, unknown> })
      perTrack.set(trackIdx, items)
    }
    for (const items of perTrack.values()) { if (items.length > 0) assignTracks(items, '_cond_track_index') }
  }
  // Containment track indices for ego vehicle
  {
    const allEgoCont: { start: number; end: number; item: Record<string, unknown> }[] = []
    for (const cont of ann.ego_vehicle?.containment || []) {
      allEgoCont.push({ start: parseTs(cont.start_timestamp), end: parseTs(cont.end_timestamp), item: cont as unknown as Record<string, unknown> })
    }
    if (allEgoCont.length > 0) assignTracks(allEgoCont, '_cont_track_index')
  }
  // Containment track indices for agents (per physical track)
  {
    const perTrack = new Map<number, { start: number; end: number; item: Record<string, unknown> }[]>()
    for (const agent of ann.agents || []) {
      const trackIdx = agent._track_index ?? 0
      const items = perTrack.get(trackIdx) || []
      for (const cont of agent.containment || []) {
        items.push({ start: parseTs(cont.start_timestamp), end: parseTs(cont.end_timestamp), item: cont as unknown as Record<string, unknown> })
      }
      perTrack.set(trackIdx, items)
    }
    for (const items of perTrack.values()) { if (items.length > 0) assignTracks(items, '_cont_track_index') }
  }
  // Physical containment track indices for traffic lights (per physical track)
  {
    const perTrack = new Map<number, { start: number; end: number; item: Record<string, unknown> }[]>()
    for (const light of ann.traffic_lights || []) {
      const trackIdx = light._track_index ?? 0
      const items = perTrack.get(trackIdx) || []
      for (const cont of light.containment || []) {
        items.push({ start: parseTs(cont.start_timestamp), end: parseTs(cont.end_timestamp), item: cont as unknown as Record<string, unknown> })
      }
      perTrack.set(trackIdx, items)
    }
    for (const items of perTrack.values()) { if (items.length > 0) assignTracks(items, '_cont_track_index') }
  }
  // Containment track indices for traffic objects (per physical track)
  {
    const perTrack = new Map<number, { start: number; end: number; item: Record<string, unknown> }[]>()
    for (const obj of ann.traffic_objects || []) {
      const trackIdx = obj._track_index ?? 0
      const items = perTrack.get(trackIdx) || []
      for (const cont of obj.containment || []) {
        items.push({ start: parseTs(cont.start_timestamp), end: parseTs(cont.end_timestamp), item: cont as unknown as Record<string, unknown> })
      }
      perTrack.set(trackIdx, items)
    }
    for (const items of perTrack.values()) { if (items.length > 0) assignTracks(items, '_cont_track_index') }
  }
  // Influence track indices for ego vehicle
  {
    const allEgoInfl: { start: number; end: number; item: Record<string, unknown> }[] = []
    for (const infl of ann.ego_vehicle?.influenced_by || []) {
      allEgoInfl.push({ start: parseTs(infl.start_timestamp), end: parseTs(infl.end_timestamp), item: infl as unknown as Record<string, unknown> })
    }
    if (allEgoInfl.length > 0) assignTracks(allEgoInfl, '_influence_track_index')
  }
  // Influence track indices for agents (per physical track)
  {
    const perTrackInfl = new Map<number, { start: number; end: number; item: Record<string, unknown> }[]>()
    for (const agent of ann.agents || []) {
      const trackIdx = agent._track_index ?? 0
      const items = perTrackInfl.get(trackIdx) || []
      for (const infl of agent.influenced_by || []) {
        items.push({ start: parseTs(infl.start_timestamp), end: parseTs(infl.end_timestamp), item: infl as unknown as Record<string, unknown> })
      }
      perTrackInfl.set(trackIdx, items)
    }
    for (const items of perTrackInfl.values()) { if (items.length > 0) assignTracks(items, '_influence_track_index') }
  }
  // Property track indices for ego vehicle
  {
    const allEgoProps: { start: number; end: number; item: Record<string, unknown> }[] = []
    for (const prop of ann.ego_vehicle?.properties || []) {
      allEgoProps.push({ start: parseTs(prop.start_timestamp), end: parseTs(prop.end_timestamp), item: prop as unknown as Record<string, unknown> })
    }
    if (allEgoProps.length > 0) assignTracks(allEgoProps, '_prop_track_index')
  }
  // Property track indices for agents (per physical track)
  {
    const perTrackProp = new Map<number, { start: number; end: number; item: Record<string, unknown> }[]>()
    for (const agent of ann.agents || []) {
      const trackIdx = agent._track_index ?? 0
      const items = perTrackProp.get(trackIdx) || []
      for (const prop of agent.properties || []) {
        items.push({ start: parseTs(prop.start_timestamp), end: parseTs(prop.end_timestamp), item: prop as unknown as Record<string, unknown> })
      }
      perTrackProp.set(trackIdx, items)
    }
    for (const items of perTrackProp.values()) { if (items.length > 0) assignTracks(items, '_prop_track_index') }
  }

  // --- Build segments (indices are now fresh on annotation objects) ---
  const segs: TimelineSegment[] = []
  let id = 0

  // Helper: build a containment label that mirrors the linked environment's label
  const envById = new Map((ann.environments || []).map(e => [e.id, e]))
  const contLabel = (envId: string | undefined, laneNumber: string | undefined, nearFlag?: boolean, omitLane?: boolean, edge?: 'left' | 'right' | null): string => {
    const parts: string[] = []
    if (nearFlag) parts.push('Near')
    if (edge === 'left') parts.push('Left edge')
    else if (edge === 'right') parts.push('Right edge')
    const prefix = parts.length ? `${parts.join(', ')}: ` : ''
    const laneSuffix = omitLane ? '' : ` Lane ${laneNumber || '?'}`
    const env = envId ? envById.get(envId) : undefined
    if (env) {
      const envLabel = `${envDisplayName(env.type).replace(' (specify)', '')} (${env.id}${env.name ? ` · "${env.name}"` : ''})`
      return `${prefix}${envLabel}${laneSuffix}`
    }
    return `${prefix}Env ${envId || '?'}${laneSuffix}`
  }

  const environmentLaneLabel = (env: SilAvAnnotation['environments'][number]) => {
    // num_lanes is pre-transition and num_out_lanes is post-transition on the
    // ego-path for merge/fork areas.
    if (env.type === 'fst:LaneMerge' || env.type === 'fst:LaneFork') {
      const inLanes = env.num_lanes
      const outLanes = env.num_out_lanes
      if (inLanes > 0 && (outLanes ?? 0) > 0) return ` [${inLanes}->${outLanes}L]`
      if (inLanes > 0) return ` [in ${inLanes}L]`
      if ((outLanes ?? 0) > 0) return ` [out ${outLanes}L]`
      return ''
    }
    return env.num_lanes > 1 ? ` [${env.num_lanes}L]` : ''
  }

  // Environments
  for (let ei = 0; ei < (ann.environments || []).length; ei++) {
    const env = ann.environments[ei]
    const trackIdx = env._track_index ?? 0
    segs.push({
      id: `env_${ei}_${id++}`,
      trackId: `env_${trackIdx}`,
      label: `${env.type === 'Other' && env.type_other_description ? `Other (${env.type_other_description})` : (envDisplayName(env.type) || 'Environment')}${environmentLaneLabel(env)} (${env.id}${env.name ? ` · "${env.name}"` : ''})`,
      t0: parseTs(env.start_timestamp),
      t1: parseTs(env.end_timestamp),
      meta: { ...env, _envIndex: ei, _objKind: 'environment' },
    })
  }

  // Conditions — nested as subtracks under parent env
  for (let ci = 0; ci < (ann.conditions || []).length; ci++) {
    const cond = ann.conditions[ci]
    // Find parent env track index by env_id
    const parentEnv = (ann.environments || []).find(e => e.id === cond.env_id)
    const parentTrackIdx = parentEnv?._track_index ?? 0
    segs.push({
      id: `cond_${ci}_${id++}`,
      trackId: `env_${parentTrackIdx}`,
      label: `${cond.type === 'Other' && cond.condition_other_description ? `Other (${cond.condition_other_description})` : (CONDITION_DISPLAY_NAMES[cond.type] ?? cond.type ?? '')} (${cond.id})`,
      t0: parseTs(cond.start_timestamp),
      t1: parseTs(cond.end_timestamp),
      meta: { ...cond, _condIndex: ci, _isCondSubtrack: true, _cond_track_index: (cond as unknown as Record<string, unknown>)._cond_track_index ?? 0 },
    })
  }

  for (let i = 0; i < (ann.ego_vehicle?.actions || []).length; i++) {
    const act = ann.ego_vehicle!.actions[i]
    segs.push({
      id: `ego_act_${id++}`,
      trackId: 'ego_act',
      label: act.type === 'Other' && act.action_other_description ? `Other (${act.action_other_description})`
        : act.type === 'Other oxd:MakeATurn' && act.turn_other_description ? `Other turn (${act.turn_other_description})`
        : (EGO_ACTION_DISPLAY_NAMES[act.type] ?? act.type) || 'Action',
      t0: parseTs(act.start_timestamp),
      t1: parseTs(act.end_timestamp),
      illegal: act.illegal_flag,
      because_of: act.because_of,
      meta: { ...act, _egoActIndex: i, _isEgoActionSubtrack: true },
    })
  }
  for (let ci = 0; ci < (ann.ego_vehicle?.containment || []).length; ci++) {
    const cont = ann.ego_vehicle!.containment![ci]
    segs.push({
      id: `ego_cont_${ci}_${id++}`,
      trackId: 'ego_act',
      label: contLabel(cont.env_id, cont.lane_number, cont.near_flag, false, cont.edge),
      t0: parseTs(cont.start_timestamp),
      t1: parseTs(cont.end_timestamp),
      illegal: cont.illegal_flag,
      meta: { ...cont, _contIndex: ci, _isEgoContSubtrack: true, _cont_track_index: (cont as unknown as Record<string, unknown>)._cont_track_index ?? 0 },
    })
  }
  for (let inflIdx = 0; inflIdx < (ann.ego_vehicle?.influenced_by || []).length; inflIdx++) {
    const infl = ann.ego_vehicle!.influenced_by![inflIdx]
    segs.push({
      id: `ego_infl_${inflIdx}_${id++}`,
      trackId: 'ego_act',
      label: `Infl: ${infl.influencers?.join(', ') || '?'}`,
      t0: parseTs(infl.start_timestamp),
      t1: parseTs(infl.end_timestamp),
      meta: { ...infl, _inflIndex: inflIdx, _isEgoInfluenceSubtrack: true, _influence_track_index: (infl as unknown as Record<string, unknown>)._influence_track_index ?? 0 },
    })
  }
  // Ego property subtracks
  for (let pi = 0; pi < (ann.ego_vehicle?.properties || []).length; pi++) {
    const prop = ann.ego_vehicle!.properties![pi]
    const propLabel = prop.property_type === 'Signal'
      ? `Signal: ${prop.signaling_details?.intent || '?'}`
      : prop.property_type === 'Other' && prop.other_description
        ? `Other (${prop.other_description})`
        : (PROPERTY_DISPLAY_NAMES[prop.property_type] ?? prop.property_type) || 'Property'
    segs.push({
      id: `ego_prop_${pi}_${id++}`,
      trackId: 'ego_act',
      label: propLabel,
      t0: parseTs(prop.start_timestamp),
      t1: parseTs(prop.end_timestamp),
      meta: { ...prop, _propIndex: pi, _isEgoPropertySubtrack: true, _prop_track_index: (prop as unknown as Record<string, unknown>)._prop_track_index ?? 0 },
    })
  }

  // Traffic objects on obj_ lanes
  for (let oi = 0; oi < (ann.traffic_objects || []).length; oi++) {
    const obj = ann.traffic_objects[oi]
    const trackIdx = obj._track_index ?? 0
    segs.push({
      id: `obj_${oi}_${id++}`,
      trackId: `obj_${trackIdx}`,
      label: `${obj.name ? `${obj.name} · ` : ''}${obj.type.startsWith('Other') && obj.other_type_description ? (TRAFFIC_OBJECT_DISPLAY_NAMES[obj.type] ?? obj.type).replace('(specify)', `(${obj.other_type_description})`) : (TRAFFIC_OBJECT_DISPLAY_NAMES[obj.type] ?? obj.type)} (${obj.id})`,
      t0: parseTs(obj.visibility_start_timestamp),
      t1: parseTs(obj.visibility_end_timestamp),
      meta: { ...obj, _objIndex: oi, _objKind: 'traffic_object' },
    })
    // ObjectStateEntry subtracks
    for (let si = 0; si < (obj.state_sequence || []).length; si++) {
      const st = obj.state_sequence[si]
      const stLabel = [st.motion_state, st.open_state].filter(Boolean).join('/') || 'State'
      segs.push({
        id: `obj_state_${oi}_${si}_${id++}`,
        trackId: `obj_${trackIdx}`,
        label: stLabel,
        t0: parseTs(st.start_timestamp),
        t1: parseTs(st.end_timestamp),
        meta: { ...st, _objIndex: oi, _stateIndex: si, _isObjStateSubtrack: true },
      })
    }
    // Object containment subtracks
    for (let ci = 0; ci < (obj.containment || []).length; ci++) {
      const cont = obj.containment![ci]
      segs.push({
        id: `obj_cont_${oi}_${ci}_${id++}`,
        trackId: `obj_${trackIdx}`,
        label: contLabel(cont.env_id, cont.lane_number, cont.near_flag, false, cont.edge),
        t0: parseTs(cont.start_timestamp),
        t1: parseTs(cont.end_timestamp),
        illegal: cont.illegal_flag,
        meta: { ...cont, _objIndex: oi, _contIndex: ci, _isObjContSubtrack: true, _cont_track_index: (cont as unknown as Record<string, unknown>)._cont_track_index ?? 0 },
      })
    }
  }

  // Traffic lights on light_ lanes
  for (let li = 0; li < (ann.traffic_lights || []).length; li++) {
    const light = ann.traffic_lights[li]
    const trackIdx = light._track_index ?? 0
    segs.push({
      id: `light_${li}_${id++}`,
      trackId: `light_${trackIdx}`,
      label: `${light.name ? `${light.name} · ` : ''}TrafficLight (${light.id})`,
      t0: parseTs(light.visibility_start_timestamp),
      t1: parseTs(light.visibility_end_timestamp),
      meta: { ...light, _lightIndex: li, _objKind: 'traffic_light' },
    })
    // Physical containment subtracks (traffic-light level location)
    for (let ci = 0; ci < (light.containment || []).length; ci++) {
      const cont = light.containment![ci]
      segs.push({
        id: `light_phys_cont_${li}_${ci}_${id++}`,
        trackId: `light_${trackIdx}`,
        label: contLabel(cont.env_id, cont.lane_number, cont.near_flag, true, cont.edge),
        t0: parseTs(cont.start_timestamp),
        t1: parseTs(cont.end_timestamp),
        illegal: cont.illegal_flag,
        meta: { ...cont, _lightIndex: li, _contIndex: ci, _isLightPhysContSubtrack: true, _cont_track_index: (cont as unknown as Record<string, unknown>)._cont_track_index ?? 0 },
      })
    }
    // SignalHead subtracks, each with LightStates and env_controlled
    for (let hi = 0; hi < (light.signal_heads || []).length; hi++) {
      const sh = light.signal_heads[hi]
      segs.push({
        id: `light_sh_${li}_${hi}_${id++}`,
        trackId: `light_${trackIdx}`,
        label: `SignalHead (${sh.id})`,
        t0: parseTs(sh.start_timestamp),
        t1: parseTs(sh.end_timestamp),
        meta: { ...sh, _lightIndex: li, _headIndex: hi, _isSignalHeadSubtrack: true },
      })
      // LightStates within this signal head
      for (let si = 0; si < (sh.state_sequence || []).length; si++) {
        const st = sh.state_sequence[si]
        const stLabel = [st.type, st.color, st.shape].filter(Boolean).join(' ') || 'Light'
        segs.push({
          id: `light_state_${li}_${hi}_${si}_${id++}`,
          trackId: `light_${trackIdx}`,
          label: stLabel,
          t0: parseTs(st.start_timestamp),
          t1: parseTs(st.end_timestamp),
          meta: { ...st, _lightIndex: li, _headIndex: hi, _stateIndex: si, _isLightStateSubtrack: true },
        })
      }
      // env_controlled containment subtracks within this signal head
      for (let ci = 0; ci < (sh.env_controlled || []).length; ci++) {
        const cont = sh.env_controlled![ci]
        segs.push({
          id: `light_cont_${li}_${hi}_${ci}_${id++}`,
          trackId: `light_${trackIdx}`,
          label: contLabel(cont.env_id, cont.lane_number, cont.near_flag, false, cont.edge),
          t0: parseTs(cont.start_timestamp),
          t1: parseTs(cont.end_timestamp),
          illegal: cont.illegal_flag,
          meta: { ...cont, _lightIndex: li, _headIndex: hi, _contIndex: ci, _isLightContSubtrack: true },
        })
      }
    }
  }

  // Agents: parent AgentObject + AgentAction subtracks (like Object/ObjectState pattern)
  for (let ai = 0; ai < (ann.agents || []).length; ai++) {
    const agent = ann.agents[ai]
    const trackIdx = agent._track_index ?? 0
    const typeLabel = agent.type === 'Other' && agent.other_type_description ? `Other (${agent.other_type_description})` : (AGENT_TYPE_DISPLAY_NAMES[agent.type] ?? agent.type)
    const prefix = `${agent.amount === 'Single' ? '' : agent.amount + ' '}${typeLabel}`

    // Compute parent visibility bounds
    let visT0: number, visT1: number
    if (agent.visibility_start_timestamp && agent.visibility_end_timestamp) {
      visT0 = parseTs(agent.visibility_start_timestamp)
      visT1 = parseTs(agent.visibility_end_timestamp)
    } else {
      visT0 = Infinity; visT1 = 0
      for (const act of agent.actions || []) {
        visT0 = Math.min(visT0, parseTs(act.start_timestamp))
        visT1 = Math.max(visT1, parseTs(act.end_timestamp))
      }
      for (const sp of agent.ego_relative_pose || []) {
        visT0 = Math.min(visT0, parseTs(sp.start_timestamp))
        visT1 = Math.max(visT1, parseTs(sp.end_timestamp))
      }
      if (!isFinite(visT0)) { visT0 = 0; visT1 = 0 }
    }

    // Parent segment
    segs.push({
      id: `agent_${ai}_${id++}`,
      trackId: `agent_${trackIdx}`,
      label: `${agent.name ? `${agent.name} · ` : ''}${prefix} [Agent]`,
      t0: visT0, t1: visT1,
      meta: { ...agent, _agentIndex: ai, _objKind: 'agent' },
    })

    // Action subtracks
    for (let actIdx = 0; actIdx < (agent.actions || []).length; actIdx++) {
      const act = agent.actions[actIdx]
      const t0 = parseTs(act.start_timestamp)
      const t1 = parseTs(act.end_timestamp)

      // Find matching ego_relative_pose (overlapping time range)
      const matchPose = (agent.ego_relative_pose || []).find(sp => {
        const st = parseTs(sp.start_timestamp), et = parseTs(sp.end_timestamp)
        return st < t1 && et > t0
      })

      // Find matching containment (overlapping time range) — used in meta for pose/cont linking
      const matchCont = (agent.containment || []).find(c => {
        const st = parseTs(c.start_timestamp), et = parseTs(c.end_timestamp)
        return st < t1 && et > t0
      })

      segs.push({
        id: `agent_action_${ai}_${actIdx}_${id++}`,
        trackId: `agent_${trackIdx}`,
        label: `${act.action_type === 'Other' && act.other_description ? `Other (${act.other_description})`
          : act.action_type === 'Other oxd:MakeATurn' && act.other_description ? `Other turn (${act.other_description})`
          : (AGENT_ACTION_DISPLAY_NAMES[act.action_type] ?? act.action_type) || 'Action'}`,
        t0, t1,
        illegal: act.illegal_flag,
        because_of: act.because_of,
        meta: {
          ...act,
          _agentIndex: ai,
          _actIndex: actIdx,
          _isAgentActionSubtrack: true,
          _segType: 'action',
          agent_type: agent.type,
          agent_prefix: prefix,
          _pose: matchPose || null,
          _poseIndex: matchPose ? (agent.ego_relative_pose || []).indexOf(matchPose) : -1,
          _cont: matchCont || null,
          _contIndex: matchCont ? (agent.containment || []).indexOf(matchCont) : -1,
        },
      })
    }

    // Agent property subtracks
    for (let pi = 0; pi < (agent.properties || []).length; pi++) {
      const prop = agent.properties![pi]
      const propLabel = prop.property_type === 'Signal'
        ? `Signal: ${prop.signaling_details?.intent || '?'}`
        : prop.property_type === 'Other' && prop.other_description
          ? `Other (${prop.other_description})`
          : (PROPERTY_DISPLAY_NAMES[prop.property_type] ?? prop.property_type) || 'Property'
      segs.push({
        id: `agent_prop_${ai}_${pi}_${id++}`,
        trackId: `agent_${trackIdx}`,
        label: propLabel,
        t0: parseTs(prop.start_timestamp),
        t1: parseTs(prop.end_timestamp),
        meta: { ...prop, _agentIndex: ai, _propIndex: pi, _isAgentPropertySubtrack: true, _segType: 'property', _prop_track_index: (prop as unknown as Record<string, unknown>)._prop_track_index ?? 0 },
      })
    }

    // Agent pose subtracks
    for (let pi = 0; pi < (agent.ego_relative_pose || []).length; pi++) {
      const pose = agent.ego_relative_pose![pi]
      const poseLabel = [pose.position_rel_to_ego, pose.direction_rel_to_ego].filter(Boolean).join(' / ') || 'Pose'
      segs.push({
        id: `agent_pose_${ai}_${pi}_${id++}`,
        trackId: `agent_${trackIdx}`,
        label: poseLabel,
        t0: parseTs(pose.start_timestamp),
        t1: parseTs(pose.end_timestamp),
        meta: { ...pose, _agentIndex: ai, _poseIndex: pi, _isAgentPoseSubtrack: true, _segType: 'pose' },
      })
    }

    // Agent containment subtracks
    for (let ci = 0; ci < (agent.containment || []).length; ci++) {
      const cont = agent.containment![ci]
      segs.push({
        id: `agent_cont_${ai}_${ci}_${id++}`,
        trackId: `agent_${trackIdx}`,
        label: contLabel(cont.env_id, cont.lane_number, cont.near_flag, false, cont.edge),
        t0: parseTs(cont.start_timestamp),
        t1: parseTs(cont.end_timestamp),
        illegal: cont.illegal_flag,
        meta: { ...cont, _agentIndex: ai, _contIndex: ci, _isAgentContSubtrack: true, _cont_track_index: (cont as unknown as Record<string, unknown>)._cont_track_index ?? 0 },
      })
    }

    // Agent influenced_by subtracks
    for (let inflIdx = 0; inflIdx < (agent.influenced_by || []).length; inflIdx++) {
      const infl = agent.influenced_by![inflIdx]
      segs.push({
        id: `agent_infl_${ai}_${inflIdx}_${id++}`,
        trackId: `agent_${trackIdx}`,
        label: `Infl: ${infl.influencers?.join(', ') || '?'}`,
        t0: parseTs(infl.start_timestamp),
        t1: parseTs(infl.end_timestamp),
        meta: { ...infl, _agentIndex: ai, _inflIndex: inflIdx, _isAgentInfluenceSubtrack: true, _influence_track_index: (infl as unknown as Record<string, unknown>)._influence_track_index ?? 0 },
      })
    }
  }

  return segs
}

/** Apply segment time update to annotation. */
export function applySegmentTimeUpdate(ann: SilAvAnnotation, segId: string, t0: number, t1: number): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  const f = formatTs
  let id = 0

  for (let ei = 0; ei < (clone.environments || []).length; ei++) {
    if (`env_${ei}_${id}` === segId) {
      const oldT0 = parseTs(clone.environments[ei].start_timestamp)
      clone.environments[ei].start_timestamp = f(t0)
      clone.environments[ei].end_timestamp = f(t1)
      // Shift child conditions by the same delta, then clamp to new bounds
      const envId = clone.environments[ei].id
      const dt0 = t0 - oldT0
      for (const cond of clone.conditions || []) {
        if (cond.env_id !== envId) continue
        let ct0 = parseTs(cond.start_timestamp) + dt0
        let ct1 = parseTs(cond.end_timestamp) + dt0
        // Clamp to new parent bounds
        if (ct0 < t0) { ct1 += t0 - ct0; ct0 = t0 }
        if (ct1 > t1) { ct0 -= ct1 - t1; ct1 = t1 }
        ct0 = Math.max(ct0, t0)
        cond.start_timestamp = f(ct0)
        cond.end_timestamp = f(ct1)
      }
      if (clone.environments[ei].keypoints?.length) {
        clone.environments[ei].keypoints = trimKeypointsToWindow(clone.environments[ei].keypoints, t0, t1)
      }
      return clone
    }
    id++
  }

  for (let ci = 0; ci < (clone.conditions || []).length; ci++) {
    if (`cond_${ci}_${id}` === segId) {
      // Clamp to parent env bounds
      const cond = clone.conditions[ci]
      const parentEnv = (clone.environments || []).find(e => e.id === cond.env_id)
      let ct0 = t0, ct1 = t1
      if (parentEnv) {
        const envT0 = parseTs(parentEnv.start_timestamp)
        const envT1 = parseTs(parentEnv.end_timestamp)
        ct0 = Math.max(ct0, envT0)
        ct1 = Math.min(ct1, envT1)
      }
      clone.conditions[ci].start_timestamp = f(ct0)
      clone.conditions[ci].end_timestamp = f(ct1)
      return clone
    }
    id++
  }

  for (let i = 0; i < (clone.ego_vehicle?.actions || []).length; i++) {
    if (`ego_act_${id}` === segId) {
      clone.ego_vehicle.actions[i].start_timestamp = f(t0)
      clone.ego_vehicle.actions[i].end_timestamp = f(t1)
      return clone
    }
    id++
  }
  for (let ci = 0; ci < (clone.ego_vehicle?.containment || []).length; ci++) {
    if (`ego_cont_${ci}_${id}` === segId) {
      clone.ego_vehicle.containment![ci].start_timestamp = f(t0)
      clone.ego_vehicle.containment![ci].end_timestamp = f(t1)
      return clone
    }
    id++
  }
  for (let inflIdx = 0; inflIdx < (clone.ego_vehicle?.influenced_by || []).length; inflIdx++) {
    if (`ego_infl_${inflIdx}_${id}` === segId) {
      clone.ego_vehicle.influenced_by![inflIdx].start_timestamp = f(t0)
      clone.ego_vehicle.influenced_by![inflIdx].end_timestamp = f(t1)
      return clone
    }
    id++
  }
  for (let pi = 0; pi < (clone.ego_vehicle?.properties || []).length; pi++) {
    if (`ego_prop_${pi}_${id}` === segId) {
      clone.ego_vehicle.properties![pi].start_timestamp = f(t0)
      clone.ego_vehicle.properties![pi].end_timestamp = f(t1)
      return clone
    }
    id++
  }

  for (let oi = 0; oi < (clone.traffic_objects || []).length; oi++) {
    if (`obj_${oi}_${id}` === segId) {
      const oldT0 = parseTs(clone.traffic_objects[oi].visibility_start_timestamp)
      clone.traffic_objects[oi].visibility_start_timestamp = f(t0)
      clone.traffic_objects[oi].visibility_end_timestamp = f(t1)
      // Shift child states by same delta, then clamp
      const dt0 = t0 - oldT0
      for (const st of clone.traffic_objects[oi].state_sequence || []) {
        let st0 = parseTs(st.start_timestamp) + dt0
        let st1 = parseTs(st.end_timestamp) + dt0
        if (st0 < t0) { st1 += t0 - st0; st0 = t0 }
        if (st1 > t1) { st0 -= st1 - t1; st1 = t1 }
        st0 = Math.max(st0, t0)
        st.start_timestamp = f(st0)
        st.end_timestamp = f(st1)
      }
      for (const c of clone.traffic_objects[oi].containment || []) {
        let ct0 = parseTs(c.start_timestamp) + dt0
        let ct1 = parseTs(c.end_timestamp) + dt0
        if (ct0 < t0) { ct1 += t0 - ct0; ct0 = t0 }
        if (ct1 > t1) { ct0 -= ct1 - t1; ct1 = t1 }
        ct0 = Math.max(ct0, t0)
        c.start_timestamp = f(ct0)
        c.end_timestamp = f(ct1)
      }
      if (clone.traffic_objects[oi].keypoints?.length) {
        clone.traffic_objects[oi].keypoints = trimKeypointsToWindow(clone.traffic_objects[oi].keypoints, t0, t1)
      }
      return clone
    }
    id++
    // State subtrack entries
    for (let si = 0; si < (clone.traffic_objects[oi]?.state_sequence || []).length; si++) {
      if (`obj_state_${oi}_${si}_${id}` === segId) {
        const obj = clone.traffic_objects[oi]
        const objT0 = parseTs(obj.visibility_start_timestamp), objT1 = parseTs(obj.visibility_end_timestamp)
        clone.traffic_objects[oi].state_sequence[si].start_timestamp = f(Math.max(t0, objT0))
        clone.traffic_objects[oi].state_sequence[si].end_timestamp = f(Math.min(t1, objT1))
        return clone
      }
      id++
    }
    // Object containment subtrack entries
    for (let ci = 0; ci < (clone.traffic_objects[oi]?.containment || []).length; ci++) {
      if (`obj_cont_${oi}_${ci}_${id}` === segId) {
        const obj = clone.traffic_objects[oi]
        const objT0 = parseTs(obj.visibility_start_timestamp), objT1 = parseTs(obj.visibility_end_timestamp)
        clone.traffic_objects[oi].containment![ci].start_timestamp = f(Math.max(t0, objT0))
        clone.traffic_objects[oi].containment![ci].end_timestamp = f(Math.min(t1, objT1))
        return clone
      }
      id++
    }
  }

  for (let li = 0; li < (clone.traffic_lights || []).length; li++) {
    if (`light_${li}_${id}` === segId) {
      const oldT0 = parseTs(clone.traffic_lights[li].visibility_start_timestamp)
      clone.traffic_lights[li].visibility_start_timestamp = f(t0)
      clone.traffic_lights[li].visibility_end_timestamp = f(t1)
      // Shift child signal heads (and their children) by same delta, then clamp
      const dt0 = t0 - oldT0
      for (const c of clone.traffic_lights[li].containment || []) {
        let ct0 = parseTs(c.start_timestamp) + dt0
        let ct1 = parseTs(c.end_timestamp) + dt0
        if (ct0 < t0) { ct1 += t0 - ct0; ct0 = t0 }
        if (ct1 > t1) { ct0 -= ct1 - t1; ct1 = t1 }
        ct0 = Math.max(ct0, t0)
        c.start_timestamp = f(ct0)
        c.end_timestamp = f(ct1)
      }
      for (const sh of clone.traffic_lights[li].signal_heads || []) {
        let sh0 = parseTs(sh.start_timestamp) + dt0
        let sh1 = parseTs(sh.end_timestamp) + dt0
        if (sh0 < t0) { sh1 += t0 - sh0; sh0 = t0 }
        if (sh1 > t1) { sh0 -= sh1 - t1; sh1 = t1 }
        sh0 = Math.max(sh0, t0)
        sh.start_timestamp = f(sh0)
        sh.end_timestamp = f(sh1)
        for (const st of sh.state_sequence || []) {
          let st0 = parseTs(st.start_timestamp) + dt0
          let st1 = parseTs(st.end_timestamp) + dt0
          if (st0 < t0) { st1 += t0 - st0; st0 = t0 }
          if (st1 > t1) { st0 -= st1 - t1; st1 = t1 }
          st0 = Math.max(st0, t0)
          st.start_timestamp = f(st0)
          st.end_timestamp = f(st1)
        }
        for (const c of sh.env_controlled || []) {
          let ct0 = parseTs(c.start_timestamp) + dt0
          let ct1 = parseTs(c.end_timestamp) + dt0
          if (ct0 < t0) { ct1 += t0 - ct0; ct0 = t0 }
          if (ct1 > t1) { ct0 -= ct1 - t1; ct1 = t1 }
          ct0 = Math.max(ct0, t0)
          c.start_timestamp = f(ct0)
          c.end_timestamp = f(ct1)
        }
        if (sh.keypoints?.length) {
          sh.keypoints = trimKeypointsToWindow(sh.keypoints, sh0, sh1)
        }
      }
      return clone
    }
    id++
    // Physical containment subtracks
    for (let ci = 0; ci < (clone.traffic_lights[li]?.containment || []).length; ci++) {
      if (`light_phys_cont_${li}_${ci}_${id}` === segId) {
        const lt = clone.traffic_lights[li]
        const ltT0 = parseTs(lt.visibility_start_timestamp), ltT1 = parseTs(lt.visibility_end_timestamp)
        const childT0 = Math.max(t0, ltT0), childT1 = Math.min(t1, ltT1)
        clone.traffic_lights[li].containment![ci].start_timestamp = f(childT0)
        clone.traffic_lights[li].containment![ci].end_timestamp = f(childT1)
        return clone
      }
      id++
    }
    // Signal head subtracks
    for (let hi = 0; hi < (clone.traffic_lights[li]?.signal_heads || []).length; hi++) {
      const sh = clone.traffic_lights[li].signal_heads[hi]
      if (`light_sh_${li}_${hi}_${id}` === segId) {
        const lt = clone.traffic_lights[li]
        const ltT0 = parseTs(lt.visibility_start_timestamp), ltT1 = parseTs(lt.visibility_end_timestamp)
        const oldShT0 = parseTs(sh.start_timestamp)
        // Clamp signal head to parent light; children will be resized to fit
        const newT0 = Math.max(t0, ltT0)
        const newT1 = Math.min(t1, ltT1)
        clone.traffic_lights[li].signal_heads[hi].start_timestamp = f(newT0)
        clone.traffic_lights[li].signal_heads[hi].end_timestamp = f(newT1)
        // Shift children by same delta, then clamp to new signal head bounds
        const dt0 = newT0 - oldShT0
        for (const st of clone.traffic_lights[li].signal_heads[hi].state_sequence || []) {
          let st0 = parseTs(st.start_timestamp) + dt0
          let st1 = parseTs(st.end_timestamp) + dt0
          if (st0 < newT0) { st1 += newT0 - st0; st0 = newT0 }
          if (st1 > newT1) { st0 -= st1 - newT1; st1 = newT1 }
          st0 = Math.max(st0, newT0)
          st.start_timestamp = f(st0)
          st.end_timestamp = f(st1)
        }
        for (const c of clone.traffic_lights[li].signal_heads[hi].env_controlled || []) {
          let ct0 = parseTs(c.start_timestamp) + dt0
          let ct1 = parseTs(c.end_timestamp) + dt0
          if (ct0 < newT0) { ct1 += newT0 - ct0; ct0 = newT0 }
          if (ct1 > newT1) { ct0 -= ct1 - newT1; ct1 = newT1 }
          ct0 = Math.max(ct0, newT0)
          c.start_timestamp = f(ct0)
          c.end_timestamp = f(ct1)
        }
        const shRef2 = clone.traffic_lights[li].signal_heads[hi]
        if (shRef2.keypoints?.length) {
          shRef2.keypoints = trimKeypointsToWindow(shRef2.keypoints, newT0, newT1)
        }
        return clone
      }
      id++
      // LightState subtracks within this signal head
      for (let si = 0; si < (sh.state_sequence || []).length; si++) {
        if (`light_state_${li}_${hi}_${si}_${id}` === segId) {
          const shRef = clone.traffic_lights[li].signal_heads[hi]
          const shT0 = parseTs(shRef.start_timestamp), shT1 = parseTs(shRef.end_timestamp)
          // Clamp child to parent signal head bounds
          const childT0 = Math.max(t0, shT0), childT1 = Math.min(t1, shT1)
          shRef.state_sequence[si].start_timestamp = f(childT0)
          shRef.state_sequence[si].end_timestamp = f(childT1)
          return clone
        }
        id++
      }
      // env_controlled containment subtracks within this signal head
      for (let ci = 0; ci < (sh.env_controlled || []).length; ci++) {
        if (`light_cont_${li}_${hi}_${ci}_${id}` === segId) {
          const shRef = clone.traffic_lights[li].signal_heads[hi]
          const shT0 = parseTs(shRef.start_timestamp), shT1 = parseTs(shRef.end_timestamp)
          // Clamp child to parent signal head bounds
          const childT0 = Math.max(t0, shT0), childT1 = Math.min(t1, shT1)
          shRef.env_controlled![ci].start_timestamp = f(childT0)
          shRef.env_controlled![ci].end_timestamp = f(childT1)
          return clone
        }
        id++
      }
    }
  }

  for (let ai = 0; ai < (clone.agents || []).length; ai++) {
    const agent = clone.agents[ai]
    // Parent agent segment
    if (`agent_${ai}_${id}` === segId) {
      const oldT0 = agent.visibility_start_timestamp ? parseTs(agent.visibility_start_timestamp) : 0
      agent.visibility_start_timestamp = f(t0)
      agent.visibility_end_timestamp = f(t1)
      // Shift child actions by same delta, then clamp
      const dt0 = t0 - oldT0
      for (const act of agent.actions || []) {
        let at0 = parseTs(act.start_timestamp) + dt0
        let at1 = parseTs(act.end_timestamp) + dt0
        if (at0 < t0) { at1 += t0 - at0; at0 = t0 }
        if (at1 > t1) { at0 -= at1 - t1; at1 = t1 }
        at0 = Math.max(at0, t0)
        act.start_timestamp = f(at0)
        act.end_timestamp = f(at1)
      }
      // Also shift pose and containment
      for (const sp of agent.ego_relative_pose || []) {
        let st0 = parseTs(sp.start_timestamp) + dt0
        let st1 = parseTs(sp.end_timestamp) + dt0
        if (st0 < t0) { st1 += t0 - st0; st0 = t0 }
        if (st1 > t1) { st0 -= st1 - t1; st1 = t1 }
        st0 = Math.max(st0, t0)
        sp.start_timestamp = f(st0)
        sp.end_timestamp = f(st1)
      }
      for (const c of agent.containment || []) {
        let ct0 = parseTs(c.start_timestamp) + dt0
        let ct1 = parseTs(c.end_timestamp) + dt0
        if (ct0 < t0) { ct1 += t0 - ct0; ct0 = t0 }
        if (ct1 > t1) { ct0 -= ct1 - t1; ct1 = t1 }
        ct0 = Math.max(ct0, t0)
        c.start_timestamp = f(ct0)
        c.end_timestamp = f(ct1)
      }
      for (const prop of agent.properties || []) {
        let pt0 = parseTs(prop.start_timestamp) + dt0
        let pt1 = parseTs(prop.end_timestamp) + dt0
        if (pt0 < t0) { pt1 += t0 - pt0; pt0 = t0 }
        if (pt1 > t1) { pt0 -= pt1 - t1; pt1 = t1 }
        pt0 = Math.max(pt0, t0)
        prop.start_timestamp = f(pt0)
        prop.end_timestamp = f(pt1)
      }
      for (const infl of agent.influenced_by || []) {
        let it0 = parseTs(infl.start_timestamp) + dt0
        let it1 = parseTs(infl.end_timestamp) + dt0
        if (it0 < t0) { it1 += t0 - it0; it0 = t0 }
        if (it1 > t1) { it0 -= it1 - t1; it1 = t1 }
        it0 = Math.max(it0, t0)
        infl.start_timestamp = f(it0)
        infl.end_timestamp = f(it1)
      }
      if (agent.keypoints?.length) {
        agent.keypoints = trimKeypointsToWindow(agent.keypoints, t0, t1)
      }
      return clone
    }
    id++
    // Action subtrack segments
    for (let i = 0; i < (agent.actions || []).length; i++) {
      if (`agent_action_${ai}_${i}_${id}` === segId) {
        agent.actions[i].start_timestamp = f(t0)
        agent.actions[i].end_timestamp = f(t1)
        // Auto-expand parent visibility if subtrack exceeds current bounds
        if (agent.visibility_start_timestamp && t0 < parseTs(agent.visibility_start_timestamp)) {
          agent.visibility_start_timestamp = f(t0)
        }
        if (agent.visibility_end_timestamp && t1 > parseTs(agent.visibility_end_timestamp)) {
          agent.visibility_end_timestamp = f(t1)
        }
        return clone
      }
      id++
    }
    // Agent property subtrack segments
    for (let pi = 0; pi < (agent.properties || []).length; pi++) {
      if (`agent_prop_${ai}_${pi}_${id}` === segId) {
        agent.properties![pi].start_timestamp = f(t0)
        agent.properties![pi].end_timestamp = f(t1)
        if (agent.visibility_start_timestamp && t0 < parseTs(agent.visibility_start_timestamp)) {
          agent.visibility_start_timestamp = f(t0)
        }
        if (agent.visibility_end_timestamp && t1 > parseTs(agent.visibility_end_timestamp)) {
          agent.visibility_end_timestamp = f(t1)
        }
        return clone
      }
      id++
    }
    // Agent pose subtrack segments
    for (let pi = 0; pi < (agent.ego_relative_pose || []).length; pi++) {
      if (`agent_pose_${ai}_${pi}_${id}` === segId) {
        agent.ego_relative_pose![pi].start_timestamp = f(t0)
        agent.ego_relative_pose![pi].end_timestamp = f(t1)
        if (agent.visibility_start_timestamp && t0 < parseTs(agent.visibility_start_timestamp)) {
          agent.visibility_start_timestamp = f(t0)
        }
        if (agent.visibility_end_timestamp && t1 > parseTs(agent.visibility_end_timestamp)) {
          agent.visibility_end_timestamp = f(t1)
        }
        return clone
      }
      id++
    }
    // Agent containment subtrack segments
    for (let ci = 0; ci < (agent.containment || []).length; ci++) {
      if (`agent_cont_${ai}_${ci}_${id}` === segId) {
        agent.containment![ci].start_timestamp = f(t0)
        agent.containment![ci].end_timestamp = f(t1)
        // Auto-expand parent visibility if subtrack exceeds current bounds
        if (agent.visibility_start_timestamp && t0 < parseTs(agent.visibility_start_timestamp)) {
          agent.visibility_start_timestamp = f(t0)
        }
        if (agent.visibility_end_timestamp && t1 > parseTs(agent.visibility_end_timestamp)) {
          agent.visibility_end_timestamp = f(t1)
        }
        return clone
      }
      id++
    }
    // Agent influenced_by subtrack segments
    for (let inflIdx = 0; inflIdx < (agent.influenced_by || []).length; inflIdx++) {
      if (`agent_infl_${ai}_${inflIdx}_${id}` === segId) {
        agent.influenced_by![inflIdx].start_timestamp = f(t0)
        agent.influenced_by![inflIdx].end_timestamp = f(t1)
        // Auto-expand parent visibility if subtrack exceeds current bounds
        if (agent.visibility_start_timestamp && t0 < parseTs(agent.visibility_start_timestamp)) {
          agent.visibility_start_timestamp = f(t0)
        }
        if (agent.visibility_end_timestamp && t1 > parseTs(agent.visibility_end_timestamp)) {
          agent.visibility_end_timestamp = f(t1)
        }
        return clone
      }
      id++
    }
  }

  return clone
}

/** Add a new segment to the annotation. */
export function addSegmentToAnnotation(ann: SilAvAnnotation, trackId: TrackId, label: string, t0: number, t1: number): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  const fmt = formatTs

  const envMatch = trackId.match(/^env_(\d+)$/)
  if (envMatch) {
    const trackIdx = parseInt(envMatch[1], 10)
    clone.environments = clone.environments || []
    const newEnvId = nextStringId('Environment', clone.environments.map(e => e.id))
    clone.environments.push({
      id: newEnvId,
      type: '',
      num_lanes: 0,
      start_timestamp: fmt(t0),
      end_timestamp: fmt(t1),
      _track_index: trackIdx,
    })
    // Auto-create a default condition spanning the full env
    clone.conditions = clone.conditions || []
    clone.conditions.push({
      id: nextStringId('Condition', clone.conditions.map(c => c.id)),
      env_id: newEnvId,
      type: '',
      start_timestamp: fmt(t0),
      end_timestamp: fmt(t1),
    })
    return clone
  }

  const objMatch = trackId.match(/^obj_(\d+)$/)
  if (objMatch) {
    const trackIdx = parseInt(objMatch[1], 10)
    clone.traffic_objects = clone.traffic_objects || []
    const objId = nextStringId('Object', clone.traffic_objects.map(o => o.id))
    const objContIds = [
      ...(clone.ego_vehicle?.containment || []).map(c => c.id || ''),
      ...(clone.agents || []).flatMap(a => (a.containment || []).map(c => c.id || '')),
      ...(clone.traffic_objects || []).flatMap(o => (o.containment || []).map(c => c.id || '')),
    ].filter(Boolean)
    clone.traffic_objects.push({
      id: objId,
      type: label,
      visibility_start_timestamp: fmt(t0),
      visibility_end_timestamp: fmt(t1),
      lane_number: 'none',
      containment: [{ id: nextStringId('ObjectContainment', objContIds), env_id: '', lane_number: 'none', start_timestamp: fmt(t0), end_timestamp: fmt(t1) }],
      state_sequence: [{ start_timestamp: fmt(t0), end_timestamp: fmt(t1) }],
      keypoints: [],
      _track_index: trackIdx,
    })
    return clone
  }

  const lightMatch = trackId.match(/^light_(\d+)$/)
  if (lightMatch) {
    const trackIdx = parseInt(lightMatch[1], 10)
    clone.traffic_lights = clone.traffic_lights || []
    const lightId = nextStringId('TrafficLight', clone.traffic_lights.map(l => l.id))
    const existingShIds = clone.traffic_lights.flatMap(l => (l.signal_heads || []).map(sh => sh.id)).filter(Boolean)
    const shId = nextStringId('SignalHead', existingShIds)
    const existingLsIds = clone.traffic_lights.flatMap(l => (l.signal_heads || []).flatMap(sh => (sh.state_sequence || []).map(s => (s as unknown as Record<string, unknown>).id as string).filter(Boolean)))
    const allContIds = [
      ...(clone.ego_vehicle?.containment || []).map(c => c.id || ''),
      ...(clone.agents || []).flatMap(a => (a.containment || []).map(c => c.id || '')),
      ...(clone.traffic_objects || []).flatMap(o => (o.containment || []).map(c => c.id || '')),
      ...(clone.traffic_lights || []).flatMap(l => [
        ...(l.containment || []).map(c => c.id || ''),
        ...(l.signal_heads || []).flatMap(sh => (sh.env_controlled || []).map(c => c.id || '')),
      ]),
    ].filter(Boolean)
    const physContId = nextStringId('LightContainment', allContIds)
    const shContId = nextStringId('LightContainment', [...allContIds, physContId])
    clone.traffic_lights.push({
      id: lightId,
      visibility_start_timestamp: fmt(t0),
      visibility_end_timestamp: fmt(t1),
      containment: [{
        id: physContId,
        env_id: '',
        lane_number: '',
        start_timestamp: fmt(t0),
        end_timestamp: fmt(t1),
      }],
      signal_heads: [{
        id: shId,
        start_timestamp: fmt(t0),
        end_timestamp: fmt(t1),
        state_sequence: [{ id: nextStringId('LightState', existingLsIds), start_timestamp: fmt(t0), end_timestamp: fmt(t1) }],
        env_controlled: [{
          id: shContId,
          env_id: '',
          lane_number: '',
          start_timestamp: fmt(t0),
          end_timestamp: fmt(t1),
        }],
        keypoints: [],
      }],
      _track_index: trackIdx,
    })
    return clone
  }

  const agentMatch = trackId.match(/^agent_(\d+)$/)
  if (agentMatch) {
    const trackIdx = parseInt(agentMatch[1], 10)
    clone.agents = clone.agents || []
    const ts0 = fmt(t0), ts1 = fmt(t1)
    const agentId = nextStringId('Agent', clone.agents.map(a => a.id))
    const contIds = [
      ...(clone.ego_vehicle?.containment || []).map(c => c.id || ''),
      ...(clone.agents || []).flatMap(a => (a.containment || []).map(c => c.id || '')),
    ].filter(Boolean)
    clone.agents.push({
      id: agentId,
      amount: 'Single', type: '',
      visibility_start_timestamp: ts0,
      visibility_end_timestamp: ts1,
      actions: [{ action_type: '', start_timestamp: ts0, end_timestamp: ts1 }],
      ego_relative_pose: [{  start_timestamp: ts0, end_timestamp: ts1 }],
      containment: [{ id: nextStringId('AgentContainment', contIds), env_id: '', lane_number: 'none', start_timestamp: ts0, end_timestamp: ts1 }],
      keypoints: [],
      _track_index: trackIdx,
    })
    return clone
  }

  // Palette fallbacks
  if (trackId === 'env') return addSegmentToAnnotation(ann, 'env_0', label, t0, t1)
  if (trackId === 'obj' || trackId === 'obj_new') return addSegmentToAnnotation(ann, 'obj_0', label, t0, t1)
  if (trackId === 'light') return addSegmentToAnnotation(ann, 'light_0', label, t0, t1)
  if (trackId === 'agent_act' || trackId === 'agent_new') return addSegmentToAnnotation(ann, 'agent_0', label, t0, t1)
  if (trackId === 'agent_type') {
    const amount = label.startsWith('Row/group') ? 'Row/group'
      : label.startsWith('Light') ? 'Light traffic'
      : label.startsWith('Medium') ? 'Medium traffic'
      : label.startsWith('Heavy') ? 'Heavy traffic'
      : 'Single'
    const agentType = label.replace(/^(Row\/group |Light traffic |Medium traffic |Heavy traffic )/, '') || label
    clone.agents = clone.agents || []
    const nextTrack = clone.agents.length > 0 ? Math.max(...clone.agents.map(a => (a._track_index ?? 0))) + 1 : 0
    const agentId = nextStringId('Agent', clone.agents.map(a => a.id))
    const defaultEnvId2 = ''
    const contIds2 = [
      ...(clone.ego_vehicle?.containment || []).map(c => c.id || ''),
      ...(clone.agents || []).flatMap(a => (a.containment || []).map(c => c.id || '')),
    ].filter(Boolean)
    clone.agents.push({
      id: agentId,
      amount, type: agentType || '',
      visibility_start_timestamp: fmt(t0),
      visibility_end_timestamp: fmt(t1),
      actions: [{ action_type: '', start_timestamp: fmt(t0), end_timestamp: fmt(t1) }],
      ego_relative_pose: [{  start_timestamp: fmt(t0), end_timestamp: fmt(t1) }],
      containment: [{ id: nextStringId('AgentContainment', contIds2), env_id: defaultEnvId2, lane_number: 'none', start_timestamp: fmt(t0), end_timestamp: fmt(t1) }],
      keypoints: [],
      _track_index: nextTrack,
    })
    return clone
  }
  if (trackId === 'agent_pose' || trackId === 'agent_cont') {
    return addSegmentToAnnotation(ann, 'agent_0', label, t0, t1)
  }

  if (trackId === 'ego_act') {
    clone.ego_vehicle = clone.ego_vehicle || { actions: [] }
    clone.ego_vehicle.actions = clone.ego_vehicle.actions || []
    const existingEgoActIds = (clone.ego_vehicle.actions || []).map(a => a.id).filter(Boolean) as string[]
    clone.ego_vehicle.actions.push({ id: nextStringId('EgoAction', existingEgoActIds), type: '', start_timestamp: fmt(t0), end_timestamp: fmt(t1) })
    return clone
  }

  if (trackId === 'ego_cont') {
    clone.ego_vehicle = clone.ego_vehicle || { actions: [] }
    clone.ego_vehicle.containment = clone.ego_vehicle.containment || []
    const laneMatch = label.match(/lane (\d+)/i)
    const egoContIds = [
      ...(clone.ego_vehicle.containment || []).map(c => c.id || ''),
      ...(clone.agents || []).flatMap(a => (a.containment || []).map(c => c.id || '')),
    ].filter(Boolean)
    clone.ego_vehicle.containment.push({ id: nextStringId('EgoContainment', egoContIds), env_id: clone.environments?.[0]?.id || 'Environment1', lane_number: laneMatch ? laneMatch[1] : 'none', start_timestamp: fmt(t0), end_timestamp: fmt(t1) })
    return clone
  }

  return ann
}

/** Add a condition subtrack to an environment segment. */
export function addConditionToEnv(ann: SilAvAnnotation, envId: string | number, t0: number, t1: number, condTrackIndex?: number): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  clone.conditions = clone.conditions || []
  const newCond: Record<string, unknown> = {
    id: nextStringId('Condition', clone.conditions.map(c => c.id)),
    env_id: String(envId),
    type: '',
    start_timestamp: formatTs(t0),
    end_timestamp: formatTs(t1),
  }
  if (condTrackIndex != null) newCond._cond_track_index = condTrackIndex
  clone.conditions.push(newCond as unknown as Condition)
  return clone
}

/** Add a state subtrack entry to a traffic object. */
export function addStateToObject(ann: SilAvAnnotation, objIndex: number, t0: number, t1: number): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  clone.traffic_objects[objIndex].state_sequence = clone.traffic_objects[objIndex].state_sequence || []
  clone.traffic_objects[objIndex].state_sequence.push({
    start_timestamp: formatTs(t0),
    end_timestamp: formatTs(t1),
  })
  return clone
}

/** Add a signal head to a traffic light. */
export function addSignalHeadToLight(ann: SilAvAnnotation, lightIndex: number, t0: number, t1: number): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  const light = clone.traffic_lights[lightIndex]
  light.signal_heads = light.signal_heads || []
  const existingShIds = clone.traffic_lights.flatMap(l => (l.signal_heads || []).map(sh => sh.id)).filter(Boolean)
  const existingLsIds = clone.traffic_lights.flatMap(l => (l.signal_heads || []).flatMap(sh => (sh.state_sequence || []).map(s => (s as unknown as Record<string, unknown>).id as string).filter(Boolean)))
  const allContIds = [
    ...(clone.ego_vehicle?.containment || []).map(c => c.id || ''),
    ...(clone.agents || []).flatMap(a => (a.containment || []).map(c => c.id || '')),
    ...(clone.traffic_objects || []).flatMap(o => (o.containment || []).map(c => c.id || '')),
    ...(clone.traffic_lights || []).flatMap(l => [
      ...(l.containment || []).map(c => c.id || ''),
      ...(l.signal_heads || []).flatMap(sh => (sh.env_controlled || []).map(c => c.id || '')),
    ]),
  ].filter(Boolean)
  light.signal_heads.push({
    id: nextStringId('SignalHead', existingShIds),
    start_timestamp: formatTs(t0),
    end_timestamp: formatTs(t1),
    state_sequence: [{ id: nextStringId('LightState', existingLsIds), start_timestamp: formatTs(t0), end_timestamp: formatTs(t1) }],
    env_controlled: [{
      id: nextStringId('LightContainment', allContIds),
      env_id: '',
      lane_number: '',
      start_timestamp: formatTs(t0),
      end_timestamp: formatTs(t1),
    }],
    keypoints: [],
  })
  return clone
}

/** Add a state subtrack entry to a signal head within a traffic light. */
export function addStateToLight(ann: SilAvAnnotation, lightIndex: number, headIndex: number, t0: number, t1: number, laneIndex?: number): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  const sh = clone.traffic_lights[lightIndex].signal_heads[headIndex]
  sh.state_sequence = sh.state_sequence || []
  const existingIds = clone.traffic_lights.flatMap(l => (l.signal_heads || []).flatMap(h => (h.state_sequence || []).map(s => (s as unknown as Record<string, unknown>).id as string).filter(Boolean)))
  const newState = {
    id: nextStringId('LightState', existingIds),
    start_timestamp: formatTs(t0),
    end_timestamp: formatTs(t1),
  } as import('./types').LightStates
  if (laneIndex != null) (newState as unknown as Record<string, unknown>)._state_track_index = laneIndex
  sh.state_sequence.push(newState)
  return clone
}

/** Add an env_controlled containment subtrack to a signal head within a traffic light. */
export function addContainmentToLight(ann: SilAvAnnotation, lightIndex: number, headIndex: number, t0: number, t1: number): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  const sh = clone.traffic_lights[lightIndex].signal_heads[headIndex]
  sh.env_controlled = sh.env_controlled || []
  const existingContIds = [
    ...(clone.ego_vehicle?.containment || []).map(c => c.id || ''),
    ...(clone.agents || []).flatMap(a => (a.containment || []).map(c => c.id || '')),
    ...(clone.traffic_objects || []).flatMap(o => (o.containment || []).map(c => c.id || '')),
    ...(clone.traffic_lights || []).flatMap(l => [
      ...(l.containment || []).map(c => c.id || ''),
      ...(l.signal_heads || []).flatMap(h => (h.env_controlled || []).map(c => c.id || '')),
    ]),
  ].filter(Boolean)
  sh.env_controlled.push({
    id: nextStringId('LightContainment', existingContIds),
    env_id: '',
    lane_number: '',
    start_timestamp: formatTs(t0),
    end_timestamp: formatTs(t1),
  })
  return clone
}

/** Add a physical containment entry to a traffic light (where the light is located). */
export function addPhysicalContainmentToLight(ann: SilAvAnnotation, lightIndex: number, t0: number, t1: number, contTrackIndex?: number): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  const light = clone.traffic_lights[lightIndex]
  light.containment = light.containment || []
  const existingContIds = [
    ...(clone.ego_vehicle?.containment || []).map(c => c.id || ''),
    ...(clone.agents || []).flatMap(a => (a.containment || []).map(c => c.id || '')),
    ...(clone.traffic_objects || []).flatMap(o => (o.containment || []).map(c => c.id || '')),
    ...(clone.traffic_lights || []).flatMap(l => [
      ...(l.containment || []).map(c => c.id || ''),
      ...(l.signal_heads || []).flatMap(h => (h.env_controlled || []).map(c => c.id || '')),
    ]),
  ].filter(Boolean)
  const newCont: import('./types').Containment = {
    id: nextStringId('LightContainment', existingContIds),
    env_id: '',
    lane_number: '',
    start_timestamp: formatTs(t0),
    end_timestamp: formatTs(t1),
  }
  if (contTrackIndex != null) (newCont as unknown as Record<string, unknown>)._cont_track_index = contTrackIndex
  light.containment.push(newCont)
  return clone
}

/** Add an action subtrack entry to an agent. */
export function addActionToAgent(ann: SilAvAnnotation, agentIndex: number, t0: number, t1: number): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  const agent = clone.agents[agentIndex]
  agent.actions = agent.actions || []
  const existingIds = clone.agents.flatMap(a => (a.actions || []).map(ac => (ac as unknown as Record<string, unknown>).id as string).filter(Boolean))
  agent.actions.push({
    id: nextStringId('AgentAction', existingIds),
    action_type: '',
    start_timestamp: formatTs(t0),
    end_timestamp: formatTs(t1),
  } as import('./types').AgentAction)
  return clone
}

/** Add a property subtrack entry to an agent. */
export function addPropertyToAgent(ann: SilAvAnnotation, agentIndex: number, t0: number, t1: number, propTrackIndex?: number): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  const agent = clone.agents[agentIndex]
  agent.properties = agent.properties || []
  const existingIds = [
    ...(clone.agents || []).flatMap(a => (a.properties || []).map(p => p.id)),
    ...(clone.ego_vehicle?.properties || []).map(p => p.id),
  ].filter(Boolean)
  const newProp: import('./types').AgentProperty = {
    id: nextStringId('AgentProperty', existingIds),
    property_type: '',
    start_timestamp: formatTs(t0),
    end_timestamp: formatTs(t1),
  }
  if (propTrackIndex != null) (newProp as unknown as Record<string, unknown>)._prop_track_index = propTrackIndex
  agent.properties.push(newProp)
  return clone
}

/** Add a property subtrack entry to the ego vehicle. */
export function addPropertyToEgo(ann: SilAvAnnotation, t0: number, t1: number, propTrackIndex?: number): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  clone.ego_vehicle = clone.ego_vehicle || { actions: [] }
  clone.ego_vehicle.properties = clone.ego_vehicle.properties || []
  const existingIds = [
    ...(clone.agents || []).flatMap(a => (a.properties || []).map(p => p.id)),
    ...(clone.ego_vehicle.properties || []).map(p => p.id),
  ].filter(Boolean)
  const newProp: import('./types').AgentProperty = {
    id: nextStringId('EgoProperty', existingIds),
    property_type: '',
    start_timestamp: formatTs(t0),
    end_timestamp: formatTs(t1),
  }
  if (propTrackIndex != null) (newProp as unknown as Record<string, unknown>)._prop_track_index = propTrackIndex
  clone.ego_vehicle.properties.push(newProp)
  return clone
}

/** Add a pose subtrack entry to an agent. */
export function addPoseToAgent(ann: SilAvAnnotation, agentIndex: number, t0: number, t1: number): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  const agent = clone.agents[agentIndex]
  agent.ego_relative_pose = agent.ego_relative_pose || []
  agent.ego_relative_pose.push({
    start_timestamp: formatTs(t0),
    end_timestamp: formatTs(t1),
  })
  return clone
}

/** Add a containment subtrack to the ego vehicle. */
export function addContainmentToEgo(ann: SilAvAnnotation, t0: number, t1: number, contTrackIndex?: number): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  clone.ego_vehicle = clone.ego_vehicle || { actions: [] }
  clone.ego_vehicle.containment = clone.ego_vehicle.containment || []
  const existingIds = [
    ...(clone.ego_vehicle.containment || []).map(c => c.id || ''),
    ...(clone.agents || []).flatMap(a => (a.containment || []).map(c => c.id || '')),
  ].filter(Boolean)
  const newCont: import('./types').Containment = {
    id: nextStringId('EgoContainment', existingIds),
    env_id: '',
    lane_number: 'none',
    start_timestamp: formatTs(t0),
    end_timestamp: formatTs(t1),
  }
  if (contTrackIndex != null) (newCont as unknown as Record<string, unknown>)._cont_track_index = contTrackIndex
  clone.ego_vehicle.containment.push(newCont)
  return clone
}

/** Add a containment subtrack to an agent. */
export function addContainmentToAgent(ann: SilAvAnnotation, agentIndex: number, t0: number, t1: number, contTrackIndex?: number): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  const agent = clone.agents[agentIndex]
  agent.containment = agent.containment || []
  const existingIds3 = [
    ...(clone.ego_vehicle?.containment || []).map(c => c.id || ''),
    ...(clone.agents || []).flatMap(a => (a.containment || []).map(c => c.id || '')),
  ].filter(Boolean)
  const newCont: import('./types').Containment = {
    id: nextStringId('AgentContainment', existingIds3),
    env_id: '',
    lane_number: 'none',
    start_timestamp: formatTs(t0),
    end_timestamp: formatTs(t1),
  }
  if (contTrackIndex != null) (newCont as unknown as Record<string, unknown>)._cont_track_index = contTrackIndex
  agent.containment.push(newCont)
  return clone
}

/** Add a containment subtrack to a traffic object. */
export function addContainmentToObject(ann: SilAvAnnotation, objectIndex: number, t0: number, t1: number, contTrackIndex?: number): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  const obj = clone.traffic_objects[objectIndex]
  obj.containment = obj.containment || []
  const existingContIds = [
    ...(clone.ego_vehicle?.containment || []).map(c => c.id || ''),
    ...(clone.agents || []).flatMap(a => (a.containment || []).map(c => c.id || '')),
    ...(clone.traffic_objects || []).flatMap(o => (o.containment || []).map(c => c.id || '')),
    ...(clone.traffic_lights || []).flatMap(l => [
      ...(l.containment || []).map(c => c.id || ''),
      ...(l.signal_heads || []).flatMap(h => (h.env_controlled || []).map(c => c.id || '')),
    ]),
  ].filter(Boolean)
  const newCont: import('./types').Containment = {
    id: nextStringId('ObjectContainment', existingContIds),
    env_id: '',
    lane_number: 'none',
    start_timestamp: formatTs(t0),
    end_timestamp: formatTs(t1),
  }
  if (contTrackIndex != null) (newCont as unknown as Record<string, unknown>)._cont_track_index = contTrackIndex
  obj.containment.push(newCont)
  return clone
}

/** Move a segment to a different track within the same group. */
export function moveSegmentToTrack(ann: SilAvAnnotation, segId: string, _fromTrackId: string, toTrackId: string): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  const segments = annotationToSegments(ann)
  const seg = segments.find(s => s.id === segId)
  if (!seg) return clone

  const toEnv = toTrackId.match(/^env_(\d+)$/)
  if (toEnv && segId.startsWith('env_')) {
    const toIdx = parseInt(toEnv[1], 10)
    const ei = (seg.meta as { _envIndex?: number })?._envIndex
    if (ei != null && ei >= 0 && ei < clone.environments.length) {
      clone.environments[ei]._track_index = toIdx
    }
    return clone
  }

  const toObj = toTrackId.match(/^obj_(\d+)$/)
  if (toObj && segId.startsWith('obj_')) {
    const toIdx = parseInt(toObj[1], 10)
    const oi = (seg.meta as { _objIndex?: number })?._objIndex
    if (oi != null && oi >= 0 && oi < (clone.traffic_objects || []).length) {
      clone.traffic_objects[oi]._track_index = toIdx
    }
    return clone
  }

  const toLight = toTrackId.match(/^light_(\d+)$/)
  if (toLight && segId.startsWith('light_')) {
    const toIdx = parseInt(toLight[1], 10)
    const li = (seg.meta as { _lightIndex?: number })?._lightIndex
    if (li != null && li >= 0 && li < (clone.traffic_lights || []).length) {
      clone.traffic_lights[li]._track_index = toIdx
    }
    return clone
  }

  const toAgent = toTrackId.match(/^agent_(\d+)$/)
  if (toAgent && (segId.startsWith('agent_') || segId.startsWith('agent_action_'))) {
    const toIdx = parseInt(toAgent[1], 10)
    const ai = (seg.meta as { _agentIndex?: number })?._agentIndex
    if (ai != null && ai >= 0 && ai < clone.agents.length) {
      clone.agents[ai]._track_index = toIdx
    }
    return clone
  }

  return clone
}

/** Remove a track and reassign items on it to track 0, shift higher tracks down. */
export function removeTrackAndReassign(ann: SilAvAnnotation, group: DynamicGroup, trackIdx: number): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation

  const reassign = (items: { _track_index?: number }[]) => {
    for (const item of items) {
      const ti = item._track_index ?? 0
      if (ti === trackIdx) item._track_index = 0
      else if (ti > trackIdx) item._track_index = ti - 1
    }
  }

  if (group === 'Environments') reassign(clone.environments || [])
  else if (group === 'Objects') reassign(clone.traffic_objects || [])
  else if (group === 'TrafficLights') reassign(clone.traffic_lights || [])
  else reassign(clone.agents || [])

  return clone
}

/** Get the group name for a track ID. */
export function getTrackGroup(trackId: string): string | null {
  if (trackId.startsWith('env_')) return 'Environments'
  if (trackId === 'ego_act') return 'Ego'
  if (trackId.startsWith('obj_')) return 'Objects'
  if (trackId.startsWith('light_')) return 'TrafficLights'
  if (trackId.startsWith('agent_')) return 'Agents'
  return null
}

/** Auto-sync influenced_agent_ids on signal heads based on because_of references.
 *  Merges auto-synced IDs with any manually-entered IDs (preserves existing entries). */
export function syncInfluencedAgentIds(ann: SilAvAnnotation): void {
  // Build a map of auto-synced IDs per signal head: key "li_hi" -> Set of agent IDs
  const autoIds = new Map<string, Set<string>>()

  const addAutoId = (li: number, hi: number, agentId: string) => {
    const key = `${li}_${hi}`
    if (!autoIds.has(key)) autoIds.set(key, new Set())
    autoIds.get(key)!.add(agentId)
  }

  // Build a set of traffic light IDs and signal head IDs
  const lightIdSet = new Set((ann.traffic_lights || []).map(l => l.id))
  const shIdMap = new Map<string, { li: number; hi: number }>()
  for (let li = 0; li < (ann.traffic_lights || []).length; li++) {
    for (let hi = 0; hi < (ann.traffic_lights[li].signal_heads || []).length; hi++) {
      const shId = ann.traffic_lights[li].signal_heads[hi].id
      if (shId) shIdMap.set(shId, { li, hi })
    }
  }

  const processRef = (ref: string, agentId: string) => {
    // Match light_state_li_hi_si or light_sh_li_hi format
    const stateMatch = ref.match(/light_state_(\d+)_(\d+)_(\d+)/)
    if (stateMatch) {
      addAutoId(parseInt(stateMatch[1], 10), parseInt(stateMatch[2], 10), agentId)
      return
    }
    const shMatch = ref.match(/light_sh_(\d+)_(\d+)/)
    if (shMatch) {
      addAutoId(parseInt(shMatch[1], 10), parseInt(shMatch[2], 10), agentId)
      return
    }
    // Match by signal head ID
    const shRef = shIdMap.get(ref)
    if (shRef) {
      addAutoId(shRef.li, shRef.hi, agentId)
      return
    }
    // Match by light ID → all signal heads
    if (lightIdSet.has(ref)) {
      const li = (ann.traffic_lights || []).findIndex(l => l.id === ref)
      if (li >= 0) {
        for (let hi = 0; hi < (ann.traffic_lights[li].signal_heads || []).length; hi++) {
          addAutoId(li, hi, agentId)
        }
      }
    }
  }

  // Check ego actions
  for (const act of ann.ego_vehicle?.actions || []) {
    for (const ref of act.because_of || []) processRef(ref, 'ego')
  }

  // Check agent actions
  for (const agent of ann.agents || []) {
    const agentId = agent.id || ''
    if (!agentId) continue
    for (const act of agent.actions || []) {
      for (const ref of act.because_of || []) processRef(ref, agentId)
    }
  }

  // Apply: merge auto-synced IDs with existing (manually-entered) IDs on signal heads
  for (let li = 0; li < (ann.traffic_lights || []).length; li++) {
    for (let hi = 0; hi < (ann.traffic_lights[li].signal_heads || []).length; hi++) {
      const sh = ann.traffic_lights[li].signal_heads[hi]
      const synced = autoIds.get(`${li}_${hi}`) || new Set<string>()
      const existing = new Set(sh.influenced_agent_ids || [])
      for (const id of synced) existing.add(id)
      sh.influenced_agent_ids = [...existing]
    }
  }
}

/** Auto-populate ego containment from environments if empty. Mutates ann in place. */
export function autoPopulateEgoContainment(ann: SilAvAnnotation): void {
  if (!ann.ego_vehicle) ann.ego_vehicle = { actions: [] }
  if ((ann.ego_vehicle.containment || []).length > 0) return
  ann.ego_vehicle.containment = (ann.environments || []).map(env => ({
    id: '',
    env_id: env.id,
    lane_number: 'none',
    start_timestamp: env.start_timestamp,
    end_timestamp: env.end_timestamp,
  }))
}

/** Migrate legacy integer IDs to string IDs. Mutates ann in place. */
export function migrateIdsToString(ann: SilAvAnnotation): void {
  const envIdMap = new Map<string, string>()
  for (const env of ann.environments || []) {
    const old = String(env.id)
    if (typeof env.id === 'number' || (typeof env.id === 'string' && /^\d+$/.test(env.id))) {
      env.id = `Environment${env.id}`
    }
    envIdMap.set(old, env.id)
  }
  for (const cond of ann.conditions || []) {
    if (typeof cond.id === 'number' || (typeof cond.id === 'string' && /^\d+$/.test(cond.id))) {
      cond.id = `Condition${cond.id}`
    }
    const mapped = envIdMap.get(String(cond.env_id))
    if (mapped) cond.env_id = mapped
  }
  for (const obj of ann.traffic_objects || []) {
    if (typeof obj.id === 'number' || (typeof obj.id === 'string' && /^\d+$/.test(obj.id))) {
      obj.id = `Object${obj.id}`
    }
  }
  for (const light of ann.traffic_lights || []) {
    if (typeof light.id === 'number' || (typeof light.id === 'string' && /^\d+$/.test(light.id))) {
      light.id = `TrafficLight${light.id}`
    }
  }
  for (const agent of ann.agents || []) {
    if (!agent.id || /^\d+$/.test(agent.id)) {
      agent.id = `Agent${(ann.agents || []).indexOf(agent) + 1}`
    }
  }
  const remapEnvId = (eid: unknown): string => {
    const s = String(eid)
    return envIdMap.get(s) || s
  }
  for (const c of ann.ego_vehicle?.containment || []) {
    (c as any).env_id = remapEnvId(c.env_id)
  }
  for (const agent of ann.agents || []) {
    for (const c of agent.containment || []) {
      (c as any).env_id = remapEnvId(c.env_id)
    }
  }
  for (const obj of ann.traffic_objects || []) {
    for (const c of obj.containment || []) {
      (c as any).env_id = remapEnvId(c.env_id)
    }
  }
  for (const light of ann.traffic_lights || []) {
    for (const c of light.containment || []) {
      (c as any).env_id = remapEnvId(c.env_id)
    }
    for (const sh of light.signal_heads || []) {
      for (const c of sh.env_controlled || []) {
        (c as any).env_id = remapEnvId(c.env_id)
      }
    }
  }
}

/** Add an influence subtrack to the ego vehicle. */
export function addInfluenceToEgo(ann: SilAvAnnotation, t0: number, t1: number, inflTrackIndex?: number): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  clone.ego_vehicle = clone.ego_vehicle || { actions: [] }
  clone.ego_vehicle.influenced_by = clone.ego_vehicle.influenced_by || []
  const existingIds = (clone.ego_vehicle.influenced_by || []).map(i => i.id).filter(Boolean)
  const newInfl: import('./types').Influence = {
    id: nextStringId('EgoInfluence', existingIds),
    influencers: [],
    start_timestamp: formatTs(t0),
    end_timestamp: formatTs(t1),
  }
  if (inflTrackIndex != null) (newInfl as unknown as Record<string, unknown>)._influence_track_index = inflTrackIndex
  clone.ego_vehicle.influenced_by.push(newInfl)
  return clone
}

/** Add an influence subtrack to an agent. */
export function addInfluenceToAgent(ann: SilAvAnnotation, agentIndex: number, t0: number, t1: number, inflTrackIndex?: number): SilAvAnnotation {
  const clone = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  const agent = clone.agents[agentIndex]
  agent.influenced_by = agent.influenced_by || []
  const existingIds = (clone.agents || []).flatMap(a => (a.influenced_by || []).map(i => i.id)).filter(Boolean)
  const newInfl: import('./types').Influence = {
    id: nextStringId('AgentInfluence', existingIds),
    influencers: [],
    start_timestamp: formatTs(t0),
    end_timestamp: formatTs(t1),
  }
  if (inflTrackIndex != null) (newInfl as unknown as Record<string, unknown>)._influence_track_index = inflTrackIndex
  agent.influenced_by.push(newInfl)
  return clone
}

/**
 * Remove a set of IDs from all cross-reference arrays in the annotation:
 * because_of, link_to, signaling_details.link_to, influenced_agent_ids, influencers.
 * Call this before splicing the entity, passing the IDs that are about to be removed.
 */
export function cleanupDeletedIds(ann: SilAvAnnotation, deletedIds: string[]): void {
  if (deletedIds.length === 0) return
  const idSet = new Set(deletedIds)
  const filterIds = (arr: string[]) => arr.filter(id => !idSet.has(id))

  for (const act of ann.ego_vehicle?.actions ?? []) {
    if (act.because_of?.length) act.because_of = filterIds(act.because_of)
    if (act.link_to?.length) act.link_to = filterIds(act.link_to)
    if (act.action_target?.length) act.action_target = filterIds(act.action_target)
  }
  for (const agent of ann.agents ?? []) {
    for (const act of agent.actions ?? []) {
      if (act.because_of?.length) act.because_of = filterIds(act.because_of)
      if (act.link_to?.length) act.link_to = filterIds(act.link_to)
      if (act.action_target?.length) act.action_target = filterIds(act.action_target)
      if (act.signaling_details?.link_to?.length) act.signaling_details.link_to = filterIds(act.signaling_details.link_to)
    }
    for (const prop of agent.properties ?? []) {
      if (prop.signaling_details?.link_to?.length) prop.signaling_details.link_to = filterIds(prop.signaling_details.link_to)
    }
    for (const infl of agent.influenced_by ?? []) {
      if (infl.influencers?.length) infl.influencers = filterIds(infl.influencers)
    }
  }
  for (const prop of ann.ego_vehicle?.properties ?? []) {
    if (prop.signaling_details?.link_to?.length) prop.signaling_details.link_to = filterIds(prop.signaling_details.link_to)
  }
  for (const infl of ann.ego_vehicle?.influenced_by ?? []) {
    if (infl.influencers?.length) infl.influencers = filterIds(infl.influencers)
  }
  for (const light of ann.traffic_lights ?? []) {
    for (const head of light.signal_heads ?? []) {
      if (head.influenced_agent_ids?.length) head.influenced_agent_ids = filterIds(head.influenced_agent_ids)
    }
  }
}

/**
 * Remove all data tied to a deleted environment: its conditions and all containment
 * entries (on agents, objects, lights, signal heads, and ego) that reference it.
 * Also removes the condition IDs from all because_of/link_to arrays.
 */
export function cleanupDeletedEnv(ann: SilAvAnnotation, envId: string): void {
  const removedConditionIds = (ann.conditions ?? []).filter(c => c.env_id === envId).map(c => c.id)
  ann.conditions = (ann.conditions ?? []).filter(c => c.env_id !== envId)
  // Unlink containments that referenced this environment (set env_id='') rather than deleting them,
  // so the user sees the "no environment" warning and can re-link them.
  const unlink = (c: { env_id: string }) => { if (c.env_id === envId) c.env_id = '' }
  for (const agent of ann.agents ?? []) {
    agent.containment?.forEach(unlink)
  }
  for (const obj of ann.traffic_objects ?? []) {
    obj.containment?.forEach(unlink)
  }
  for (const light of ann.traffic_lights ?? []) {
    light.containment?.forEach(unlink)
    for (const head of light.signal_heads ?? []) {
      head.env_controlled?.forEach(unlink)
    }
  }
  ann.ego_vehicle?.containment?.forEach(unlink)
  if (removedConditionIds.length > 0) cleanupDeletedIds(ann, removedConditionIds)
}
