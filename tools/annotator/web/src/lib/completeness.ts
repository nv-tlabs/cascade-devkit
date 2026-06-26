// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import type { SilAvAnnotation, TimelineSegment } from './types'
import { annotationToSegments } from './timeline-utils'
import { ACTION_LINK_TO_CONFIG, QUANTITY_ELIGIBLE_OBJECT_TYPES } from './attribute-cycling'

export type CompletenessResult = { missingCount: number; issues: string[] }

// Env types whose lane count changes across the segment; for these we require
// both pre (num_lanes) and post (num_out_lanes) counts on the ego-path.
const SPLIT_LANE_TYPES = new Set(['fst:LaneMerge', 'fst:LaneFork'])

export function checkSegmentCompleteness(seg: TimelineSegment, ann: SilAvAnnotation): CompletenessResult {
  const issues: string[] = []
  const m = (seg.meta || {}) as Record<string, unknown>

  // Environment
  if (seg.trackId.startsWith('env_') && !m._isCondSubtrack) {
    if (!m.id) issues.push('Missing: id')
    if (!m.type) issues.push('Missing: type')
    if (m.type === 'Other' && !m.type_other_description) issues.push('Missing: other type description')
    if (m.type === 'Other Intersection' && !m.type_other_description) issues.push('Missing: other intersection description')
    // For MergeArea, num_lanes is the incoming side; num_out_lanes is required
    // in addition. Every other env type just needs num_lanes. Messages are
    // keyed to match the UI field labels so the RightPanel highlights line up.
    if (SPLIT_LANE_TYPES.has(m.type as string)) {
      if (m.num_lanes == null || m.num_lanes === 0) issues.push('Missing: in_lanes')
      if (m.num_out_lanes == null || m.num_out_lanes === 0) issues.push('Missing: out_lanes')
    } else if (m.num_lanes == null || m.num_lanes === 0) {
      issues.push('Missing: num_lanes')
    }
    const ei = m._envIndex as number
    if (ei != null) {
      const env = ann.environments?.[ei]
      if (!env?.keypoints?.length) issues.push('Missing: keypoints (need >= 1)')
    }
  }

  // Condition
  if (m._isCondSubtrack) {
    if (!m.id) issues.push('Missing: id')
    if (!m.env_id) issues.push('Missing: env_id')
    if (!m.type) issues.push('Missing: type')
    if (m.type === 'Other' && !m.condition_other_description) issues.push('Missing: other condition description')
  }

  // Traffic Object (parent)
  if (seg.trackId.startsWith('obj_') && !m._isObjStateSubtrack && !m._isObjContSubtrack && m._objKind === 'traffic_object') {
    if (!m.id) issues.push('Missing: id')
    if (!m.type) issues.push('Missing: type')
    if (typeof m.type === 'string' && m.type.startsWith('Other') && !m.other_type_description) issues.push('Missing: other type description')
    // Quantity is mandatory only for portable traffic-control objects whose
    // intentional arrangement matters. Fixed-point types and incidental
    // debris/fallen objects skip this check.
    if (typeof m.type === 'string' && QUANTITY_ELIGIBLE_OBJECT_TYPES.has(m.type) && !m.quantity) {
      issues.push('Missing: quantity')
    }
    // Check for at least 1 state. Containment is optional because recent
    // batches may annotate objects without environments.
    const oi = m._objIndex as number
    if (oi != null) {
      const obj = ann.traffic_objects?.[oi]
      if (!obj?.state_sequence?.length) issues.push('Missing: state_sequence (need >= 1)')
      if (!obj?.keypoints?.length) issues.push('Missing: keypoints (need >= 1)')
    }
  }

  // Traffic Light (parent)
  if (seg.trackId.startsWith('light_') && !m._isSignalHeadSubtrack && !m._isLightStateSubtrack && !m._isLightContSubtrack && !m._isLightPhysContSubtrack) {
    if (!m.id) issues.push('Missing: id')
    const li = m._lightIndex as number
    if (li != null) {
      const light = ann.traffic_lights?.[li]
      if (!light?.signal_heads?.length) issues.push('Missing: signal_heads (need >= 1)')
    }
  }

  // Signal Head subtrack
  if (m._isSignalHeadSubtrack) {
    if (!m.id) issues.push('Missing: id')
    const li = m._lightIndex as number
    const hi = m._headIndex as number
    if (li != null && hi != null) {
      const sh = ann.traffic_lights?.[li]?.signal_heads?.[hi]
      if (!sh?.state_sequence?.length) issues.push('Missing: state_sequence (need >= 1)')
      if (!sh?.keypoints?.length) issues.push('Missing: keypoints (need >= 1)')
    }
  }

  // Light State subtrack
  if (m._isLightStateSubtrack) {
    if (!m.color) issues.push('Missing: color')
    if (!m.type) issues.push('Missing: type')
    if (!m.shape) issues.push('Missing: shape')
    if ((m.color === 'Other' || m.shape === 'Other') && !m.other_condition_description) issues.push('Missing: other description')
    if (m.color === 'Yellow') {
      if (m.yellow_on_ego_path == null) issues.push('Missing: yellow on ego path')
      if (m.yellow_on_ego_path === true && m.ego_in_intersection_on_yellow == null) issues.push('Missing: ego in intersection on yellow')
      if (m.yellow_on_ego_path === true && m.ego_in_intersection_on_yellow === false && m.ego_could_have_cleared_safely == null) issues.push('Missing: could ego have cleared safely')
    }
  }

  // Object State subtrack
  if (m._isObjStateSubtrack) {
    if (!m.motion_state) issues.push('Missing: motion_state')
  }

  // Property (agent or ego)
  if (m._isAgentPropertySubtrack || m._isEgoPropertySubtrack) {
    if (!m.property_type) issues.push('Missing: property_type')
    if (m.property_type === 'Other' && !m.other_description) issues.push('Missing: other description')
    if (m.property_type === 'Signal') {
      const sd = m.signaling_details as Record<string, unknown> | undefined
      if (!sd?.source) issues.push('Missing: signal source')
      if (sd?.source === 'Other' && !sd?.other_source_description) issues.push('Missing: other source description')
      if (!sd?.intent) issues.push('Missing: signal intent')
      if (sd?.intent === 'Other' && !sd?.other_intent_description) issues.push('Missing: other intent description')
      if (sd?.source === 'Holding sign' && !sd?.sign_type) issues.push('Missing: signal sign type')
      if (sd?.source === 'Holding sign' && sd?.sign_type === 'Other' && !sd?.other_sign_description) issues.push('Missing: other sign type description')
    }
  }

  // Ego Action
  if (seg.trackId === 'ego_act' && !m._isEgoContSubtrack && !m._isEgoInfluenceSubtrack && !m._isEgoPropertySubtrack && !m._isEgoMain) {
    if (!m.type || m.type === 'none') issues.push('Missing: type')
    if (m.type === 'Other' && !m.action_other_description) issues.push('Missing: other action description')
    if (m.type === 'Other oxd:MakeATurn' && !m.turn_other_description) issues.push('Missing: other turn description')
    const config = ACTION_LINK_TO_CONFIG[m.type as string]
    if (config?.required) {
      const actionTarget = m.action_target as string[] | undefined
      if (!actionTarget?.length) issues.push(`Missing: ${config.missingLabel}`)
    }
  }

  // Agent (parent)
  if (seg.trackId.startsWith('agent_') && m._objKind === 'agent' && !m._isAgentActionSubtrack && !m._isAgentPropertySubtrack && !m._isAgentContSubtrack && !m._isAgentPoseSubtrack) {
    if (!m.id) issues.push('Missing: id')
    if (!m.type) issues.push('Missing: type')
    if (m.type === 'Other' && !m.other_type_description) issues.push('Missing: other type description')
    if (m.type === 'Pedestrian (Other)' && !m.other_type_description) issues.push('Missing: other pedestrian description')
    if (m.type === 'oxd:Animal' && !m.other_type_description) issues.push('Missing: animal description')
    if (!m.amount) issues.push('Missing: amount')
    const ai = m._agentIndex as number
    if (ai != null) {
      const agent = ann.agents?.[ai]
      if (!agent?.actions?.length) issues.push('Missing: actions (need >= 1)')
      if (!agent?.ego_relative_pose?.length) issues.push('Missing: pose (need >= 1)')
      if (!agent?.keypoints?.length) issues.push('Missing: keypoints (need >= 1)')
    }
  }

  // Agent Action
  if (m._isAgentActionSubtrack) {
    if (!m.action_type) issues.push('Missing: action_type')
    if (m.action_type === 'Other' && !m.other_description) issues.push('Missing: other action description')
    if (m.action_type === 'Other oxd:MakeATurn' && !m.other_description) issues.push('Missing: other turn description')
    const config = ACTION_LINK_TO_CONFIG[m.action_type as string]
    if (config?.required) {
      const actionTarget = m.action_target as string[] | undefined
      if (!actionTarget?.length) issues.push(`Missing: ${config.missingLabel}`)
    }
  }

  // Agent Pose
  if (m._isAgentPoseSubtrack) {
    if (!m.position_rel_to_ego) issues.push('Missing: position_rel_to_ego')
    if (!m.direction_rel_to_ego) issues.push('Missing: direction_rel_to_ego')
  }

  // Influence (ego or agent)
  if (m._isEgoInfluenceSubtrack || m._isAgentInfluenceSubtrack) {
    const influencers = m.influencers as string[] | undefined
    if (!influencers?.length) issues.push('Missing: influencers')
  }

  // Containment is optional, and when present may be left unassigned to an
  // environment. Only require the free-text detail when "Other" is selected.
  if (m._isEgoContSubtrack || m._isAgentContSubtrack || m._isObjContSubtrack || m._isLightContSubtrack || m._isLightPhysContSubtrack) {
    if (m.env_id === 'Other' && (m.other == null || m.other === '')) issues.push('Missing: other environment description')
  }

  return { missingCount: issues.length, issues }
}

export function buildCompletenessMap(ann: SilAvAnnotation | undefined): Map<string, CompletenessResult> {
  const map = new Map<string, CompletenessResult>()
  if (!ann) return map
  const segs = annotationToSegments(ann)
  for (const seg of segs) {
    const result = checkSegmentCompleteness(seg, ann)
    if (result.missingCount > 0) {
      map.set(seg.id, result)
    }
  }
  return map
}
