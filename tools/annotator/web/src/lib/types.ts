// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// --- Shared sub-structures ---

export interface Containment {
  id: string
  env_id: string
  lane_number?: string
  illegal_flag?: boolean
  near_flag?: boolean
  edge?: 'left' | 'right' | null
  other?: string | null
  start_timestamp: string
  end_timestamp: string
  _track_index?: number
}

export interface Influence {
  id: string
  influencers: string[]
  comment?: string
  start_timestamp: string
  end_timestamp: string
  _influence_track_index?: number
}

// --- Environment ---

export interface Environment {
  id: string
  name?: string
  type: string
  type_other_description?: string
  // For MergeArea (`fst:LaneMerge`) and BranchArea (`fst:LaneFork`), num_lanes
  // is the ego-path lane count BEFORE the transition and num_out_lanes is the
  // count AFTER, both measured on the continuation that ego takes. For every
  // other environment type num_lanes is "lanes in ego-direction" and
  // num_out_lanes is unused.
  num_lanes: number
  num_out_lanes?: number
  one_way?: boolean
  start_timestamp: string
  end_timestamp: string
  keypoints?: Keypoint[]
  _track_index?: number
}

// --- Condition ---

export interface Condition {
  id: string
  env_id: string
  type: string
  condition_other_description?: string
  start_timestamp: string
  end_timestamp: string
  _track_index?: number
  _cond_track_index?: number
}

// --- Traffic Object ---

export interface ObjectStateEntry {
  id?: string
  start_timestamp: string
  end_timestamp: string
  open_state?: string
  motion_state?: string
  other_condition_description?: string
}

// Spatial arrangement; only meaningful for portable traffic-control objects
// whose intentional arrangement matters. Unset for fixed-point types and
// incidental debris/fallen objects.
export type TrafficObjectQuantity = 'Single' | 'Line / Row' | 'Channelizing Line' | 'Perimeter' | 'Group'

export interface TrafficObject {
  id: string
  name?: string
  type: string
  other_type_description?: string
  visibility_start_timestamp: string
  visibility_end_timestamp: string
  lane_number?: string
  quantity?: TrafficObjectQuantity
  containment?: Containment[]
  state_sequence: ObjectStateEntry[]
  keypoints?: Keypoint[]
  _track_index?: number
}

// --- Traffic Light ---

export interface LightStates {
  id?: string
  type?: string
  color?: string
  shape?: string
  start_timestamp: string
  end_timestamp: string
  other_condition_description?: string
  yellow_on_ego_path?: boolean | null
  ego_in_intersection_on_yellow?: boolean | null
  ego_could_have_cleared_safely?: boolean | null
  _state_track_index?: number
}

export interface SignalHead {
  id: string
  state_sequence: LightStates[]
  env_controlled?: Containment[]
  influenced_agent_ids?: string[]
  affects_ego?: string
  start_timestamp: string
  end_timestamp: string
  keypoints?: Keypoint[]
}

export interface TrafficLight {
  id: string
  name?: string
  type?: string
  other_type_description?: string
  visibility_start_timestamp: string
  visibility_end_timestamp: string
  containment?: Containment[]
  signal_heads: SignalHead[]
  _track_index?: number
}

// --- Ego Vehicle ---

export interface EgoAction {
  id?: string
  type: string
  turn_other_description?: string
  action_other_description?: string
  because_of?: string[]
  link_to?: string[]
  action_target?: string[]
  illegal_flag?: boolean
  is_aggressive_or_cut_in?: boolean
  start_timestamp: string
  end_timestamp: string
}

export interface EgoVehicle {
  name?: string
  actions: EgoAction[]
  properties?: AgentProperty[]
  containment?: Containment[]
  influenced_by?: Influence[]
  driving_judgment?: string
}

// --- Properties (shared by Agent and Ego) ---

export interface AgentProperty {
  id: string
  property_type: string
  other_description?: string
  start_timestamp: string
  end_timestamp: string
  // Signal-specific fields (only when property_type starts with 'Signal')
  signaling_details?: SignalingDetails
}

// --- Agent ---

export interface SignalingDetails {
  source?: string
  intent?: string
  sign_type?: string
  other_source_description?: string
  other_intent_description?: string
  other_sign_description?: string
  link_to?: string[]
  not_facing_ego?: boolean
  /** @deprecated Use link_to instead */
  target_agent_ids?: string[]
}

export interface AgentAction {
  id?: string
  action_type: string
  other_description?: string
  because_of?: string[]
  link_to?: string[]
  action_target?: string[]
  start_timestamp: string
  end_timestamp: string
  turn_protected?: boolean
  change_where?: string
  jaywalk_flag?: boolean
  erratic_flag?: boolean
  ego_lane_flag?: boolean
  nudge_magnitude?: string
  is_aggressive_or_cut_in?: boolean
  illegal_flag?: boolean
  signaling_details?: SignalingDetails
}

export interface EgoRelativePose {
  position_rel_to_ego?: string
  direction_rel_to_ego?: string
  start_timestamp: string
  end_timestamp: string
}

export interface Agent {
  id: string
  name?: string
  amount: string
  type: string
  other_type_description?: string
  visibility_start_timestamp?: string
  visibility_end_timestamp?: string
  actions: AgentAction[]
  properties?: AgentProperty[]
  ego_relative_pose?: EgoRelativePose[]
  containment?: Containment[]
  influenced_by?: Influence[]
  keypoints?: Keypoint[]
  _track_index?: number
}

// --- Keypoint (sparse trajectory point) ---

export interface Keypoint {
  timestamp: string
  x: number
  y: number
}

// --- Top-level annotation ---

export interface SilAvAnnotation {
  eventful?: boolean | null
  brief_description: string
  environments: Environment[]
  conditions: Condition[]
  traffic_objects: TrafficObject[]
  traffic_lights: TrafficLight[]
  ego_vehicle: EgoVehicle
  agents: Agent[]
}

export interface VideoMeta {
  clip_id: string
  source?: string
  fps: number
  duration_s: number
}

export interface UiConfig {
  envTrackCount?: number
  objectTrackCount?: number
  lightTrackCount?: number
  agentTrackCount?: number
  egoContTrackCount?: number
  zoomLevel?: number
  scrollOffset?: number
}

export interface AnnotationBundle {
  schema_version: string
  video: VideoMeta
  annotation: SilAvAnnotation
  status: string
  provenance: Record<string, string>
  qa_issues?: { type: string; severity: string; message: string }[]
  _ui_config?: UiConfig
  review_round?: number
}

export interface VideoListItem {
  clip_id: string
  filename: string
  size_mb: number
  status: string
  has_annotation: boolean
  env_count: number
  agent_count: number
  ego_action_count: number
}

export type TrackId = string

export interface TimelineSegment {
  id: string
  trackId: TrackId
  label: string
  t0: number
  t1: number
  illegal?: boolean
  because_of?: string[]
  meta?: unknown
}
