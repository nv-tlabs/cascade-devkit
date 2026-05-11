import type { TimelineSegment, SilAvAnnotation } from './types'

// --- Action-specific link-to configuration ---
// Defines which actions have a specific "target" link-to (in addition to the general optional link_to).
// entityTypes: 'agent' = agents + Ego, 'object' = traffic objects
export interface ActionLinkToConfig {
  label: string         // shown in the UI as the picker question
  missingLabel: string  // noun phrase used in completeness error messages
  entityTypes: ('agent' | 'object')[]
  required: boolean
}

export const ACTION_LINK_TO_CONFIG: Record<string, ActionLinkToConfig> = {
  'fst:Enter':                                  { label: 'What?',           missingLabel: 'target to enter/embark',   entityTypes: ['agent', 'object'], required: true },
  'fst:Exit':                                   { label: 'What?',           missingLabel: 'target to exit/disembark', entityTypes: ['agent', 'object'], required: true },
  'oxd:FollowRoadUser':                         { label: 'What road user?', missingLabel: 'road user being followed', entityTypes: ['agent'],           required: true },
  'oxd:Overtake':                               { label: 'What?',           missingLabel: 'road user being overtaken', entityTypes: ['agent'],          required: true },
  'oxd:Overtake (using ego lane)':              { label: 'What?',           missingLabel: 'road user being overtaken', entityTypes: ['agent'],          required: true },
  'oxd:Overtake (not using ego lane)':          { label: 'What?',           missingLabel: 'road user being overtaken', entityTypes: ['agent'],          required: true },
  'fst:Nudge (in lane)':                        { label: 'Around what?',    missingLabel: 'obstacle being nudged around', entityTypes: ['agent', 'object'], required: false },
  'fst:Nudge (out of lane)':                    { label: 'Around what?',    missingLabel: 'obstacle being nudged around', entityTypes: ['agent', 'object'], required: false },
  'fst:Nudge (out of lane: not into ego lane)': { label: 'Around what?',    missingLabel: 'obstacle being nudged around', entityTypes: ['agent', 'object'], required: false },
  'fst:Nudge (out of lane: into ego lane)':     { label: 'Around what?',    missingLabel: 'obstacle being nudged around', entityTypes: ['agent', 'object'], required: false },
  'fst:Yield':                                  { label: 'To what?',        missingLabel: 'target being yielded to',  entityTypes: ['agent'],           required: false },
}

// --- Ordered enum arrays (matching backend/models/sil_av.py) ---

export const ENVIRONMENT_TYPES = [
  'oxd:Road', 'fst:LaneMerge', 'fst:LaneFork',
  'oxd:TIntersection', 'oxd:YIntersection', 'oxd:CrossRoad',
  '5-way', '6-way', '6+-way', 'Other Intersection',
  'oxd:Roundabout', 'oxd:Tunnel', 'oxd:Bridge',
  'oxd:PavedShoulder', 'oxd:GrassShoulder',
  'oxd:Sidewalk', 'oxd:PedestrianCrossing',
  'oxd:RailCrossing', 'oxd:CycleLane', 'Other',
]

export const CONDITION_TYPES = [
  'Construction Zone', 'Temporarily marked',
  'oxd:snowyRoadCondition', 'oxd:wetRoadCondition',
  'Overgrown', 'Shared marked center lane',
  'No direction divider', 'Lanes obscured / unmarked',
  'Other',
]

export const EGO_ACTION_TYPES = [
  'fst:ManeuverAbort', 'oxd:Stop', 'oxd:NotMove', 'fst:Enter', 'fst:Exit',
  'fst:Creep', 'fst:Yield', 'oxd:Decelerate',
  'fst:DrivingInLane', 'oxd:FollowRoadUser',
  'fst:Nudge (in lane)', 'fst:Nudge (out of lane)',
  'oxd:Overtake',
  'oxd:ChangeLane (left)', 'oxd:ChangeLane (right)',
  'oxd:MakeALeftTurn (unprotected)', 'oxd:MakeARightTurn (unprotected)',
  'fst:MakeAUTurn (unprotected)',
  'oxd:MakeALeftTurn (protected)', 'oxd:MakeARightTurn (protected)',
  'fst:MakeAUTurn (protected)', 'Other oxd:MakeATurn',
  'fst:Reverse',
  'Other',
]

export const TRAFFIC_OBJECT_TYPES = [
  'fst:StopSign', 'fst:YieldSign', 'fst:SpeedLimitSign',
  'Merge ahead', 'Adjacent lanes ahead', 'Do not enter',
  'Other oxd:TrafficSign',
  'oxd:RailCrossing', 'Garage', 'oxd:TollPlaza', 'Bollard',
  'oxd:Roadblocks', 'oxd:TrafficCone', 'oxd:WarningSign',
  'PortableDisplay', 'Barrier', 'Other Small Portable Traffic Indicator',
  'Toy', 'Ball', 'Other fst:FallenObject',
  'Dirt', 'Trash', 'Other oxd:Debris', 'Not identifiable', 'Other',
]

export const TRAFFIC_OBJECT_QUANTITIES = ['Single', 'Line / Row', 'Channelizing Line', 'Perimeter', 'Group']

// Portable traffic-control objects that are intentionally placed in an
// arrangement (single, channelizing line, perimeter, or unordered group).
// Excludes fixed-point types (regulatory signs, infrastructure) and also
// excludes fallen objects / debris — those aren't "arranged", and only
// Single vs Group would be meaningful, which is a different axis.
export const QUANTITY_ELIGIBLE_OBJECT_TYPES = new Set<string>([
  'oxd:Roadblocks', 'oxd:TrafficCone', 'PortableDisplay', 'Other Small Portable Traffic Indicator',
  'Barrier', 'Bollard',
  'Other',
])

export const AGENT_ACTION_TYPES = [
  'fst:ManeuverAbort', 'fst:Park',
  'oxd:Stop',
  'oxd:NotMove',
  'fst:Enter', 'fst:Exit', 'fst:Creep', 'fst:Yield', 'oxd:Decelerate',
  'fst:DrivingInLane', 'oxd:FollowRoadUser',
  'fst:Nudge (in lane)', 'fst:Nudge (out of lane: not into ego lane)',
  'fst:Nudge (out of lane: into ego lane)',
  'oxd:Overtake (using ego lane)', 'oxd:Overtake (not using ego lane)',
  'oxd:ChangeLane (left)', 'oxd:ChangeLane (right)',
  'oxd:MakeALeftTurn (unprotected)', 'oxd:MakeARightTurn (unprotected)',
  'fst:MakeAUTurn (unprotected)', 'oxd:MakeALeftTurn (protected)',
  'oxd:MakeARightTurn (protected)', 'fst:MakeAUTurn (protected)',
  'Other oxd:MakeATurn',
  'fst:Reverse', 'Other',
]

export const AGENT_ACTION_TYPES_PEDESTRIAN = [
  'fst:ManeuverAbort', 'oxd:Stop',
  'oxd:Stand',
  'fst:Enter', 'fst:Exit', 'fst:Creep', 'fst:Yield', 'oxd:Decelerate', 'oxd:FollowRoadUser',
  'oxd:Walk', 'oxd:Walk (jaywalk)', 'oxd:Walk (erratic)', 'oxd:Walk (jaywalk, erratic)',
  'oxd:Run', 'oxd:Run (jaywalk)', 'oxd:Run (erratic)', 'oxd:Run (jaywalk, erratic)',
  'Other',
]

export function getAgentActionTypes(agentType: string): string[] {
  return PEDESTRIAN_OR_ANIMAL_TYPES.has(agentType) ? AGENT_ACTION_TYPES_PEDESTRIAN : AGENT_ACTION_TYPES
}

export function getAgentActionTypeGroups(agentType: string): OptionGroupEntry[] | undefined {
  return PEDESTRIAN_OR_ANIMAL_TYPES.has(agentType) ? undefined : AGENT_ACTION_TYPE_GROUPS
}

export const AGENT_TYPES = [
  'Vehicle', 'oxd:Car', 'Heavy-duty vehicle', 'oxd:EmergencyVehicle',
  'oxd:Truck', 'PublicBus',
  'oxd:Bicycle', 'oxd:Motorcycle', 'fst:Scooter',
  'Pedestrian', 'Pedestrian (Officer)', 'Pedestrian (Personnel)',
  'Pedestrian (Adult)', 'Pedestrian (Kid/Teen)',
  'Pedestrian (Stroller)', 'Pedestrian (oxd:Wheelchair)',
  'Pedestrian (Other)', 'oxd:Animal', 'Other',
]

export type OptionGroupEntry =
  | string  // standalone selectable option
  | { header: string; options: string[]; selectable?: boolean }  // selectable:true flattens (header is a selectable parent, children indented); default is decorative optgroup

/** @deprecated Use OptionGroupEntry instead */
export type AgentTypeGroupEntry = OptionGroupEntry

export const AGENT_TYPE_GROUPS: OptionGroupEntry[] = [
  { header: 'Vehicle', selectable: true, options: ['oxd:Car', 'Heavy-duty vehicle', 'oxd:EmergencyVehicle', 'oxd:Truck', 'PublicBus', 'oxd:Bicycle', 'oxd:Motorcycle', 'fst:Scooter'] },
  { header: 'Pedestrian', selectable: true, options: ['Pedestrian (Officer)', 'Pedestrian (Personnel)', 'Pedestrian (Adult)', 'Pedestrian (Kid/Teen)', 'Pedestrian (Stroller)', 'Pedestrian (oxd:Wheelchair)', 'Pedestrian (Other)'] },
  'oxd:Animal',
  'Other',
]

export const TRAFFIC_OBJECT_TYPE_GROUPS: OptionGroupEntry[] = [
  { header: 'Traffic signs', options: ['fst:StopSign', 'fst:YieldSign', 'fst:SpeedLimitSign', 'Merge ahead', 'Adjacent lanes ahead', 'Do not enter', 'oxd:WarningSign', 'Other oxd:TrafficSign'] },
  { header: 'Infrastructure', options: ['oxd:RailCrossing', 'Garage', 'oxd:TollPlaza', 'Bollard'] },
  { header: 'Portable indicators', options: ['oxd:Roadblocks', 'oxd:TrafficCone', 'PortableDisplay', 'Barrier', 'Other Small Portable Traffic Indicator'] },
  { header: 'Fallen objects', options: ['Toy', 'Ball', 'Other fst:FallenObject'] },
  { header: 'Road debris', options: ['Dirt', 'Trash', 'Other oxd:Debris'] },
  'Not identifiable',
  'Other',
]

export const ENVIRONMENT_TYPE_GROUPS: OptionGroupEntry[] = [
  'oxd:Road', 'fst:LaneMerge', 'fst:LaneFork',
  { header: 'Intersection', options: ['oxd:TIntersection', 'oxd:YIntersection', 'oxd:CrossRoad', '5-way', '6-way', '6+-way', 'Other Intersection'] },
  'oxd:Roundabout',
  'oxd:Tunnel', 'oxd:Bridge',
  'oxd:PavedShoulder', 'oxd:GrassShoulder',
  'oxd:Sidewalk', 'oxd:PedestrianCrossing',
  'oxd:RailCrossing', 'oxd:CycleLane', 'Other',
]

export const EGO_ACTION_TYPE_GROUPS: OptionGroupEntry[] = [
  'fst:ManeuverAbort', 'oxd:Stop', 'oxd:NotMove', 'fst:Enter', 'fst:Exit',
  'fst:Creep', 'fst:Yield', 'oxd:Decelerate',
  'fst:DrivingInLane', 'oxd:FollowRoadUser',
  'fst:Nudge (in lane)', 'fst:Nudge (out of lane)',
  'oxd:Overtake',
  'oxd:ChangeLane (left)', 'oxd:ChangeLane (right)',
  { header: 'Turns', options: [
    'oxd:MakeALeftTurn (unprotected)', 'oxd:MakeARightTurn (unprotected)',
    'fst:MakeAUTurn (unprotected)',
    'oxd:MakeALeftTurn (protected)', 'oxd:MakeARightTurn (protected)',
    'fst:MakeAUTurn (protected)', 'Other oxd:MakeATurn',
  ]},
  'fst:Reverse',
  'Other',
]

export const AGENT_ACTION_TYPE_GROUPS: OptionGroupEntry[] = [
  'fst:ManeuverAbort', 'fst:Park',
  'oxd:Stop',
  'oxd:NotMove',
  'fst:Enter', 'fst:Exit', 'fst:Creep', 'fst:Yield', 'oxd:Decelerate',
  'fst:DrivingInLane', 'oxd:FollowRoadUser',
  'fst:Nudge (in lane)', 'fst:Nudge (out of lane: not into ego lane)',
  'fst:Nudge (out of lane: into ego lane)',
  'oxd:Overtake (using ego lane)', 'oxd:Overtake (not using ego lane)',
  'oxd:ChangeLane (left)', 'oxd:ChangeLane (right)',
  { header: 'Turns', options: [
    'oxd:MakeALeftTurn (unprotected)', 'oxd:MakeARightTurn (unprotected)',
    'fst:MakeAUTurn (unprotected)', 'oxd:MakeALeftTurn (protected)',
    'oxd:MakeARightTurn (protected)', 'fst:MakeAUTurn (protected)',
    'Other oxd:MakeATurn',
  ]},
  'fst:Reverse', 'Other',
]

const PEDESTRIAN_OR_ANIMAL_TYPES = new Set([
  'Pedestrian', 'Pedestrian (Officer)', 'Pedestrian (Personnel)',
  'Pedestrian (Adult)', 'Pedestrian (Kid/Teen)',
  'Pedestrian (Stroller)', 'Pedestrian (oxd:Wheelchair)',
  'Pedestrian (Other)', 'oxd:Animal',
])

const ALL_AMOUNTS = ['Single', 'Row/group', 'Light traffic', 'Medium traffic', 'Heavy traffic']
const PED_AMOUNTS = ['Single', 'Row/group']

export function getAllowedAmounts(agentType: string): string[] {
  return PEDESTRIAN_OR_ANIMAL_TYPES.has(agentType) ? PED_AMOUNTS : ALL_AMOUNTS
}

export const POSITION_REL_TO_EGO = ['In front', 'Left', 'Right', 'Behind']

export const DIRECTION_REL_TO_EGO = ['Same', 'Opposite', 'Perpendicular-L-R', 'Perpendicular-R-L']

export const LIGHT_STATE_TYPES = ['Fixed', 'Flashing', 'OFF']
export const LIGHT_COLORS = ['Green', 'Yellow', 'Red', 'Other']
export const LIGHT_SHAPES = ['Round', 'Arrow_Left', 'Arrow_Right', 'Arrow_Up', 'Arrow_Down', 'Other']

export const MOTION_STATES = ['Static', 'Moving / Rolling']
export const OPEN_STATES = ['Open', 'Closed']
export const AFFECTS_EGO = ['True', 'False']

export const SIGNAL_SOURCES = ['Flashing light', 'Hand gesture', 'Holding sign', 'Other']
export const SIGNAL_INTENTS = ['Turn', 'Stop', 'Slow Down', 'Proceed', 'Follow', 'Caution', 'Danger', 'Unclear / Incorrectly used', 'Other']
export const SIGN_TYPES = ['Stop Sign', 'Yield Sign', 'Slow Sign', 'Not identifiable', 'Other']

// --- Property types (for Agent and Ego properties subtrack) ---

export const PROPERTY_TYPES = [
  'Slow', 'Fast', 'Aggressive', 'Erratic', 'Emergency', 'On Duty', 'Double Parked',
  'Signal', 'Outside Camera View', 'Other',
]

export const PROPERTY_DISPLAY_NAMES: Record<string, string> = {
  'Slow': 'Slow',
  'Fast': 'Fast',
  'Aggressive': 'Aggressive',
  'Erratic': 'Erratic',
  'Emergency': 'Emergency',
  'On Duty': 'On Duty',
  'Double Parked': 'Double Parked',
  'Signal': 'Signal',
  'Outside Camera View': 'Outside Camera View',
  'Other': 'Other (specify)',
}

// --- Display names ---

export const ENVIRONMENT_DISPLAY_NAMES: Record<string, string> = {
  'oxd:Road': 'Road',
  'fst:LaneMerge': 'Merge area',
  'fst:LaneFork': 'Branch area',
  'oxd:TIntersection': 'T-intersection',
  'oxd:YIntersection': 'Y-intersection (3-way)',
  'oxd:CrossRoad': 'Crossroad (4-way)',
  '5-way': '5-way',
  '6-way': '6-way',
  '6+-way': '6+-way',
  'Other Intersection': 'Other intersection',
  'oxd:Roundabout': 'Roundabout',
  'oxd:Tunnel': 'Tunnel / Underpass',
  'oxd:Bridge': 'Bridge / Overpass',
  'oxd:PavedShoulder': 'Paved shoulder',
  'oxd:GrassShoulder': 'Grass shoulder',
  'oxd:Sidewalk': 'Sidewalk',
  'oxd:PedestrianCrossing': 'Pedestrian crossing',
  'oxd:RailCrossing': 'Railroad crossing',
  'oxd:CycleLane': 'Bike lane',
  'Other': 'Other (specify)',
}

export const CONDITION_DISPLAY_NAMES: Record<string, string> = {
  'Construction Zone': 'Construction zone',
  'Temporarily marked': 'Temporarily marked',
  'oxd:snowyRoadCondition': 'Snowy road',
  'oxd:wetRoadCondition': 'Wet road',
  'Overgrown': 'Overgrown',
  'Other': 'Other (specify)',
}

export const EGO_ACTION_DISPLAY_NAMES: Record<string, string> = {
  'fst:ManeuverAbort': 'Maneuver abort',
  'oxd:Stop': 'Stop',
  'oxd:NotMove': 'Not moving',
  'fst:Enter': 'Embark/Enter',
  'fst:Exit': 'Disembark/Exit',
  'fst:Creep': 'Creep',
  'fst:Yield': 'Yield',
  'oxd:Decelerate': 'Decelerate',
  'fst:DrivingInLane': 'Driving in lane',
  'oxd:FollowRoadUser': 'Follow road user',
  'fst:Nudge (in lane)': 'Nudge (in lane)',
  'fst:Nudge (out of lane)': 'Nudge (out of lane)',
  'oxd:Overtake': 'Overtake',
  'oxd:ChangeLane (left)': 'Change lane (left)',
  'oxd:ChangeLane (right)': 'Change lane (right)',
  'oxd:MakeALeftTurn (unprotected)': 'Left turn (unprotected)',
  'oxd:MakeARightTurn (unprotected)': 'Right turn (unprotected)',
  'fst:MakeAUTurn (unprotected)': 'U-turn (unprotected)',
  'oxd:MakeALeftTurn (protected)': 'Left turn (protected)',
  'oxd:MakeARightTurn (protected)': 'Right turn (protected)',
  'fst:MakeAUTurn (protected)': 'U-turn (protected)',
  'Other oxd:MakeATurn': 'Other turn (specify)',
  'fst:Reverse': 'Reverse',
  'Other': 'Other (specify)',
}

export const TRAFFIC_OBJECT_DISPLAY_NAMES: Record<string, string> = {
  'fst:StopSign': 'Stop sign',
  'fst:YieldSign': 'Yield sign',
  'fst:SpeedLimitSign': 'Speed limit sign',
  'Merge ahead': 'Merge ahead',
  'Adjacent lanes ahead': 'Adjacent lanes ahead',
  'Do not enter': 'Do not enter',
  'Other oxd:TrafficSign': 'Other traffic sign (specify)',
  'oxd:RailCrossing': 'Rail crossing',
  'Garage': 'Garage',
  'oxd:TollPlaza': 'Toll plaza',
  'Bollard': 'Bollard',
  'Barrier': 'Barrier',
  'oxd:Roadblocks': 'Roadblocks',
  'oxd:TrafficCone': 'Traffic cone / pole',
  'oxd:WarningSign': 'Warning sign',
  'PortableDisplay': 'Portable display',
  'Other Small Portable Traffic Indicator': 'Other portable indicator (specify)',
  'Toy': 'Toy',
  'Ball': 'Ball',
  'Other fst:FallenObject': 'Other fallen object (specify)',
  'Dirt': 'Dirt',
  'Trash': 'Trash',
  'Other oxd:Debris': 'Other debris (specify)',
  'Not identifiable': 'Not identifiable',
  'Other': 'Other (specify)',
}

export const AGENT_TYPE_DISPLAY_NAMES: Record<string, string> = {
  'oxd:Car': 'Car',
  'Heavy-duty vehicle': 'Heavy-duty vehicle',
  'oxd:EmergencyVehicle': 'Emergency vehicle',
  'oxd:Truck': 'Van/Truck',
  'PublicBus': 'Public bus',
  'oxd:Bicycle': 'Bicycle',
  'oxd:Motorcycle': 'Motorcycle',
  'fst:Scooter': 'Scooter',
  'Pedestrian (Officer)': 'Pedestrian (Officer)',
  'Pedestrian (Personnel)': 'Pedestrian (Personnel)',
  'Pedestrian (Adult)': 'Pedestrian (Adult)',
  'Pedestrian (Kid/Teen)': 'Pedestrian (Kid/Teen)',
  'Pedestrian (Stroller)': 'Pedestrian (Stroller)',
  'Pedestrian (oxd:Wheelchair)': 'Pedestrian (Wheelchair)',
  'Pedestrian (Other)': 'Pedestrian (Other)',
  'oxd:Animal': 'Animal',
  'Other': 'Other (specify)',
}

export const AGENT_ACTION_DISPLAY_NAMES: Record<string, string> = {
  'fst:ManeuverAbort': 'Maneuver abort',
  'fst:Park': 'Park',
  'oxd:Stop': 'Stop',
  'oxd:NotMove': 'Not moving',
  'fst:Enter': 'Embark/Enter',
  'fst:Exit': 'Disembark/Exit',
  'fst:Creep': 'Creep',
  'fst:Yield': 'Yield',
  'oxd:Decelerate': 'Decelerate',
  'fst:DrivingInLane': 'Driving in lane',
  'oxd:FollowRoadUser': 'Follow road user',
  'fst:Nudge (in lane)': 'Nudge (in lane)',
  'fst:Nudge (out of lane: not into ego lane)': 'Nudge (out, not into ego lane)',
  'fst:Nudge (out of lane: into ego lane)': 'Nudge (out, into ego lane)',
  'oxd:Overtake (using ego lane)': 'Overtake (using ego lane)',
  'oxd:Overtake (not using ego lane)': 'Overtake (not using ego lane)',
  'oxd:ChangeLane (left)': 'Change lane (left)',
  'oxd:ChangeLane (right)': 'Change lane (right)',
  'oxd:MakeALeftTurn (unprotected)': 'Left turn (unprotected)',
  'oxd:MakeARightTurn (unprotected)': 'Right turn (unprotected)',
  'fst:MakeAUTurn (unprotected)': 'U-turn (unprotected)',
  'oxd:MakeALeftTurn (protected)': 'Left turn (protected)',
  'oxd:MakeARightTurn (protected)': 'Right turn (protected)',
  'fst:MakeAUTurn (protected)': 'U-turn (protected)',
  'Other oxd:MakeATurn': 'Other turn (specify)',
  'fst:Reverse': 'Reverse',
  'Signals': 'Signals',
  'Other': 'Other (specify)',
  // Pedestrian-specific
  'oxd:Stand': 'Stand',
  'oxd:Walk': 'Walk',
  'oxd:Walk (jaywalk)': 'Walk (jaywalk)',
  'oxd:Walk (erratic)': 'Walk (erratic)',
  'oxd:Walk (jaywalk, erratic)': 'Walk (jaywalk, erratic)',
  'oxd:Run': 'Run',
  'oxd:Run (jaywalk)': 'Run (jaywalk)',
  'oxd:Run (erratic)': 'Run (erratic)',
  'oxd:Run (jaywalk, erratic)': 'Run (jaywalk, erratic)',
}

export function envDisplayName(enumVal: string): string {
  return ENVIRONMENT_DISPLAY_NAMES[enumVal] || enumVal
}

// --- Helpers ---

function cycleValue<T>(list: T[], current: T, direction: 1 | -1): T {
  const idx = list.indexOf(current)
  if (idx < 0) return list[0]
  return list[(idx + direction + list.length) % list.length]
}

type Meta = Record<string, unknown>

// --- Primary cycling (Left/Right) ---

export function cyclePrimary(
  seg: TimelineSegment,
  direction: 1 | -1
): { field: string; value: string | string[] } | null {
  const meta = (seg.meta || {}) as Meta

  if ((meta as Record<string, unknown>)._isCondSubtrack) {
    const cur = (meta.type as string) || CONDITION_TYPES[0]
    return { field: 'type', value: cycleValue(CONDITION_TYPES, cur, direction) }
  }
  if (seg.trackId.startsWith('env_')) {
    const cur = (meta.type as string) || ''
    return { field: 'type', value: cycleValue(ENVIRONMENT_TYPES, cur, direction) }
  }
  if ((meta as Record<string, unknown>)._isObjStateSubtrack) {
    const cur = (meta.motion_state as string) || 'Static'
    return { field: 'motion_state', value: cycleValue(MOTION_STATES, cur, direction) }
  }
  if ((meta as Record<string, unknown>)._isLightStateSubtrack) {
    const cur = (meta.color as string) || 'Green'
    return { field: 'color', value: cycleValue(LIGHT_COLORS, cur, direction) }
  }
  if (seg.trackId.startsWith('light_')) {
    return null
  }
  if ((meta as Record<string, unknown>)._isEgoContSubtrack || (meta as Record<string, unknown>)._isAgentContSubtrack ||
      (meta as Record<string, unknown>)._isObjContSubtrack || (meta as Record<string, unknown>)._isLightContSubtrack ||
      (meta as Record<string, unknown>)._isLightPhysContSubtrack) {
    const cur = (meta.lane_number as string) || 'none'
    if (cur === 'none') {
      return direction === 1 ? { field: 'lane_number', value: '1' } : null
    }
    const num = parseInt(cur, 10)
    if (isNaN(num)) return { field: 'lane_number', value: '1' }
    const next = num + direction
    return { field: 'lane_number', value: next < 1 ? 'none' : String(next) }
  }
  if (seg.trackId === 'ego_act') {
    const cur = (meta.type as string) || ''
    return { field: 'type', value: cycleValue(EGO_ACTION_TYPES, cur, direction) }
  }
  if (seg.trackId.startsWith('obj_')) {
    const cur = (meta.type as string) || ''
    return { field: 'type', value: cycleValue(TRAFFIC_OBJECT_TYPES, cur, direction) }
  }
  if (seg.trackId.startsWith('agent_')) {
    if ((meta as Record<string, unknown>)._isAgentPropertySubtrack) {
      const cur = (meta.property_type as string) || ''
      return { field: 'property_type', value: cycleValue(PROPERTY_TYPES, cur, direction) }
    }
    if ((meta as Record<string, unknown>)._isAgentActionSubtrack) {
      const cur = (meta.action_type as string) || ''
      const agentType = (meta.agent_type as string) || ''
      return { field: 'action_type', value: cycleValue(getAgentActionTypes(agentType), cur, direction) }
    }
    // Parent agent segment — cycle agent type
    const cur = (meta.type as string) || ''
    return { field: 'agent_type_enum', value: cycleValue(AGENT_TYPES, cur, direction) }
  }
  if (seg.trackId === 'ego_act' && (meta as Record<string, unknown>)._isEgoPropertySubtrack) {
    const cur = (meta.property_type as string) || ''
    return { field: 'property_type', value: cycleValue(PROPERTY_TYPES, cur, direction) }
  }
  return null
}

// --- Secondary cycling (Up/Down) ---

export function cycleSecondary(
  seg: TimelineSegment,
  direction: 1 | -1
): { field: string; value: unknown } | null {
  const meta = (seg.meta || {}) as Meta

  if (seg.trackId.startsWith('env_')) {
    const cur = (meta.num_lanes as number) || 0
    const next = Math.max(1, cur + direction)
    return { field: 'num_lanes', value: next }
  }

  if ((meta as Record<string, unknown>)._isCondSubtrack || seg.trackId.startsWith('light_')) {
    return null
  }

  if ((meta as Record<string, unknown>)._isEgoContSubtrack || (meta as Record<string, unknown>)._isAgentContSubtrack ||
      (meta as Record<string, unknown>)._isObjContSubtrack || (meta as Record<string, unknown>)._isLightContSubtrack ||
      (meta as Record<string, unknown>)._isLightPhysContSubtrack) {
    return { field: 'illegal_flag', value: !meta.illegal_flag }
  }

  if (seg.trackId === 'ego_act') {
    return { field: 'illegal_flag', value: !meta.illegal_flag }
  }

  if ((meta as Record<string, unknown>)._isObjStateSubtrack) {
    const cur = (meta.open_state as string) || ''
    return cur ? { field: 'open_state', value: cycleValue(OPEN_STATES, cur, direction) } : { field: 'open_state', value: 'Open' }
  }
  if ((meta as Record<string, unknown>)._isLightStateSubtrack) {
    const cur = (meta.shape as string) || 'Round'
    return { field: 'shape', value: cycleValue(LIGHT_SHAPES, cur, direction) }
  }

  if (seg.trackId.startsWith('obj_')) {
    // Default: motion state
    const state = (meta.state_sequence as { motion_state?: string }[])?.[0]
    const cur = state?.motion_state || 'Static'
    return { field: 'motion_state', value: cycleValue(MOTION_STATES, cur, direction) }
  }

  if (seg.trackId.startsWith('agent_')) {
    if ((meta as Record<string, unknown>)._isAgentPropertySubtrack) {
      const propType = (meta.property_type as string) || ''
      if (propType === 'Signal') {
        const cur = (meta.signaling_details as { intent?: string })?.intent || 'Stop'
        return { field: 'signal_intent', value: cycleValue(SIGNAL_INTENTS, cur, direction) }
      }
      return null
    }
    if ((meta as Record<string, unknown>)._isAgentActionSubtrack) {
      return { field: 'illegal_flag', value: !meta.illegal_flag }
    }
    // Parent agent segment — cycle amount
    const cur = (meta.amount as string) || 'Single'
    const agentType = (meta.type as string) || ''
    return { field: 'agent_amount', value: cycleValue(getAllowedAmounts(agentType), cur, direction) }
  }
  if (seg.trackId === 'ego_act' && (meta as Record<string, unknown>)._isEgoPropertySubtrack) {
    const propType = (meta.property_type as string) || ''
    if (propType === 'Signal') {
      const cur = (meta.signaling_details as { intent?: string })?.intent || 'Stop'
      return { field: 'signal_intent', value: cycleValue(SIGNAL_INTENTS, cur, direction) }
    }
    return null
  }
  return null
}

// --- Apply mutation to annotation ---

export function applyAttributeCycle(
  ann: SilAvAnnotation,
  seg: TimelineSegment,
  segments: TimelineSegment[],
  field: string,
  value: unknown
): SilAvAnnotation {
  const updated = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
  const meta = (seg.meta || {}) as Meta

  if (meta._isCondSubtrack) {
    const ci = meta._condIndex as number
    if (ci != null && ci >= 0 && ci < updated.conditions.length) {
      (updated.conditions[ci] as unknown as Record<string, unknown>)[field] = value
    }
    return updated
  }

  if (seg.trackId.startsWith('env_')) {
    const ei = meta._envIndex as number
    if (ei != null && ei >= 0 && ei < updated.environments.length) {
      (updated.environments[ei] as unknown as Record<string, unknown>)[field] = value
    }
    return updated
  }

  if (seg.trackId.startsWith('light_')) {
    const li = meta._lightIndex as number
    if (li != null && li >= 0 && li < (updated.traffic_lights || []).length) {
      (updated.traffic_lights[li] as unknown as Record<string, unknown>)[field] = value
    }
    return updated
  }

  if (meta._isEgoContSubtrack) {
    const ci = (meta._contIndex as number) ?? -1
    if (ci >= 0) {
      const cont = updated.ego_vehicle?.containment?.[ci]
      if (cont) (cont as unknown as Record<string, unknown>)[field] = value
    }
    return updated
  }

  if (meta._isAgentContSubtrack) {
    const ai = (meta._agentIndex as number) ?? -1
    const ci = (meta._contIndex as number) ?? -1
    if (ai >= 0 && ci >= 0 && ai < (updated.agents?.length || 0)) {
      const cont = updated.agents[ai]?.containment?.[ci]
      if (cont) (cont as unknown as Record<string, unknown>)[field] = value
    }
    return updated
  }

  if (seg.trackId === 'ego_act') {
    const idx = segments.filter(s => s.trackId === 'ego_act' && !(s.meta as Record<string, unknown>)?._isEgoContSubtrack).findIndex(s => s.id === seg.id)
    if (idx >= 0 && idx < updated.ego_vehicle.actions.length) {
      (updated.ego_vehicle.actions[idx] as unknown as Record<string, unknown>)[field] = value
    }
    return updated
  }

  if (meta._isObjStateSubtrack) {
    const oi = meta._objIndex as number
    const si = meta._stateIndex as number
    if (oi != null && si != null && oi >= 0 && si >= 0) {
      const st = updated.traffic_objects[oi]?.state_sequence?.[si]
      if (st) (st as unknown as Record<string, unknown>)[field] = value
    }
    return updated
  }
  if (meta._isLightStateSubtrack) {
    const li = meta._lightIndex as number
    const hi = meta._headIndex as number
    const si = meta._stateIndex as number
    if (li != null && hi != null && si != null && li >= 0 && hi >= 0 && si >= 0) {
      const st = updated.traffic_lights[li]?.signal_heads?.[hi]?.state_sequence?.[si]
      if (st) (st as unknown as Record<string, unknown>)[field] = value
    }
    return updated
  }

  if (seg.trackId.startsWith('obj_')) {
    const oi = meta._objIndex as number
    if (oi != null && oi >= 0 && oi < updated.traffic_objects.length) {
      const obj = updated.traffic_objects[oi]
      if (field === 'type') {
        obj.type = value as string
      } else if (field === 'motion_state') {
        if (!obj.state_sequence || obj.state_sequence.length === 0) {
          obj.state_sequence = [{ start_timestamp: obj.visibility_start_timestamp, end_timestamp: obj.visibility_end_timestamp }]
        }
        obj.state_sequence[0].motion_state = value as string
      }
    }
    return updated
  }

  if (meta._isEgoPropertySubtrack) {
    const pi = (meta._propIndex as number) ?? -1
    if (pi >= 0 && pi < (updated.ego_vehicle?.properties || []).length) {
      const prop = updated.ego_vehicle.properties![pi]
      if (field === 'signal_intent') {
        prop.signaling_details = prop.signaling_details || {}
        prop.signaling_details.intent = value as string
      } else {
        (prop as unknown as Record<string, unknown>)[field] = value
      }
    }
    return updated
  }

  if (seg.trackId.startsWith('agent_')) {
    const ai = meta._agentIndex as number
    if (ai == null || ai < 0 || ai >= (updated.agents?.length || 0)) return updated
    const agent = updated.agents[ai]

    if ((meta as Record<string, unknown>)._isAgentPropertySubtrack) {
      const pi = (meta._propIndex as number) ?? -1
      if (pi >= 0 && pi < (agent.properties || []).length) {
        const prop = agent.properties![pi]
        if (field === 'signal_intent') {
          prop.signaling_details = prop.signaling_details || {}
          prop.signaling_details.intent = value as string
        } else {
          (prop as unknown as Record<string, unknown>)[field] = value
        }
      }
      return updated
    }

    if ((meta as Record<string, unknown>)._isAgentActionSubtrack) {
      const actIdx = (meta._actIndex as number) ?? -1
      if (actIdx >= 0 && actIdx < agent.actions.length) {
        const act = agent.actions[actIdx]
        if (field === 'signal_intent') {
          act.signaling_details = act.signaling_details || {}
          act.signaling_details.intent = value as string
        } else {
          (act as unknown as Record<string, unknown>)[field] = value
        }
      }
    } else {
      // Parent agent segment
      if (field === 'agent_amount') (agent as unknown as Record<string, unknown>).amount = value
      else if (field === 'agent_type_enum') {
        ;(agent as unknown as Record<string, unknown>).type = value
        const allowed = getAllowedAmounts(value as string)
        if (!allowed.includes((agent as unknown as Record<string, unknown>).amount as string)) {
          ;(agent as unknown as Record<string, unknown>).amount = ''
        }
      }
      else if (field === 'agent_id') (agent as unknown as Record<string, unknown>).id = value
      else (agent as unknown as Record<string, unknown>)[field] = value
    }
    return updated
  }

  return updated
}
