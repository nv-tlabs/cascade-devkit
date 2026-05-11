import { useState } from 'react'
import { ChevronDown, ChevronRight, GripVertical } from 'lucide-react'
import { useStore } from '../lib/store'
import type { TrackId } from '../lib/types'

interface LabelCategory {
  trackId: TrackId
  name: string
  color: string
  labels: string[]
}

const LABEL_CATEGORIES: LabelCategory[] = [
  {
    trackId: 'light',
    name: 'Traffic Lights',
    color: '#ef4444',
    labels: ['TrafficLight'],
  },
  {
    trackId: 'env',
    name: 'Environments',
    color: '#22c55e',
    labels: [
      'oxd:Road', 'fst:LaneMerge', 'fst:LaneFork',
      'oxd:TIntersection', 'oxd:YIntersection', 'oxd:CrossRoad',
      '5-way', '6-way', '6+-way',
      'oxd:Roundabout', 'oxd:Tunnel', 'oxd:Bridge',
      'oxd:PavedShoulder', 'oxd:GrassShoulder',
      'oxd:Sidewalk', 'oxd:PedestrianCrossing',
      'oxd:RailCrossing', 'oxd:CycleLane',
    ],
  },
  {
    trackId: 'ego_act',
    name: 'Ego Actions',
    color: '#3b82f6',
    labels: [
      'oxd:Stop', 'oxd:NotMove', 'fst:Enter', 'fst:Exit',
      'fst:Creep', 'fst:Yield', 'oxd:Decelerate',
      'fst:DrivingInLane', 'oxd:FollowRoadUser',
      'fst:Nudge (in lane)', 'fst:Nudge (out of lane)',
      'oxd:Overtake', 'oxd:ChangeLane',
      'oxd:MakeALeftTurn', 'oxd:MakeARightTurn', 'fst:MakeAUTurn',
      'fst:Reverse',
    ],
  },
  {
    trackId: 'obj',
    name: 'Objects',
    color: '#f59e0b',
    labels: [
      'fst:StopSign', 'fst:YieldSign', 'fst:SpeedLimitSign',
      'Merge ahead', 'Adjacent lanes ahead', 'Do not enter',
      'oxd:RailCrossing', 'Garage', 'oxd:TollPlaza',
      'oxd:Roadblocks', 'oxd:TrafficCone', 'oxd:WarningSign', 'PortableDisplay',
      'Toy', 'Ball', 'Dirt', 'Trash',
    ],
  },
  {
    trackId: 'agent_pose',
    name: 'Agent Types',
    color: '#a855f7',
    labels: [
      'oxd:Car', 'oxd:Truck', 'PublicBus',
      'oxd:EmergencyVehicle', 'Heavy-duty vehicle',
      'oxd:Bicycle', 'oxd:Motorcycle', 'fst:Scooter',
      'Pedestrian (Adult)', 'Pedestrian (Kid/Teen)',
      'Pedestrian (Officer)', 'oxd:Animal',
    ],
  },
  {
    trackId: 'agent_act',
    name: 'Agent Actions',
    color: '#c084fc',
    labels: [
      'fst:Park', 'oxd:Stop', 'oxd:NotMove',
      'fst:Yield', 'oxd:Decelerate',
      'fst:DrivingInLane', 'oxd:FollowRoadUser',
      'oxd:Overtake', 'oxd:ChangeLane',
      'oxd:Walk', 'oxd:Run', 'oxd:Stand',
    ],
  },
]

const DRAG_DATA_KEY = 'application/x-silav-label'

export function LabelPalette() {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const { selectedClipId } = useStore()

  const toggleCategory = (name: string) => {
    setExpanded((prev) => ({ ...prev, [name]: !prev[name] }))
  }

  const handleDragStart = (e: React.DragEvent, trackId: TrackId, label: string) => {
    e.dataTransfer.setData(DRAG_DATA_KEY, JSON.stringify({ trackId, label }))
    e.dataTransfer.effectAllowed = 'copy'
    e.dataTransfer.setData('text/plain', label)
  }

  if (!selectedClipId) {
    return (
      <div className="flex-shrink-0 h-9 flex items-center px-3 bg-[#16162b] border-b border-[#2a2a45]">
        <span className="text-[10px] text-[#666] uppercase tracking-wider">Label palette — select a clip</span>
      </div>
    )
  }

  return (
    <div className="flex-shrink-0 min-h-[36px] bg-[#16162b] border-b border-[#2a2a45] overflow-hidden">
      <div className="flex flex-wrap items-start gap-x-4 gap-y-2 px-3 py-2">
        {LABEL_CATEGORIES.map((cat) => {
          const isExpanded = expanded[cat.name] ?? false
          return (
            <div key={cat.name} className="flex flex-col gap-1">
              <button
                type="button"
                onClick={() => toggleCategory(cat.name)}
                className="flex items-center gap-1 text-[10px] font-medium text-[#a0a0c0] hover:text-[#e0e0f0] transition-colors"
              >
                {isExpanded ? (
                  <ChevronDown className="w-3 h-3" />
                ) : (
                  <ChevronRight className="w-3 h-3" />
                )}
                <span>{cat.name}</span>
              </button>
              {isExpanded && (
                <div className="flex flex-wrap gap-1">
                  {cat.labels.map((label) => (
                    <span
                      key={label}
                      draggable
                      onDragStart={(e) => handleDragStart(e, cat.trackId, label)}
                      className="inline-flex items-center gap-0.5 px-2 py-0.5 rounded-full text-[10px] cursor-grab active:cursor-grabbing border border-transparent hover:border-current/40 transition-colors"
                      style={{
                        backgroundColor: `${cat.color}20`,
                        color: cat.color,
                      }}
                    >
                      <GripVertical className="w-2.5 h-2.5 opacity-60 flex-shrink-0" />
                      {label}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
