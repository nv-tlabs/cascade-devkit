import React, { useState, useEffect, useRef } from 'react'
import { useStore } from '../lib/store'
import { saveCurrentBundle } from '../lib/save'
import { annotationToSegments, applySegmentTimeUpdate, clampToAvoidOverlap, syncInfluencedAgentIds, addSignalHeadToLight, addPhysicalContainmentToLight, addContainmentToEgo, addContainmentToAgent, addContainmentToObject, addInfluenceToEgo, addInfluenceToAgent, addPropertyToAgent, addPropertyToEgo, addConditionToEnv, parseTs, cleanupDeletedIds, cleanupDeletedEnv } from '../lib/timeline-utils'
import {
  ENVIRONMENT_TYPES, EGO_ACTION_TYPES, TRAFFIC_OBJECT_TYPES,
  TRAFFIC_OBJECT_QUANTITIES, QUANTITY_ELIGIBLE_OBJECT_TYPES,
  POSITION_REL_TO_EGO, DIRECTION_REL_TO_EGO, SIGNAL_INTENTS,
  AGENT_TYPES, AGENT_TYPE_GROUPS, CONDITION_TYPES, LIGHT_COLORS, LIGHT_SHAPES,
  SIGN_TYPES, SIGNAL_SOURCES, LIGHT_STATE_TYPES, PROPERTY_TYPES, PROPERTY_DISPLAY_NAMES,
  ENVIRONMENT_DISPLAY_NAMES, EGO_ACTION_DISPLAY_NAMES,
  TRAFFIC_OBJECT_DISPLAY_NAMES, AGENT_TYPE_DISPLAY_NAMES,
  AGENT_ACTION_DISPLAY_NAMES, CONDITION_DISPLAY_NAMES,
  TRAFFIC_OBJECT_TYPE_GROUPS, ENVIRONMENT_TYPE_GROUPS,
  EGO_ACTION_TYPE_GROUPS,
  envDisplayName, getAllowedAmounts, getAgentActionTypes, getAgentActionTypeGroups,
  ACTION_LINK_TO_CONFIG,
} from '../lib/attribute-cycling'
import type { SilAvAnnotation, AnnotationBundle } from '../lib/types'
import { AlertTriangle, Trash2, Link2, BarChart3, FileText, Download, Upload, Save, ChevronDown, ChevronRight, X } from 'lucide-react'
import { checkSegmentCompleteness } from '../lib/completeness'

// --- Shared UI helpers ---
const inputCls = 'w-full px-3 py-2 text-xs bg-[#1a1a35] text-white rounded-lg border border-[#2a2a50] focus:border-blue-500/50 focus:outline-none'
const selectCls = inputCls
const labelCls = 'text-[10px] text-[#556] block mb-1'
const SPLIT_LANE_TYPES = new Set(['fst:LaneMerge', 'fst:LaneFork'])
const fieldCls = (highlight?: boolean) =>
  `w-full px-3 py-2 text-xs bg-[#1a1a35] text-white rounded-lg border ${highlight ? 'border-orange-500/60' : 'border-[#2a2a50]'} focus:border-blue-500/50 focus:outline-none`

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <label className="block"><span className={labelCls}>{label}</span>{children}</label>
}

import type { OptionGroupEntry } from '../lib/attribute-cycling'

function SelectField({ label, value, options, onChange, allowEmpty, displayNames, groups, highlight }: { label: string; value: string; options: readonly string[]; onChange: (v: string) => void; allowEmpty?: boolean; displayNames?: Record<string, string>; groups?: OptionGroupEntry[]; highlight?: boolean }) {
  const dn = (v: string) => displayNames?.[v] ?? v
  return (
    <Field label={label}>
      <select value={value} onChange={e => onChange(e.target.value)} className={fieldCls(highlight)}>
        {allowEmpty && <option value="">-- none --</option>}
        {groups
          ? groups.flatMap(entry => {
              if (typeof entry === 'string') return [<option key={entry} value={entry}>{dn(entry)}</option>]
              if (entry.selectable) {
                return [
                  <option key={`hdr_${entry.header}`} value={entry.header}>{dn(entry.header)}</option>,
                  ...entry.options.map(o => <option key={o} value={o}>{`\u00A0\u00A0\u00A0\u00A0${dn(o)}`}</option>),
                ]
              }
              return [
                <optgroup key={`grp_${entry.header}`} label={dn(entry.header)}>
                  {entry.options.map(o => <option key={o} value={o}>{dn(o)}</option>)}
                </optgroup>,
              ]
            })
          : options.map(o => <option key={o} value={o}>{dn(o)}</option>)
        }
      </select>
    </Field>
  )
}

function TextField({ label, value, onChange, placeholder, highlight }: { label: string; value: string; onChange: (v: string) => void; placeholder?: string; highlight?: boolean }) {
  return <Field label={label}><input type="text" value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} className={fieldCls(highlight)} /></Field>
}

function LaneNumberField({ label, value, onChange, step, highlight }: { label: string; value: number | string; onChange: (v: number | string) => void; step?: number; highlight?: boolean }) {
  return (
    <Field label={label}>
      <input
        type="number" min={1} step={step ?? 1} value={value}
        onChange={e => onChange(e.target.value === "" ? "" : Math.trunc(+e.target.value))}
        onBlur={() => (value === "" || +value < 1) && onChange(1)}
        className={fieldCls(highlight)}
      />
    </Field>
  );
}

function LaneIdField({ value, onChange, highlight }: { value: string; onChange: (v: string) => void; highlight?: boolean }) {
  const inputVal = (!value || value === 'none') ? '' : value
  return (
    <Field label="Lane Number">
      <input type="number" step={1} value={inputVal} placeholder="none"
        onChange={e => onChange(e.target.value === '' ? 'none' : String(Math.trunc(+e.target.value)))}
        className={fieldCls(highlight)} />
    </Field>
  )
}

function ToggleField({ label, value, onChange, activeColor, highlight }: { label: string; value: boolean; onChange: (v: boolean) => void; activeColor?: string; highlight?: boolean }) {
  const active = activeColor || 'bg-blue-500/20 text-blue-400 border-blue-500/40'
  const inactive = highlight ? 'bg-[#1a1a35] text-[#555] border border-orange-500/60 hover:border-orange-500/80' : 'bg-[#1a1a35] text-[#555] border border-[#2a2a50] hover:border-[#3a3a60]'
  return (
    <button type="button" onClick={() => onChange(!value)}
      className={`w-full flex items-center gap-2 px-3 py-2 mt-2 rounded-lg text-[11px] font-medium transition-all ${
        value ? active : inactive
      }`}>
      <div className={`w-3 h-3 rounded-sm border ${value ? 'bg-current border-current' : 'border-[#555]'}`} />
      {label}
    </button>
  )
}

function SectionHeader({ color, children }: { color: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 mb-3">
      <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
      <h4 className="text-sm font-bold text-white flex-1 truncate">{children}</h4>
    </div>
  )
}

function AgentIdPicker({ label, value, onChange, ann }: { label: string; value: string[]; onChange: (v: string[]) => void; ann: SilAvAnnotation | undefined }) {
  const agentMap = new Map((ann?.agents || []).filter(a => a.id).map(a => [a.id, a]))
  const options = ['Ego', ...(ann?.agents || []).map(a => a.id).filter(Boolean)]
  const nameFor = (id: string) => { const a = agentMap.get(id); return a ? `${id} (Agent${a.name ? ` · "${a.name}"` : ''})` : id }
  const add = (id: string) => { if (id && !value.includes(id)) onChange([...value, id]) }
  const remove = (id: string) => onChange(value.filter(v => v !== id))
  const available = options.filter(o => !value.includes(o))
  const hasUnknown = value.includes('Unknown')
  return (
    <div className="block">
      <span className={labelCls}>{label}</span>
      <div className="flex flex-wrap gap-1 mb-1.5 min-h-[24px]">
        {value.map(id => (
          <span key={id} className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-blue-500/20 text-blue-300 text-[10px] border border-blue-500/30">
            {nameFor(id)}
            <button type="button" onClick={() => remove(id)} className="hover:text-red-400 transition-colors"><X className="w-2.5 h-2.5" /></button>
          </span>
        ))}
        {value.length === 0 && <span className="text-[10px] text-[#444] italic">None</span>}
      </div>
      <select value="" onChange={e => { add(e.target.value); e.target.value = '' }} className={selectCls} disabled={available.length === 0 && hasUnknown}>
        <option value="">+ Add agent&hellip;</option>
        {available.map(o => <option key={o} value={o}>{nameFor(o)}</option>)}
        {!hasUnknown && <option value="Unknown">Unknown</option>}
      </select>
    </div>
  )
}

function ActionTargetPicker({ label, value, onChange, ann, entityTypes, required, highlight }: { label: string; value: string[]; onChange: (v: string[]) => void; ann: SilAvAnnotation | undefined; entityTypes: ('agent' | 'object')[]; required?: boolean; highlight?: boolean }) {
  const existing = new Set(value)
  const includeAgents = entityTypes.includes('agent')
  const includeObjects = entityTypes.includes('object')
  const agentOpts = includeAgents
    ? ['Ego', ...(ann?.agents || []).map(a => a.id).filter(Boolean)].filter(id => !existing.has(id))
    : []
  const agentMap = new Map((ann?.agents || []).filter(a => a.id).map(a => [a.id, a]))
  const objectOpts = includeObjects
    ? (ann?.traffic_objects || []).filter(to => to.id && !existing.has(to.id)).map(to => ({ id: to.id, label: `${to.id} · ${to.type || 'Object'}${to.name ? ` "${to.name}"` : ''}` }))
    : []
  const hasOptions = agentOpts.length > 0 || objectOpts.length > 0
  const hasUnknown = value.includes('Unknown')
  const add = (id: string) => { if (id && !value.includes(id)) onChange([...value, id]) }
  const remove = (id: string) => onChange(value.filter(v => v !== id))
  const nameFor = (id: string) => {
    if (id === 'Ego') return 'Ego'
    if (id === 'Unknown') return 'Unknown'
    const a = agentMap.get(id)
    if (a) return `${id} (Agent${a.name ? ` · "${a.name}"` : ''})`
    const obj = (ann?.traffic_objects || []).find(to => to.id === id)
    if (obj) return `${obj.id} · ${obj.type || 'Object'}${obj.name ? ` "${obj.name}"` : ''}`
    return id
  }
  return (
    <div className="block">
      <span className={labelCls}>
        {required ? label : (
          <>
            <span className="text-[#666]">Optional: </span>
            {label}
          </>
        )}
      </span>
      <div className="flex flex-wrap gap-1 mb-1.5 min-h-[24px]">
        {value.map(id => (
          <span key={id} className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 text-[10px] border border-emerald-500/30">
            {nameFor(id)}
            <button type="button" onClick={() => remove(id)} className="hover:text-red-400 transition-colors"><X className="w-2.5 h-2.5" /></button>
          </span>
        ))}
        {value.length === 0 && <span className="text-[10px] text-[#444] italic">None</span>}
      </div>
      <select value="" onChange={e => { add(e.target.value); e.target.value = '' }} className={fieldCls(highlight)} disabled={!hasOptions && hasUnknown}>
        <option value="">+ Add&hellip;</option>
        {includeAgents && agentOpts.length > 0 && <optgroup label="Agents">{agentOpts.map(o => <option key={o} value={o}>{nameFor(o)}</option>)}</optgroup>}
        {includeObjects && objectOpts.length > 0 && <optgroup label="Objects">{objectOpts.map(o => <option key={o.id} value={o.id}>{o.label}</option>)}</optgroup>}
        {!hasUnknown && <option value="Unknown">Unknown</option>}
      </select>
    </div>
  )
}

function InfluencerIdPicker({ label, value, onChange, ann, highlight, showAgents }: { label: string; value: string[]; onChange: (v: string[]) => void; ann: SilAvAnnotation | undefined; highlight?: boolean; showAgents?: boolean }) {
  const existing = new Set(value)
  // Objects grouped
  const objectOpts = (ann?.traffic_objects || [])
    .filter(to => to.id && !existing.has(to.id))
    .map(to => ({ id: to.id, label: `${to.id} · ${to.type || 'Object'}${to.name ? ` "${to.name}"` : ''}` }))
  // Traffic lights grouped
  const tlOpts = (ann?.traffic_lights || [])
    .filter(tl => tl.id && !existing.has(tl.id))
    .map(tl => ({ id: tl.id, label: `${tl.id}${tl.name ? ` "${tl.name}"` : ''}${tl.type ? ` · ${tl.type}` : ''}` }))
  // Agents grouped (only when enabled)
  const agentOpts = showAgents ? (ann?.agents || [])
    .filter(a => a.id && !existing.has(a.id))
    .map(a => ({ id: a.id, label: `${a.id} · ${AGENT_TYPE_DISPLAY_NAMES[a.type] ?? a.type}${a.name ? ` "${a.name}"` : ''}` })) : []
  const hasOptions = objectOpts.length > 0 || tlOpts.length > 0 || agentOpts.length > 0
  const hasUnknown = value.includes('Unknown')
  const add = (id: string) => { if (id && !value.includes(id)) onChange([...value, id]) }
  const remove = (id: string) => onChange(value.filter(v => v !== id))
  return (
    <div className="block">
      <span className={labelCls}>{label}</span>
      <div className="flex flex-wrap gap-1 mb-1.5 min-h-[24px]">
        {value.map(id => (
          <span key={id} className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 text-[10px] border border-amber-500/30">
            {id}
            <button type="button" onClick={() => remove(id)} className="hover:text-red-400 transition-colors"><X className="w-2.5 h-2.5" /></button>
          </span>
        ))}
        {value.length === 0 && <span className="text-[10px] text-[#444] italic">None</span>}
      </div>
      <select value="" onChange={e => { add(e.target.value); e.target.value = '' }} className={fieldCls(highlight)} disabled={!hasOptions && hasUnknown}>
        <option value="">+ Add entity&hellip;</option>
        {agentOpts.length > 0 && <optgroup label="Agents">{agentOpts.map(o => <option key={o.id} value={o.id}>{o.label}</option>)}</optgroup>}
        {objectOpts.length > 0 && <optgroup label="Objects">{objectOpts.map(o => <option key={o.id} value={o.id}>{o.label}</option>)}</optgroup>}
        {tlOpts.length > 0 && <optgroup label="Traffic Lights">{tlOpts.map(o => <option key={o.id} value={o.id}>{o.label}</option>)}</optgroup>}
        {!hasUnknown && <option value="Unknown">Unknown</option>}
      </select>
    </div>
  )
}

// --- Helpers ---
function unionLen(intervals: { t0: number; t1: number }[]) {
  if (!intervals.length) return 0
  const s = [...intervals].sort((a, b) => a.t0 - b.t0)
  let total = 0, end = s[0].t0
  for (const { t0, t1 } of s) { if (t1 > end) { total += t1 - Math.max(t0, end); end = t1 } }
  return total
}

/** Sub-panel for editing state_sequence of a traffic object */
// ============================================================
// Main Panel
// ============================================================

export function RightPanel() {
  const { bundle, selectedClipId, selectedPath, updateBundle, selectPath, playheadTime } = useStore()
  const locked = useStore(s => s.locked)
  const serverReadOnly = useStore(s => s.serverReadOnly)
  const dirty = useStore(s => s.dirty)
  const editsBlocked = locked
  const [briefEdit, setBriefEdit] = useState<string | null>(null)
  const [attrsOpen, setAttrsOpen] = useState(false)
  const [saving, setSaving] = useState(false)
  const [becauseOtherMode, setBecauseOtherMode] = useState(false)
  const [becauseOtherText, setBecauseOtherText] = useState('')
  const ann = bundle?.annotation
  const segments = annotationToSegments(ann)
  const sel = selectedPath ? segments.find(s => s.id === selectedPath) : null
  const comp = sel && ann ? checkSegmentCompleteness(sel, ann) : null
  const missing = (key: string) => !!(comp?.issues.some(i => i === `Missing: ${key}` || i.startsWith(`Missing: ${key} (`)))

  useEffect(() => { setBecauseOtherMode(false); setBecauseOtherText('') }, [sel?.id, selectedClipId])

  // Auto-follow playhead: when playhead leaves the selected segment, switch to
  // the segment on the same track that contains the playhead time.
  const lastManualSelRef = useRef<string | null>(null)
  useEffect(() => {
    // Track user manual selections
    lastManualSelRef.current = selectedPath
  }, [selectedPath])
  useEffect(() => {
    if (!sel || !ann) return
    // Only follow if playhead is outside the current segment
    if (playheadTime >= sel.t0 && playheadTime < sel.t1) return
    // Signal-head subtracks: each row maps to exactly one head, so there's
    // nothing meaningful to auto-switch to when the playhead leaves its window.
    if ((sel.meta as Record<string, unknown>)?._isSignalHeadSubtrack) return
    const sameTrack = segments.filter(s => s.trackId === sel.trackId)
    // Match specific subtrack kind — prevents jumping between categories
    const getSubtrackKind = (m: Record<string, unknown>): string => {
      if (m._isCondSubtrack) return 'cond'
      if (m._isObjStateSubtrack) return 'objState'
      if (m._isLightStateSubtrack) return 'lightState'
      if (m._isAgentActionSubtrack) return 'agentAction'
      if (m._isAgentPropertySubtrack) return 'agentProperty'
      if (m._isEgoPropertySubtrack) return 'egoProperty'
      if (m._isAgentPoseSubtrack) return 'agentPose'
      if (m._isEgoInfluenceSubtrack) return 'egoInfluence'
      if (m._isAgentInfluenceSubtrack) return 'agentInfluence'
      if (m._isEgoContSubtrack) return 'egoCont'
      if (m._isAgentContSubtrack) return 'agentCont'
      if (m._isObjContSubtrack) return 'objCont'
      if (m._isLightContSubtrack) return 'lightCont'
      if (m._isLightPhysContSubtrack) return 'lightPhysCont'
      return 'main'
    }
    const selMeta = sel.meta as Record<string, unknown>
    const selKind = getSubtrackKind(selMeta)
    const atTime = sameTrack.find(s => {
      if (playheadTime < s.t0 || playheadTime >= s.t1) return false
      const m = s.meta as Record<string, unknown>
      return getSubtrackKind(m) === selKind
    })
    if (atTime && atTime.id !== sel.id) {
      selectPath(atTime.id)
    }
  }, [playheadTime]) // eslint-disable-line react-hooks/exhaustive-deps
  const dur = bundle?.video?.duration_s ?? 0
  const status = bundle?.status ?? 'pending'

  const egoActCov = dur > 0 ? (unionLen(segments.filter(s => s.trackId === 'ego_act')) / dur) * 100 : 0
  const egoContCov = dur > 0 ? (unionLen(segments.filter(s => !!(s.meta as Record<string, unknown>)?._isEgoContSubtrack)) / dur) * 100 : 0

  const causal: { id: string; label: string; cause: string; track: string }[] = []
  for (const s of segments) {
    if ((s.trackId === 'ego_act' || s.trackId.startsWith('agent_')) && s.because_of?.length) {
      for (const c of s.because_of) causal.push({ id: s.id, label: s.label, cause: c, track: s.trackId })
    }
  }

  // ---- Generic persist helper ----
  // Mutates the in-memory bundle and flags it dirty via updateBundle().
  // The actual write to disk happens via the explicit Save button (Step 6).
  const persist = async (updated: SilAvAnnotation) => {
    if (!bundle || !selectedClipId || useStore.getState().editsBlocked()) return
    const nb = { ...bundle, annotation: updated }
    updateBundle(nb)
  }

  // ---- Guarded save ----
  // Step 4 stub: edits while unlocked update the in-memory bundle (and flip
  // dirty) but do not auto-save. Step 6 wires the explicit Save button.
  const guardedSave = async (_clipId: string, data: unknown): Promise<AnnotationBundle> => {
    return data as AnnotationBundle
  }

  // ---- Generic field updater (mutates a clone, then persists) ----
  const updateField = (field: string, value: unknown) => {
    if (!sel || !ann || editsBlocked) return
    const u = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
    const m = sel.meta as Record<string, unknown>

    const set = (item: Record<string, unknown>) => {
      if (field === 'label') {
        if ('type' in item) item.type = value
        else if ('action_type' in item) item.action_type = value
        else if ('position_rel_to_ego' in item) item.position_rel_to_ego = value
      } else if (field === 'conditions') {
        item.conditions = Array.isArray(value) ? value : String(value).split(',').map(s => s.trim()).filter(Boolean)
      } else if (field === 'color' || field === 'shape') {
        item[field] = value
        const otherField = field === 'color' ? 'shape' : 'color'
        if (value !== 'Other' && item[otherField] !== 'Other' && 'other_condition_description' in item) {
          item.other_condition_description = ''
        }
        if (field === 'color' && value !== 'Yellow') {
          item.yellow_on_ego_path = null
          item.ego_in_intersection_on_yellow = null
          item.ego_could_have_cleared_safely = null
        }
      } else if (field === 'env_id_or_other') {
        if (value === '__other__') { item.env_id = 'Other'; item.other = '' }
        else { item.env_id = value; item.other = null }
      } else if (field === 'type') {
        item.type = value
        const hasOther = Array.isArray(value) ? value.some((v: string) => v.startsWith('Other')) : String(value).startsWith('Other')
        if (!hasOther) {
          if ('type_other_description' in item) item.type_other_description = ''
          if ('other_type_description' in item) item.other_type_description = ''
          if ('other_condition_description' in item) item.other_condition_description = ''
          if ('condition_other_description' in item) item.condition_other_description = ''
        }
        if ('quantity' in item && !QUANTITY_ELIGIBLE_OBJECT_TYPES.has(String(value))) {
          item.quantity = undefined
        }
      } else {
        item[field] = value
      }
    }

    if ((m as Record<string, unknown>)._isEgoInfluenceSubtrack) {
      const inflIdx = (m._inflIndex as number) ?? -1
      if (inflIdx >= 0) {
        const infl = u.ego_vehicle?.influenced_by?.[inflIdx]
        if (infl) set(infl as unknown as Record<string, unknown>)
      }
    } else if ((m as Record<string, unknown>)._isAgentInfluenceSubtrack) {
      const ai = (m._agentIndex as number) ?? -1
      const inflIdx = (m._inflIndex as number) ?? -1
      if (ai >= 0 && inflIdx >= 0 && ai < (u.agents?.length || 0)) {
        const infl = u.agents[ai]?.influenced_by?.[inflIdx]
        if (infl) set(infl as unknown as Record<string, unknown>)
      }
    } else if ((m as Record<string, unknown>)._isEgoContSubtrack) {
      const ci = (m._contIndex as number) ?? -1
      if (ci >= 0) {
        const cont = u.ego_vehicle?.containment?.[ci]
        if (cont) set(cont as unknown as Record<string, unknown>)
      }
    } else if ((m as Record<string, unknown>)._isAgentContSubtrack) {
      const ai = (m._agentIndex as number) ?? -1
      const ci = (m._contIndex as number) ?? -1
      if (ai >= 0 && ci >= 0 && ai < (u.agents?.length || 0)) {
        const cont = u.agents[ai]?.containment?.[ci]
        if (cont) set(cont as unknown as Record<string, unknown>)
      }
    } else if ((m as Record<string, unknown>)._isCondSubtrack) {
      const ci = (m._condIndex as number) ?? -1
      if (ci >= 0) set(u.conditions[ci] as unknown as Record<string, unknown>)
    } else if ((m as Record<string, unknown>)._isObjStateSubtrack) {
      const oi = (m._objIndex as number) ?? -1
      const si = (m._stateIndex as number) ?? -1
      if (oi >= 0 && si >= 0) set(u.traffic_objects[oi].state_sequence[si] as unknown as Record<string, unknown>)
    } else if ((m as Record<string, unknown>)._isSignalHeadSubtrack) {
      const li = (m._lightIndex as number) ?? -1
      const hi = (m._headIndex as number) ?? -1
      if (li >= 0 && hi >= 0) set(u.traffic_lights[li].signal_heads[hi] as unknown as Record<string, unknown>)
    } else if ((m as Record<string, unknown>)._isLightStateSubtrack) {
      const li = (m._lightIndex as number) ?? -1
      const hi = (m._headIndex as number) ?? -1
      const si = (m._stateIndex as number) ?? -1
      if (li >= 0 && hi >= 0 && si >= 0) set(u.traffic_lights[li].signal_heads[hi].state_sequence[si] as unknown as Record<string, unknown>)
    } else if ((m as Record<string, unknown>)._isObjContSubtrack) {
      const oi = (m._objIndex as number) ?? -1
      const ci = (m._contIndex as number) ?? -1
      if (oi >= 0 && ci >= 0 && oi < (u.traffic_objects?.length || 0)) {
        const cont = u.traffic_objects[oi]?.containment?.[ci]
        if (cont) set(cont as unknown as Record<string, unknown>)
      }
    } else if ((m as Record<string, unknown>)._isLightPhysContSubtrack) {
      const li = (m._lightIndex as number) ?? -1
      const ci = (m._contIndex as number) ?? -1
      if (li >= 0 && ci >= 0 && li < (u.traffic_lights?.length || 0)) {
        const cont = u.traffic_lights[li]?.containment?.[ci]
        if (cont) set(cont as unknown as Record<string, unknown>)
      }
    } else if ((m as Record<string, unknown>)._isLightContSubtrack) {
      const li = (m._lightIndex as number) ?? -1
      const hi = (m._headIndex as number) ?? -1
      const ci = (m._contIndex as number) ?? -1
      if (li >= 0 && hi >= 0 && ci >= 0 && li < (u.traffic_lights?.length || 0)) {
        const cont = u.traffic_lights[li]?.signal_heads?.[hi]?.env_controlled?.[ci]
        if (cont) set(cont as unknown as Record<string, unknown>)
      }
    } else if (sel.trackId.startsWith('env_')) {
      const ei = (m._envIndex as number) ?? -1
      if (ei >= 0) set(u.environments[ei] as unknown as Record<string, unknown>)
    } else if (sel.trackId.startsWith('light_')) {
      const li = (m._lightIndex as number) ?? -1
      if (li >= 0) set(u.traffic_lights[li] as unknown as Record<string, unknown>)
    } else if ((m as Record<string, unknown>)._isEgoPropertySubtrack) {
      const pi = (m._propIndex as number) ?? -1
      if (pi >= 0 && pi < (u.ego_vehicle?.properties || []).length) {
        const prop = u.ego_vehicle.properties![pi]
        if (field === 'signal_source') { prop.signaling_details = prop.signaling_details || {}; prop.signaling_details.source = value as string; if (value !== 'Other') prop.signaling_details.other_source_description = ''; if (value !== 'Holding sign') { prop.signaling_details.sign_type = ''; prop.signaling_details.other_sign_description = ''; prop.signaling_details.not_facing_ego = false } }
        else if (field === 'signal_other_source') { prop.signaling_details = prop.signaling_details || {}; prop.signaling_details.other_source_description = value as string }
        else if (field === 'signal_intent') { prop.signaling_details = prop.signaling_details || {}; prop.signaling_details.intent = value as string; if (value !== 'Other') prop.signaling_details.other_intent_description = '' }
        else if (field === 'signal_other_intent') { prop.signaling_details = prop.signaling_details || {}; prop.signaling_details.other_intent_description = value as string }
        else if (field === 'signal_sign_type') { prop.signaling_details = prop.signaling_details || {}; prop.signaling_details.sign_type = value as string; if (value !== 'Other') prop.signaling_details.other_sign_description = '' }
        else if (field === 'signal_other_sign') { prop.signaling_details = prop.signaling_details || {}; prop.signaling_details.other_sign_description = value as string }
        else if (field === 'signal_not_facing_ego') { prop.signaling_details = prop.signaling_details || {}; prop.signaling_details.not_facing_ego = value as boolean }
        else if (field === 'signal_target_ids') {
          prop.signaling_details = prop.signaling_details || {}
          const ids = Array.isArray(value) ? value as string[] : (value as string).split(',').map(s => s.trim()).filter(Boolean)
          prop.signaling_details.link_to = ids
        }
        else if (field === 'property_type') {
          prop.property_type = value as string
          if (value !== 'Other') prop.other_description = ''
          if (value !== 'Signal') prop.signaling_details = undefined
        }
        else set(prop as unknown as Record<string, unknown>)
      }
    } else if (sel.trackId === 'ego_act') {
      const eai = (m._egoActIndex as number) ?? -1
      if (eai >= 0 && eai < (u.ego_vehicle?.actions || []).length) {
        set(u.ego_vehicle.actions[eai] as unknown as Record<string, unknown>)
        // Clear description fields when switching away from Other / Other turn
        if (field === 'type') {
          const act = u.ego_vehicle.actions[eai]
          if (!act.type.startsWith('Other oxd:MakeATurn')) act.turn_other_description = ''
          if (act.type !== 'Other') act.action_other_description = ''
          // Clear action_target when switching to a type that doesn't have one
          if (!ACTION_LINK_TO_CONFIG[act.type]) act.action_target = undefined
        }
      }
    } else if (sel.trackId.startsWith('obj_')) {
      const oi = (m._objIndex as number) ?? -1
      const objKind = m._objKind as string
      if (oi >= 0) {
        if (objKind === 'traffic_light') {
          set(u.traffic_lights[oi] as unknown as Record<string, unknown>)
        } else {
          set(u.traffic_objects[oi] as unknown as Record<string, unknown>)
        }
      }
    } else if ((m as Record<string, unknown>)._isAgentPoseSubtrack) {
      const ai = (m._agentIndex as number) ?? -1
      const pi = (m._poseIndex as number) ?? -1
      if (ai >= 0 && pi >= 0 && ai < u.agents.length && pi < (u.agents[ai].ego_relative_pose || []).length) {
        set(u.agents[ai].ego_relative_pose![pi] as unknown as Record<string, unknown>)
      }
    } else if ((m as Record<string, unknown>)._isAgentPropertySubtrack) {
      const ai = (m._agentIndex as number) ?? -1
      const pi = (m._propIndex as number) ?? -1
      if (ai >= 0 && pi >= 0 && ai < u.agents.length && pi < (u.agents[ai].properties || []).length) {
        const prop = u.agents[ai].properties![pi]
        if (field === 'signal_source') { prop.signaling_details = prop.signaling_details || {}; prop.signaling_details.source = value as string; if (value !== 'Other') prop.signaling_details.other_source_description = ''; if (value !== 'Holding sign') { prop.signaling_details.sign_type = ''; prop.signaling_details.other_sign_description = ''; prop.signaling_details.not_facing_ego = false } }
        else if (field === 'signal_other_source') { prop.signaling_details = prop.signaling_details || {}; prop.signaling_details.other_source_description = value as string }
        else if (field === 'signal_intent') { prop.signaling_details = prop.signaling_details || {}; prop.signaling_details.intent = value as string; if (value !== 'Other') prop.signaling_details.other_intent_description = '' }
        else if (field === 'signal_other_intent') { prop.signaling_details = prop.signaling_details || {}; prop.signaling_details.other_intent_description = value as string }
        else if (field === 'signal_sign_type') { prop.signaling_details = prop.signaling_details || {}; prop.signaling_details.sign_type = value as string; if (value !== 'Other') prop.signaling_details.other_sign_description = '' }
        else if (field === 'signal_other_sign') { prop.signaling_details = prop.signaling_details || {}; prop.signaling_details.other_sign_description = value as string }
        else if (field === 'signal_not_facing_ego') { prop.signaling_details = prop.signaling_details || {}; prop.signaling_details.not_facing_ego = value as boolean }
        else if (field === 'signal_target_ids') {
          prop.signaling_details = prop.signaling_details || {}
          const ids = Array.isArray(value) ? value as string[] : (value as string).split(',').map(s => s.trim()).filter(Boolean)
          prop.signaling_details.link_to = ids
        }
        else if (field === 'property_type') {
          prop.property_type = value as string
          if (value !== 'Other') prop.other_description = ''
          if (value !== 'Signal') prop.signaling_details = undefined
        }
        else set(prop as unknown as Record<string, unknown>)
      }
    } else if (sel.trackId.startsWith('agent_')) {
      const ai = (m._agentIndex as number) ?? -1
      if (ai >= 0 && ai < u.agents.length) {
        // Fields on the parent agent (amount, type, id)
        if (field === 'agent_amount' || field === 'agent_type_enum' || field === 'agent_id' || field === 'agent_other_type_description') {
          const agent = u.agents[ai] as unknown as Record<string, unknown>
          if (field === 'agent_amount') agent.amount = value
          else if (field === 'agent_type_enum') {
            agent.type = value
            agent.other_type_description = ''
            // Reset amount if not allowed for new type
            const allowed = getAllowedAmounts(value as string)
            if (!allowed.includes(agent.amount as string)) agent.amount = ''
          }
          else if (field === 'agent_id') agent.id = value
          else if (field === 'agent_other_type_description') agent.other_type_description = value
        } else if (m._isAgentActionSubtrack) {
          const actIdx = (m._actIndex as number) ?? -1
          if (actIdx >= 0) {
            if (field === 'signal_source') {
              const act = u.agents[ai].actions[actIdx]
              act.signaling_details = act.signaling_details || {}
              act.signaling_details.source = value as string
              if (value !== 'Holding sign') { act.signaling_details.sign_type = ''; act.signaling_details.other_sign_description = ''; act.signaling_details.not_facing_ego = false }
            } else if (field === 'signal_intent') {
              const act = u.agents[ai].actions[actIdx]
              act.signaling_details = act.signaling_details || {}
              act.signaling_details.intent = value as string
            } else if (field === 'signal_target_ids') {
              const act = u.agents[ai].actions[actIdx]
              act.signaling_details = act.signaling_details || {}
              const ids = Array.isArray(value) ? value as string[] : (value as string).split(',').map(s => s.trim()).filter(Boolean)
              act.signaling_details.link_to = ids
              act.link_to = ids
            } else if (field === 'signal_sign_type') {
              const act = u.agents[ai].actions[actIdx]
              act.signaling_details = act.signaling_details || {}
              act.signaling_details.sign_type = value as string
            } else if (field === 'signal_not_facing_ego') {
              const act = u.agents[ai].actions[actIdx]
              act.signaling_details = act.signaling_details || {}
              act.signaling_details.not_facing_ego = value as boolean
            } else {
              set(u.agents[ai].actions[actIdx] as unknown as Record<string, unknown>)
              // Sync link_to ↔ signaling_details.link_to
              if (field === 'link_to') {
                const act = u.agents[ai].actions[actIdx]
                if (act.signaling_details) act.signaling_details.link_to = act.link_to || []
              }
              // Clear other_description when switching away from Other / Other turn
              if (field === 'action_type') {
                const act = u.agents[ai].actions[actIdx]
                if (act.action_type !== 'Other' && act.action_type !== 'Other oxd:MakeATurn') {
                  act.other_description = ''
                }
                // Clear action_target when switching to a type that doesn't have one
                if (!ACTION_LINK_TO_CONFIG[act.action_type]) act.action_target = undefined
              }
            }
          }
        } else {
          // Parent agent segment — set fields directly on agent
          set(u.agents[ai] as unknown as Record<string, unknown>)
        }
      }
    }
    persist(u)
  }

  // ---- Handlers ----
  // The single save entry-point. saveCurrentBundle() lives in lib/save.ts so
  // the Cmd+S keybinding and the Sidebar clip-switch dialog hit identical
  // dirty-clearing, error-surfacing, and lock-policy logic.
  const save = async () => {
    setSaving(true)
    await saveCurrentBundle()
    setSaving(false)
  }
  const loadJsonRef = useRef<HTMLInputElement>(null)
  const loadJson = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !bundle || ann?.eventful === false) return
    const reader = new FileReader()
    reader.onload = () => {
      try {
        const raw = JSON.parse(reader.result as string)
        const data: SilAvAnnotation = raw.annotation && (raw.annotation.ego_vehicle || raw.annotation.environments || raw.annotation.agents)
          ? raw.annotation
          : raw
        if (!data.ego_vehicle && !data.environments && !data.agents) {
          alert('Invalid annotation JSON: missing expected fields.')
          return
        }
        updateBundle({ ...bundle, annotation: data })
      } catch {
        alert('Failed to parse JSON file.')
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }
  const handleBriefBlur = async () => {
    if (briefEdit === null || !ann || !selectedClipId || !bundle || editsBlocked) return
    const nb = { ...bundle, annotation: { ...ann, brief_description: briefEdit } }
    try { updateBundle(await guardedSave(selectedClipId, nb)) } catch {}
    setBriefEdit(null)
  }

  const handleTimeChange = async (t0: number, t1: number) => {
    if (!sel || !ann || !selectedClipId || !bundle || t0 >= t1 || editsBlocked) return
    // Determine resize mode based on which edge changed
    const mode = Math.abs(t0 - sel.t0) > 0.001 ? 'left' as const : 'right' as const
    const selIsCond = !!(sel.meta as Record<string, unknown>)?._isCondSubtrack
    const selIsObjState = !!(sel.meta as Record<string, unknown>)?._isObjStateSubtrack
    const selIsLightState = !!(sel.meta as Record<string, unknown>)?._isLightStateSubtrack
    const selIsAgentAction = !!(sel.meta as Record<string, unknown>)?._isAgentActionSubtrack
    const selIsAgentPose = !!(sel.meta as Record<string, unknown>)?._isAgentPoseSubtrack
    const selIsEgoCont = !!(sel.meta as Record<string, unknown>)?._isEgoContSubtrack
    const selIsAgentCont = !!(sel.meta as Record<string, unknown>)?._isAgentContSubtrack
    const isSubtrack: boolean | 'agent_action' | 'agent_pose' | 'agent_cont' = selIsAgentAction ? 'agent_action'
      : selIsAgentPose ? 'agent_pose'
      : selIsAgentCont ? 'agent_cont'
      : (selIsCond || selIsObjState || selIsLightState || selIsEgoCont)
    const clamped = clampToAvoidOverlap(segments, sel.trackId, sel.id, t0, t1, mode, isSubtrack)
    if (!clamped) return
    const u = applySegmentTimeUpdate(ann, sel.id, clamped.t0, clamped.t1)
    const nb = { ...bundle, annotation: u }
    updateBundle(nb)
    try { await guardedSave(selectedClipId, nb) } catch {}
  }

  const handleDelete = async () => {
    if (!sel || !ann || !selectedClipId || !bundle || editsBlocked) return
    const u = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
    const m = sel.meta as Record<string, unknown>
    if ((m as Record<string, unknown>)._isCondSubtrack) {
      const ci = (m._condIndex as number) ?? -1
      if (ci >= 0) {
        const condId = u.conditions[ci]?.id
        if (condId) cleanupDeletedIds(u, [condId])
        u.conditions.splice(ci, 1)
      }
    } else if ((m as Record<string, unknown>)._isObjContSubtrack) {
      const oi = (m._objIndex as number) ?? -1
      const ci = (m._contIndex as number) ?? -1
      if (oi >= 0 && ci >= 0) u.traffic_objects[oi]?.containment?.splice(ci, 1)
    } else if ((m as Record<string, unknown>)._isLightContSubtrack) {
      const li = (m._lightIndex as number) ?? -1
      const hi = (m._headIndex as number) ?? -1
      const ci = (m._contIndex as number) ?? -1
      if (li >= 0 && hi >= 0 && ci >= 0) u.traffic_lights[li]?.signal_heads?.[hi]?.env_controlled?.splice(ci, 1)
    } else if ((m as Record<string, unknown>)._isObjStateSubtrack) {
      const oi = (m._objIndex as number) ?? -1
      const si = (m._stateIndex as number) ?? -1
      if (oi >= 0 && si >= 0) {
        const stateId = u.traffic_objects[oi].state_sequence[si]?.id
        if (stateId) cleanupDeletedIds(u, [stateId])
        u.traffic_objects[oi].state_sequence.splice(si, 1)
      }
    } else if ((m as Record<string, unknown>)._isSignalHeadSubtrack) {
      const li = (m._lightIndex as number) ?? -1
      const hi = (m._headIndex as number) ?? -1
      if (li >= 0 && hi >= 0) {
        const head = u.traffic_lights[li].signal_heads[hi]
        const headIds = [head.id, ...(head.state_sequence.map(s => s.id).filter(Boolean) as string[])]
        cleanupDeletedIds(u, headIds)
        u.traffic_lights[li].signal_heads.splice(hi, 1)
      }
    } else if ((m as Record<string, unknown>)._isLightStateSubtrack) {
      const li = (m._lightIndex as number) ?? -1
      const hi = (m._headIndex as number) ?? -1
      const si = (m._stateIndex as number) ?? -1
      if (li >= 0 && hi >= 0 && si >= 0) {
        const stateId = u.traffic_lights[li].signal_heads[hi].state_sequence[si]?.id
        if (stateId) cleanupDeletedIds(u, [stateId])
        u.traffic_lights[li].signal_heads[hi].state_sequence.splice(si, 1)
      }
    } else if (sel.trackId.startsWith('light_')) {
      const li = (m._lightIndex as number) ?? -1
      if (li >= 0) {
        const light = u.traffic_lights[li]
        const lightIds = [light.id, ...light.signal_heads.flatMap(h => [h.id, ...(h.state_sequence.map(s => s.id).filter(Boolean) as string[])])]
        cleanupDeletedIds(u, lightIds)
        u.traffic_lights.splice(li, 1)
      }
    } else if (sel.trackId.startsWith('obj_')) {
      const oi = (m._objIndex as number) ?? -1
      if (oi >= 0) {
        const obj = u.traffic_objects[oi]
        const objIds = [obj.id, ...(obj.state_sequence.map(s => s.id).filter(Boolean) as string[])]
        cleanupDeletedIds(u, objIds)
        u.traffic_objects.splice(oi, 1)
      }
    } else if (m._isEgoInfluenceSubtrack) {
      const inflIdx = (m._inflIndex as number) ?? -1
      if (inflIdx >= 0) u.ego_vehicle?.influenced_by?.splice(inflIdx, 1)
    } else if (m._isAgentInfluenceSubtrack) {
      const ai = (m._agentIndex as number) ?? -1
      const inflIdx = (m._inflIndex as number) ?? -1
      if (ai >= 0 && inflIdx >= 0) u.agents[ai]?.influenced_by?.splice(inflIdx, 1)
    } else if (m._isEgoContSubtrack) {
      const ci = (m._contIndex as number) ?? -1
      if (ci >= 0) u.ego_vehicle?.containment?.splice(ci, 1)
    } else if (m._isAgentContSubtrack) {
      const ai = (m._agentIndex as number) ?? -1
      const ci = (m._contIndex as number) ?? -1
      if (ai >= 0 && ci >= 0 && ai < u.agents.length) {
        u.agents[ai].containment?.splice(ci, 1)
      }
    } else if ((m as Record<string, unknown>)._isAgentPoseSubtrack) {
      const ai = (m._agentIndex as number) ?? -1
      const pi = (m._poseIndex as number) ?? -1
      if (ai >= 0 && pi >= 0 && ai < u.agents.length) {
        u.agents[ai].ego_relative_pose?.splice(pi, 1)
      }
    } else if (sel.trackId.startsWith('agent_')) {
      const ai = (m._agentIndex as number) ?? -1
      if (ai >= 0 && ai < u.agents.length) {
        if (m._isAgentActionSubtrack) {
          const actIdx = (m._actIndex as number) ?? -1
          if (actIdx >= 0) {
            const actionId = u.agents[ai].actions[actIdx]?.id
            if (actionId) cleanupDeletedIds(u, [actionId])
            u.agents[ai].actions.splice(actIdx, 1)
          }
        } else {
          const agent = u.agents[ai]
          const agentIds = [agent.id, ...(agent.actions.map(a => a.id).filter(Boolean) as string[])]
          cleanupDeletedIds(u, agentIds)
          u.agents.splice(ai, 1)
        }
      }
    } else if (sel.trackId.startsWith('env_')) {
      const ei = (m._envIndex as number) ?? -1
      if (ei >= 0) {
        const envId = u.environments[ei]?.id
        u.environments.splice(ei, 1)
        if (envId) cleanupDeletedEnv(u, envId)
      }
    } else if (sel.trackId === 'ego_act') {
      const eai = (m._egoActIndex as number) ?? -1
      if (eai >= 0 && eai < (u.ego_vehicle?.actions || []).length) u.ego_vehicle.actions.splice(eai, 1)
    }
    const nb = { ...bundle, annotation: u }
    try { updateBundle(await guardedSave(selectedClipId, nb)); selectPath(null) } catch { updateBundle(nb); selectPath(null) }
  }

  const handleToggleIllegal = () => {
    if (!sel || !ann || editsBlocked) return
    const u = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
    const newVal = !sel.illegal
    const m = sel.meta as Record<string, unknown>
    if (m._isEgoContSubtrack) {
      const ci = (m._contIndex as number) ?? -1
      if (ci >= 0) { const cont = u.ego_vehicle?.containment?.[ci]; if (cont) cont.illegal_flag = newVal }
    } else if (m._isAgentContSubtrack) {
      const ai = (m._agentIndex as number) ?? -1
      const ci = (m._contIndex as number) ?? -1
      if (ai >= 0 && ci >= 0 && ai < u.agents.length) {
        const cont = u.agents[ai]?.containment?.[ci]
        if (cont) cont.illegal_flag = newVal
      }
    } else if (m._isObjContSubtrack) {
      const oi = (m._objIndex as number) ?? -1; const ci = (m._contIndex as number) ?? -1
      if (oi >= 0 && ci >= 0) { const c = u.traffic_objects[oi]?.containment?.[ci]; if (c) c.illegal_flag = newVal }
    } else if (sel.trackId === 'ego_act') {
      const eai = (m._egoActIndex as number) ?? -1
      if (eai >= 0 && eai < (u.ego_vehicle?.actions || []).length) u.ego_vehicle.actions[eai].illegal_flag = newVal
    } else if (sel.trackId.startsWith('agent_') && m._isAgentActionSubtrack) {
      const ai = (m._agentIndex as number) ?? -1
      const actIdx = (m._actIndex as number) ?? -1
      if (ai >= 0 && actIdx >= 0 && ai < u.agents.length) u.agents[ai].actions[actIdx].illegal_flag = newVal
    }
    persist(u)
  }

  const handleAddBecauseById = async (causeId: string) => {
    if (!causeId || !sel || !ann || !selectedClipId || !bundle || editsBlocked) return
    const isEgoAct = sel.trackId === 'ego_act'
    const isAgentAct = sel.trackId.startsWith('agent_') && !!(sel.meta as Record<string, unknown>)?._isAgentActionSubtrack
    if (!isEgoAct && !isAgentAct) return
    const u = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
    if (isEgoAct) {
      const eai = ((sel.meta as Record<string, unknown>)?._egoActIndex as number) ?? -1
      if (eai >= 0 && eai < (u.ego_vehicle?.actions || []).length) {
        u.ego_vehicle.actions[eai].because_of = [...(u.ego_vehicle.actions[eai].because_of || []), causeId]
      }
    } else {
      const ai = ((sel.meta as Record<string, unknown>)?._agentIndex as number) ?? -1
      const actIdx = ((sel.meta as Record<string, unknown>)?._actIndex as number) ?? -1
      if (ai >= 0 && actIdx >= 0 && ai < u.agents.length) {
        u.agents[ai].actions[actIdx].because_of = [...(u.agents[ai].actions[actIdx].because_of || []), causeId]
      }
    }
    syncInfluencedAgentIds(u)
    const nb = { ...bundle, annotation: u }
    try { updateBundle(await guardedSave(selectedClipId, nb)) } catch { updateBundle(nb) }
  }

  const handleRemoveBecause = async (causeId: string) => {
    if (!sel || !ann || !selectedClipId || !bundle || editsBlocked) return
    const isEgoAct = sel.trackId === 'ego_act'
    const isAgentAct = sel.trackId.startsWith('agent_') && !!(sel.meta as Record<string, unknown>)?._isAgentActionSubtrack
    if (!isEgoAct && !isAgentAct) return
    const u = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
    if (isEgoAct) {
      const eai = ((sel.meta as Record<string, unknown>)?._egoActIndex as number) ?? -1
      if (eai >= 0 && eai < (u.ego_vehicle?.actions || []).length) {
        const arr = u.ego_vehicle.actions[eai].because_of
        if (arr) {
          const i = arr.indexOf(causeId)
          if (i >= 0) arr.splice(i, 1)
        }
      }
    } else {
      const ai = ((sel.meta as Record<string, unknown>)?._agentIndex as number) ?? -1
      const actIdx = ((sel.meta as Record<string, unknown>)?._actIndex as number) ?? -1
      if (ai >= 0 && actIdx >= 0 && ai < u.agents.length) {
        const arr = u.agents[ai].actions[actIdx].because_of
        if (arr) {
          const i = arr.indexOf(causeId)
          if (i >= 0) arr.splice(i, 1)
        }
      }
    }
    syncInfluencedAgentIds(u)
    const nb = { ...bundle, annotation: u }
    try { updateBundle(await guardedSave(selectedClipId, nb)) } catch { updateBundle(nb) }
  }

  const handleExport = () => {
    if (!selectedClipId || !ann) return
    const exportData = { eventful: ann.eventful ?? null, brief_description: ann.brief_description || '', environments: ann.environments || [], conditions: ann.conditions || [], traffic_objects: ann.traffic_objects || [], traffic_lights: ann.traffic_lights || [], ego_vehicle: ann.ego_vehicle || { actions: [] }, agents: ann.agents || [] }
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `${selectedClipId}_sil_av.json`; a.click()
  }

  const handleClear = async () => {
    if (!selectedClipId || !bundle || editsBlocked) return
    if (!window.confirm('Clear ALL annotations for this clip?')) return
    const empty: SilAvAnnotation = { eventful: null, brief_description: '', environments: [], conditions: [], traffic_objects: [], traffic_lights: [], ego_vehicle: { actions: [] }, agents: [] }
    const cleared = { ...bundle, annotation: empty, status: 'pending' }
    // Optimistically update the UI immediately
    updateBundle(cleared)
    selectPath(null)
    try {
      const saved = await guardedSave(selectedClipId, cleared)
      updateBundle(saved)
    } catch (e) {
      console.error('Failed to clear annotations:', e)
    }
  }


  const statusColor = status === 'approved' ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' : status === 'disapproved' ? 'bg-red-500/20 text-red-400 border-red-500/30' : status === 'needs_revision' ? 'bg-orange-500/20 text-orange-400 border-orange-500/30' : status === 'annotating' ? 'bg-blue-500/20 text-blue-400 border-blue-500/30' : 'bg-zinc-500/20 text-zinc-400 border-zinc-500/30'

  // Determine entity info for the selected segment
  const selMeta = (sel?.meta ?? {}) as Record<string, unknown>
  const canHaveIllegal = sel && !selMeta._isEgoPropertySubtrack && !selMeta._isAgentPropertySubtrack && (sel.trackId === 'ego_act' || selMeta._isEgoContSubtrack || selMeta._isAgentContSubtrack || selMeta._isObjContSubtrack || (sel.trackId.startsWith('agent_') && selMeta._segType === 'action'))
  const canHaveBecause = sel && !selMeta._isEgoContSubtrack && !selMeta._isAgentContSubtrack && !selMeta._isEgoInfluenceSubtrack && !selMeta._isAgentInfluenceSubtrack && !selMeta._isEgoPropertySubtrack && !selMeta._isAgentPropertySubtrack && (sel.trackId === 'ego_act' || (sel.trackId.startsWith('agent_') && selMeta._segType === 'action'))
  // ---- Render property editors per entity type ----
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function renderProperties(): any {
    // Ego main bar (not in annotationToSegments, handled specially)
    if (selectedPath === 'ego_main' && ann) {
      const egoDur = bundle?.video?.duration_s || 10
      return (
        <>
          <SectionHeader color="#3b82f6">Ego</SectionHeader>
          <button
            className="mt-2 w-full px-3 py-1.5 rounded text-xs font-medium bg-cyan-700 hover:bg-cyan-600 text-white transition-colors"
            onClick={() => persist(addContainmentToEgo(ann, 0, egoDur))}
          >+ Add Containment</button>
          <button
            className="mt-1 w-full px-3 py-1.5 rounded text-xs font-medium bg-amber-800 hover:bg-amber-700 text-white transition-colors"
            onClick={() => persist(addInfluenceToEgo(ann, 0, egoDur))}
          >+ Add Influence</button>
          <button
            className="mt-1 w-full px-3 py-1.5 rounded text-xs font-medium bg-fuchsia-800 hover:bg-fuchsia-700 text-white transition-colors"
            onClick={() => persist(addPropertyToEgo(ann, 0, egoDur))}
          >+ Add Property</button>
        </>
      )
    }
    if (!sel) return null
    const m = selMeta
    const envOptions = (ann?.environments || []).filter(e => e.id)

    const renderEnvSelect = (meta: Record<string, unknown>, label = 'Contained in Environment') => (
      <>
        <label className="block">
          <span className={labelCls}>{label}</span>
          <select value={meta.other != null || meta.env_id === 'Other' ? '__other__' : String(meta.env_id ?? '')} onChange={e => {
            updateField('env_id_or_other', e.target.value)
          }} className={fieldCls(missing('env_id'))}>
            <option value="">-- Select environment --</option>
            {envOptions.map(e => (
              <option key={e.id} value={e.id}>{e.id} · {envDisplayName(e.type).replace(' (specify)', '')}{e.name ? ` ("${e.name}")` : ''}</option>
            ))}
            <option value="__other__">Other (specify)</option>
          </select>
        </label>
        {meta.env_id === 'Other' && <TextField label="Other description" value={(meta.other as string) || ''} onChange={v => updateField('other', v || '')} highlight={missing('other environment description')} />}
      </>
    )

    const renderSpatialContainmentFields = (meta: Record<string, unknown>) => {
      const edge = meta.edge as 'left' | 'right' | null | undefined
      return (
        <>
          <ToggleField label="Near (not fully inside)" value={!!meta.near_flag} onChange={v => updateField('near_flag', v)} activeColor="bg-amber-500/20 text-amber-400 border border-amber-500/40" />
          <ToggleField label="Left edge" value={edge === 'left'} onChange={v => updateField('edge', v ? 'left' : null)} activeColor="bg-amber-500/20 text-amber-400 border border-amber-500/40" />
          <ToggleField label="Right edge" value={edge === 'right'} onChange={v => updateField('edge', v ? 'right' : null)} activeColor="bg-amber-500/20 text-amber-400 border border-amber-500/40" />
        </>
      )
    }

    // --- Ego Influence subtrack ---
    if ((m as Record<string, unknown>)._isEgoInfluenceSubtrack) {
      return (
        <>
          <SectionHeader color="#92400e">Ego Influence</SectionHeader>
          <Field label="ID"><span className="text-sm text-gray-300">{(m.id as string) || ''}</span></Field>
          <InfluencerIdPicker label="Influencers (Shift+click on timeline)" value={(m.influencers as string[]) || []}
            onChange={v => updateField('influencers', v)} ann={ann} highlight={missing('influencers')} />
          <TextField label="Comment (optional)" value={(m.comment as string) || ''} onChange={v => updateField('comment', v)} />
        </>
      )
    }

    // --- Agent Influence subtrack ---
    if ((m as Record<string, unknown>)._isAgentInfluenceSubtrack) {
      return (
        <>
          <SectionHeader color="#92400e">Agent Influence</SectionHeader>
          <Field label="ID"><span className="text-sm text-gray-300">{(m.id as string) || ''}</span></Field>
          <InfluencerIdPicker label="Influencers (Shift+click on timeline)" value={(m.influencers as string[]) || []}
            onChange={v => updateField('influencers', v)} ann={ann} highlight={missing('influencers')} showAgents />
          <TextField label="Comment (optional)" value={(m.comment as string) || ''} onChange={v => updateField('comment', v)} />
        </>
      )
    }

    // --- Ego Containment subtrack ---
    if ((m as Record<string, unknown>)._isEgoContSubtrack) {
      return (
        <>
          <SectionHeader color="#22c55e">Ego Containment</SectionHeader>
          <Field label="Containment ID"><span className="text-sm text-gray-300">{(m.id as string) || ''}</span></Field>
          {renderEnvSelect(m)}
          <LaneIdField value={(m.lane_number as string) || 'none'} onChange={v => updateField('lane_number', v)} highlight={missing('lane_number')} />
          <ToggleField label="Illegal containment" value={(m.illegal_flag as boolean) ?? false} onChange={v => updateField('illegal_flag', v)} activeColor="bg-red-500/20 text-red-400 border border-red-500/40" />
          {renderSpatialContainmentFields(m)}
        </>
      )
    }

    // --- Agent Containment subtrack ---
    if ((m as Record<string, unknown>)._isAgentContSubtrack) {
      return (
        <>
          <SectionHeader color="#22c55e">Agent Containment</SectionHeader>
          <Field label="Containment ID"><span className="text-sm text-gray-300">{(m.id as string) || ''}</span></Field>
          {renderEnvSelect(m)}
          <LaneIdField value={(m.lane_number as string) || 'none'} onChange={v => updateField('lane_number', v)} highlight={missing('lane_number')} />
          <ToggleField label="Illegal containment" value={(m.illegal_flag as boolean) ?? false} onChange={v => updateField('illegal_flag', v)} activeColor="bg-red-500/20 text-red-400 border border-red-500/40" />
          {renderSpatialContainmentFields(m)}
        </>
      )
    }

    // --- Object Containment subtrack ---
    if ((m as Record<string, unknown>)._isObjContSubtrack) {
      return (
        <>
          <SectionHeader color="#f59e0b">Object Containment</SectionHeader>
          <Field label="Containment ID"><span className="text-sm text-gray-300">{(m.id as string) || ''}</span></Field>
          {renderEnvSelect(m)}
          <LaneIdField value={(m.lane_number as string) || 'none'} onChange={v => updateField('lane_number', v)} highlight={missing('lane_number')} />
          <ToggleField label="Illegal containment" value={(m.illegal_flag as boolean) ?? false} onChange={v => updateField('illegal_flag', v)} activeColor="bg-red-500/20 text-red-400 border border-red-500/40" />
          {renderSpatialContainmentFields(m)}
        </>
      )
    }

    // --- Traffic Light Physical Containment subtrack ---
    if ((m as Record<string, unknown>)._isLightPhysContSubtrack) {
      return (
        <>
          <SectionHeader color="#22c55e">Light Location</SectionHeader>
          <Field label="Containment ID"><span className="text-sm text-gray-300">{(m.id as string) || ''}</span></Field>
          {renderEnvSelect(m, 'Controlling Environment')}
        </>
      )
    }

    // --- Light Containment subtrack ---
    if ((m as Record<string, unknown>)._isLightContSubtrack) {
      return (
        <>
          <SectionHeader color="#ef4444">Light Containment</SectionHeader>
          <Field label="Containment ID"><span className="text-sm text-gray-300">{(m.id as string) || ''}</span></Field>
          {renderEnvSelect(m, 'Affects Environment')}
          <TextField label="Affected lane numbers (comma separated)" value={(m.lane_number as string) || ''} onChange={v => updateField('lane_number', v)} highlight={missing('lane_number')} />
        </>
      )
    }

    // --- Condition (must be checked before env since conditions share env_ trackIds) ---
    if ((m as Record<string, unknown>)._isCondSubtrack) {
      const condId = m.id as string
      const typeStr = (m.type as string) || ''
      return (
        <>
          <SectionHeader color="#06b6d4">Special condition ({condId})</SectionHeader>
          <Field label="ID"><span className="text-sm text-gray-300">{String(condId ?? '')}</span></Field>
          <Field label="Environment ID"><span className="text-sm text-gray-300">{String(m.env_id ?? '')}{(() => { const env = envOptions.find(e => e.id === m.env_id); return env ? ` · ${envDisplayName(env.type)}${env.name ? ` ("${env.name}")` : ''}` : '' })()}</span></Field>
          <SelectField label="Type" value={typeStr} options={CONDITION_TYPES} onChange={v => updateField('type', v)} allowEmpty displayNames={CONDITION_DISPLAY_NAMES} highlight={missing('type')} />
          {typeStr === 'Other' && <TextField label="Other description" value={(m.condition_other_description as string) || ''} onChange={v => updateField('condition_other_description', v)} highlight={missing('other condition description')} />}
        </>
      )
    }

    // --- Object State subtrack ---
    if ((m as Record<string, unknown>)._isObjStateSubtrack) {
      return (
        <>
          <SectionHeader color="#d97706">Object State</SectionHeader>
          <Field label="ID"><span className="text-sm text-gray-300">{(m.id as string) || ''}</span></Field>
          <SelectField label="Motion State" value={(m.motion_state as string) || ''} options={['Static', 'Moving / Rolling']} onChange={v => updateField('motion_state', v)} allowEmpty highlight={missing('motion_state')} />
          <SelectField label="Open State" value={(m.open_state as string) || ''} options={['Open', 'Closed']} onChange={v => updateField('open_state', v)} allowEmpty />
          <TextField label="Other condition description" value={(m.other_condition_description as string) || ''}
            onChange={v => updateField('other_condition_description', v)} />
        </>
      )
    }

    // --- Signal Head subtrack ---
    if ((m as Record<string, unknown>)._isSignalHeadSubtrack) {
      return (
        <>
          <SectionHeader color="#f97316">Signal Head ({(m.id as string) || ''})</SectionHeader>
          <Field label="ID"><span className="text-sm text-gray-300">{(m.id as string) || ''}</span></Field>
        </>
      )
    }

    // --- Light State subtrack ---
    if ((m as Record<string, unknown>)._isLightStateSubtrack) {
      const handleLightStateTypeChange = (newType: string) => {
        if (!sel || !ann) return
        const u = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
        const li = (m._lightIndex as number) ?? -1
        const hi = (m._headIndex as number) ?? -1
        const si = (m._stateIndex as number) ?? -1
        if (li >= 0 && hi >= 0 && si >= 0) {
          (u.traffic_lights[li].signal_heads[hi].state_sequence[si] as unknown as Record<string, unknown>).type = newType
          if (newType === 'OFF') (u.traffic_lights[li].signal_heads[hi].state_sequence[si] as unknown as Record<string, unknown>).color = 'Other'
        }
        persist(u)
      }
      return (
        <>
          <SectionHeader color="#dc2626">Light State</SectionHeader>
          <Field label="ID"><span className="text-sm text-gray-300">{(m.id as string) || ''}</span></Field>
          <SelectField label="Type" value={(m.type as string) || ''} options={LIGHT_STATE_TYPES} onChange={handleLightStateTypeChange} allowEmpty highlight={missing('type')} />
          <SelectField label="Color" value={(m.color as string) || ''} options={LIGHT_COLORS} onChange={v => updateField('color', v)} allowEmpty highlight={missing('color')} />
          <SelectField label="Shape" value={(m.shape as string) || ''} options={LIGHT_SHAPES} onChange={v => updateField('shape', v)} allowEmpty highlight={missing('shape')} />
          {((m.shape as string) === 'Other' || (m.color as string) === 'Other') && (
            <TextField label="Other condition description" value={(m.other_condition_description as string) || ''}
              onChange={v => updateField('other_condition_description', v)} highlight={missing('other description')} />
          )}
          {(m.color as string) === 'Yellow' && (() => {
            const boolToStr = (v: unknown): string => v === true ? 'Yes' : v === false ? 'No' : ''
            const strToBool = (v: string): boolean | null => v === 'Yes' ? true : v === 'No' ? false : null
            const updateLightStateFields = (fields: Record<string, boolean | null>) => {
              if (!sel || !ann) return
              const u = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
              const li = (m._lightIndex as number) ?? -1
              const hi = (m._headIndex as number) ?? -1
              const si = (m._stateIndex as number) ?? -1
              if (li >= 0 && hi >= 0 && si >= 0) {
                const state = u.traffic_lights[li].signal_heads[hi].state_sequence[si] as unknown as Record<string, unknown>
                for (const [k, v] of Object.entries(fields)) state[k] = v
              }
              persist(u)
            }
            return (
              <>
                <SelectField label="On ego's path?" value={boolToStr(m.yellow_on_ego_path)} options={['Yes', 'No']} onChange={v => updateLightStateFields({ yellow_on_ego_path: strToBool(v), ...(v !== 'Yes' ? { ego_in_intersection_on_yellow: null, ego_could_have_cleared_safely: null } : {}) })} allowEmpty highlight={missing('yellow on ego path')} />
                {m.yellow_on_ego_path === true && (
                  <SelectField label="Is the ego in the intersection on yellow?" value={boolToStr(m.ego_in_intersection_on_yellow)} options={['Yes', 'No']} onChange={v => updateLightStateFields({ ego_in_intersection_on_yellow: strToBool(v), ...(v !== 'No' ? { ego_could_have_cleared_safely: null } : {}) })} allowEmpty highlight={missing('ego in intersection on yellow')} />
                )}
                {m.yellow_on_ego_path === true && m.ego_in_intersection_on_yellow === false && (
                  <SelectField label="Could the ego have made it safely into the intersection on yellow?" value={boolToStr(m.ego_could_have_cleared_safely)} options={['Yes', 'No']} onChange={v => updateLightStateFields({ ego_could_have_cleared_safely: strToBool(v) })} allowEmpty highlight={missing('could ego have cleared safely')} />
                )}
              </>
            )
          })()}
        </>
      )
    }

    // --- Environment ---
    if (sel.trackId.startsWith('env_')) {
      const envId = m.id as string
      return (
        <>
          <SectionHeader color="#22c55e">Environment ({envId})</SectionHeader>
          <Field label="ID"><span className="text-sm text-gray-300">{String(envId ?? '')}</span></Field>
          <TextField label="Name" value={(m.name as string) || ''} onChange={v => updateField('name', v)} />
          <SelectField label="Type" value={(m.type as string) || ''} options={ENVIRONMENT_TYPES} groups={ENVIRONMENT_TYPE_GROUPS} onChange={v => updateField('type', v)} allowEmpty displayNames={ENVIRONMENT_DISPLAY_NAMES} highlight={missing('type')} />
          {(m.type === 'Other') && <TextField label="Other type description" value={(m.type_other_description as string) || ''} onChange={v => updateField('type_other_description', v)} highlight={missing('other type description')} />}
          {(m.type === 'Other Intersection') && <TextField label="Other intersection description" value={(m.type_other_description as string) || ''} onChange={v => updateField('type_other_description', v)} highlight={missing('other intersection description')} />}
          {SPLIT_LANE_TYPES.has(m.type as string) ? (
            <div className="grid grid-cols-2 gap-2">
              <LaneNumberField
                label="In Lanes"
                value={(m.num_lanes as number) || ''}
                onChange={v => updateField('num_lanes', v)}
                highlight={missing('in_lanes')}
              />
              <LaneNumberField
                label="Out Lanes"
                value={(m.num_out_lanes as number) || ''}
                onChange={v => updateField('num_out_lanes', v)}
                highlight={missing('out_lanes')}
              />
            </div>
          ) : (
            <LaneNumberField label="Number of Lanes in ego-direction" value={(m.num_lanes as number) || ''} onChange={v => updateField('num_lanes', v)} highlight={missing('num_lanes')} />
          )}
          <ToggleField label="One way" value={(m.one_way as boolean) ?? false} onChange={v => updateField('one_way', v)} />
          {ann && <button
            className="mt-2 w-full px-3 py-1.5 rounded text-xs font-medium bg-cyan-700 hover:bg-cyan-600 text-white transition-colors"
            onClick={() => persist(addConditionToEnv(ann, String(envId), sel.t0, sel.t1))}
          >+ Add Condition</button>}
        </>
      )
    }

    // --- Traffic Light ---
    if (sel.trackId.startsWith('light_')) {
      const lightId = m.id as string
      const li = m._lightIndex as number
      return (
        <>
          <SectionHeader color="#ef4444">Traffic Light ({lightId})</SectionHeader>
          <Field label="ID"><span className="text-sm text-gray-300">{String(lightId ?? '')}</span></Field>
          <TextField label="Name" value={(m.name as string) || ''} onChange={v => updateField('name', v)} />
          {ann && li != null && li >= 0 && (
            <>
              <button
                className="w-full mt-2 px-3 py-1.5 text-xs bg-[#1a1a35] text-white rounded-lg border border-[#2a2a50] hover:bg-[#2a2a50]"
                onClick={() => {
                  const light = ann.traffic_lights[li]
                  const t0 = parseTs(light.visibility_start_timestamp)
                  const t1 = parseTs(light.visibility_end_timestamp)
                  persist(addPhysicalContainmentToLight(ann, li, t0, t1))
                }}
              >+ Add Containment</button>
              <button
                className="w-full mt-1 px-3 py-1.5 text-xs bg-[#1a1a35] text-white rounded-lg border border-[#2a2a50] hover:bg-[#2a2a50]"
                onClick={() => {
                  const light = ann.traffic_lights[li]
                  const t0 = parseTs(light.visibility_start_timestamp)
                  const t1 = parseTs(light.visibility_end_timestamp)
                  persist(addSignalHeadToLight(ann, li, t0, t1))
                }}
              >+ Add Signal Head</button>
            </>
          )}
        </>
      )
    }

    // --- Ego Property (subtrack) ---
    if (sel.trackId === 'ego_act' && m._isEgoPropertySubtrack) {
      const propType = (m.property_type as string) || ''
      const isSignal = propType === 'Signal'
      const sd = (m.signaling_details || {}) as Record<string, unknown>
      return (
        <>
          <SectionHeader color="#93c5fd">Ego — Property</SectionHeader>
          <Field label="Property ID"><span className="text-sm text-gray-300">{(m.id as string) || ''}</span></Field>
          <SelectField label="Property Type" value={propType} options={PROPERTY_TYPES} onChange={v => updateField('property_type', v)} allowEmpty displayNames={PROPERTY_DISPLAY_NAMES} highlight={missing('property_type')} />
          {propType === 'Other' && (
            <TextField label="Other Description" value={(m.other_description as string) || ''} onChange={v => updateField('other_description', v)} placeholder="Describe the property" highlight={missing('other description')} />
          )}
          {isSignal && (
            <div className="space-y-2 pl-2 border-l-2 border-purple-500/30">
              <span className="text-[9px] text-[#556] font-bold uppercase">Signaling Details</span>
              <SelectField label="Source" value={(sd.source as string) || ''} options={SIGNAL_SOURCES} onChange={v => updateField('signal_source', v)} allowEmpty highlight={missing('signal source')} />
              {sd.source === 'Other' && <TextField label="Other Source" value={(sd.other_source_description as string) || ''} onChange={v => updateField('signal_other_source', v)} placeholder="Describe the source" highlight={missing('other source description')} />}
              <SelectField label="Intent" value={(sd.intent as string) || ''} options={SIGNAL_INTENTS} onChange={v => updateField('signal_intent', v)} allowEmpty highlight={missing('signal intent')} />
              {sd.intent === 'Other' && <TextField label="Other Intent" value={(sd.other_intent_description as string) || ''} onChange={v => updateField('signal_other_intent', v)} placeholder="Describe the intent" highlight={missing('other intent description')} />}
              {sd.source === 'Holding sign' && <SelectField label="Sign Type" value={(sd.sign_type as string) || ''} options={SIGN_TYPES} onChange={v => updateField('signal_sign_type', v)} allowEmpty highlight={missing('signal sign type')} />}
              {sd.source === 'Holding sign' && sd.sign_type === 'Other' && <TextField label="Other Sign Type" value={(sd.other_sign_description as string) || ''} onChange={v => updateField('signal_other_sign', v)} placeholder="Describe the sign type" highlight={missing('other sign type description')} />}
              {sd.source === 'Holding sign' && (
                <label className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" checked={!!sd.not_facing_ego} onChange={e => updateField('signal_not_facing_ego', e.target.checked)} className="accent-blue-500" />
                  <span className={labelCls}>Not facing ego</span>
                </label>
              )}
              <AgentIdPicker label="Signaling To" value={(sd.link_to as string[]) || []}
                onChange={v => updateField('signal_target_ids', v)} ann={ann} />
            </div>
          )}
        </>
      )
    }

    // --- Ego Action ---
    if (sel.trackId === 'ego_act') {
      const actType = (m.type as string) || ''
      const isTurnOther = actType.startsWith('Other oxd:MakeATurn')
      return (
        <>
          <SectionHeader color="#3b82f6">Ego Action</SectionHeader>
          <SelectField label="Action Type" value={actType} options={EGO_ACTION_TYPES} groups={EGO_ACTION_TYPE_GROUPS} onChange={v => updateField('type', v)} allowEmpty displayNames={EGO_ACTION_DISPLAY_NAMES} highlight={missing('type')} />
          {isTurnOther && <TextField label="Turn Other Description" value={(m.turn_other_description as string) || ''} onChange={v => updateField('turn_other_description', v)} placeholder="Describe the turn" highlight={missing('other turn description')} />}
          {actType === 'Other' && <TextField label="Other Description" value={(m.action_other_description as string) || ''} onChange={v => updateField('action_other_description', v)} placeholder="Describe the action" highlight={missing('other action description')} />}
          {ACTION_LINK_TO_CONFIG[actType] && (
            <ActionTargetPicker label={ACTION_LINK_TO_CONFIG[actType].label} value={(m.action_target as string[]) || []}
              onChange={v => updateField('action_target', v)} ann={ann} entityTypes={ACTION_LINK_TO_CONFIG[actType].entityTypes} required={ACTION_LINK_TO_CONFIG[actType].required}
              highlight={ACTION_LINK_TO_CONFIG[actType].required && missing(ACTION_LINK_TO_CONFIG[actType].missingLabel)} />
          )}
        </>
      )
    }

    // (Ego Containment is now rendered via _isEgoContSubtrack above)

    // --- Traffic Object ---
    if (sel.trackId.startsWith('obj_')) {
      const objId = m.id as string
      const oi = m._objIndex as number | undefined
      return (
        <>
          <SectionHeader color="#f59e0b">Object ({objId})</SectionHeader>
          <Field label="ID"><span className="text-sm text-gray-300">{String(objId ?? '')}</span></Field>
          <TextField label="Name" value={(m.name as string) || ''} onChange={v => updateField('name', v)} />
          <SelectField label="Type" value={(m.type as string) || ''} options={TRAFFIC_OBJECT_TYPES} groups={TRAFFIC_OBJECT_TYPE_GROUPS} onChange={v => updateField('type', v)} allowEmpty displayNames={TRAFFIC_OBJECT_DISPLAY_NAMES} highlight={missing('type')} />
          {(typeof m.type === 'string' && m.type.startsWith('Other')) && <TextField label="Other type description" value={(m.other_type_description as string) || ''} onChange={v => updateField('other_type_description', v)} highlight={missing('other type description')} />}
          {QUANTITY_ELIGIBLE_OBJECT_TYPES.has(m.type as string) && (
            <SelectField
              label="Quantity"
              value={(m.quantity as string) || ''}
              options={TRAFFIC_OBJECT_QUANTITIES}
              onChange={v => updateField('quantity', v)}
              allowEmpty
              highlight={missing('quantity')}
            />
          )}
          {oi != null && <button
            className="mt-2 w-full px-3 py-1.5 rounded text-xs font-medium bg-cyan-700 hover:bg-cyan-600 text-white transition-colors"
            onClick={() => {
              const o = ann!.traffic_objects[oi]
              const t0 = o.visibility_start_timestamp ? parseTs(o.visibility_start_timestamp) : 0
              const t1 = o.visibility_end_timestamp ? parseTs(o.visibility_end_timestamp) : (bundle?.video?.duration_s || 300)
              persist(addContainmentToObject(ann!, oi, t0, t1))
            }}
          >+ Add Containment</button>}
        </>
      )
    }

    // --- Agent Pose subtrack ---
    if ((m as Record<string, unknown>)._isAgentPoseSubtrack) {
      return (
        <>
          <SectionHeader color="#14b8a6">Agent Pose (relative to Ego)</SectionHeader>
          <SelectField label="Position" value={(m.position_rel_to_ego as string) || ''} options={POSITION_REL_TO_EGO} onChange={v => updateField('position_rel_to_ego', v)} allowEmpty highlight={missing('position_rel_to_ego')} />
          <SelectField label="Direction" value={(m.direction_rel_to_ego as string) || ''} options={DIRECTION_REL_TO_EGO} onChange={v => updateField('direction_rel_to_ego', v)} allowEmpty highlight={missing('direction_rel_to_ego')} />
        </>
      )
    }

    // --- Agent Property (subtrack) ---
    if (sel.trackId.startsWith('agent_') && m._isAgentPropertySubtrack) {
      const ai = m._agentIndex as number
      const agent = ann?.agents?.[ai]
      const propType = (m.property_type as string) || ''
      const isSignal = propType === 'Signal'
      const sd = (m.signaling_details || {}) as Record<string, unknown>
      return (
        <>
          <SectionHeader color="#d8b4fe">Agent ({agent?.id || '...'}) — Property</SectionHeader>
          <Field label="Property ID"><span className="text-sm text-gray-300">{(m.id as string) || ''}</span></Field>
          <SelectField label="Property Type" value={propType} options={PROPERTY_TYPES} onChange={v => updateField('property_type', v)} allowEmpty displayNames={PROPERTY_DISPLAY_NAMES} highlight={missing('property_type')} />
          {propType === 'Other' && (
            <TextField label="Other Description" value={(m.other_description as string) || ''} onChange={v => updateField('other_description', v)} placeholder="Describe the property" highlight={missing('other description')} />
          )}
          {isSignal && (
            <div className="space-y-2 pl-2 border-l-2 border-purple-500/30">
              <span className="text-[9px] text-[#556] font-bold uppercase">Signaling Details</span>
              <SelectField label="Source" value={(sd.source as string) || ''} options={SIGNAL_SOURCES} onChange={v => updateField('signal_source', v)} allowEmpty highlight={missing('signal source')} />
              {sd.source === 'Other' && <TextField label="Other Source" value={(sd.other_source_description as string) || ''} onChange={v => updateField('signal_other_source', v)} placeholder="Describe the source" highlight={missing('other source description')} />}
              <SelectField label="Intent" value={(sd.intent as string) || ''} options={SIGNAL_INTENTS} onChange={v => updateField('signal_intent', v)} allowEmpty highlight={missing('signal intent')} />
              {sd.intent === 'Other' && <TextField label="Other Intent" value={(sd.other_intent_description as string) || ''} onChange={v => updateField('signal_other_intent', v)} placeholder="Describe the intent" highlight={missing('other intent description')} />}
              {sd.source === 'Holding sign' && <SelectField label="Sign Type" value={(sd.sign_type as string) || ''} options={SIGN_TYPES} onChange={v => updateField('signal_sign_type', v)} allowEmpty highlight={missing('signal sign type')} />}
              {sd.source === 'Holding sign' && sd.sign_type === 'Other' && <TextField label="Other Sign Type" value={(sd.other_sign_description as string) || ''} onChange={v => updateField('signal_other_sign', v)} placeholder="Describe the sign type" highlight={missing('other sign type description')} />}
              {sd.source === 'Holding sign' && (
                <label className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" checked={!!sd.not_facing_ego} onChange={e => updateField('signal_not_facing_ego', e.target.checked)} className="accent-blue-500" />
                  <span className={labelCls}>Not facing ego</span>
                </label>
              )}
              <AgentIdPicker label="Signaling To" value={(sd.link_to as string[]) || []}
                onChange={v => updateField('signal_target_ids', v)} ann={ann} />
            </div>
          )}
        </>
      )
    }

    // --- Agent (parent) ---
    if (sel.trackId.startsWith('agent_') && m._objKind === 'agent' && !m._isAgentActionSubtrack && !m._isAgentPropertySubtrack) {
      const ai = m._agentIndex as number
      const agent = ann?.agents?.[ai]
      return (
        <>
          <SectionHeader color="#a855f7">Agent ({agent?.id || '...'})</SectionHeader>
          <Field label="Agent ID"><span className="text-sm text-gray-300">{agent?.id || ''}</span></Field>
          <TextField label="Name" value={(m.name as string) || ''} onChange={v => updateField('name', v)} />
          <SelectField label="Amount" value={agent?.amount || ''} options={getAllowedAmounts(agent?.type || '')} onChange={v => updateField('agent_amount', v)} allowEmpty highlight={missing('amount')} />
          <SelectField label="Agent Type" value={agent?.type || ''} options={AGENT_TYPES} groups={AGENT_TYPE_GROUPS} onChange={v => updateField('agent_type_enum', v)} allowEmpty displayNames={AGENT_TYPE_DISPLAY_NAMES} highlight={missing('type')} />
          {agent?.type === 'Other' && <TextField label="Other type description" value={agent?.other_type_description || ''} onChange={v => updateField('agent_other_type_description', v)} highlight={missing('other type description')} />}
          {agent?.type === 'Pedestrian (Other)' && <TextField label="Other pedestrian description" value={agent?.other_type_description || ''} onChange={v => updateField('agent_other_type_description', v)} highlight={missing('other pedestrian description')} />}
          {agent?.type === 'oxd:Animal' && <TextField label="Animal description" value={agent?.other_type_description || ''} onChange={v => updateField('agent_other_type_description', v)} highlight={missing('animal description')} />}
          {ai != null && <button
            className="mt-2 w-full px-3 py-1.5 rounded text-xs font-medium bg-green-700 hover:bg-green-600 text-white transition-colors"
            onClick={() => {
              const a = ann!.agents[ai]
              const t0 = a.visibility_start_timestamp ? parseTs(a.visibility_start_timestamp) : 0
              const t1 = a.visibility_end_timestamp ? parseTs(a.visibility_end_timestamp) : (bundle?.video?.duration_s || 300)
              persist(addContainmentToAgent(ann!, ai, t0, t1))
            }}
          >+ Add Containment</button>}
          {ai != null && <button
            className="mt-1 w-full px-3 py-1.5 rounded text-xs font-medium bg-amber-800 hover:bg-amber-700 text-white transition-colors"
            onClick={() => {
              const a = ann!.agents[ai]
              const t0 = a.visibility_start_timestamp ? parseTs(a.visibility_start_timestamp) : 0
              const t1 = a.visibility_end_timestamp ? parseTs(a.visibility_end_timestamp) : (bundle?.video?.duration_s || 300)
              persist(addInfluenceToAgent(ann!, ai, t0, t1))
            }}
          >+ Add Influence</button>}
          {ai != null && <button
            className="mt-1 w-full px-3 py-1.5 rounded text-xs font-medium bg-fuchsia-800 hover:bg-fuchsia-700 text-white transition-colors"
            onClick={() => {
              const a = ann!.agents[ai]
              const t0 = a.visibility_start_timestamp ? parseTs(a.visibility_start_timestamp) : 0
              const t1 = a.visibility_end_timestamp ? parseTs(a.visibility_end_timestamp) : (bundle?.video?.duration_s || 300)
              persist(addPropertyToAgent(ann!, ai, t0, t1))
            }}
          >+ Add Property</button>}
        </>
      )
    }

    // --- Agent Action (subtrack) ---
    if (sel.trackId.startsWith('agent_') && m._isAgentActionSubtrack) {
      const ai = m._agentIndex as number
      const agent = ann?.agents?.[ai]
      const actionType = (m.action_type as string) || ''
      return (
        <>
          <SectionHeader color="#a855f7">Agent ({agent?.id || '...'}) — Action</SectionHeader>
          <Field label="Action ID"><span className="text-sm text-gray-300">{(m.id as string) || ''}</span></Field>
          <SelectField label="Action Type" value={actionType} options={getAgentActionTypes(agent?.type || '')} groups={getAgentActionTypeGroups(agent?.type || '')} onChange={v => updateField('action_type', v)} allowEmpty displayNames={AGENT_ACTION_DISPLAY_NAMES} highlight={missing('action_type')} />
          {actionType === 'Other oxd:MakeATurn' && (
            <TextField label="Turn Other Description" value={(m.other_description as string) || ''} onChange={v => updateField('other_description', v)} placeholder="Describe the turn" highlight={missing('other turn description')} />
          )}
          {actionType === 'Other' && (
            <TextField label="Other Description" value={(m.other_description as string) || ''} onChange={v => updateField('other_description', v)} placeholder="Describe the action" highlight={missing('other action description')} />
          )}
          {ACTION_LINK_TO_CONFIG[actionType] && (
            <ActionTargetPicker label={ACTION_LINK_TO_CONFIG[actionType].label} value={(m.action_target as string[]) || []}
              onChange={v => updateField('action_target', v)} ann={ann} entityTypes={ACTION_LINK_TO_CONFIG[actionType].entityTypes} required={ACTION_LINK_TO_CONFIG[actionType].required}
              highlight={ACTION_LINK_TO_CONFIG[actionType].required && missing(ACTION_LINK_TO_CONFIG[actionType].missingLabel)} />
          )}
        </>
      )
    }

    return null
  }

  // ================= JSX =================
  return (
    <aside className="h-full flex flex-col bg-[#111128] overflow-y-auto overflow-x-hidden" style={{ scrollbarGutter: 'stable' }}>

      {/* Status + Save */}
      <div className="p-4 border-b border-[#1e1e38]">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-[11px] font-bold uppercase tracking-widest text-[#556]">Status</h3>
          <span className={`px-3 py-1 rounded-full text-[11px] font-semibold border ${statusColor}`}>{status === 'needs_revision' ? 'revision requested' : status}</span>
        </div>
        <div className="flex gap-2 mb-3">
          <button
            onClick={save}
            disabled={!bundle || saving || !dirty || locked || serverReadOnly}
            className="flex-1 flex items-center justify-center gap-2.5 px-5 py-3 rounded-2xl bg-blue-600/20 text-blue-400 border border-blue-600/30 hover:bg-blue-600/30 text-sm font-semibold transition-all disabled:opacity-30 shadow-sm"
          >
            <Save className="w-5 h-5" /> {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>

      <div>
      {/* === Details view === */}

      {/* Relevancy */}
      {ann && (
        <div className="p-4 border-b border-[#1e1e38]">
          <h3 className={`text-[11px] font-bold uppercase tracking-widest mb-2 ${ann.eventful == null ? 'text-red-400' : 'text-[#556]'}`}>Relevancy</h3>
          <p className="text-[11px] text-[#888] mb-2">Does the clip show eventful, non-nominal driving?</p>
          <div className="flex gap-2">
            {[
              { val: true, label: 'Yes' },
              { val: false, label: 'No' },
            ].map(opt => {
              const active = ann.eventful === opt.val
              return (
                <button key={String(opt.val)} onClick={() => {
                  if (!ann || !bundle || editsBlocked) return
                  const u = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
                  u.eventful = opt.val
                  persist(u)
                }}
                  className={`flex-1 px-3 py-2 rounded-xl text-xs font-medium transition-all border ${
                    active ? 'bg-blue-500/20 text-blue-400 border-blue-500/40' : 'bg-[#1a1a35] text-[#555] border-[#2a2a50] hover:border-[#3a3a60]'
                  }`}>
                  {opt.label}
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Brief Description */}
      <div className="p-4 border-b border-[#1e1e38]">
        <h3 className="text-[11px] font-bold uppercase tracking-widest text-[#556] mb-2 flex items-center gap-1.5"><FileText className="w-3.5 h-3.5" /> Description</h3>
        {briefEdit !== null ? (
          <textarea value={briefEdit} onChange={e => setBriefEdit(e.target.value)} onBlur={handleBriefBlur} autoFocus className="w-full h-16 px-3 py-2 text-sm bg-[#1a1a35] text-[#ddd] rounded-xl border border-[#2a2a50] focus:border-blue-500/50 focus:outline-none resize-none" />
        ) : (
          <p className="text-sm text-[#999] cursor-pointer hover:text-white transition-colors min-h-[2rem] px-3 py-2 bg-[#1a1a35] rounded-xl border border-[#1e1e38]" onClick={() => setBriefEdit(ann?.brief_description ?? '')}>{ann?.brief_description || 'Click to add...'}</p>
        )}
      </div>

      {/* Ego Driving Judgment */}
      {ann && (
        <div className="p-4 border-b border-[#1e1e38]">
          <h3 className="text-[11px] font-bold uppercase tracking-widest text-[#556] mb-2">Ego Driving Judgment</h3>
          <div className="flex gap-2">
            {[
              { val: 'good', icon: '😊', label: 'Good' },
              { val: 'neutral', icon: '😐', label: 'Neutral' },
              { val: 'bad', icon: '🙁', label: 'Bad' },
            ].map(opt => {
              const active = ann.ego_vehicle?.driving_judgment === opt.val
              return (
                <button key={opt.val} onClick={() => {
                  if (!ann || !bundle) return
                  const u = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
                  if (!u.ego_vehicle) u.ego_vehicle = { actions: [] }
                  u.ego_vehicle.driving_judgment = opt.val
                  persist(u)
                }}
                  className={`flex-1 flex flex-col items-center gap-1 px-3 py-2.5 rounded-xl text-xs font-medium transition-all border ${
                    active ? 'bg-blue-500/20 text-blue-400 border-blue-500/40' : 'bg-[#1a1a35] text-[#555] border-[#2a2a50] hover:border-[#3a3a60]'
                  }`}>
                  <span className="text-lg">{opt.icon}</span>
                  <span>{opt.label}</span>
                </button>
              )
            })}
          </div>
        </div>
      )}

      {ann?.eventful !== false && <>
      {/* Selected Segment — Properties */}
      <div className="p-5 border-b border-[#1e1e38]">
        <h3 className="text-[11px] font-bold uppercase tracking-widest text-[#556] mb-3">Properties</h3>
        {(sel || selectedPath === 'ego_main') ? (
          <div className="flex flex-col gap-1">
            {/* Completeness warning */}
            {(() => {
              if (!comp || comp.missingCount === 0) return null
              return (
                <div className="mb-3 p-2.5 rounded-xl bg-yellow-500/10 border border-yellow-500/20">
                  <div className="text-[10px] text-yellow-400 font-bold uppercase mb-1">Missing fields ({comp.missingCount})</div>
                  <ul className="text-[10px] text-yellow-300/70 space-y-0.5">
                    {comp.issues.map((issue, i) => <li key={i}>{issue}</li>)}
                  </ul>
                </div>
              )
            })()}
            {/* Entity-specific property fields */}
            {renderProperties()}

            {sel && <>
            {/* Time inputs (common to all) */}
            <div className="flex gap-3 pt-4 border-t border-[#1e1e38]">
              <label className="flex-1">
                <span className={labelCls}>Start (s)</span>
                <input type="number" step={0.1} value={sel.t0.toFixed(1)} onChange={e => handleTimeChange(+e.target.value, sel.t1)} className={inputCls} />
              </label>
              <label className="flex-1">
                <span className={labelCls}>End (s)</span>
                <input type="number" step={0.1} value={sel.t1.toFixed(1)} onChange={e => handleTimeChange(sel.t0, +e.target.value)} className={inputCls} />
              </label>
            </div>

            {/* Illegal toggle */}
            {canHaveIllegal && (
              <div className="pt-2">
                <button onClick={handleToggleIllegal} className={`w-full flex items-center justify-center gap-2.5 px-5 py-3.5 rounded-2xl text-sm font-semibold transition-all shadow-sm ${sel.illegal ? 'bg-red-500/20 text-red-400 border border-red-500/40 shadow-[0_0_12px_rgba(239,68,68,0.2)]' : 'bg-[#1a1a35] text-[#666] border border-[#2a2a50] hover:border-red-500/30 hover:text-red-400'}`}>
                  <AlertTriangle className="w-5 h-5" /> {sel.illegal ? 'ILLEGAL — click to remove' : 'Mark as Illegal'}
                </button>
              </div>
            )}

            {/* Because_of (causality) */}
            {canHaveBecause && (
              <div className="pt-3 border-t border-[#1e1e38]">
                <span className="text-[11px] text-[#556] font-bold uppercase tracking-wider block mb-2">Because of (causality)</span>
                <div className="flex flex-wrap gap-1.5 mb-3">
                  {(sel.because_of || []).map((c, i) => (
                    <span key={i} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[11px] bg-red-500/15 text-red-400 border border-red-500/20">
                      {c}
                      <button onClick={() => handleRemoveBecause(c)} className="hover:text-white"><X className="w-3.5 h-3.5" /></button>
                    </span>
                  ))}
                  {!(sel.because_of?.length) && <span className="text-[11px] text-[#444] italic">Use the dropdown below or Ctrl+click on the timeline.</span>}
                </div>
                {(() => {
                  const existing = new Set(sel.because_of || [])
                  // Actions grouped by ego + agent
                  const agentActionGroups: { groupLabel: string; opts: { id: string; label: string }[] }[] = []
                  const egoActs = (ann?.ego_vehicle?.actions || []).filter(act => act.id && !existing.has(act.id!))
                  if (egoActs.length > 0) {
                    agentActionGroups.push({ groupLabel: 'Ego', opts: egoActs.map(act => ({ id: act.id!, label: `${act.id} · ${act.type === 'Other' && act.action_other_description ? `Other (${act.action_other_description})` : (EGO_ACTION_DISPLAY_NAMES[act.type] ?? act.type)}` })) })
                  }
                  for (const a of ann?.agents || []) {
                    const acts = (a.actions || []).filter(act => act.id && !existing.has(act.id!))
                    if (acts.length > 0) {
                      const agentTypeLabel = a.type === 'Other' && a.other_type_description ? `Other (${a.other_type_description})` : (a.type === 'Pedestrian (Other)' && a.other_type_description ? `Pedestrian (Other: ${a.other_type_description})` : (AGENT_TYPE_DISPLAY_NAMES[a.type] ?? a.type))
                      const groupLabel = `${a.id} · ${agentTypeLabel}${a.name ? ` "${a.name}"` : ''}`
                      agentActionGroups.push({ groupLabel, opts: acts.map(act => ({ id: act.id!, label: `${act.id} · ${(act.action_type === 'Other' || act.action_type === 'Other oxd:MakeATurn') && act.other_description ? `Other (${act.other_description})` : (AGENT_ACTION_DISPLAY_NAMES[act.action_type] ?? act.action_type)}` })) })
                    }
                  }
                  // TL States grouped by TL + signal head (+ base TL options)
                  const tlBaseOpts: { id: string; label: string }[] = []
                  const tlStateGroups: { groupLabel: string; opts: { id: string; label: string }[] }[] = []
                  for (const tl of ann?.traffic_lights || []) {
                    const tlLabel = `${tl.id}${tl.name ? ` "${tl.name}"` : ''}${tl.type ? ` · ${tl.type}` : ''}`
                    if (tl.id && !existing.has(tl.id)) tlBaseOpts.push({ id: tl.id, label: tlLabel })
                    for (const sh of tl.signal_heads || []) {
                      const states = (sh.state_sequence || []).filter(ls => ls.id && !existing.has(ls.id!))
                      if (states.length > 0) {
                        const groupLabel = `${tlLabel} — ${sh.id}`
                        tlStateGroups.push({ groupLabel, opts: states.map(ls => { const detail = [ls.color, ls.type, ls.shape].filter(Boolean).join('/'); return { id: ls.id!, label: detail ? `${ls.id} · ${detail}` : ls.id! } }) })
                      }
                    }
                  }
                  // Properties grouped by agent + ego
                  const propertyGroups: { groupLabel: string; opts: { id: string; label: string }[] }[] = []
                  const egoProps = (ann?.ego_vehicle?.properties || []).filter(p => p.id && !existing.has(p.id))
                  if (egoProps.length > 0) {
                    propertyGroups.push({ groupLabel: 'Ego', opts: egoProps.map(p => ({ id: p.id, label: `${p.id} · ${(PROPERTY_DISPLAY_NAMES[p.property_type] ?? p.property_type)}${p.property_type === 'Other' && p.other_description ? ` (${p.other_description})` : ''}` })) })
                  }
                  for (const a of ann?.agents || []) {
                    const props = (a.properties || []).filter(p => p.id && !existing.has(p.id))
                    if (props.length > 0) {
                      const agentTypeLabel = a.type === 'Other' && a.other_type_description ? `Other (${a.other_type_description})` : (a.type === 'Pedestrian (Other)' && a.other_type_description ? `Pedestrian (Other: ${a.other_type_description})` : (AGENT_TYPE_DISPLAY_NAMES[a.type] ?? a.type))
                      const groupLabel = `${a.id} · ${agentTypeLabel}${a.name ? ` "${a.name}"` : ''}`
                      propertyGroups.push({ groupLabel, opts: props.map(p => ({ id: p.id, label: `${p.id} · ${(PROPERTY_DISPLAY_NAMES[p.property_type] ?? p.property_type)}${p.property_type === 'Other' && p.other_description ? ` (${p.other_description})` : ''}` })) })
                    }
                  }
                  // Objects
                  const objectOpts = (ann?.traffic_objects || [])
                    .filter(to => to.id && !existing.has(to.id))
                    .map(to => ({ id: to.id, label: `${to.id} · ${to.type.startsWith('Other') && to.other_type_description ? (TRAFFIC_OBJECT_DISPLAY_NAMES[to.type] ?? to.type).replace('(specify)', `(${to.other_type_description})`) : (TRAFFIC_OBJECT_DISPLAY_NAMES[to.type] ?? to.type)}${to.name ? ` "${to.name}"` : ''}` }))
                  // Environments (parent only — conditions intentionally excluded for now)
                  const envBaseOpts: { id: string; label: string }[] = []
                  for (const env of ann?.environments || []) {
                    const envLabel = `${env.id}${env.name ? ` "${env.name}"` : ''} · ${envDisplayName(env.type).replace(' (specify)', '')}${env.type === 'Other' && env.type_other_description ? ` (${env.type_other_description})` : ''}`
                    if (env.id && !existing.has(env.id)) envBaseOpts.push({ id: env.id, label: envLabel })
                  }
                  const hasUnknown = existing.has('Unknown')
                  return (
                    <>
                      <select
                        value=""
                        onChange={e => {
                          const val = e.target.value
                          if (val === '__other__') { setBecauseOtherMode(true); setBecauseOtherText('') }
                          else if (val) { handleAddBecauseById(val); setBecauseOtherMode(false); setBecauseOtherText('') }
                        }}
                        className={selectCls}
                      >
                        <option value="">+ Add cause…</option>
                        {agentActionGroups.length > 0 && <option disabled>· · · Actions · · ·</option>}
                        {agentActionGroups.map(g => (
                          <optgroup key={g.groupLabel} label={g.groupLabel}>
                            {g.opts.map(o => <option key={o.id} value={o.id}>{o.label}</option>)}
                          </optgroup>
                        ))}
                        {(tlBaseOpts.length > 0 || tlStateGroups.length > 0) && <option disabled>· · · Traffic Lights · · ·</option>}
                        {tlBaseOpts.map(o => <option key={o.id} value={o.id}>{o.label}</option>)}
                        {tlStateGroups.map(g => (
                          <optgroup key={g.groupLabel} label={g.groupLabel}>
                            {g.opts.map(o => <option key={o.id} value={o.id}>{o.label}</option>)}
                          </optgroup>
                        ))}
                        {propertyGroups.length > 0 && <option disabled>· · · Properties · · ·</option>}
                        {propertyGroups.map(g => (
                          <optgroup key={g.groupLabel} label={g.groupLabel}>
                            {g.opts.map(o => <option key={o.id} value={o.id}>{o.label}</option>)}
                          </optgroup>
                        ))}
                        {objectOpts.length > 0 && <option disabled>· · · Objects · · ·</option>}
                        {objectOpts.length > 0 && <optgroup label="Objects">{objectOpts.map(o => <option key={o.id} value={o.id}>{o.label}</option>)}</optgroup>}
                        {envBaseOpts.length > 0 && <option disabled>· · · Environments · · ·</option>}
                        {envBaseOpts.map(o => <option key={o.id} value={o.id}>{o.label}</option>)}
                        <option disabled>· · · · · ·</option>
                        {!hasUnknown && <option value="Unknown">Unknown</option>}
                        <option value="__other__">Other (specify)…</option>
                      </select>
                      {becauseOtherMode && (
                        <div className="flex gap-2 mt-2">
                          <input
                            type="text"
                            autoFocus
                            value={becauseOtherText}
                            onChange={e => setBecauseOtherText(e.target.value)}
                            onKeyDown={e => {
                              if (e.key === 'Enter' && becauseOtherText.trim()) {
                                handleAddBecauseById(`other:${becauseOtherText.trim()}`)
                                setBecauseOtherMode(false); setBecauseOtherText('')
                              } else if (e.key === 'Escape') {
                                setBecauseOtherMode(false); setBecauseOtherText('')
                              }
                            }}
                            placeholder="Short description…"
                            className="flex-1 bg-[#1a1a35] border border-[#2a2a50] rounded-lg px-3 py-1.5 text-[12px] text-white placeholder-[#444] focus:outline-none focus:border-red-500/50"
                          />
                          <button
                            onClick={() => {
                              if (becauseOtherText.trim()) {
                                handleAddBecauseById(`other:${becauseOtherText.trim()}`)
                                setBecauseOtherMode(false); setBecauseOtherText('')
                              }
                            }}
                            className="px-3 py-1.5 rounded-lg bg-red-500/20 text-red-400 border border-red-500/30 text-[12px] hover:bg-red-500/30"
                          >Add</button>
                          <button
                            onClick={() => { setBecauseOtherMode(false); setBecauseOtherText('') }}
                            className="px-2 py-1.5 rounded-lg bg-[#1a1a35] text-[#666] border border-[#2a2a50] text-[12px] hover:text-white"
                          ><X className="w-3 h-3" /></button>
                        </div>
                      )}
                    </>
                  )
                })()}
              </div>
            )}

            {/* Link_to (Shift+click on timeline to add) */}
            {canHaveBecause && (
              <div className="pt-3 border-t border-[#1e1e38]">
                <span className="text-[11px] text-[#556] font-bold uppercase tracking-wider block mb-2">Link To <span className="font-normal normal-case">(optional)</span></span>
                <AgentIdPicker label="Link To (Shift+click on timeline)" value={(selMeta.link_to as string[]) || []}
                  onChange={v => updateField('link_to', v)} ann={ann} />
              </div>
            )}

            {/* Delete + Raw attrs — well separated */}
            <div className="pt-5 mt-2 border-t border-[#1e1e38] space-y-3">
              <div className="flex items-center gap-3">
                <button onClick={handleDelete} className="flex-1 flex items-center justify-center gap-2.5 px-5 py-3.5 rounded-2xl text-sm font-semibold bg-red-600/10 text-red-400 border border-red-600/20 hover:bg-red-600/20 hover:border-red-600/40 transition-all shadow-sm">
                  <Trash2 className="w-5 h-5" /> Delete Segment
                </button>
                <button onClick={() => setAttrsOpen(!attrsOpen)} className="flex items-center gap-1.5 px-4 py-3.5 rounded-2xl text-[11px] text-[#556] hover:text-[#888] bg-[#1a1a35] border border-[#2a2a50] transition-all">
                  {attrsOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />} Raw
                </button>
              </div>
              {attrsOpen && <pre className="text-[9px] text-[#666] bg-[#0e0e20] p-3 rounded-xl overflow-auto max-h-32">{JSON.stringify(sel.meta, null, 2)}</pre>}
            </div>
            </>}
          </div>
        ) : (
          <p className="text-[11px] text-[#444] italic">Click a segment on the timeline</p>
        )}
      </div>

      {/* Coverage */}
      <div className="p-4 border-b border-[#1e1e38]">
        <h3 className="text-[11px] font-bold uppercase tracking-widest text-[#556] mb-2 flex items-center gap-1.5"><BarChart3 className="w-3.5 h-3.5" /> Coverage</h3>
        <div className="space-y-2">
          {[{ label: 'Ego Actions', pct: egoActCov, color: 'bg-blue-500' }, { label: 'Ego Containment', pct: egoContCov, color: 'bg-cyan-500' }].map(m => (
            <div key={m.label}>
              <div className="flex justify-between text-[10px] text-[#666] mb-1"><span>{m.label}</span><span>{m.pct.toFixed(0)}%</span></div>
              <div className="h-2 bg-[#1a1a35] rounded-full overflow-hidden"><div className={`h-full ${m.color} rounded-full transition-all`} style={{ width: `${Math.min(100, m.pct)}%` }} /></div>
            </div>
          ))}
        </div>
      </div>

      {/* Causality */}
      {causal.length > 0 && (
        <div className="p-4 border-b border-[#1e1e38]">
          <h3 className="text-[11px] font-bold uppercase tracking-widest text-[#556] mb-2 flex items-center gap-1.5"><Link2 className="w-3.5 h-3.5" /> Causality</h3>
          <div className="space-y-1 max-h-28 overflow-y-auto">
            {causal.map((c, i) => (
              <button key={i} onClick={() => selectPath(c.id)} className="w-full text-left flex items-center gap-1.5 px-2 py-1 rounded-lg text-[10px] hover:bg-[#1a1a35] transition-colors">
                <span className={`w-1.5 h-1.5 rounded-full ${c.track === 'ego_act' ? 'bg-blue-500' : 'bg-purple-500'}`} />
                <span className="text-[#ccc] truncate">{c.label}</span>
                <span className="text-[#444]">-&gt;</span>
                <span className="text-[#999] truncate">{c.cause}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      </>}

      </div>

      {/* Bottom actions */}
      <div>
      <div className="p-5 border-t border-[#1e1e38] space-y-3">
        <button onClick={handleExport} disabled={!selectedClipId || !bundle} className="w-full flex items-center justify-center gap-2.5 px-5 py-3.5 rounded-2xl bg-[#1a1a35] text-[#999] border border-[#2a2a50] hover:bg-[#222245] hover:text-white text-sm font-semibold transition-all disabled:opacity-30 shadow-sm">
          <Download className="w-5 h-5" /> Export JSON
        </button>
        <button onClick={() => loadJsonRef.current?.click()} disabled={!bundle} className="w-full flex items-center justify-center gap-2.5 px-5 py-3.5 rounded-2xl bg-[#1a1a35] text-[#999] border border-[#2a2a50] hover:bg-[#222245] hover:text-white text-sm font-semibold transition-all disabled:opacity-30 shadow-sm">
          <Upload className="w-5 h-5" /> Load JSON
        </button>
        <input ref={loadJsonRef} type="file" accept=".json" onChange={loadJson} className="hidden" />
        <button onClick={handleClear} disabled={!selectedClipId || !bundle} className="w-full flex items-center justify-center gap-2.5 px-5 py-3.5 rounded-2xl bg-[#1a1a35] text-[#555] border border-[#2a2a50] hover:bg-red-500/10 hover:text-red-400 hover:border-red-500/30 text-sm font-semibold transition-all disabled:opacity-30 shadow-sm">
          <Trash2 className="w-5 h-5" /> Clear All Annotations
        </button>
      </div>
      </div>
    </aside>
  )
}
