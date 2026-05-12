// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { Fragment, useRef, useEffect, useCallback, useMemo, useState } from 'react'
import { useStore } from '../lib/store'
import {
  annotationToSegments,
  buildTrackList,
  applySegmentTimeUpdate,
  parseTs,
  addSegmentToAnnotation,
  addConditionToEnv,
  addStateToObject,
  addStateToLight,
  addContainmentToLight,
  addPhysicalContainmentToLight,
  addActionToAgent,
  addPoseToAgent,
  addContainmentToEgo,
  addContainmentToAgent,
  addContainmentToObject,
  addInfluenceToEgo,
  addInfluenceToAgent,
  addPropertyToAgent,
  addPropertyToEgo,
  moveSegmentToTrack,
  removeTrackAndReassign,
  getTrackGroup,
  clampToAvoidOverlap,
  syncInfluencedAgentIds,
  cleanupDeletedIds,
  cleanupDeletedEnv,
  type DynamicGroup,
} from '../lib/timeline-utils'
import { removeKeypointAt } from '../lib/keypoint-utils'
import { buildCompletenessMap } from '../lib/completeness'
import { ACTION_LINK_TO_CONFIG } from '../lib/attribute-cycling'
import type { TimelineSegment, TrackId, SilAvAnnotation } from '../lib/types'
import { ZoomIn, ZoomOut, Maximize2, Plus, Minus, ChevronRight, ChevronDown } from 'lucide-react'
import * as Tooltip from '@radix-ui/react-tooltip'

const TRACK_HEIGHT = 32
const ENV_TRACK_HEIGHT = 50
const EGO_TRACK_HEIGHT = 42
const EGO_SUBTRACK_BASE = 24
const AGENT_TRACK_HEIGHT = 84
const HEADER_HEIGHT = 36
const SCRUBBER_LANE_HEIGHT = 16
const LABEL_PANEL_WIDTH = 140
const COND_SUBTRACK_HEIGHT = 18
const ENV_MAIN_HEIGHT = 30
const AGENT_SUBTRACK_BASE = ENV_MAIN_HEIGHT  // 30 — subtracks start below the main bar
// Object tracks: main + N containment rows + 1 state row (no gap). State row Y
// depends on the number of containment rows for that track.
const OBJ_CONT_ROW_Y = ENV_MAIN_HEIGHT                          // 30 (containment first)
const getObjStateRowY = (contRows: number) => OBJ_CONT_ROW_Y + Math.max(1, contRows) * COND_SUBTRACK_HEIGHT
const getObjTrackHeight = (contRows: number) => getObjStateRowY(contRows) + COND_SUBTRACK_HEIGHT
const OBJ_LIGHT_TRACK_HEIGHT = getObjTrackHeight(1) // 66 — single-containment-row default (used for lights)
// Light tracks: now have signal heads, each with containment + bar + states
// Layout per light: [Main] [PhysCont] [gap] [SH1 bar] [SH1 cont] [SH1 states...] [gap] [SH2 bar] [SH2 cont] [SH2 states...] ...
const SH_GAP = 4
const COLLAPSED_GROUP_HEIGHT = 24
// Height reserved above each non-collapsed group's first track for the
// category header pill ("AGENTS", "OBJECTS", ...). Lives in the gap
// region so the track row's top slot stays free for the track-name row
// alone — otherwise the name row would collide with the first subtrack
// label ("containment") which is absolutely-positioned at
// AGENT_SUBTRACK_BASE / ENV_MAIN_HEIGHT (=30). Collapsed groups already
// render the header as their only row, so no extra space is reserved
// when the next group's first track is collapsed.
const CATEGORY_HEADER_HEIGHT = 24

/** Compute total subtrack rows for a traffic light based on its signal heads. */
function getLightPhysContRows(light: { containment?: unknown[] }): number {
  let max = 0
  for (const c of light.containment || []) {
    const idx = (c as Record<string, unknown>)._cont_track_index as number ?? 0
    if (idx + 1 > max) max = idx + 1
  }
  return Math.max(1, max) // minimum 1 row reserved
}

function getLightSubtrackRows(light: { signal_heads?: { state_sequence?: unknown[]; env_controlled?: unknown[] }[]; containment?: unknown[] }, physContOverride?: number): number {
  let rows = physContOverride ?? getLightPhysContRows(light)
  for (const sh of light.signal_heads || []) {
    const stateLanes = Math.max(1, (sh as unknown as Record<string, unknown>)._state_lane_count as number || 1)
    rows += 1 + stateLanes + 1 // SH bar + state lanes + cont row
  }
  return rows
}

/** Find which signal head component (bar/state/cont) is at a given localY within a light track. */
function findLightHeadAtY(light: { signal_heads?: { state_sequence?: unknown[]; env_controlled?: unknown[] }[]; containment?: unknown[] }, localY: number, maxPhysContRows: number): {
  headIdx: number; inSHBar: boolean; inState: boolean; inCont: boolean; stateLane: number
} | null {
  let y = ENV_MAIN_HEIGHT + maxPhysContRows * COND_SUBTRACK_HEIGHT // skip the physical containment rows
  for (let hi = 0; hi < (light.signal_heads || []).length; hi++) {
    const sh = light.signal_heads![hi]
    const stateLanes = Math.max(1, (sh as unknown as Record<string, unknown>)._state_lane_count as number || 1)
    y += SH_GAP // gap before this signal head
    const shBarEnd = y + COND_SUBTRACK_HEIGHT
    const contEnd = shBarEnd + COND_SUBTRACK_HEIGHT
    const stateEnd = contEnd + stateLanes * COND_SUBTRACK_HEIGHT
    if (localY >= y && localY < shBarEnd) return { headIdx: hi, inSHBar: true, inState: false, inCont: false, stateLane: 0 }
    if (localY >= shBarEnd && localY < contEnd) return { headIdx: hi, inSHBar: false, inState: false, inCont: true, stateLane: 0 }
    if (localY >= contEnd && localY < stateEnd) {
      const lane = Math.floor((localY - contEnd) / COND_SUBTRACK_HEIGHT)
      return { headIdx: hi, inSHBar: false, inState: true, inCont: false, stateLane: lane }
    }
    y = stateEnd
  }
  return null
}

/** Get the Y offset (from track top) of a signal head's SH bar within a light. */
function getSignalHeadY(light: { signal_heads?: { state_sequence?: unknown[]; env_controlled?: unknown[] }[]; containment?: unknown[] }, headIdx: number, maxPhysContRows: number): number {
  let y = ENV_MAIN_HEIGHT + maxPhysContRows * COND_SUBTRACK_HEIGHT // after physical containment rows
  for (let hi = 0; hi < headIdx; hi++) {
    const sh = light.signal_heads![hi]
    const stateLanes = Math.max(1, (sh as unknown as Record<string, unknown>)._state_lane_count as number || 1)
    y += SH_GAP + (2 + stateLanes) * COND_SUBTRACK_HEIGHT
  }
  return y + SH_GAP // gap before this head's bar
}

const GROUP_GAP = 6
const TRACK_GAP = 4
const BOTTOM_PADDING = 40

// Canvas colors are resolved from CSS variables so they flip with the
// theme. The hard-coded fallbacks here only apply when the canvas paints
// during SSR or before the document is ready (impossible in practice but
// kept for safety).
const CANVAS_BG_FALLBACK = '#111125'
const TRACK_BG_A_FALLBACK = '#151530'
const TRACK_BG_B_FALLBACK = '#181840'
const RULER_BG_FALLBACK = '#0d0d1a'
const SEGMENT_RADIUS = 3
const LABEL_MIN_WIDTH = 40

function readCssColor(name: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return v || fallback
}

function adjustColor(hex: string, amount: number): string {
  const num = parseInt(hex.slice(1), 16)
  const r = Math.max(0, Math.min(255, ((num >> 16) & 0xff) + amount))
  const g = Math.max(0, Math.min(255, ((num >> 8) & 0xff) + amount))
  const b = Math.max(0, Math.min(255, (num & 0xff) + amount))
  return `rgb(${r},${g},${b})`
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.lineTo(x + w - r, y)
  ctx.quadraticCurveTo(x + w, y, x + w, y + r)
  ctx.lineTo(x + w, y + h - r)
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h)
  ctx.lineTo(x + r, y + h)
  ctx.quadraticCurveTo(x, y + h, x, y + h - r)
  ctx.lineTo(x, y + r)
  ctx.quadraticCurveTo(x, y, x + r, y)
  ctx.closePath()
}

function getTrackHeight(group: string, trackId?: string, lightSubRowCounts?: Map<string, number>, collapsed?: boolean): number {
  if (collapsed) return COLLAPSED_GROUP_HEIGHT
  if (group === 'Agents') {
    if (trackId && lightSubRowCounts) {
      const contRows = lightSubRowCounts.get(trackId) ?? 1
      const inflRows = lightSubRowCounts.get(`${trackId}_infl`) ?? 0
      const propRows = lightSubRowCounts.get(`${trackId}_prop`) ?? 0
      return AGENT_SUBTRACK_BASE + (contRows + inflRows + 1 + propRows) * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT
    }
    return AGENT_TRACK_HEIGHT
  }
  if (group === 'Ego') {
    if (lightSubRowCounts) {
      const contRows = lightSubRowCounts.get('ego_act') ?? 1
      const inflRows = lightSubRowCounts.get('ego_act_infl') ?? 0
      const propRows = lightSubRowCounts.get('ego_act_prop') ?? 0
      return EGO_SUBTRACK_BASE + contRows * COND_SUBTRACK_HEIGHT + inflRows * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT + propRows * COND_SUBTRACK_HEIGHT
    }
    return EGO_TRACK_HEIGHT
  }
  if (group === 'Environments') {
    if (trackId && lightSubRowCounts) {
      const condRows = lightSubRowCounts.get(`${trackId}_cond`) ?? 1
      return ENV_MAIN_HEIGHT + condRows * COND_SUBTRACK_HEIGHT + 2
    }
    return ENV_TRACK_HEIGHT
  }
  if (group === 'TrafficLights' && trackId && lightSubRowCounts) {
    const rows = lightSubRowCounts.get(trackId) ?? 2
    const shCount = lightSubRowCounts.get(`${trackId}_shcount`) ?? 0
    return ENV_MAIN_HEIGHT + rows * COND_SUBTRACK_HEIGHT + shCount * SH_GAP
  }
  if (group === 'Objects') {
    if (trackId && lightSubRowCounts) {
      const contRows = lightSubRowCounts.get(trackId) ?? 1
      return getObjTrackHeight(contRows)
    }
    return OBJ_LIGHT_TRACK_HEIGHT
  }
  if (group === 'TrafficLights') return OBJ_LIGHT_TRACK_HEIGHT
  return TRACK_HEIGHT
}

/** True when the main track row has any parent entity assigned.
 * Subtrack labels and dividers are only shown when this is true. Ego is
 * treated as always having a main-track entity (the ego_vehicle). */
function hasMainTrackEntity(group: string, trackId: string, ann?: SilAvAnnotation): boolean {
  if (group === 'Ego') return !!ann?.ego_vehicle
  const m = trackId.match(/_(\d+)$/)
  const trackIdx = m ? parseInt(m[1], 10) : 0
  if (group === 'Environments') return (ann?.environments || []).some(e => (e._track_index ?? 0) === trackIdx)
  if (group === 'Objects') return (ann?.traffic_objects || []).some(o => (o._track_index ?? 0) === trackIdx)
  if (group === 'TrafficLights') return (ann?.traffic_lights || []).some(l => (l._track_index ?? 0) === trackIdx)
  if (group === 'Agents') return (ann?.agents || []).some(a => (a._track_index ?? 0) === trackIdx)
  return false
}

/** Returns a list of (y, label, color) descriptors for subtrack row labels
 * to display in the label panel, one per visible subtrack row. Labels are
 * emitted for every subtrack row whenever the main track has at least one
 * parent entity — even if the particular subtrack is empty. When the main
 * track has no parent entity, no labels are emitted. */
function getSubtrackRowLabels(group: string, trackId: string, lightLaneCounts: Map<string, number>, ann?: SilAvAnnotation): { y: number; label: string; color: string }[] {
  const out: { y: number; label: string; color: string }[] = []
  if (!hasMainTrackEntity(group, trackId, ann)) return out
  if (group === 'Environments') {
    const condRows = lightLaneCounts.get(`${trackId}_cond`) ?? 1
    for (let i = 0; i < condRows; i++) out.push({ y: ENV_MAIN_HEIGHT + i * COND_SUBTRACK_HEIGHT, label: i === 0 ? 'conditions' : '', color: '#06b6d4' })
  } else if (group === 'Ego') {
    const contRows = lightLaneCounts.get('ego_act') ?? 1
    const inflRows = lightLaneCounts.get('ego_act_infl') ?? 0
    const propRows = lightLaneCounts.get('ego_act_prop') ?? 0
    let y = EGO_SUBTRACK_BASE
    for (let i = 0; i < contRows; i++) { out.push({ y, label: i === 0 ? 'containment' : '', color: '#22c55e' }); y += COND_SUBTRACK_HEIGHT }
    for (let i = 0; i < inflRows; i++) { out.push({ y, label: i === 0 ? 'influences' : '', color: '#f97316' }); y += COND_SUBTRACK_HEIGHT }
    out.push({ y, label: 'actions', color: '#3b82f6' }); y += COND_SUBTRACK_HEIGHT
    for (let i = 0; i < propRows; i++) { out.push({ y, label: i === 0 ? 'properties' : '', color: '#93c5fd' }); y += COND_SUBTRACK_HEIGHT }
  } else if (group === 'Agents') {
    const contRows = lightLaneCounts.get(trackId) ?? 1
    const inflRows = lightLaneCounts.get(`${trackId}_infl`) ?? 0
    const propRows = lightLaneCounts.get(`${trackId}_prop`) ?? 0
    let y = AGENT_SUBTRACK_BASE
    for (let i = 0; i < contRows; i++) { out.push({ y, label: i === 0 ? 'containment' : '', color: '#22c55e' }); y += COND_SUBTRACK_HEIGHT }
    out.push({ y, label: 'pose', color: '#14b8a6' }); y += COND_SUBTRACK_HEIGHT
    for (let i = 0; i < inflRows; i++) { out.push({ y, label: i === 0 ? 'influences' : '', color: '#f97316' }); y += COND_SUBTRACK_HEIGHT }
    out.push({ y, label: 'actions', color: '#c084fc' }); y += COND_SUBTRACK_HEIGHT
    for (let i = 0; i < propRows; i++) { out.push({ y, label: i === 0 ? 'properties' : '', color: '#d8b4fe' }); y += COND_SUBTRACK_HEIGHT }
  } else if (group === 'Objects') {
    const contRows = lightLaneCounts.get(trackId) ?? 1
    for (let i = 0; i < contRows; i++) out.push({ y: OBJ_CONT_ROW_Y + i * COND_SUBTRACK_HEIGHT, label: i === 0 ? 'containment' : '', color: '#22c55e' })
    out.push({ y: getObjStateRowY(contRows), label: 'states', color: '#d97706' })
  } else if (group === 'TrafficLights') {
    const physContMax = lightLaneCounts.get(`${trackId}_physcontmax`) ?? 1
    for (let i = 0; i < physContMax; i++) out.push({ y: ENV_MAIN_HEIGHT + i * COND_SUBTRACK_HEIGHT, label: i === 0 ? 'TL control' : '', color: '#22c55e' })
    // Signal heads: use the light with the most signal heads as reference (matches divider logic)
    const divLightIdx = lightLaneCounts.get(`${trackId}_divlight`)
    const light = ann?.traffic_lights?.[divLightIdx ?? -1]
    const shCount = light ? (light.signal_heads || []).length : (lightLaneCounts.get(`${trackId}_shcount`) ?? 0)
    for (let hi = 0; hi < shCount; hi++) {
      const shY = light ? getSignalHeadY(light, hi, physContMax) : ENV_MAIN_HEIGHT + physContMax * COND_SUBTRACK_HEIGHT + SH_GAP + hi * 3 * COND_SUBTRACK_HEIGHT
      out.push({ y: shY, label: 'signal head', color: '#f97316' })
      out.push({ y: shY + COND_SUBTRACK_HEIGHT, label: 'containment', color: '#22c55e' })
      const stateLanes = light ? Math.max(1, (light.signal_heads![hi] as unknown as Record<string, unknown>)._state_lane_count as number || 1) : 1
      for (let sl = 0; sl < stateLanes; sl++) out.push({ y: shY + (2 + sl) * COND_SUBTRACK_HEIGHT, label: sl === 0 ? 'states' : '', color: '#dc2626' })
    }
  }
  return out
}

/** Gap after track i (group gap or agent-to-agent gap).
 *
 * When the next track starts a new group and is non-collapsed, the gap
 * is widened by CATEGORY_HEADER_HEIGHT so the panel can render the
 * group's header pill in that space rather than steal space from the
 * track row itself (which would collide with the first subtrack label).
 * Collapsed groups already use their single row as their header, so no
 * extra space is reserved when the next track is collapsed.
 */
function getTrackGap(trackList: { group: string; id: string; _collapsed?: boolean }[], i: number): number {
  if (i + 1 >= trackList.length) return 0
  if (trackList[i].group !== trackList[i + 1].group) {
    return GROUP_GAP + (trackList[i + 1]._collapsed ? 0 : CATEGORY_HEADER_HEIGHT)
  }
  if (!trackList[i]._collapsed && trackList[i].group === trackList[i + 1].group) return TRACK_GAP
  return 0
}

function getTrackY(trackList: { group: string; id: string; _collapsed?: boolean }[], rowIdx: number, lightLaneCounts?: Map<string, number>): number {
  let y = HEADER_HEIGHT
  // The first track is always first-in-group; reserve header space for
  // non-collapsed groups so the canvas paint and panel render stay aligned.
  if (trackList.length > 0 && !trackList[0]._collapsed) y += CATEGORY_HEADER_HEIGHT
  for (let i = 0; i < rowIdx; i++) {
    y += getTrackHeight(trackList[i].group, trackList[i].id, lightLaneCounts, trackList[i]._collapsed)
    y += getTrackGap(trackList, i)
  }
  return y
}

/**
 * Find a gap adjacent to existing conditions within env bounds to place a new condition.
 * Returns {t0, t1} or null if no space.
 */
function findAdjacentCondGap(
  condSegs: TimelineSegment[],
  envT0: number,
  envT1: number,
  clickT: number
): { t0: number; t1: number } | null {
  const DEFAULT_LEN = 2.0

  // Right-click on an existing segment: do nothing. Users opt-in to creation
  // by clicking in empty space on the row.
  for (const cs of condSegs) {
    if (clickT >= cs.t0 && clickT <= cs.t1) return null
  }

  if (condSegs.length === 0) {
    // Place a DEFAULT_LEN segment starting at click, clamped to bounds
    let t0 = Math.max(envT0, clickT)
    let t1 = t0 + DEFAULT_LEN
    if (t1 > envT1) { t1 = envT1; t0 = Math.max(envT0, t1 - DEFAULT_LEN) }
    return { t0, t1 }
  }

  // Build gaps within env bounds between/around existing conditions
  const gaps: [number, number][] = []
  let prev = envT0
  for (const cs of condSegs) {
    if (cs.t0 > prev + 0.05) gaps.push([prev, cs.t0])
    prev = Math.max(prev, cs.t1)
  }
  if (prev < envT1 - 0.05) gaps.push([prev, envT1])

  if (gaps.length === 0) return null

  // Pick the gap that contains or is closest to the click time
  let bestGap = gaps[0]
  let bestDist = Infinity
  for (const [g0, g1] of gaps) {
    const dist = clickT >= g0 && clickT <= g1 ? 0 : Math.min(Math.abs(g0 - clickT), Math.abs(g1 - clickT))
    if (dist < bestDist) { bestDist = dist; bestGap = [g0, g1] }
  }

  const gapLen = bestGap[1] - bestGap[0]
  if (gapLen <= DEFAULT_LEN) {
    return { t0: bestGap[0], t1: bestGap[1] }
  }
  // Place starting at click time, clamped within the gap
  let t0 = Math.max(bestGap[0], clickT)
  let t1 = t0 + DEFAULT_LEN
  if (t1 > bestGap[1]) { t1 = bestGap[1]; t0 = Math.max(bestGap[0], t1 - DEFAULT_LEN) }
  return { t0, t1 }
}

function drawBecauseOfCurve(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number, color: string, label: string, glow = false) {
  const midX = (x1 + x2) / 2
  const ctrlY = Math.min(y1, y2) - 30
  const angle = Math.atan2(y2 - ctrlY, x2 - midX)
  const arrowLen = 8
  if (glow) {
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 4.5
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.quadraticCurveTo(midX, ctrlY, x2, y2); ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(x2, y2); ctx.lineTo(x2 - arrowLen * Math.cos(angle - 0.4), y2 - arrowLen * Math.sin(angle - 0.4))
    ctx.moveTo(x2, y2); ctx.lineTo(x2 - arrowLen * Math.cos(angle + 0.4), y2 - arrowLen * Math.sin(angle + 0.4))
    ctx.stroke()
    ctx.lineCap = 'butt'
    ctx.lineJoin = 'miter'
  }
  ctx.strokeStyle = color
  ctx.lineWidth = glow ? 3 : 2
  ctx.beginPath()
  ctx.moveTo(x1, y1)
  ctx.quadraticCurveTo(midX, ctrlY, x2, y2)
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(x2, y2)
  ctx.lineTo(x2 - arrowLen * Math.cos(angle - 0.4), y2 - arrowLen * Math.sin(angle - 0.4))
  ctx.moveTo(x2, y2)
  ctx.lineTo(x2 - arrowLen * Math.cos(angle + 0.4), y2 - arrowLen * Math.sin(angle + 0.4))
  ctx.stroke()
  ctx.font = '8px sans-serif'
  const tw = ctx.measureText(label).width
  const pillW = tw + 8
  const pillX = midX - pillW / 2
  const apexY = 0.25 * (y1 + y2) + 0.5 * ctrlY
  const pillY = apexY - 6
  ctx.fillStyle = color
  ctx.fillRect(pillX, pillY, pillW, 12)
  ctx.fillStyle = '#fff'
  ctx.textAlign = 'center'
  ctx.fillText(label, midX, pillY + 9)
}

// --- Toolbar helpers ---------------------------------------------------------

function ZoomButton({
  icon: Icon,
  label,
  onClick,
}: {
  icon: React.ComponentType<{ className?: string }>
  label: string
  onClick: () => void
}) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>
        <button
          type="button"
          onClick={onClick}
          aria-label={label}
          className="h-7 px-3 inline-flex items-center gap-1.5 rounded-md text-text-muted hover:text-text-primary hover:bg-surface-hover text-xs transition-colors"
        >
          <Icon className="w-3.5 h-3.5" />
          <span>{label}</span>
        </button>
      </Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          sideOffset={6}
          className="z-50 px-3 py-1.5 rounded-md bg-surface-overlay text-text-primary border border-border-default text-xs font-medium shadow-md"
        >
          {label}
          <Tooltip.Arrow className="fill-[var(--color-border-default)]" />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  )
}

function Kb({ keys, desc }: { keys: string; desc: string }) {
  return (
    <div className="flex items-center gap-2 text-[11px] text-text-muted">
      <kbd className="font-mono text-[10px] h-6 px-3 inline-flex items-center bg-surface-overlay border border-border-default rounded-md text-text-secondary">
        {keys}
      </kbd>
      <span>{desc}</span>
    </div>
  )
}

export function Timeline() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const {
    bundle, selectedClipId, duration, playheadTime, selectedPath,
    zoomLevel, scrollOffset,
    envTrackCount, objectTrackCount, lightTrackCount, agentTrackCount, egoContTrackCount,
    arrowTypes, keypointsVisible,
    effectiveTheme,
    setPlayhead, selectPath, updateBundle, undo, setZoom, setScroll,
    setEnvTrackCount, setObjectTrackCount, setLightTrackCount, setAgentTrackCount, setEgoContTrackCount: _setEgoContTrackCount,
  } = useStore()

  const scrollOffsetRef = useRef(scrollOffset)
  useEffect(() => { scrollOffsetRef.current = scrollOffset }, [scrollOffset])

  const getLiveScrollOffset = useCallback(() => {
    const el = containerRef.current
    if (!el) return scrollOffsetRef.current
    const pps = (el.clientWidth * zoomLevel) / Math.max(duration, 1)
    return pps > 0 ? el.scrollLeft / pps : scrollOffsetRef.current
  }, [duration, zoomLevel])

  const commitNativeScroll = useCallback(() => {
    const liveScrollOffset = getLiveScrollOffset()
    scrollOffsetRef.current = liveScrollOffset
    const currentScrollOffset = useStore.getState().scrollOffset
    if (Math.abs(currentScrollOffset - liveScrollOffset) > 0.01) setScroll(liveScrollOffset)
  }, [getLiveScrollOffset, setScroll])

  const scrollSyncRafRef = useRef<number | null>(null)
  const scheduleNativeScrollCommit = useCallback(() => {
    if (scrollSyncRafRef.current != null) return
    scrollSyncRafRef.current = window.requestAnimationFrame(() => {
      scrollSyncRafRef.current = null
      commitNativeScroll()
    })
  }, [commitNativeScroll])

  const pendingScrubTimeRef = useRef<number | null>(null)
  const scrubRafRef = useRef<number | null>(null)
  const scrubTo = useCallback((time: number) => {
    pendingScrubTimeRef.current = Math.max(0, Math.min(duration, time))
    if (scrubRafRef.current != null) return
    scrubRafRef.current = window.requestAnimationFrame(() => {
      scrubRafRef.current = null
      const next = pendingScrubTimeRef.current
      pendingScrubTimeRef.current = null
      if (next != null) setPlayhead(next)
    })
  }, [duration, setPlayhead])

  useEffect(() => {
    return () => {
      if (scrollSyncRafRef.current != null) window.cancelAnimationFrame(scrollSyncRafRef.current)
      if (scrubRafRef.current != null) window.cancelAnimationFrame(scrubRafRef.current)
    }
  }, [])

  // No-op until Step 6 wires the explicit Save button. `updateBundle` already
  // flips `dirty` in the store, so callers passing through here just need
  // their save promise to resolve.
  const guardedSave = useCallback((_clipId: string, _data: unknown) => {
    return Promise.resolve()
  }, [])

  const [dragState, setDragState] = useState<{
    seg: TimelineSegment
    mode: 'left' | 'right' | 'move'
    startX: number
    origT0: number
    origT1: number
    curT0: number
    curT1: number
    targetTrackId?: string
  } | null>(null)
  const dragRef = useRef(dragState)
  dragRef.current = dragState
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null)
  const [verticalScroll, setVerticalScroll] = useState(0)
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set())

  const baseSegments = useMemo(() => {
    const segs = annotationToSegments(bundle?.annotation)
    const dur = bundle?.video?.duration_s ?? 0
    if (dur > 0) {
      segs.push({ id: 'ego_main', trackId: 'ego_act', label: 'Ego', t0: 0, t1: dur, meta: { _isEgoMain: true } })
    }
    return segs
  }, [bundle?.annotation, bundle?.video?.duration_s])
  const baseSegmentsRef = useRef(baseSegments); baseSegmentsRef.current = baseSegments

  const segments = useMemo(() => {
    if (dragState) {
      return baseSegments.map((seg) =>
        seg.id === dragState.seg.id
          ? { ...seg, t0: dragState.curT0, t1: dragState.curT1, trackId: dragState.targetTrackId || seg.trackId }
          : seg
      )
    }
    return baseSegments
  }, [baseSegments, dragState])

  const { tracks, allTrackList } = useMemo(() => {
    const byTrack = new Map<TrackId, TimelineSegment[]>()
    for (const seg of segments) {
      const arr = byTrack.get(seg.trackId) || []
      arr.push(seg)
      byTrack.set(seg.trackId, arr)
    }
    const trackList = buildTrackList(bundle?.annotation, envTrackCount, objectTrackCount, agentTrackCount, egoContTrackCount, lightTrackCount)
    return { tracks: byTrack, allTrackList: trackList }
    // effectiveTheme is included so entity colors (read from CSS vars in
    // buildTrackList) flip atomically with the theme.
  }, [segments, bundle?.annotation, envTrackCount, objectTrackCount, lightTrackCount, agentTrackCount, egoContTrackCount, effectiveTheme])

  const trackList = useMemo(() => {
    if (collapsedGroups.size === 0) return allTrackList
    const result: typeof allTrackList = []
    const seenCollapsed = new Set<string>()
    for (const track of allTrackList) {
      if (collapsedGroups.has(track.group)) {
        if (!seenCollapsed.has(track.group)) {
          seenCollapsed.add(track.group)
          result.push({ ...track, _collapsed: true })
        }
      } else {
        result.push(track)
      }
    }
    return result
  }, [allTrackList, collapsedGroups])

  const toggleGroupCollapse = useCallback((group: string) => {
    setCollapsedGroups(prev => {
      const next = new Set(prev)
      if (next.has(group)) next.delete(group)
      else next.add(group)
      return next
    })
  }, [])

  const completenessMap = useMemo(() => buildCompletenessMap(bundle?.annotation), [bundle?.annotation])

  const lightLaneCounts = useMemo(() => {
    const map = new Map<string, number>()
    // First pass: compute max phys cont rows and max signal head count per track
    for (const light of bundle?.annotation?.traffic_lights || []) {
      const trackId = `light_${light._track_index ?? 0}`
      const shCount = (light.signal_heads || []).length
      map.set(`${trackId}_shcount`, Math.max(map.get(`${trackId}_shcount`) ?? 0, shCount))
      const physCont = getLightPhysContRows(light)
      map.set(`${trackId}_physcontmax`, Math.max(map.get(`${trackId}_physcontmax`) ?? 1, physCont))
    }
    // Second pass: compute total subtrack rows using the track-level max phys cont rows,
    // and find the light with the most rows per track (for divider drawing)
    const lights = bundle?.annotation?.traffic_lights || []
    for (let li = 0; li < lights.length; li++) {
      const light = lights[li]
      const trackId = `light_${light._track_index ?? 0}`
      const trackPhysCont = map.get(`${trackId}_physcontmax`) ?? 1
      const rows = getLightSubtrackRows(light, trackPhysCont)
      if (rows > (map.get(trackId) ?? 2)) {
        map.set(`${trackId}_divlight`, li)
      }
      map.set(trackId, Math.max(map.get(trackId) ?? 2, rows))
    }
    // Condition row counts per environment track
    {
      const envIdToTrackIdx = new Map<string, number>()
      for (const env of bundle?.annotation?.environments || []) {
        envIdToTrackIdx.set(env.id, env._track_index ?? 0)
      }
      for (const cond of bundle?.annotation?.conditions || []) {
        const trackIdx = envIdToTrackIdx.get(cond.env_id) ?? 0
        const key = `env_${trackIdx}_cond`
        const idx = ((cond as unknown as Record<string, unknown>)._cond_track_index as number) ?? 0
        map.set(key, Math.max(map.get(key) ?? 1, idx + 1))
      }
      // Ensure every env track has at least 1 cond row reserved
      for (const env of bundle?.annotation?.environments || []) {
        const key = `env_${env._track_index ?? 0}_cond`
        map.set(key, Math.max(map.get(key) ?? 1, 1))
      }
    }
    // Ego containment row count
    let egoContMax = 0
    for (const cont of bundle?.annotation?.ego_vehicle?.containment || []) {
      egoContMax = Math.max(egoContMax, ((cont as unknown as Record<string, unknown>)._cont_track_index as number ?? 0) + 1)
    }
    map.set('ego_act', Math.max(1, egoContMax))
    // Ego influence row count
    let egoInflMax = 0
    for (const infl of bundle?.annotation?.ego_vehicle?.influenced_by || []) {
      egoInflMax = Math.max(egoInflMax, ((infl as unknown as Record<string, unknown>)._influence_track_index as number ?? 0) + 1)
    }
    map.set('ego_act_infl', Math.max(1, egoInflMax))
    // Agent containment row counts per track
    for (const agent of bundle?.annotation?.agents || []) {
      const trackId = `agent_${agent._track_index ?? 0}`
      for (const cont of agent.containment || []) {
        const idx = ((cont as unknown as Record<string, unknown>)._cont_track_index as number) ?? 0
        map.set(trackId, Math.max(map.get(trackId) ?? 1, idx + 1))
      }
    }
    // Object containment row counts per track (min 1 row reserved)
    for (const obj of bundle?.annotation?.traffic_objects || []) {
      const trackId = `obj_${obj._track_index ?? 0}`
      map.set(trackId, Math.max(map.get(trackId) ?? 1, 1))
      for (const cont of obj.containment || []) {
        const idx = ((cont as unknown as Record<string, unknown>)._cont_track_index as number) ?? 0
        map.set(trackId, Math.max(map.get(trackId) ?? 1, idx + 1))
      }
    }
    // Agent influence row counts per track
    for (const agent of bundle?.annotation?.agents || []) {
      const trackId = `agent_${agent._track_index ?? 0}`
      let agentInflMax = 0
      for (const infl of (agent as unknown as Record<string, unknown>).influenced_by as unknown[] || []) {
        agentInflMax = Math.max(agentInflMax, ((infl as Record<string, unknown>)._influence_track_index as number ?? 0) + 1)
      }
      map.set(`${trackId}_infl`, Math.max(1, map.get(`${trackId}_infl`) ?? 0, agentInflMax))
    }
    // Ego property row count
    let egoPropMax = 0
    for (const prop of bundle?.annotation?.ego_vehicle?.properties || []) {
      egoPropMax = Math.max(egoPropMax, ((prop as unknown as Record<string, unknown>)._prop_track_index as number ?? 0) + 1)
    }
    map.set('ego_act_prop', Math.max(1, egoPropMax))
    // Agent property row counts per track
    for (const agent of bundle?.annotation?.agents || []) {
      const trackId = `agent_${agent._track_index ?? 0}`
      let agentPropMax = 0
      for (const prop of agent.properties || []) {
        agentPropMax = Math.max(agentPropMax, ((prop as unknown as Record<string, unknown>)._prop_track_index as number ?? 0) + 1)
      }
      const key = `${trackId}_prop`
      map.set(key, Math.max(1, map.get(key) ?? 0, agentPropMax))
    }
    return map
  }, [bundle?.annotation])

  // --- Persist helper ---
  const persistBundle = useCallback((newBundle: typeof bundle) => {
    if (!newBundle) return
    updateBundle(newBundle)
    if (selectedClipId) guardedSave(selectedClipId, newBundle).catch(() => {})
  }, [selectedClipId, updateBundle, guardedSave])

  // --- Track add/remove ---
  const handleAddTrack = useCallback((group: DynamicGroup) => {
    if (useStore.getState().editsBlocked()) return
    if (group === 'Environments') setEnvTrackCount(envTrackCount + 1)
    else if (group === 'Objects') setObjectTrackCount(objectTrackCount + 1)
    else if (group === 'TrafficLights') setLightTrackCount(lightTrackCount + 1)
    else setAgentTrackCount(agentTrackCount + 1)
  }, [envTrackCount, objectTrackCount, lightTrackCount, agentTrackCount, setEnvTrackCount, setObjectTrackCount, setLightTrackCount, setAgentTrackCount])

  const handleRemoveTrack = useCallback((group: DynamicGroup, trackIdx: number) => {
    if (useStore.getState().editsBlocked()) return
    const count = group === 'Environments' ? envTrackCount : group === 'Objects' ? objectTrackCount : group === 'TrafficLights' ? lightTrackCount : agentTrackCount
    if (count <= 1) return
    const ann = bundle?.annotation
    if (ann) {
      const updated = removeTrackAndReassign(ann, group, trackIdx)
      persistBundle({ ...bundle!, annotation: updated })
    }
    if (group === 'Environments') setEnvTrackCount(count - 1)
    else if (group === 'Objects') setObjectTrackCount(count - 1)
    else if (group === 'TrafficLights') setLightTrackCount(count - 1)
    else setAgentTrackCount(count - 1)
    selectPath(null)
  }, [bundle, envTrackCount, objectTrackCount, lightTrackCount, agentTrackCount, persistBundle, selectPath, setEnvTrackCount, setObjectTrackCount, setLightTrackCount, setAgentTrackCount])

  // Canvas palette resolved from CSS variables. Keyed on effectiveTheme so
  // theme flips produce a fresh palette and force a redraw via the dep
  // array of the draw effect.
  const canvasColors = useMemo(() => ({
    bg: readCssColor('--color-surface-raised', CANVAS_BG_FALLBACK),
    trackA: readCssColor('--color-surface-overlay', TRACK_BG_A_FALLBACK),
    trackB: readCssColor('--color-surface-hover', TRACK_BG_B_FALLBACK),
    ruler: readCssColor('--color-surface-sunken', RULER_BG_FALLBACK),
    raised: readCssColor('--color-surface-raised', '#13132a'),
    muted: readCssColor('--color-text-muted', '#8a8aaa'),
  }), [effectiveTheme])

  // --- Drawing ---
  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let rafId: number
    const draw = () => {
      const dpr = window.devicePixelRatio || 1
      const w = container.clientWidth
      const h = container.clientHeight
      canvas.width = w * dpr
      canvas.height = h * dpr
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)
      ctx.fillStyle = canvasColors.bg
      ctx.fillRect(0, 0, w, h)

      const pps = (w * zoomLevel) / Math.max(duration, 1)
      const liveScrollOffset = getLiveScrollOffset()
      const t2x = (t: number) => (t - liveScrollOffset) * pps
      const drawGetTrackY = (rowIdx: number) => getTrackY(trackList, rowIdx, lightLaneCounts) - verticalScroll

      // Header — split into scrubber lane (top) + ruler lane (bottom)
      const rulerTop = SCRUBBER_LANE_HEIGHT
      ctx.fillStyle = canvasColors.raised
      ctx.fillRect(0, 0, w, SCRUBBER_LANE_HEIGHT)
      ctx.fillStyle = canvasColors.ruler
      ctx.fillRect(0, rulerTop, w, HEADER_HEIGHT - rulerTop)
      ctx.strokeStyle = canvasColors.trackA
      ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(0, rulerTop + 0.5); ctx.lineTo(w, rulerTop + 0.5); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(0, HEADER_HEIGHT + 0.5); ctx.lineTo(w, HEADER_HEIGHT + 0.5); ctx.stroke()
      // Scrubber rail
      const railY = SCRUBBER_LANE_HEIGHT / 2
      const railX0 = t2x(0)
      const railX1 = t2x(duration)
      ctx.strokeStyle = '#3a3a5a'; ctx.lineWidth = 2
      ctx.lineCap = 'round'
      ctx.beginPath(); ctx.moveTo(Math.max(0, railX0), railY); ctx.lineTo(Math.min(w, railX1), railY); ctx.stroke()
      ctx.lineCap = 'butt'
      ctx.font = '10px sans-serif'
      ctx.textAlign = 'center'
      for (let t = 0; t <= duration; t += 0.5) {
        const x = t2x(t)
        if (x >= -20 && x <= w + 20) {
          const major = t % 1 === 0
          ctx.strokeStyle = major ? canvasColors.trackB : canvasColors.trackA
          const tickLen = major ? 5 : 3
          ctx.beginPath(); ctx.moveTo(x, rulerTop); ctx.lineTo(x, rulerTop + tickLen); ctx.stroke()
          if (major) {
            ctx.fillStyle = canvasColors.muted
            const label = t === 0 || t + 1 > duration ? `${t}` : `${t}s`
            const halfW = ctx.measureText(label).width / 2
            ctx.textAlign = x - halfW < 0 ? 'left' : x + halfW > w ? 'right' : 'center'
            ctx.fillText(label, x, rulerTop + 15)
            ctx.textAlign = 'center'
          }
        }
      }

      // Clip subsequent track/segment drawing so it cannot bleed into the header.
      ctx.save()
      ctx.beginPath()
      ctx.rect(0, HEADER_HEIGHT, w, Math.max(0, h - HEADER_HEIGHT))
      ctx.clip()

      // Track backgrounds
      trackList.forEach((track, rowIdx) => {
        const y = drawGetTrackY(rowIdx)
        const h = getTrackHeight(track.group, track.id, lightLaneCounts, track._collapsed)
        // Draw gap above track (group gap or agent track gap)
        if (rowIdx > 0) {
          const gap = getTrackGap(trackList, rowIdx - 1)
          if (gap > 0) { ctx.fillStyle = canvasColors.bg; ctx.fillRect(0, y - gap, w, gap) }
        }
        ctx.fillStyle = rowIdx % 2 === 0 ? canvasColors.trackA : canvasColors.trackB
        ctx.fillRect(0, y, w, h)
        ctx.strokeStyle = '#1a1a35'; ctx.lineWidth = 1; ctx.strokeRect(0, y, w, h)
        if (track._collapsed) {
          // Draw collapsed group indicator with colored accent
          const midY = y + h / 2
          // Colored left accent bar
          ctx.fillStyle = track.color
          ctx.globalAlpha = 0.5
          ctx.fillRect(0, y + 2, 3, h - 4)
          ctx.globalAlpha = 1
          // Dashed center line
          ctx.strokeStyle = track.color
          ctx.globalAlpha = 0.25
          ctx.lineWidth = 1
          ctx.setLineDash([4, 4])
          ctx.beginPath(); ctx.moveTo(10, midY); ctx.lineTo(w - 10, midY); ctx.stroke()
          ctx.setLineDash([])
          ctx.globalAlpha = 1
          // Chevron icon (▶) and label
          ctx.fillStyle = track.color
          ctx.globalAlpha = 0.6
          ctx.font = 'bold 9px sans-serif'
          ctx.textAlign = 'left'
          ctx.fillText(`▶ ${track.group}`, 10, midY + 3)
          ctx.globalAlpha = 1
          return
        }
        // Draw subtrack divider lines — only when the main track has a parent entity
        const _mainPopulated = hasMainTrackEntity(track.group, track.id, bundle?.annotation)
        if (_mainPopulated && track.group === 'Environments') {
          ctx.strokeStyle = canvasColors.bg; ctx.lineWidth = 0.5
          ctx.beginPath(); ctx.moveTo(0, y + ENV_MAIN_HEIGHT); ctx.lineTo(w, y + ENV_MAIN_HEIGHT); ctx.stroke()
          const envCondRows = lightLaneCounts.get(`${track.id}_cond`) ?? 1
          for (let ri = 1; ri < envCondRows; ri++) {
            const divY = y + ENV_MAIN_HEIGHT + ri * COND_SUBTRACK_HEIGHT
            ctx.beginPath(); ctx.moveTo(0, divY); ctx.lineTo(w, divY); ctx.stroke()
          }
        }
        if (_mainPopulated && track.group === 'Objects') {
          ctx.strokeStyle = canvasColors.bg; ctx.lineWidth = 0.5
          const _objContRows = lightLaneCounts.get(track.id) ?? 1
          const _objStateY = getObjStateRowY(_objContRows)
          ctx.beginPath(); ctx.moveTo(0, y + OBJ_CONT_ROW_Y); ctx.lineTo(w, y + OBJ_CONT_ROW_Y); ctx.stroke()
          for (let ri = 1; ri < _objContRows; ri++) {
            const rowY = y + OBJ_CONT_ROW_Y + ri * COND_SUBTRACK_HEIGHT
            ctx.beginPath(); ctx.moveTo(0, rowY); ctx.lineTo(w, rowY); ctx.stroke()
          }
          ctx.beginPath(); ctx.moveTo(0, y + _objStateY); ctx.lineTo(w, y + _objStateY); ctx.stroke()
        }
        if (_mainPopulated && track.group === 'TrafficLights') {
          ctx.strokeStyle = canvasColors.bg; ctx.lineWidth = 0.5
          // Draw divider at main/subtrack boundary
          ctx.beginPath(); ctx.moveTo(0, y + ENV_MAIN_HEIGHT); ctx.lineTo(w, y + ENV_MAIN_HEIGHT); ctx.stroke()
          // Draw physical containment row dividers
          const trackPhysContMax = lightLaneCounts.get(`${track.id}_physcontmax`) ?? 1
          const divLightIdx = lightLaneCounts.get(`${track.id}_divlight`)
          const lightData = divLightIdx != null ? bundle?.annotation?.traffic_lights?.[divLightIdx] : undefined
          for (let ri = 1; ri <= trackPhysContMax; ri++) {
            const divY = y + ENV_MAIN_HEIGHT + ri * COND_SUBTRACK_HEIGHT
            ctx.beginPath(); ctx.moveTo(0, divY); ctx.lineTo(w, divY); ctx.stroke()
          }
          // Draw per-signal-head dividers with gaps
          if (lightData) {
            for (let hi = 0; hi < (lightData.signal_heads || []).length; hi++) {
              const sh = lightData.signal_heads![hi]
              const stateLanes = Math.max(1, (sh as unknown as Record<string, unknown>)._state_lane_count as number || 1)
              const shY = y + getSignalHeadY(lightData, hi, trackPhysContMax)
              // SH bar / cont / state dividers
              for (let ri = 1; ri <= 2 + stateLanes; ri++) {
                const divY = shY + ri * COND_SUBTRACK_HEIGHT
                ctx.beginPath(); ctx.moveTo(0, divY); ctx.lineTo(w, divY); ctx.stroke()
              }
            }
          }
        }
        if (_mainPopulated && track.group === 'Ego') {
          ctx.strokeStyle = canvasColors.bg; ctx.lineWidth = 0.5
          const egoContRows = lightLaneCounts.get('ego_act') ?? 1
          const egoInflRowsDiv = lightLaneCounts.get('ego_act_infl') ?? 0
          // Containment base divider
          ctx.beginPath(); ctx.moveTo(0, y + EGO_SUBTRACK_BASE); ctx.lineTo(w, y + EGO_SUBTRACK_BASE); ctx.stroke()
          // Containment row dividers
          for (let ri = 1; ri < egoContRows; ri++) {
            const divY = y + EGO_SUBTRACK_BASE + ri * COND_SUBTRACK_HEIGHT
            ctx.beginPath(); ctx.moveTo(0, divY); ctx.lineTo(w, divY); ctx.stroke()
          }
          // Influence base divider (after containment rows)
          const egoInflBase = y + EGO_SUBTRACK_BASE + egoContRows * COND_SUBTRACK_HEIGHT
          ctx.beginPath(); ctx.moveTo(0, egoInflBase); ctx.lineTo(w, egoInflBase); ctx.stroke()
          // Influence row dividers
          for (let ri = 1; ri < egoInflRowsDiv; ri++) {
            const divY = egoInflBase + ri * COND_SUBTRACK_HEIGHT
            ctx.beginPath(); ctx.moveTo(0, divY); ctx.lineTo(w, divY); ctx.stroke()
          }
          // Action base divider (after influence rows)
          if (egoInflRowsDiv > 0) {
            const actionBaseY = egoInflBase + egoInflRowsDiv * COND_SUBTRACK_HEIGHT
            ctx.beginPath(); ctx.moveTo(0, actionBaseY); ctx.lineTo(w, actionBaseY); ctx.stroke()
          }
        }
        if (_mainPopulated && track.group === 'Agents') {
          ctx.strokeStyle = canvasColors.bg; ctx.lineWidth = 0.5
          const agentContRows = lightLaneCounts.get(track.id) ?? 1
          const agentInflRowsDiv = lightLaneCounts.get(`${track.id}_infl`) ?? 0
          // Containment base divider
          ctx.beginPath(); ctx.moveTo(0, y + AGENT_SUBTRACK_BASE); ctx.lineTo(w, y + AGENT_SUBTRACK_BASE); ctx.stroke()
          // Containment row dividers
          for (let ri = 1; ri < agentContRows; ri++) {
            const divY = y + AGENT_SUBTRACK_BASE + ri * COND_SUBTRACK_HEIGHT
            ctx.beginPath(); ctx.moveTo(0, divY); ctx.lineTo(w, divY); ctx.stroke()
          }
          // Pose base divider (after containment rows)
          const agentPoseBase = y + AGENT_SUBTRACK_BASE + agentContRows * COND_SUBTRACK_HEIGHT
          ctx.beginPath(); ctx.moveTo(0, agentPoseBase); ctx.lineTo(w, agentPoseBase); ctx.stroke()
          // Influence base divider (after pose row)
          const agentInflBase = agentPoseBase + COND_SUBTRACK_HEIGHT
          ctx.beginPath(); ctx.moveTo(0, agentInflBase); ctx.lineTo(w, agentInflBase); ctx.stroke()
          // Influence row dividers
          for (let ri = 1; ri < agentInflRowsDiv; ri++) {
            const divY = agentInflBase + ri * COND_SUBTRACK_HEIGHT
            ctx.beginPath(); ctx.moveTo(0, divY); ctx.lineTo(w, divY); ctx.stroke()
          }
          // Action base divider (after influence rows)
          const agentActionBase = agentInflBase + agentInflRowsDiv * COND_SUBTRACK_HEIGHT
          ctx.beginPath(); ctx.moveTo(0, agentActionBase); ctx.lineTo(w, agentActionBase); ctx.stroke()
        }
      })

      // Segments
      trackList.forEach((trackItem, rowIdx) => {
        if (trackItem._collapsed) return
        const { id, color, group } = trackItem
        const y = drawGetTrackY(rowIdx)
        const trackH = getTrackHeight(group, id, lightLaneCounts)
        const hasSubtrack = group === 'Environments' || group === 'Objects' || group === 'TrafficLights' || group === 'Agents' || group === 'Ego'
        for (const seg of tracks.get(id) || []) {
          const segMeta = seg.meta as Record<string, unknown>
          const isCond = !!segMeta?._isCondSubtrack
          const isObjState = !!segMeta?._isObjStateSubtrack
          const isSignalHead = !!segMeta?._isSignalHeadSubtrack
          const isLightState = !!segMeta?._isLightStateSubtrack
          const isObjCont = !!segMeta?._isObjContSubtrack
          const isLightCont = !!segMeta?._isLightContSubtrack
          const isLightPhysCont = !!segMeta?._isLightPhysContSubtrack
          const isAgentAction = !!segMeta?._isAgentActionSubtrack
          const isAgentProperty = !!segMeta?._isAgentPropertySubtrack
          const isAgentPose = !!segMeta?._isAgentPoseSubtrack
          const isEgoCont = !!segMeta?._isEgoContSubtrack
          const isAgentCont = !!segMeta?._isAgentContSubtrack
          const isEgoAction = !!segMeta?._isEgoActionSubtrack
          const isEgoProperty = !!segMeta?._isEgoPropertySubtrack
          const isEgoInfluence = !!segMeta?._isEgoInfluenceSubtrack
          const isAgentInfluence = !!segMeta?._isAgentInfluenceSubtrack
          const isSubtrack = isCond || isObjState || isSignalHead || isLightState || isObjCont || isLightCont || isLightPhysCont || isAgentAction || isAgentProperty || isAgentPose || isEgoCont || isAgentCont || isEgoInfluence || isAgentInfluence || isEgoAction || isEgoProperty
          const x0 = t2x(seg.t0), x1 = t2x(seg.t1)
          if (x1 < -10 || x0 > w + 10) continue
          const left = Math.max(0, x0), width = Math.min(x1, w) - left
          if (width <= 0) continue

          const confidence = (seg.meta as { confidence?: number } | undefined)?.confidence
          const lowConf = confidence != null && confidence < 0.7
          if (lowConf) ctx.globalAlpha = 0.6

          if (isSubtrack) {
            // Determine subtrack row Y offset
            let subY: number
            let subColor: string
            if (isEgoProperty) {
              const _epContRows = lightLaneCounts.get('ego_act') ?? 1
              const _epInflRows = lightLaneCounts.get('ego_act_infl') ?? 0
              const _epPropIdx = segMeta._prop_track_index as number ?? 0
              subY = y + EGO_SUBTRACK_BASE + _epContRows * COND_SUBTRACK_HEIGHT + _epInflRows * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT + _epPropIdx * COND_SUBTRACK_HEIGHT
              subColor = '#93c5fd'
            } else if (isEgoAction) {
              const _egoContRows = lightLaneCounts.get('ego_act') ?? 1
              const _egoInflRows = lightLaneCounts.get('ego_act_infl') ?? 0
              subY = y + EGO_SUBTRACK_BASE + _egoContRows * COND_SUBTRACK_HEIGHT + _egoInflRows * COND_SUBTRACK_HEIGHT
              subColor = '#3b82f6'
            } else if (isEgoInfluence) {
              const inflIdx = segMeta._influence_track_index as number ?? 0
              const _egoContRows2 = lightLaneCounts.get('ego_act') ?? 1
              subY = y + EGO_SUBTRACK_BASE + _egoContRows2 * COND_SUBTRACK_HEIGHT + inflIdx * COND_SUBTRACK_HEIGHT
              subColor = '#f97316'
            } else if (isAgentInfluence) {
              const inflIdx = segMeta._influence_track_index as number ?? 0
              const _agentContRows = lightLaneCounts.get(id) ?? 1
              subY = y + AGENT_SUBTRACK_BASE + _agentContRows * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT + inflIdx * COND_SUBTRACK_HEIGHT
              subColor = '#f97316'
            } else if (isEgoCont) {
              const contIdx = segMeta._cont_track_index as number ?? 0
              subY = y + EGO_SUBTRACK_BASE + contIdx * COND_SUBTRACK_HEIGHT
              subColor = '#22c55e'
            } else if (isAgentCont) {
              const contIdx = segMeta._cont_track_index as number ?? 0
              subY = y + AGENT_SUBTRACK_BASE + contIdx * COND_SUBTRACK_HEIGHT
              subColor = '#22c55e'
            } else if (isObjCont) {
              const contIdx = segMeta._cont_track_index as number ?? 0
              subY = y + OBJ_CONT_ROW_Y + contIdx * COND_SUBTRACK_HEIGHT
              subColor = '#22c55e'
            } else if (isLightPhysCont) {
              subY = y + ENV_MAIN_HEIGHT + (segMeta._cont_track_index as number ?? 0) * COND_SUBTRACK_HEIGHT
              subColor = '#22c55e'
            } else if (isSignalHead) {
              const hi = segMeta._headIndex as number ?? 0
              const light = bundle?.annotation?.traffic_lights?.[(segMeta._lightIndex as number) ?? 0]
              const trackPCMax = lightLaneCounts.get(`${id}_physcontmax`) ?? 1
              subY = y + (light ? getSignalHeadY(light, hi, trackPCMax) : ENV_MAIN_HEIGHT)
              subColor = '#f97316' // orange for signal head
            } else if (isLightCont) {
              const hi = segMeta._headIndex as number ?? 0
              const light = bundle?.annotation?.traffic_lights?.[(segMeta._lightIndex as number) ?? 0]
              const trackPCMax = lightLaneCounts.get(`${id}_physcontmax`) ?? 1
              subY = y + (light ? getSignalHeadY(light, hi, trackPCMax) + COND_SUBTRACK_HEIGHT : ENV_MAIN_HEIGHT + COND_SUBTRACK_HEIGHT)
              subColor = '#22c55e'
            } else if (isAgentPose) {
              const _apContRows = lightLaneCounts.get(id) ?? 1
              subY = y + AGENT_SUBTRACK_BASE + _apContRows * COND_SUBTRACK_HEIGHT
              subColor = '#14b8a6'
            } else if (isAgentProperty) {
              const _apropContRows = lightLaneCounts.get(id) ?? 1
              const _apropInflRows = lightLaneCounts.get(`${id}_infl`) ?? 0
              const _apropPropIdx = segMeta._prop_track_index as number ?? 0
              subY = y + AGENT_SUBTRACK_BASE + _apropContRows * COND_SUBTRACK_HEIGHT + _apropInflRows * COND_SUBTRACK_HEIGHT + 2 * COND_SUBTRACK_HEIGHT + _apropPropIdx * COND_SUBTRACK_HEIGHT
              subColor = '#d8b4fe'
            } else if (isAgentAction) {
              const _aaContRows = lightLaneCounts.get(id) ?? 1
              const _aaInflRows = lightLaneCounts.get(`${id}_infl`) ?? 0
              subY = y + AGENT_SUBTRACK_BASE + _aaContRows * COND_SUBTRACK_HEIGHT + _aaInflRows * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT
              subColor = '#c084fc'
            } else if (isObjState) {
              subY = y + getObjStateRowY(lightLaneCounts.get(id) ?? 1)
              subColor = '#d97706'
            } else if (isLightState) {
              const hi = segMeta._headIndex as number ?? 0
              const stLaneIdx = segMeta._state_track_index as number ?? 0
              const light = bundle?.annotation?.traffic_lights?.[(segMeta._lightIndex as number) ?? 0]
              const trackPCMax = lightLaneCounts.get(`${id}_physcontmax`) ?? 1
              const shY = light ? getSignalHeadY(light, hi, trackPCMax) : ENV_MAIN_HEIGHT
              subY = y + shY + 2 * COND_SUBTRACK_HEIGHT + stLaneIdx * COND_SUBTRACK_HEIGHT // cont + SH bar + lane offset
              const lightColor = (segMeta.color as string || '').toLowerCase()
              subColor = lightColor === 'green' ? '#22c55e' : lightColor === 'yellow' ? '#eab308' : lightColor === 'red' ? '#ef4444' : lightColor === 'other' ? '#888' : '#dc2626'
            } else {
              // Cond (environment conditions)
              const condIdx = segMeta._cond_track_index as number ?? 0
              subY = y + ENV_MAIN_HEIGHT + condIdx * COND_SUBTRACK_HEIGHT
              subColor = '#06b6d4'
            }
            ctx.fillStyle = subColor
            roundRect(ctx, left, subY + 1, width, COND_SUBTRACK_HEIGHT - 2, 2)
            ctx.fill()
            const fullWidth = x1 - x0
            if (fullWidth >= 30) {
              ctx.fillStyle = '#fff'; ctx.font = '10px sans-serif'; ctx.textAlign = 'left'
              const maxChars = Math.max(4, Math.floor((fullWidth - 6) / 6))
              const text = seg.label.length > maxChars ? seg.label.slice(0, maxChars - 1) + '\u2026' : seg.label
              ctx.fillText(text, x0 + 3, subY + COND_SUBTRACK_HEIGHT / 2 + 4)
            }
            if (seg.id === selectedPath) {
              ctx.shadowColor = 'rgba(255,255,255,0.5)'; ctx.shadowBlur = 4
              ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5
              roundRect(ctx, left, subY + 1, width, COND_SUBTRACK_HEIGHT - 2, 2); ctx.stroke()
              ctx.shadowBlur = 0
            }

            // Keypoint diamonds on the signal-head subtrack
            if (isSignalHead && keypointsVisible) {
              const li = segMeta?._lightIndex as number | undefined
              const hi = segMeta?._headIndex as number | undefined
              const shKps = li != null && hi != null
                ? bundle?.annotation?.traffic_lights?.[li]?.signal_heads?.[hi]?.keypoints
                : undefined
              if (shKps && shKps.length > 0) {
                const cy = subY + COND_SUBTRACK_HEIGHT / 2
                const r = Math.max(3, Math.min(4, Math.floor(COND_SUBTRACK_HEIGHT / 4)))
                ctx.save()
                ctx.lineWidth = 1
                for (const kp of shKps) {
                  const kpX = t2x(parseTs(kp.timestamp))
                  if (kpX < -r || kpX > w + r) continue
                  ctx.beginPath()
                  ctx.moveTo(kpX, cy - r)
                  ctx.lineTo(kpX + r, cy)
                  ctx.lineTo(kpX, cy + r)
                  ctx.lineTo(kpX - r, cy)
                  ctx.closePath()
                  ctx.fillStyle = '#fff'
                  ctx.fill()
                  ctx.strokeStyle = 'rgba(0,0,0,0.7)'
                  ctx.stroke()
                }
                ctx.restore()
              }
            }
          } else {
            // Main segment
            const isEgoMain = !!segMeta?._isEgoMain
            const segH = isEgoMain ? EGO_SUBTRACK_BASE : (hasSubtrack ? ENV_MAIN_HEIGHT : trackH)
            const gradient = ctx.createLinearGradient(left, y, left, y + segH)
            gradient.addColorStop(0, adjustColor(color, 30))
            gradient.addColorStop(1, adjustColor(color, -20))
            ctx.fillStyle = gradient
            roundRect(ctx, left, y + 2, width, segH - 4, SEGMENT_RADIUS)
            ctx.fill()

            if (seg.illegal) { ctx.fillStyle = '#ef4444'; ctx.fillRect(left, y + 2, 4, segH - 4) }

            const fullWidthMain = x1 - x0
            if (fullWidthMain >= LABEL_MIN_WIDTH) {
              ctx.fillStyle = '#fff'; ctx.font = '10px sans-serif'; ctx.textAlign = 'left'
              const maxChars = Math.max(4, Math.floor((fullWidthMain - 12) / 6))
              const text = seg.label.length > maxChars ? seg.label.slice(0, maxChars - 1) + '\u2026' : seg.label
              ctx.fillText(text, x0 + 6, y + segH / 2 + 4)
            }

            if (seg.id === selectedPath) {
              ctx.shadowColor = 'rgba(255,255,255,0.5)'; ctx.shadowBlur = 6
              ctx.strokeStyle = '#fff'; ctx.lineWidth = 2
              roundRect(ctx, left, y + 2, width, segH - 4, SEGMENT_RADIUS); ctx.stroke()
              ctx.shadowBlur = 0
              if (!isEgoMain) {
                ctx.fillStyle = '#fff'
                ctx.fillRect(left - 1, y + 6, 3, segH - 12)
                ctx.fillRect(left + width - 2, y + 6, 3, segH - 12)
              }
            }

            // Keypoint markers on the main parent bar (agent / traffic_object / environment)
            const objKind = segMeta?._objKind as string | undefined
            let entityKeypoints: { timestamp: string; x: number; y: number }[] | undefined
            if (keypointsVisible) {
              if (objKind === 'agent' && segMeta?._agentIndex != null) {
                entityKeypoints = bundle?.annotation?.agents?.[segMeta._agentIndex as number]?.keypoints
              } else if (objKind === 'traffic_object' && segMeta?._objIndex != null) {
                entityKeypoints = bundle?.annotation?.traffic_objects?.[segMeta._objIndex as number]?.keypoints
              } else if (objKind === 'environment' && segMeta?._envIndex != null) {
                entityKeypoints = bundle?.annotation?.environments?.[segMeta._envIndex as number]?.keypoints
              }
            }
            if (entityKeypoints && entityKeypoints.length > 0) {
              const cy = y + segH / 2
              const r = Math.max(3, Math.min(4, Math.floor(segH / 6)))
              ctx.save()
              ctx.lineWidth = 1
              for (const kp of entityKeypoints) {
                const kpX = t2x(parseTs(kp.timestamp))
                if (kpX < -r || kpX > w + r) continue
                ctx.beginPath()
                ctx.moveTo(kpX, cy - r)
                ctx.lineTo(kpX + r, cy)
                ctx.lineTo(kpX, cy + r)
                ctx.lineTo(kpX - r, cy)
                ctx.closePath()
                ctx.fillStyle = '#fff'
                ctx.fill()
                ctx.strokeStyle = 'rgba(0,0,0,0.7)'
                ctx.stroke()
              }
              ctx.restore()
            }
          }
          if (lowConf) ctx.globalAlpha = 1

          // Completeness badge
          const compResult = completenessMap.get(seg.id)
          if (compResult && compResult.missingCount > 0) {
            const badgeR = 7
            const badgeColor = compResult.missingCount >= 3 ? '#ef4444' : '#f59e0b'
            // Position at top-right of the segment
            let segTop: number, segRight: number
            if (isSubtrack) {
              // For subtracks, use their specific Y position
              let subYBadge: number
              if (isEgoProperty) { const _ecrBp = lightLaneCounts.get('ego_act') ?? 1; const _eirBp = lightLaneCounts.get('ego_act_infl') ?? 0; const _epidxBp = segMeta._prop_track_index as number ?? 0; subYBadge = y + EGO_SUBTRACK_BASE + _ecrBp * COND_SUBTRACK_HEIGHT + _eirBp * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT + _epidxBp * COND_SUBTRACK_HEIGHT }
              else if (isEgoAction) { const _ecrB = lightLaneCounts.get('ego_act') ?? 1; const _eirB = lightLaneCounts.get('ego_act_infl') ?? 0; subYBadge = y + EGO_SUBTRACK_BASE + _ecrB * COND_SUBTRACK_HEIGHT + _eirB * COND_SUBTRACK_HEIGHT }
              else if (isEgoInfluence) subYBadge = y + EGO_SUBTRACK_BASE + (lightLaneCounts.get('ego_act') ?? 1) * COND_SUBTRACK_HEIGHT + (segMeta._influence_track_index as number ?? 0) * COND_SUBTRACK_HEIGHT
              else if (isAgentInfluence) subYBadge = y + AGENT_SUBTRACK_BASE + (lightLaneCounts.get(id) ?? 1) * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT + (segMeta._influence_track_index as number ?? 0) * COND_SUBTRACK_HEIGHT
              else if (isEgoCont) { subYBadge = y + EGO_SUBTRACK_BASE + (segMeta._cont_track_index as number ?? 0) * COND_SUBTRACK_HEIGHT }
              else if (isObjCont) subYBadge = y + OBJ_CONT_ROW_Y + (segMeta._cont_track_index as number ?? 0) * COND_SUBTRACK_HEIGHT
              else if (isLightPhysCont) subYBadge = y + ENV_MAIN_HEIGHT + (segMeta._cont_track_index as number ?? 0) * COND_SUBTRACK_HEIGHT
              else if (isSignalHead || isLightCont || isLightState) {
                const lhi = segMeta._headIndex as number ?? 0
                const ll = bundle?.annotation?.traffic_lights?.[(segMeta._lightIndex as number) ?? 0]
                const trackPCMax = lightLaneCounts.get(`${id}_physcontmax`) ?? 1
                const shBaseY = ll ? getSignalHeadY(ll, lhi, trackPCMax) : ENV_MAIN_HEIGHT
                if (isSignalHead) subYBadge = y + shBaseY
                else if (isLightCont) subYBadge = y + shBaseY + COND_SUBTRACK_HEIGHT
                else { const stIdx = segMeta._state_track_index as number ?? 0; subYBadge = y + shBaseY + 2 * COND_SUBTRACK_HEIGHT + stIdx * COND_SUBTRACK_HEIGHT }
              }
              else if (isAgentCont) { subYBadge = y + AGENT_SUBTRACK_BASE + (segMeta._cont_track_index as number ?? 0) * COND_SUBTRACK_HEIGHT }
              else if (isAgentProperty) { const _acrBp = lightLaneCounts.get(id) ?? 1; const _airBp = lightLaneCounts.get(`${id}_infl`) ?? 0; const _pidxBp = segMeta._prop_track_index as number ?? 0; subYBadge = y + AGENT_SUBTRACK_BASE + _acrBp * COND_SUBTRACK_HEIGHT + _airBp * COND_SUBTRACK_HEIGHT + 2 * COND_SUBTRACK_HEIGHT + _pidxBp * COND_SUBTRACK_HEIGHT }
              else if (isAgentAction) { const _acrB = lightLaneCounts.get(id) ?? 1; const _airB = lightLaneCounts.get(`${id}_infl`) ?? 0; subYBadge = y + AGENT_SUBTRACK_BASE + _acrB * COND_SUBTRACK_HEIGHT + _airB * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT }
              else if (isObjState) subYBadge = y + getObjStateRowY(lightLaneCounts.get(id) ?? 1)
              else if (isAgentPose) { const _aprB = lightLaneCounts.get(id) ?? 1; subYBadge = y + AGENT_SUBTRACK_BASE + _aprB * COND_SUBTRACK_HEIGHT }
              else if (isCond) subYBadge = y + ENV_MAIN_HEIGHT + (segMeta._cond_track_index as number ?? 0) * COND_SUBTRACK_HEIGHT
              else subYBadge = y + ENV_MAIN_HEIGHT
              segTop = subYBadge + 1
              segRight = x1
            } else {
              segTop = y + 2
              segRight = x1
            }
            const cx = segRight - badgeR - 2
            const cy = segTop + badgeR + 1
            ctx.fillStyle = badgeColor
            ctx.beginPath()
            ctx.arc(cx, cy, badgeR, 0, Math.PI * 2)
            ctx.fill()
            ctx.fillStyle = '#fff'
            ctx.font = 'bold 9px sans-serif'
            ctx.textAlign = 'center'
            ctx.fillText(String(compResult.missingCount), cx, cy + 3)
          }
        }
      })

      // "Outside Camera View" gray overlay — draw over entire agent track height
      trackList.forEach((trackItem, rowIdx) => {
        if (trackItem._collapsed) return
        const { id, group } = trackItem
        if (group !== 'Agents') return
        const trackIdx = parseInt(id.replace('agent_', ''), 10)
        if (isNaN(trackIdx)) return
        const agents = bundle?.annotation?.agents || []
        for (const agent of agents) {
          if ((agent._track_index ?? 0) !== trackIdx) continue
          for (const prop of agent.properties || []) {
            if (prop.property_type !== 'Outside Camera View') continue
            const pt0 = parseTs(prop.start_timestamp), pt1 = parseTs(prop.end_timestamp)
            const ox0 = t2x(pt0), ox1 = t2x(pt1)
            if (ox1 < 0 || ox0 > w) continue
            const oleft = Math.max(0, ox0), owidth = Math.min(ox1, w) - oleft
            if (owidth <= 0) continue
            const oy = drawGetTrackY(rowIdx)
            const oh = getTrackHeight(group, id, lightLaneCounts)
            ctx.fillStyle = 'rgba(80, 80, 80, 0.45)'
            ctx.fillRect(oleft, oy, owidth, oh)
          }
        }
      })

      // arrow curves
      const anyArrow = arrowTypes.becauseOf || arrowTypes.linkTo || arrowTypes.containedIn || arrowTypes.influencedBy || arrowTypes.actionTarget
      if (anyArrow) {
      const segById = new Map(segments.map((s) => [s.id, s]))
      const entityToSeg = new Map<string, typeof segments[0]>()
      for (const seg of segments) {
        const m = seg.meta as Record<string, unknown> | undefined
        if (!m) continue
        // Register subtracks with IDs for target resolution
        if ((m._isCondSubtrack || m._isAgentActionSubtrack || m._isAgentPropertySubtrack || m._isEgoActionSubtrack || m._isEgoPropertySubtrack || m._isLightStateSubtrack || m._isObjStateSubtrack || m._isSignalHeadSubtrack) && m.id) {
          entityToSeg.set(m.id as string, seg)
          continue
        }
        if (m._isObjContSubtrack || m._isLightContSubtrack || m._isLightPhysContSubtrack ||
            m._isEgoContSubtrack || m._isAgentContSubtrack || m._isAgentPoseSubtrack || m._isEgoInfluenceSubtrack || m._isAgentInfluenceSubtrack) continue
        const eid = m.id as string | undefined
        if (eid) entityToSeg.set(eid, seg)
      }
      const resolveTarget = (targetId: string, _srcMidT: number): typeof segments[0] | undefined => {
        const direct = segById.get(targetId) || entityToSeg.get(targetId)
        if (direct) return direct
        if (targetId === 'Ego') {
          const egoMain = segById.get('ego_main')
          if (egoMain) return egoMain
        }
        return undefined
      }
      const trackIndex = new Map(trackList.map((t, i) => [t.id, i]))

      // Map every segment ID → its top-level entity segment ID (the main segment that owns it)
      const isSubtrackSeg = (m: Record<string, unknown> | undefined) =>
        !!m && !!(m._isCondSubtrack || m._isAgentActionSubtrack || m._isAgentPropertySubtrack || m._isLightStateSubtrack ||
          m._isObjStateSubtrack || m._isSignalHeadSubtrack || m._isObjContSubtrack ||
          m._isLightContSubtrack || m._isLightPhysContSubtrack || m._isEgoContSubtrack || m._isAgentContSubtrack ||
          m._isAgentPoseSubtrack || m._isEgoInfluenceSubtrack || m._isEgoPropertySubtrack || m._isAgentInfluenceSubtrack)
      const mainSegs = segments.filter(s => !isSubtrackSeg(s.meta as Record<string, unknown> | undefined))
      const segToEntity = new Map<string, string>()
      for (const s of mainSegs) segToEntity.set(s.id, s.id)
      for (const s of segments) {
        if (segToEntity.has(s.id)) continue
        const owner = mainSegs.find(m => m.trackId === s.trackId && m.t0 <= s.t0 && m.t1 >= s.t1)
        segToEntity.set(s.id, owner?.id ?? s.id)
      }

      const selectedEntityId = selectedPath ? (segToEntity.get(selectedPath) ?? null) : null
      const selectedIsSubtrack = selectedPath ? isSubtrackSeg(segments.find(s => s.id === selectedPath)?.meta as Record<string, unknown> | undefined) : false
      const arrowAlpha = (srcId: string, tgtId: string) => {
        if (!selectedPath) return 1
        if (selectedIsSubtrack)
          return srcId === selectedPath || tgtId === selectedPath ? 1 : 0.15
        return segToEntity.get(srcId) === selectedEntityId || segToEntity.get(tgtId) === selectedEntityId ? 1 : 0.15
      }

      // Compute the Y center of a segment, accounting for subtrack rows
      const getSegCenterY = (seg: TimelineSegment, rowIdx: number): number => {
        const ty = drawGetTrackY(rowIdx)
        const m = seg.meta as Record<string, unknown> | undefined
        if (!m) return ty + ENV_MAIN_HEIGHT / 2
        if (m._isEgoActionSubtrack) { const _ecr = lightLaneCounts.get('ego_act') ?? 1; const _eir = lightLaneCounts.get('ego_act_infl') ?? 0; return ty + EGO_SUBTRACK_BASE + _ecr * COND_SUBTRACK_HEIGHT + _eir * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT / 2 }
        if (m._isEgoPropertySubtrack) { const _epcr = lightLaneCounts.get('ego_act') ?? 1; const _epir = lightLaneCounts.get('ego_act_infl') ?? 0; const _eppi = m._prop_track_index as number ?? 0; return ty + EGO_SUBTRACK_BASE + _epcr * COND_SUBTRACK_HEIGHT + _epir * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT + _eppi * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT / 2 }
        if (m._isAgentPropertySubtrack) { const _aprcr = lightLaneCounts.get(seg.trackId) ?? 1; const _aprir = lightLaneCounts.get(`${seg.trackId}_infl`) ?? 0; const _aprpi = m._prop_track_index as number ?? 0; return ty + AGENT_SUBTRACK_BASE + _aprcr * COND_SUBTRACK_HEIGHT + _aprir * COND_SUBTRACK_HEIGHT + 2 * COND_SUBTRACK_HEIGHT + _aprpi * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT / 2 }
        if (m._isAgentActionSubtrack) { const _aacr = lightLaneCounts.get(seg.trackId) ?? 1; const _aair = lightLaneCounts.get(`${seg.trackId}_infl`) ?? 0; return ty + AGENT_SUBTRACK_BASE + _aacr * COND_SUBTRACK_HEIGHT + _aair * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT / 2 }
        if (m._isAgentPoseSubtrack) { const _apcr = lightLaneCounts.get(seg.trackId) ?? 1; return ty + AGENT_SUBTRACK_BASE + _apcr * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT / 2 }
        if (m._isEgoInfluenceSubtrack) { const _eicr = lightLaneCounts.get('ego_act') ?? 1; return ty + EGO_SUBTRACK_BASE + _eicr * COND_SUBTRACK_HEIGHT + (m._influence_track_index as number ?? 0) * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT / 2 }
        if (m._isAgentInfluenceSubtrack) { const _aicr = lightLaneCounts.get(seg.trackId) ?? 1; return ty + AGENT_SUBTRACK_BASE + _aicr * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT + (m._influence_track_index as number ?? 0) * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT / 2 }
        if (m._isAgentContSubtrack) { return ty + AGENT_SUBTRACK_BASE + (m._cont_track_index as number ?? 0) * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT / 2 }
        if (m._isEgoContSubtrack) { return ty + EGO_SUBTRACK_BASE + (m._cont_track_index as number ?? 0) * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT / 2 }
        if (m._isObjContSubtrack) return ty + OBJ_CONT_ROW_Y + (m._cont_track_index as number ?? 0) * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT / 2
        if (m._isObjStateSubtrack) return ty + getObjStateRowY(lightLaneCounts.get(seg.trackId) ?? 1) + COND_SUBTRACK_HEIGHT / 2
        if (m._isSignalHeadSubtrack) {
          const hi = m._headIndex as number ?? 0
          const ll = bundle?.annotation?.traffic_lights?.[(m._lightIndex as number) ?? 0]
          const trackPCMax = lightLaneCounts.get(`${seg.trackId}_physcontmax`) ?? 1
          const shBaseY = ll ? getSignalHeadY(ll, hi, trackPCMax) : ENV_MAIN_HEIGHT
          return ty + shBaseY + COND_SUBTRACK_HEIGHT / 2
        }
        if (m._isLightContSubtrack) {
          const hi = m._headIndex as number ?? 0
          const ll = bundle?.annotation?.traffic_lights?.[(m._lightIndex as number) ?? 0]
          const trackPCMax = lightLaneCounts.get(`${seg.trackId}_physcontmax`) ?? 1
          const shBaseY = ll ? getSignalHeadY(ll, hi, trackPCMax) : ENV_MAIN_HEIGHT
          return ty + shBaseY + COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT / 2
        }
        if (m._isLightStateSubtrack) {
          const hi = m._headIndex as number ?? 0
          const stIdx = (m._state_track_index as number) ?? 0
          const ll = bundle?.annotation?.traffic_lights?.[(m._lightIndex as number) ?? 0]
          const trackPCMax = lightLaneCounts.get(`${seg.trackId}_physcontmax`) ?? 1
          const shBaseY = ll ? getSignalHeadY(ll, hi, trackPCMax) : ENV_MAIN_HEIGHT
          return ty + shBaseY + 2 * COND_SUBTRACK_HEIGHT + stIdx * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT / 2
        }
        if (m._isLightPhysContSubtrack) {
          return ty + ENV_MAIN_HEIGHT + (m._cont_track_index as number ?? 0) * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT / 2
        }
        if (m._isCondSubtrack) return ty + ENV_MAIN_HEIGHT + (m._cond_track_index as number ?? 0) * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT / 2
        return ty + ENV_MAIN_HEIGHT / 2
      }

      if (arrowTypes.becauseOf) for (const seg of segments) {
        if (!seg.because_of?.length) continue
        const srcRow = trackIndex.get(seg.trackId); if (srcRow == null) continue
        const srcY = getSegCenterY(seg, srcRow)
        const srcMidT = (seg.t0 + seg.t1) / 2
        const srcX = t2x(srcMidT)
        if (srcX < -50 || srcX > w + 50) continue
        for (const targetId of seg.because_of) {
          const target = resolveTarget(targetId, srcMidT); if (!target) continue
          const tgtRow = trackIndex.get(target.trackId); if (tgtRow == null) continue
          const tgtY = getSegCenterY(target, tgtRow)
          const tgtX = (t2x(target.t0) + t2x(target.t1)) / 2
          if (tgtX < -50 || tgtX > w + 50) continue
          const a = arrowAlpha(seg.id, target.id)
          ctx.globalAlpha = a
          drawBecauseOfCurve(ctx, srcX, srcY, tgtX, tgtY, '#b45309', 'BECAUSE', !!selectedEntityId && a === 1)
        }
      }

      // containment → environment arrows (green)
      if (arrowTypes.containedIn) for (const seg of segments) {
        const m = seg.meta as Record<string, unknown> | undefined
        if (!m) continue
        const isCont = (m._isEgoContSubtrack || m._isAgentContSubtrack || m._isObjContSubtrack || m._isLightContSubtrack || m._isLightPhysContSubtrack) && !m._isEgoInfluenceSubtrack && !m._isAgentInfluenceSubtrack
        if (!isCont) continue
        const envId = m.env_id as string | undefined
        if (!envId) continue
        const envSeg = entityToSeg.get(envId)
        if (!envSeg) continue
        const srcRow = trackIndex.get(seg.trackId); if (srcRow == null) continue
        const tgtRow = trackIndex.get(envSeg.trackId); if (tgtRow == null) continue
        const srcY = getSegCenterY(seg, srcRow)
        const tgtY = getSegCenterY(envSeg, tgtRow)
        const srcX = t2x((seg.t0 + seg.t1) / 2)
        const tgtX = t2x((envSeg.t0 + envSeg.t1) / 2)
        if (srcX < -50 || srcX > w + 50) continue
        const a = arrowAlpha(seg.id, envSeg.id)
        ctx.globalAlpha = a
        const isHighlighted = !!selectedEntityId && a === 1
        ctx.setLineDash(isHighlighted ? [] : [4, 4])
        drawBecauseOfCurve(ctx, srcX, srcY, tgtX, tgtY, '#14532d', 'CONTAINED_IN', isHighlighted)
        ctx.setLineDash([])
      }

      // influence ← influencer arrows (light brown)
      if (arrowTypes.influencedBy) for (const seg of segments) {
        const m = seg.meta as Record<string, unknown> | undefined
        if (!m) continue
        const isInfl = m._isEgoInfluenceSubtrack || m._isAgentInfluenceSubtrack
        if (!isInfl) continue
        const influencers = m.influencers as string[] | undefined
        if (!influencers?.length) continue
        const tgtRow = trackIndex.get(seg.trackId); if (tgtRow == null) continue
        const tgtY = getSegCenterY(seg, tgtRow)
        const tgtMidT = (seg.t0 + seg.t1) / 2
        const tgtX = t2x(tgtMidT)
        if (tgtX < -50 || tgtX > w + 50) continue
        for (const influencerId of influencers) {
          const src = resolveTarget(influencerId, tgtMidT); if (!src) continue
          const srcRow = trackIndex.get(src.trackId); if (srcRow == null) continue
          const srcY = getSegCenterY(src, srcRow)
          const srcX = (t2x(src.t0) + t2x(src.t1)) / 2
          if (srcX < -50 || srcX > w + 50) continue
          const a = arrowAlpha(seg.id, src.id)
          ctx.globalAlpha = a
          drawBecauseOfCurve(ctx, tgtX, tgtY, srcX, srcY, '#78350f', 'INFLUENCED_BY', !!selectedEntityId && a === 1)
        }
      }

      // link_to arrows (blue)
      if (arrowTypes.linkTo) for (const seg of segments) {
        const m = seg.meta as Record<string, unknown> | undefined
        if (!m) continue
        const isAction = (seg.trackId === 'ego_act' && !m._isEgoContSubtrack && !m._isEgoInfluenceSubtrack && !m._isEgoPropertySubtrack) || m._isAgentActionSubtrack
        const isProperty = !!m._isAgentPropertySubtrack || !!m._isEgoPropertySubtrack
        if (!isAction && !isProperty) continue
        const linkTo = isProperty
          ? (m.signaling_details as { link_to?: string[] } | undefined)?.link_to
          : m.link_to as string[] | undefined
        if (!linkTo?.length) continue
        const srcRow = trackIndex.get(seg.trackId); if (srcRow == null) continue
        const srcY = getSegCenterY(seg, srcRow)
        const srcMidT = (seg.t0 + seg.t1) / 2
        const srcX = t2x(srcMidT)
        if (srcX < -50 || srcX > w + 50) continue
        for (const targetId of linkTo) {
          const target = resolveTarget(targetId, srcMidT); if (!target) continue
          const tgtRow = trackIndex.get(target.trackId); if (tgtRow == null) continue
          const tgtY = getSegCenterY(target, tgtRow)
          const tgtX = (t2x(target.t0) + t2x(target.t1)) / 2
          if (tgtX < -50 || tgtX > w + 50) continue
          const a = arrowAlpha(seg.id, target.id)
          ctx.globalAlpha = a
          drawBecauseOfCurve(ctx, srcX, srcY, tgtX, tgtY, '#1d4ed8', 'LINK_TO', !!selectedEntityId && a === 1)
        }
      }

      // action_target arrows (emerald)
      if (arrowTypes.actionTarget) for (const seg of segments) {
        const m = seg.meta as Record<string, unknown> | undefined
        if (!m) continue
        const isAction = (seg.trackId === 'ego_act' && !m._isEgoContSubtrack && !m._isEgoInfluenceSubtrack && !m._isEgoPropertySubtrack) || m._isAgentActionSubtrack
        if (!isAction) continue
        const actionTarget = m.action_target as string[] | undefined
        if (!actionTarget?.length) continue
        const srcRow = trackIndex.get(seg.trackId); if (srcRow == null) continue
        const srcY = getSegCenterY(seg, srcRow)
        const srcMidT = (seg.t0 + seg.t1) / 2
        const srcX = t2x(srcMidT)
        if (srcX < -50 || srcX > w + 50) continue
        for (const targetId of actionTarget) {
          const target = resolveTarget(targetId, srcMidT); if (!target) continue
          const tgtRow = trackIndex.get(target.trackId); if (tgtRow == null) continue
          const tgtY = getSegCenterY(target, tgtRow)
          const tgtX = (t2x(target.t0) + t2x(target.t1)) / 2
          if (tgtX < -50 || tgtX > w + 50) continue
          const a = arrowAlpha(seg.id, target.id)
          ctx.globalAlpha = a
          const actionType = (m.action_type ?? m.type) as string
          const targetLabel = ACTION_LINK_TO_CONFIG[actionType]?.label ?? 'Target'
          drawBecauseOfCurve(ctx, srcX, srcY, tgtX, tgtY, '#10b981', targetLabel, !!selectedEntityId && a === 1)
        }
      }

      ctx.globalAlpha = 1
      } // end showArrows

      // Release header clip so the playhead and end-of-video marker can extend through it.
      ctx.restore()

      // End-of-video marker
      const endX = t2x(duration)
      if (endX >= -20 && endX <= w + 20) {
        ctx.strokeStyle = '#8a8aaa'; ctx.lineWidth = 1
        ctx.setLineDash([4, 4])
        ctx.beginPath(); ctx.moveTo(endX, 0); ctx.lineTo(endX, h); ctx.stroke()
        ctx.setLineDash([])
      }

      // Playhead — vertical line drops out of a draggable circle handle in the scrubber lane
      const phX = t2x(playheadTime)
      if (phX >= -20 && phX <= w + 20) {
        const handleY = SCRUBBER_LANE_HEIGHT / 2
        const handleR = 6
        ctx.save()
        ctx.globalAlpha = 0.85
        ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 1.5
        ctx.beginPath(); ctx.moveTo(phX, handleY); ctx.lineTo(phX, h); ctx.stroke()
        ctx.restore()
        ctx.fillStyle = '#ef4444'
        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2
        ctx.beginPath(); ctx.arc(phX, handleY, handleR, 0, Math.PI * 2); ctx.fill(); ctx.stroke()
      }

      rafId = requestAnimationFrame(draw)
    }
    draw()
    return () => cancelAnimationFrame(rafId)
  }, [duration, playheadTime, selectedPath, trackList, tracks, zoomLevel, getLiveScrollOffset, mousePos, segments, arrowTypes, verticalScroll, lightLaneCounts, keypointsVisible, bundle, canvasColors])

  // --- Mouse down ---
  const handleCanvasMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (e.button !== 0) return // only left-click for drag/select
      const canvas = canvasRef.current, container = containerRef.current
      if (!canvas || !container) return
      const rect = canvas.getBoundingClientRect()
      const mx = e.clientX - rect.left, myRaw = e.clientY - rect.top
      const my = myRaw + verticalScroll
      const w = container.clientWidth
      const pps = (w * zoomLevel) / Math.max(duration, 1)
      const liveScrollOffset = getLiveScrollOffset()
      const t2x = (t: number) => (t - liveScrollOffset) * pps
      const x2t = (x: number) => x / pps + liveScrollOffset

      if (myRaw < HEADER_HEIGHT) {
        e.preventDefault()
        let previousUserSelect = ''
        if (document.body) {
          previousUserSelect = document.body.style.userSelect
          document.body.style.userSelect = 'none'
        }
        canvas.style.cursor = 'col-resize'
        const updateFromClientX = (clientX: number) => {
          const r = canvas.getBoundingClientRect()
          const edgeOverflow =
            clientX < r.left ? clientX - r.left :
            clientX > r.right ? clientX - r.right :
            0
          if (edgeOverflow !== 0 && zoomLevel > 1) {
            container.scrollBy({ left: edgeOverflow, behavior: 'auto' })
            const livePps = (container.clientWidth * zoomLevel) / Math.max(duration, 1)
            scrollOffsetRef.current = livePps > 0 ? container.scrollLeft / livePps : scrollOffsetRef.current
          }
          const livePps = (container.clientWidth * zoomLevel) / Math.max(duration, 1)
          const liveOffset = getLiveScrollOffset()
          const cx = Math.max(0, Math.min(container.clientWidth, clientX - r.left))
          scrubTo(cx / livePps + liveOffset)
        }
        updateFromClientX(e.clientX)
        const onMove = (ev: MouseEvent) => {
          ev.preventDefault()
          updateFromClientX(ev.clientX)
        }
        const onUp = () => {
          commitNativeScroll()
          canvas.style.cursor = ''
          if (document.body) document.body.style.userSelect = previousUserSelect
          window.removeEventListener('mousemove', onMove)
          window.removeEventListener('mouseup', onUp)
          window.removeEventListener('blur', onUp)
        }
        window.addEventListener('mousemove', onMove)
        window.addEventListener('mouseup', onUp)
        window.addEventListener('blur', onUp)
        return
      }

      // Find which track row was clicked using variable heights
      let rowIdx = -1
      let cumY = HEADER_HEIGHT
      for (let i = 0; i < trackList.length; i++) {
        const h = getTrackHeight(trackList[i].group, trackList[i].id, lightLaneCounts, trackList[i]._collapsed)
        if (my >= cumY && my < cumY + h) { rowIdx = i; break }
        cumY += h + getTrackGap(trackList, i)
      }
      if (rowIdx < 0 || rowIdx >= trackList.length) { selectPath(null); return }
      if (trackList[rowIdx]._collapsed) { toggleGroupCollapse(trackList[rowIdx].group); return }
      const trackId = trackList[rowIdx].id
      const trackSegs = tracks.get(trackId) || []
      const y = getTrackY(trackList, rowIdx, lightLaneCounts)
      const trackH = getTrackHeight(trackList[rowIdx].group, trackList[rowIdx].id, lightLaneCounts, trackList[rowIdx]._collapsed)
      const grp = trackList[rowIdx].group
      const hasSubtrack = grp === 'Environments' || grp === 'Objects' || grp === 'TrafficLights' || grp === 'Agents' || grp === 'Ego'

      // Left-click on a keypoint diamond → select the owning entity and scrub the playhead to that keypoint's time.
      if (keypointsVisible) {
        const localY = my - y
        const HIT_PX = 6
        const t = x2t(mx)
        const onMainBar = (grp === 'Agents' || grp === 'Objects' || grp === 'Environments') && localY >= 0 && localY < ENV_MAIN_HEIGHT
        if (onMainBar) {
          for (const seg of trackSegs) {
            const sm = seg.meta as Record<string, unknown>
            let kps: { timestamp: string; x: number; y: number }[] | undefined
            if (sm?._objKind === 'agent' && sm?._agentIndex != null && t >= seg.t0 && t <= seg.t1) {
              kps = bundle?.annotation?.agents?.[sm._agentIndex as number]?.keypoints
            } else if (sm?._objKind === 'traffic_object' && sm?._objIndex != null && t >= seg.t0 && t <= seg.t1) {
              kps = bundle?.annotation?.traffic_objects?.[sm._objIndex as number]?.keypoints
            } else if (sm?._objKind === 'environment' && sm?._envIndex != null && t >= seg.t0 && t <= seg.t1) {
              kps = bundle?.annotation?.environments?.[sm._envIndex as number]?.keypoints
            }
            if (!kps?.length) continue
            let hitKp: { timestamp: string; x: number; y: number } | null = null
            let bestDist = HIT_PX
            for (const kp of kps) {
              const d = Math.abs(t2x(parseTs(kp.timestamp)) - mx)
              if (d <= bestDist) { bestDist = d; hitKp = kp }
            }
            if (hitKp) {
              selectPath(seg.id)
              setPlayhead(Math.max(0, Math.min(duration, parseTs(hitKp.timestamp))))
              return
            }
          }
        }
        if (grp === 'TrafficLights') {
          for (const seg of trackSegs) {
            const sm = seg.meta as Record<string, unknown>
            if (!sm?._isSignalHeadSubtrack) continue
            if (!(t >= seg.t0 && t <= seg.t1)) continue
            const li = sm._lightIndex as number | undefined
            const hi = sm._headIndex as number | undefined
            if (li == null || hi == null) continue
            const light = bundle?.annotation?.traffic_lights?.[li]
            if (!light) continue
            const trackPCMax = lightLaneCounts.get(`${trackList[rowIdx].id}_physcontmax`) ?? 1
            const shTop = getSignalHeadY(light, hi, trackPCMax)
            if (!(localY >= shTop && localY < shTop + COND_SUBTRACK_HEIGHT)) continue
            const kps = light.signal_heads?.[hi]?.keypoints
            if (!kps?.length) continue
            let hitKp: { timestamp: string; x: number; y: number } | null = null
            let bestDist = HIT_PX
            for (const kp of kps) {
              const d = Math.abs(t2x(parseTs(kp.timestamp)) - mx)
              if (d <= bestDist) { bestDist = d; hitKp = kp }
            }
            if (hitKp) {
              selectPath(seg.id)
              setPlayhead(Math.max(0, Math.min(duration, parseTs(hitKp.timestamp))))
              return
            }
          }
        }
      }
      const getSegZone = (seg: TimelineSegment): 'main' | 'subtrack_pose' | 'subtrack1' | 'subtrack2' => {
        const m = seg.meta as Record<string, unknown>
        if (m?._isEgoActionSubtrack) return 'subtrack1'
        if (m?._isEgoInfluenceSubtrack) return 'subtrack1'
        if (m?._isAgentInfluenceSubtrack) return 'subtrack1'
        if (m?._isEgoContSubtrack) return 'subtrack1'
        if (m?._isAgentContSubtrack) return 'subtrack2'
        if (m?._isObjContSubtrack || m?._isLightContSubtrack || m?._isLightPhysContSubtrack) return 'subtrack2'
        if (m?._isAgentPoseSubtrack) return 'subtrack_pose'
        if (m?._isAgentPropertySubtrack || m?._isEgoPropertySubtrack) return 'subtrack1'
        if (m?._isAgentActionSubtrack) return 'subtrack1'
        if (m?._isCondSubtrack || m?._isObjStateSubtrack || m?._isLightStateSubtrack || m?._isSignalHeadSubtrack) return 'subtrack1'
        return 'main'
      }
      const getSegBounds = (seg: TimelineSegment): { top: number; bottom: number } => {
        const sm = seg.meta as Record<string, unknown>
        if (sm?._isEgoMain) return { top: y, bottom: y + EGO_SUBTRACK_BASE }
        const zone = getSegZone(seg)
        if (zone === 'main') return { top: y, bottom: y + (hasSubtrack ? ENV_MAIN_HEIGHT : trackH) }
        if (zone === 'subtrack_pose') {
          const _spCr = lightLaneCounts.get(trackId) ?? 1
          const poseY = AGENT_SUBTRACK_BASE + _spCr * COND_SUBTRACK_HEIGHT
          return { top: y + poseY, bottom: y + poseY + COND_SUBTRACK_HEIGHT }
        }
        if (grp === 'TrafficLights' && sm?._isLightPhysContSubtrack) {
          const pcY = y + ENV_MAIN_HEIGHT + (sm._cont_track_index as number ?? 0) * COND_SUBTRACK_HEIGHT
          return { top: pcY, bottom: pcY + COND_SUBTRACK_HEIGHT }
        }
        if (grp === 'TrafficLights' && (sm?._isSignalHeadSubtrack || sm?._isLightStateSubtrack || sm?._isLightContSubtrack)) {
          const lhi = sm._headIndex as number ?? 0
          const ll = bundle?.annotation?.traffic_lights?.[(sm._lightIndex as number) ?? 0]
          const trackPCMax = lightLaneCounts.get(`${trackId}_physcontmax`) ?? 1
          const shBaseY = ll ? getSignalHeadY(ll, lhi, trackPCMax) : ENV_MAIN_HEIGHT
          if (sm._isSignalHeadSubtrack) return { top: y + shBaseY, bottom: y + shBaseY + COND_SUBTRACK_HEIGHT }
          if (sm._isLightContSubtrack) return { top: y + shBaseY + COND_SUBTRACK_HEIGHT, bottom: y + shBaseY + 2 * COND_SUBTRACK_HEIGHT }
          if (sm._isLightStateSubtrack) {
            const stIdx = sm._state_track_index as number ?? 0
            const stY = shBaseY + 2 * COND_SUBTRACK_HEIGHT + stIdx * COND_SUBTRACK_HEIGHT
            return { top: y + stY, bottom: y + stY + COND_SUBTRACK_HEIGHT }
          }
        }
        if (sm?._isEgoActionSubtrack) {
          const _eaCr = lightLaneCounts.get('ego_act') ?? 1; const _eaIr = lightLaneCounts.get('ego_act_infl') ?? 0
          const actionY = EGO_SUBTRACK_BASE + _eaCr * COND_SUBTRACK_HEIGHT + _eaIr * COND_SUBTRACK_HEIGHT
          return { top: y + actionY, bottom: y + actionY + COND_SUBTRACK_HEIGHT }
        }
        if (sm?._isEgoInfluenceSubtrack) {
          const inflIdx = sm._influence_track_index as number ?? 0
          const _eiCr = lightLaneCounts.get('ego_act') ?? 1
          const inflY = EGO_SUBTRACK_BASE + _eiCr * COND_SUBTRACK_HEIGHT + inflIdx * COND_SUBTRACK_HEIGHT
          return { top: y + inflY, bottom: y + inflY + COND_SUBTRACK_HEIGHT }
        }
        if (sm?._isAgentInfluenceSubtrack) {
          const inflIdx = sm._influence_track_index as number ?? 0
          const _aiCr = lightLaneCounts.get(trackId) ?? 1
          const inflY = AGENT_SUBTRACK_BASE + _aiCr * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT + inflIdx * COND_SUBTRACK_HEIGHT
          return { top: y + inflY, bottom: y + inflY + COND_SUBTRACK_HEIGHT }
        }
        if (sm?._isAgentPropertySubtrack) {
          const propIdx = sm._prop_track_index as number ?? 0
          const _apCr = lightLaneCounts.get(trackId) ?? 1
          const _apIr = lightLaneCounts.get(`${trackId}_infl`) ?? 0
          const propY = AGENT_SUBTRACK_BASE + _apCr * COND_SUBTRACK_HEIGHT + _apIr * COND_SUBTRACK_HEIGHT + 2 * COND_SUBTRACK_HEIGHT + propIdx * COND_SUBTRACK_HEIGHT
          return { top: y + propY, bottom: y + propY + COND_SUBTRACK_HEIGHT }
        }
        if (sm?._isEgoPropertySubtrack) {
          const propIdx = sm._prop_track_index as number ?? 0
          const _epCr = lightLaneCounts.get('ego_act') ?? 1
          const _epIr = lightLaneCounts.get('ego_act_infl') ?? 0
          const propY = EGO_SUBTRACK_BASE + _epCr * COND_SUBTRACK_HEIGHT + _epIr * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT + propIdx * COND_SUBTRACK_HEIGHT
          return { top: y + propY, bottom: y + propY + COND_SUBTRACK_HEIGHT }
        }
        if (zone === 'subtrack2') {
          if (sm?._isAgentContSubtrack) {
            const contIdx = sm._cont_track_index as number ?? 0
            const contY = AGENT_SUBTRACK_BASE + contIdx * COND_SUBTRACK_HEIGHT
            return { top: y + contY, bottom: y + contY + COND_SUBTRACK_HEIGHT }
          }
          if (sm?._isObjContSubtrack) {
            const contIdx = sm._cont_track_index as number ?? 0
            const contY = OBJ_CONT_ROW_Y + contIdx * COND_SUBTRACK_HEIGHT
            return { top: y + contY, bottom: y + contY + COND_SUBTRACK_HEIGHT }
          }
          const contY = grp === 'Objects' ? OBJ_CONT_ROW_Y : AGENT_SUBTRACK_BASE
          return { top: y + contY, bottom: y + contY + COND_SUBTRACK_HEIGHT }
        }
        if (sm?._isEgoContSubtrack) {
          const contIdx = sm._cont_track_index as number ?? 0
          const contY = EGO_SUBTRACK_BASE + contIdx * COND_SUBTRACK_HEIGHT
          return { top: y + contY, bottom: y + contY + COND_SUBTRACK_HEIGHT }
        }
        if (grp === 'Agents') { const _aaCr = lightLaneCounts.get(trackId) ?? 1; const _aaIr = lightLaneCounts.get(`${trackId}_infl`) ?? 0; const aaY = AGENT_SUBTRACK_BASE + _aaCr * COND_SUBTRACK_HEIGHT + _aaIr * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT; return { top: y + aaY, bottom: y + aaY + COND_SUBTRACK_HEIGHT } }
        if (grp === 'Objects') { const _ocrY = getObjStateRowY(lightLaneCounts.get(trackId) ?? 1); return { top: y + _ocrY, bottom: y + _ocrY + COND_SUBTRACK_HEIGHT } }
        if (sm?._isCondSubtrack) {
          const condIdx = sm._cond_track_index as number ?? 0
          const condY = ENV_MAIN_HEIGHT + condIdx * COND_SUBTRACK_HEIGHT
          return { top: y + condY, bottom: y + condY + COND_SUBTRACK_HEIGHT }
        }
        return { top: y + ENV_MAIN_HEIGHT, bottom: y + ENV_MAIN_HEIGHT + COND_SUBTRACK_HEIGHT }
      }

      // Ctrl+click: add entity ID to the selected action's because_of
      if (e.ctrlKey && selectedPath && !useStore.getState().editsBlocked()) {
        const selSeg = segments.find(s => s.id === selectedPath)
        const selMeta = selSeg?.meta as Record<string, unknown> | undefined
        const isParentAgentCtrl = selSeg && selSeg.trackId.startsWith('agent_') && selMeta?._objKind === 'agent' && !selMeta?._isAgentActionSubtrack && !selMeta?._isAgentPropertySubtrack && !selMeta?._isAgentContSubtrack && !selMeta?._isAgentInfluenceSubtrack
        if (selSeg && (selSeg.trackId === 'ego_act' || selMeta?._isAgentActionSubtrack || isParentAgentCtrl)) {
          for (const seg of trackSegs) {
            const x0 = t2x(seg.t0), x1 = t2x(seg.t1)
            if (mx >= x0 && mx <= x1 && (({ top, bottom }) => my >= top && my < bottom)(getSegBounds(seg))) {
              const clickedMeta = seg.meta as Record<string, unknown> | undefined
              if (!clickedMeta) continue
              // Only allow the same entities as the because_of dropdown:
              // agent actions, TL states, traffic lights (parent), traffic objects (parent), and properties
              let entityId: string | undefined
              if ((clickedMeta._isAgentActionSubtrack || clickedMeta._isEgoActionSubtrack) && clickedMeta.id) {
                entityId = clickedMeta.id as string
              } else if (clickedMeta._isLightStateSubtrack && clickedMeta.id) {
                entityId = clickedMeta.id as string
              } else if (clickedMeta._objKind === 'traffic_light' && clickedMeta.id) {
                entityId = clickedMeta.id as string
              } else if ((clickedMeta._isAgentPropertySubtrack || clickedMeta._isEgoPropertySubtrack) && clickedMeta.id) {
                entityId = clickedMeta.id as string
              } else if (seg.trackId.startsWith('obj_') && !clickedMeta._isObjStateSubtrack && !clickedMeta._isObjContSubtrack) {
                entityId = String((clickedMeta as { id?: string }).id ?? '')
              } else if (clickedMeta._objKind === 'environment' && clickedMeta.id) {
                entityId = clickedMeta.id as string
              }
              if (!entityId) continue
              const ann = bundle?.annotation; if (!ann) return
              const updated = JSON.parse(JSON.stringify(ann)) as typeof ann
              if (selSeg.trackId === 'ego_act' && !selMeta?._isEgoContSubtrack && !selMeta?._isEgoInfluenceSubtrack) {
                const egoIdx = segments.filter(s => s.trackId === 'ego_act' && !(s.meta as Record<string, unknown>)?._isEgoContSubtrack && !(s.meta as Record<string, unknown>)?._isEgoInfluenceSubtrack).findIndex(s => s.id === selectedPath)
                const act = updated.ego_vehicle?.actions?.[egoIdx]
                if (act && !act.because_of?.includes(entityId)) act.because_of = [...(act.because_of || []), entityId]
              } else if (isParentAgentCtrl) {
                const ai = (selMeta!._agentIndex as number) ?? -1
                const actIdx = (updated.agents?.[ai]?.actions || []).findIndex(a => {
                  const at0 = parseTs(a.start_timestamp), at1 = parseTs(a.end_timestamp)
                  return playheadTime >= at0 && playheadTime <= at1
                })
                if (actIdx >= 0) {
                  const act = updated.agents![ai].actions[actIdx]
                  if (!act.because_of?.includes(entityId)) act.because_of = [...(act.because_of || []), entityId]
                }
              } else {
                const ai = (selMeta!._agentIndex as number) ?? -1
                const actIdx = (selMeta!._actIndex as number) ?? -1
                const act = updated.agents?.[ai]?.actions?.[actIdx]
                if (act && !act.because_of?.includes(entityId)) act.because_of = [...(act.because_of || []), entityId]
              }
              syncInfluencedAgentIds(updated)
              updateBundle({ ...bundle!, annotation: updated })
              return
            }
          }
        }
      }

      // Shift+click: link_to, containment→env, influencers
      if (e.shiftKey && selectedPath && !useStore.getState().editsBlocked()) {
        const selSeg = segments.find(s => s.id === selectedPath)
        const selMeta = selSeg?.meta as Record<string, unknown> | undefined

        // Shift+click influencers: if selected is influence subtrack, add object or traffic light to influencers
        const isInfluenceSubtrack = selMeta?._isEgoInfluenceSubtrack || selMeta?._isAgentInfluenceSubtrack
        if (selSeg && isInfluenceSubtrack) {
          for (const seg of trackSegs) {
            const x0 = t2x(seg.t0), x1 = t2x(seg.t1)
            if (mx >= x0 && mx <= x1 && (({ top, bottom }) => my >= top && my < bottom)(getSegBounds(seg))) {
              const clickedMeta = seg.meta as Record<string, unknown> | undefined
              if (!clickedMeta) continue
              let entityId: string | undefined
              if (seg.trackId.startsWith('obj_') && !clickedMeta._isObjStateSubtrack && !clickedMeta._isObjContSubtrack) {
                entityId = String((clickedMeta as { id?: string }).id ?? '')
              } else if (seg.trackId.startsWith('light_') && !clickedMeta._isLightStateSubtrack && !clickedMeta._isLightContSubtrack && !clickedMeta._isLightPhysContSubtrack && !clickedMeta._isSignalHeadSubtrack) {
                entityId = String((clickedMeta as { id?: string }).id ?? '')
              } else if (selMeta?._isAgentInfluenceSubtrack && seg.trackId.startsWith('agent_') && !clickedMeta._isAgentActionSubtrack && !clickedMeta._isAgentPropertySubtrack && !clickedMeta._isAgentPoseSubtrack && !clickedMeta._isAgentInfluenceSubtrack && !clickedMeta._isAgentContSubtrack) {
                entityId = String((clickedMeta as { id?: string }).id ?? '')
              }
              if (!entityId) continue
              const ann = bundle?.annotation; if (!ann) return
              const updated = JSON.parse(JSON.stringify(ann)) as typeof ann
              if (selMeta!._isEgoInfluenceSubtrack) {
                const inflIdx = (selMeta!._inflIndex as number) ?? -1
                const infl = updated.ego_vehicle?.influenced_by?.[inflIdx]
                if (infl && !infl.influencers.includes(entityId)) infl.influencers = [...infl.influencers, entityId]
              } else {
                const ai = (selMeta!._agentIndex as number) ?? -1
                const inflIdx = (selMeta!._inflIndex as number) ?? -1
                const infl = updated.agents?.[ai]?.influenced_by?.[inflIdx]
                if (infl && !infl.influencers.includes(entityId)) infl.influencers = [...infl.influencers, entityId]
              }
              updateBundle({ ...bundle!, annotation: updated })
              return
            }
          }
        }

        // Shift+click containment linking: if selected is containment subtrack and target is environment
        const isContSubtrack = selMeta?._isEgoContSubtrack || selMeta?._isAgentContSubtrack || selMeta?._isObjContSubtrack || selMeta?._isLightContSubtrack || selMeta?._isLightPhysContSubtrack
        if (selSeg && isContSubtrack) {
          for (const seg of trackSegs) {
            const x0 = t2x(seg.t0), x1 = t2x(seg.t1)
            if (mx >= x0 && mx <= x1 && (({ top, bottom }) => my >= top && my < bottom)(getSegBounds(seg))) {
              const clickedMeta = seg.meta as Record<string, unknown> | undefined
              if (!clickedMeta) continue
              // Target must be a parent environment segment
              if (!seg.trackId.startsWith('env_') || clickedMeta._isCondSubtrack) continue
              const envId = clickedMeta.id as string
              if (!envId) continue
              const ann = bundle?.annotation; if (!ann) return
              const updated = JSON.parse(JSON.stringify(ann)) as typeof ann
              if (selMeta!._isEgoContSubtrack) {
                const ci = (selMeta!._contIndex as number) ?? -1
                if (ci >= 0) { const c = updated.ego_vehicle?.containment?.[ci]; if (c) c.env_id = envId }
              } else if (selMeta!._isAgentContSubtrack) {
                const ai = (selMeta!._agentIndex as number) ?? -1
                const ci = (selMeta!._contIndex as number) ?? -1
                if (ai >= 0 && ci >= 0) { const c = updated.agents?.[ai]?.containment?.[ci]; if (c) c.env_id = envId }
              } else if (selMeta!._isObjContSubtrack) {
                const oi = (selMeta!._objIndex as number) ?? -1
                const ci = (selMeta!._contIndex as number) ?? -1
                if (oi >= 0 && ci >= 0) { const c = updated.traffic_objects?.[oi]?.containment?.[ci]; if (c) c.env_id = envId }
              } else if (selMeta!._isLightContSubtrack) {
                const li = (selMeta!._lightIndex as number) ?? -1
                const hi = (selMeta!._headIndex as number) ?? -1
                const ci = (selMeta!._contIndex as number) ?? -1
                if (li >= 0 && hi >= 0 && ci >= 0) { const c = updated.traffic_lights?.[li]?.signal_heads?.[hi]?.env_controlled?.[ci]; if (c) c.env_id = envId }
              } else if (selMeta!._isLightPhysContSubtrack) {
                const li = (selMeta!._lightIndex as number) ?? -1
                const ci = (selMeta!._contIndex as number) ?? -1
                if (li >= 0 && ci >= 0) { const c = updated.traffic_lights?.[li]?.containment?.[ci]; if (c) c.env_id = envId }
              }
              persistBundle({ ...bundle!, annotation: updated })
              return
            }
          }
        }

        // Shift+click link_to: if selected is action or signal property and target is agent or ego → add to link_to
        const isAgentActionShift = selSeg && selMeta?._isAgentActionSubtrack
        const isEgoActionShift = selSeg && selSeg.trackId === 'ego_act' && !selMeta?._isEgoContSubtrack && !selMeta?._isEgoInfluenceSubtrack && !selMeta?._isEgoPropertySubtrack
        const isAgentPropertyShift = selSeg && selMeta?._isAgentPropertySubtrack
        const isEgoPropertyShift = selSeg && selMeta?._isEgoPropertySubtrack
        if (selSeg && (isAgentActionShift || isEgoActionShift || isAgentPropertyShift || isEgoPropertyShift)) {
          for (const seg of trackSegs) {
            const x0 = t2x(seg.t0), x1 = t2x(seg.t1)
            if (mx >= x0 && mx <= x1 && (({ top, bottom }) => my >= top && my < bottom)(getSegBounds(seg))) {
              const clickedMeta = seg.meta as Record<string, unknown> | undefined
              if (!clickedMeta) continue
              // Target must be a parent agent segment or ego
              let targetId: string | undefined
              if (seg.trackId.startsWith('agent_') && clickedMeta._objKind === 'agent' && !clickedMeta._isAgentActionSubtrack && !clickedMeta._isAgentPropertySubtrack && !clickedMeta._isAgentContSubtrack) {
                targetId = clickedMeta.id as string
              } else if (seg.trackId === 'ego_act' && !clickedMeta._isEgoContSubtrack && !clickedMeta._isEgoActionSubtrack && !clickedMeta._isEgoInfluenceSubtrack && !clickedMeta._isEgoPropertySubtrack) {
                targetId = 'Ego'
              }
              if (!targetId) continue
              const ann = bundle?.annotation; if (!ann) return
              const updated = JSON.parse(JSON.stringify(ann)) as typeof ann
              if (isEgoPropertyShift) {
                const pi = (selMeta!._propIndex as number) ?? -1
                const prop = updated.ego_vehicle?.properties?.[pi]
                if (prop) {
                  prop.signaling_details = prop.signaling_details || {}
                  if (!prop.signaling_details.link_to?.includes(targetId)) prop.signaling_details.link_to = [...(prop.signaling_details.link_to || []), targetId]
                }
              } else if (isAgentPropertyShift) {
                const ai = (selMeta!._agentIndex as number) ?? -1
                const pi = (selMeta!._propIndex as number) ?? -1
                const prop = updated.agents?.[ai]?.properties?.[pi]
                if (prop) {
                  prop.signaling_details = prop.signaling_details || {}
                  if (!prop.signaling_details.link_to?.includes(targetId)) prop.signaling_details.link_to = [...(prop.signaling_details.link_to || []), targetId]
                }
              } else if (isEgoActionShift) {
                const egoIdx = segments.filter(s => s.trackId === 'ego_act' && !(s.meta as Record<string, unknown>)?._isEgoContSubtrack && !(s.meta as Record<string, unknown>)?._isEgoInfluenceSubtrack && !(s.meta as Record<string, unknown>)?._isEgoPropertySubtrack).findIndex(s => s.id === selectedPath)
                const act = updated.ego_vehicle?.actions?.[egoIdx]
                if (act && !act.link_to?.includes(targetId)) act.link_to = [...(act.link_to || []), targetId]
              } else {
                const ai = (selMeta!._agentIndex as number) ?? -1
                const actIdx = (selMeta!._actIndex as number) ?? -1
                const act = updated.agents?.[ai]?.actions?.[actIdx]
                if (act && !act.link_to?.includes(targetId)) {
                  act.link_to = [...(act.link_to || []), targetId]
                  if (act.signaling_details) act.signaling_details.link_to = act.link_to
                }
              }
              updateBundle({ ...bundle!, annotation: updated })
              return
            }
          }
        }
      }

      const EDGE_ZONE = 8

      // Check all segments for hits — collect all edge candidates first
      type EdgeHit = { seg: TimelineSegment; mode: 'left' | 'right'; dist: number; edgeX: number }
      const edgeHits: EdgeHit[] = []
      let bodyHit: TimelineSegment | null = null

      for (const seg of trackSegs) {
        const { top: segTop, bottom: segBottom } = getSegBounds(seg)
        if (my < segTop || my > segBottom) continue
        const x0 = t2x(seg.t0), x1 = t2x(seg.t1)
        if (mx < x0 - EDGE_ZONE || mx > x1 + EDGE_ZONE) continue
        const isEgoMainSeg = !!(seg.meta as Record<string, unknown>)?._isEgoMain
        if (!isEgoMainSeg && Math.abs(mx - x0) <= EDGE_ZONE) {
          edgeHits.push({ seg, mode: 'left', dist: Math.abs(mx - x0), edgeX: x0 })
        }
        if (!isEgoMainSeg && Math.abs(mx - x1) <= EDGE_ZONE) {
          edgeHits.push({ seg, mode: 'right', dist: Math.abs(mx - x1), edgeX: x1 })
        }
        if (!bodyHit && mx >= x0 && mx <= x1) bodyHit = seg
      }

      if (edgeHits.length > 0) {
        // When multiple edges coincide (adjacent segments), disambiguate:
        // prefer the left-edge of the right segment (allows creating a gap between them)
        edgeHits.sort((a, b) => {
          // Prefer the currently selected segment's edges
          const aSel = a.seg.id === selectedPath ? -1 : 0
          const bSel = b.seg.id === selectedPath ? -1 : 0
          if (aSel !== bSel) return aSel - bSel
          // If edges are at different positions, pick closest to mouse
          if (Math.abs(a.edgeX - b.edgeX) > 1) return a.dist - b.dist
          // Edges coincide: prefer left-edge over right-edge
          if (a.mode === 'left' && b.mode === 'right') return -1
          if (a.mode === 'right' && b.mode === 'left') return 1
          return a.dist - b.dist
        })
        const best = edgeHits[0]
        selectPath(best.seg.id); if (!useStore.getState().editsBlocked()) setDragState({ seg: best.seg, mode: best.mode, startX: mx, origT0: best.seg.t0, origT1: best.seg.t1, curT0: best.seg.t0, curT1: best.seg.t1 }); return
      }
      if (bodyHit) {
        if ((bodyHit.meta as Record<string, unknown>)?._isEgoMain) { selectPath(bodyHit.id); return }
        selectPath(bodyHit.id); if (!useStore.getState().editsBlocked()) setDragState({ seg: bodyHit, mode: 'move', startX: mx, origT0: bodyHit.t0, origT1: bodyHit.t1, curT0: bodyHit.t0, curT1: bodyHit.t1, targetTrackId: bodyHit.trackId }); return
      }
      selectPath(null)
    },
    [bundle, duration, getLiveScrollOffset, commitNativeScroll, scrubTo, segments, updateBundle, setPlayhead, selectPath, selectedPath, tracks, trackList, zoomLevel, verticalScroll, lightLaneCounts, toggleGroupCollapse, keypointsVisible]
  )

  const bundleRef = useRef(bundle); bundleRef.current = bundle
  const selectedClipIdRef = useRef(selectedClipId); selectedClipIdRef.current = selectedClipId
  const trackListRef = useRef(trackList); trackListRef.current = trackList
  const verticalScrollRef = useRef(verticalScroll); verticalScrollRef.current = verticalScroll

  // --- Drag effect ---
  useEffect(() => {
    if (!dragState) return
    const container = containerRef.current
    const isCond = !!(dragState.seg.meta as Record<string, unknown>)?._isCondSubtrack
    const isObjState = !!(dragState.seg.meta as Record<string, unknown>)?._isObjStateSubtrack
    const isSignalHead = !!(dragState.seg.meta as Record<string, unknown>)?._isSignalHeadSubtrack
    const isLightState = !!(dragState.seg.meta as Record<string, unknown>)?._isLightStateSubtrack
    const isObjCont = !!(dragState.seg.meta as Record<string, unknown>)?._isObjContSubtrack
    const isLightCont = !!(dragState.seg.meta as Record<string, unknown>)?._isLightContSubtrack
    const isLightPhysCont = !!(dragState.seg.meta as Record<string, unknown>)?._isLightPhysContSubtrack
    const isAgentAction = !!(dragState.seg.meta as Record<string, unknown>)?._isAgentActionSubtrack
    const isAgentProperty = !!(dragState.seg.meta as Record<string, unknown>)?._isAgentPropertySubtrack
    const isAgentPose = !!(dragState.seg.meta as Record<string, unknown>)?._isAgentPoseSubtrack
    const isEgoCont = !!(dragState.seg.meta as Record<string, unknown>)?._isEgoContSubtrack
    const isAgentCont = !!(dragState.seg.meta as Record<string, unknown>)?._isAgentContSubtrack
    const isEgoAction = !!(dragState.seg.meta as Record<string, unknown>)?._isEgoActionSubtrack
    const isEgoProperty = !!(dragState.seg.meta as Record<string, unknown>)?._isEgoPropertySubtrack
    const isEgoInfluence = !!(dragState.seg.meta as Record<string, unknown>)?._isEgoInfluenceSubtrack
    const isAgentInfluence = !!(dragState.seg.meta as Record<string, unknown>)?._isAgentInfluenceSubtrack
    const isSubtrackDrag = isCond || isObjState || isSignalHead || isLightState || isObjCont || isLightCont || isLightPhysCont || isAgentAction || isAgentProperty || isAgentPose || isEgoCont || isAgentCont || isEgoInfluence || isEgoProperty || isAgentInfluence || isEgoAction
    // Compute the actual subtrack type for clampToAvoidOverlap filtering
    const dragSubType: boolean | 'agent_action' | 'agent_property' | 'agent_pose' | 'agent_cont' | 'obj_cont' | 'light_cont' | 'light_phys_cont' | 'signal_head' | 'ego_influence' | 'ego_property' | 'agent_influence' | 'ego_action' =
      isEgoAction ? 'ego_action' : isEgoProperty ? 'ego_property' : isAgentProperty ? 'agent_property' : isAgentAction ? 'agent_action' : isAgentPose ? 'agent_pose' : isAgentCont ? 'agent_cont'
      : isObjCont ? 'obj_cont' : isLightCont ? 'light_cont' : isLightPhysCont ? 'light_phys_cont' : isSignalHead ? 'signal_head'
      : isEgoInfluence ? 'ego_influence' : isAgentInfluence ? 'agent_influence'
      : isSubtrackDrag

    // For subtrack segments, find parent bounds and sibling overlap set
    const getSubtrackClampRange = (): { parentT0: number; parentT1: number; siblingSegs: TimelineSegment[] } | null => {
      if (!isSubtrackDrag) return null
      const segs = baseSegmentsRef.current
      const meta = dragState.seg.meta as Record<string, unknown>

      if (isCond) {
        const envId = meta.env_id as number
        const condLane = meta._cond_track_index as number ?? 0
        const envSeg = segs.find(s => {
          if ((s.meta as Record<string, unknown>)?._isCondSubtrack) return false
          return s.trackId === dragState.seg.trackId && (s.meta as Record<string, unknown>)?.id === envId
        })
        if (!envSeg) return null
        const condSegs = segs.filter(s => {
          const sm = s.meta as Record<string, unknown>
          return !!sm?._isCondSubtrack &&
            sm?.env_id === envId &&
            (sm._cond_track_index as number ?? 0) === condLane
        })
        return { parentT0: envSeg.t0, parentT1: envSeg.t1, siblingSegs: condSegs }
      }

      if (isObjState) {
        const objIndex = meta._objIndex as number
        const parentSeg = segs.find(s =>
          !(s.meta as Record<string, unknown>)?._isObjStateSubtrack &&
          !(s.meta as Record<string, unknown>)?._isObjContSubtrack &&
          s.trackId === dragState.seg.trackId &&
          (s.meta as Record<string, unknown>)?._objIndex === objIndex
        )
        if (!parentSeg) return null
        const stateSegs = segs.filter(s =>
          !!(s.meta as Record<string, unknown>)?._isObjStateSubtrack &&
          (s.meta as Record<string, unknown>)?._objIndex === objIndex
        )
        return { parentT0: parentSeg.t0, parentT1: parentSeg.t1, siblingSegs: stateSegs }
      }

      if (isObjCont) {
        const objIndex = meta._objIndex as number
        const contLane = meta._cont_track_index as number ?? 0
        const parentSeg = segs.find(s =>
          !(s.meta as Record<string, unknown>)?._isObjStateSubtrack &&
          !(s.meta as Record<string, unknown>)?._isObjContSubtrack &&
          s.trackId === dragState.seg.trackId &&
          (s.meta as Record<string, unknown>)?._objIndex === objIndex
        )
        if (!parentSeg) return null
        const contSegs = segs.filter(s => {
          const sm = s.meta as Record<string, unknown>
          return !!sm?._isObjContSubtrack &&
            (sm._objIndex as number) === objIndex &&
            (sm._cont_track_index as number ?? 0) === contLane
        })
        return { parentT0: parentSeg.t0, parentT1: parentSeg.t1, siblingSegs: contSegs }
      }

      if (isSignalHead) {
        const lightIndex = meta._lightIndex as number
        const parentSeg = segs.find(s => {
          const sm = s.meta as Record<string, unknown>
          return !sm?._isLightStateSubtrack && !sm?._isLightContSubtrack && !sm?._isLightPhysContSubtrack && !sm?._isSignalHeadSubtrack &&
            s.trackId === dragState.seg.trackId && sm?._lightIndex === lightIndex
        })
        if (!parentSeg) return null
        // Signal heads are allowed to overlap — no sibling collision
        return { parentT0: parentSeg.t0, parentT1: parentSeg.t1, siblingSegs: [] }
      }

      if (isLightState) {
        const lightIndex = meta._lightIndex as number
        const headIndex = meta._headIndex as number
        const laneIdx = (meta._state_track_index as number) ?? 0
        // Clamp to parent signal head bounds
        const shSeg = segs.find(s => {
          const sm = s.meta as Record<string, unknown>
          return !!sm?._isSignalHeadSubtrack &&
            (sm._lightIndex as number) === lightIndex &&
            (sm._headIndex as number) === headIndex
        })
        if (!shSeg) return null
        const stateSegs = segs.filter(s =>
          !!(s.meta as Record<string, unknown>)?._isLightStateSubtrack &&
          (s.meta as Record<string, unknown>)?._lightIndex === lightIndex &&
          (s.meta as Record<string, unknown>)?._headIndex === headIndex &&
          ((s.meta as Record<string, unknown>)?._state_track_index as number ?? 0) === laneIdx
        )
        return { parentT0: shSeg.t0, parentT1: shSeg.t1, siblingSegs: stateSegs }
      }

      if (isLightCont) {
        const lightIndex = meta._lightIndex as number
        const headIndex = meta._headIndex as number
        // Clamp to parent signal head bounds
        const shSeg = segs.find(s => {
          const sm = s.meta as Record<string, unknown>
          return !!sm?._isSignalHeadSubtrack &&
            (sm._lightIndex as number) === lightIndex &&
            (sm._headIndex as number) === headIndex
        })
        if (!shSeg) return null
        const contSegs = segs.filter(s =>
          !!(s.meta as Record<string, unknown>)?._isLightContSubtrack &&
          (s.meta as Record<string, unknown>)?._lightIndex === lightIndex &&
          (s.meta as Record<string, unknown>)?._headIndex === headIndex
        )
        return { parentT0: shSeg.t0, parentT1: shSeg.t1, siblingSegs: contSegs }
      }

      if (isLightPhysCont) {
        const lightIndex = meta._lightIndex as number
        // Clamp to parent traffic light bounds
        const lightSeg = segs.find(s => {
          const sm = s.meta as Record<string, unknown>
          return !sm?._isSignalHeadSubtrack && !sm?._isLightStateSubtrack &&
            !sm?._isLightContSubtrack && !sm?._isLightPhysContSubtrack &&
            s.trackId === dragState.seg.trackId && (sm?._lightIndex as number) === lightIndex
        })
        if (!lightSeg) return null
        const contTrackIdx = meta._cont_track_index as number ?? 0
        const physContSegs = segs.filter(s =>
          !!(s.meta as Record<string, unknown>)?._isLightPhysContSubtrack &&
          (s.meta as Record<string, unknown>)?._lightIndex === lightIndex &&
          ((s.meta as Record<string, unknown>)?._cont_track_index as number ?? 0) === contTrackIdx
        )
        return { parentT0: lightSeg.t0, parentT1: lightSeg.t1, siblingSegs: physContSegs }
      }

      if (isAgentAction) {
        const agentIndex = meta._agentIndex as number
        const parentSeg = segs.find(s =>
          !(s.meta as Record<string, unknown>)?._isAgentActionSubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentPropertySubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentPoseSubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentContSubtrack &&
          s.trackId === dragState.seg.trackId &&
          (s.meta as Record<string, unknown>)?._agentIndex === agentIndex
        )
        if (!parentSeg) return null
        const actionSegs = segs.filter(s =>
          !!(s.meta as Record<string, unknown>)?._isAgentActionSubtrack &&
          (s.meta as Record<string, unknown>)?._agentIndex === agentIndex
        )
        const agentData = bundleRef.current?.annotation?.agents?.[agentIndex]
        const hasExplicitVis = !!(agentData?.visibility_start_timestamp && agentData?.visibility_end_timestamp)
        const pT0 = hasExplicitVis ? parentSeg.t0 : 0
        const pT1 = hasExplicitVis ? parentSeg.t1 : (duration || 300)
        return { parentT0: pT0, parentT1: pT1, siblingSegs: actionSegs }
      }

      if (isAgentProperty) {
        const agentIndex = meta._agentIndex as number
        const propLane = meta._prop_track_index as number ?? 0
        const parentSeg = segs.find(s =>
          !(s.meta as Record<string, unknown>)?._isAgentActionSubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentPropertySubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentPoseSubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentContSubtrack &&
          s.trackId === dragState.seg.trackId &&
          (s.meta as Record<string, unknown>)?._agentIndex === agentIndex
        )
        if (!parentSeg) return null
        const propSegs = segs.filter(s => {
          const sm = s.meta as Record<string, unknown>
          return !!sm?._isAgentPropertySubtrack &&
            (sm._agentIndex as number) === agentIndex &&
            (sm._prop_track_index as number ?? 0) === propLane
        })
        const agentData = bundleRef.current?.annotation?.agents?.[agentIndex]
        const hasExplicitVis = !!(agentData?.visibility_start_timestamp && agentData?.visibility_end_timestamp)
        const pT0 = hasExplicitVis ? parentSeg.t0 : 0
        const pT1 = hasExplicitVis ? parentSeg.t1 : (duration || 300)
        return { parentT0: pT0, parentT1: pT1, siblingSegs: propSegs }
      }

      if (isEgoProperty) {
        const propLane = meta._prop_track_index as number ?? 0
        const propSegs = segs.filter(s => {
          const sm = s.meta as Record<string, unknown>
          return !!sm?._isEgoPropertySubtrack && (sm._prop_track_index as number ?? 0) === propLane
        })
        return { parentT0: 0, parentT1: duration || 300, siblingSegs: propSegs }
      }

      if (isAgentPose) {
        const agentIndex = meta._agentIndex as number
        const parentSeg = segs.find(s =>
          !(s.meta as Record<string, unknown>)?._isAgentActionSubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentPropertySubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentPoseSubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentContSubtrack &&
          s.trackId === dragState.seg.trackId &&
          (s.meta as Record<string, unknown>)?._agentIndex === agentIndex
        )
        if (!parentSeg) return null
        const poseSegs = segs.filter(s =>
          !!(s.meta as Record<string, unknown>)?._isAgentPoseSubtrack &&
          (s.meta as Record<string, unknown>)?._agentIndex === agentIndex
        )
        const agentData = bundleRef.current?.annotation?.agents?.[agentIndex]
        const hasExplicitVis = !!(agentData?.visibility_start_timestamp && agentData?.visibility_end_timestamp)
        const pT0 = hasExplicitVis ? parentSeg.t0 : 0
        const pT1 = hasExplicitVis ? parentSeg.t1 : (duration || 300)
        return { parentT0: pT0, parentT1: pT1, siblingSegs: poseSegs }
      }

      if (isEgoAction) {
        const actionSegs = segs.filter(s => !!(s.meta as Record<string, unknown>)?._isEgoActionSubtrack)
        return { parentT0: 0, parentT1: duration || 300, siblingSegs: actionSegs }
      }

      if (isEgoCont) {
        // Ego containment: filter by same lane, no strict parent bounds
        const contLane = meta._cont_track_index as number ?? 0
        const contSegs = segs.filter(s => {
          const sm = s.meta as Record<string, unknown>
          return !!sm?._isEgoContSubtrack && (sm._cont_track_index as number ?? 0) === contLane
        })
        return { parentT0: 0, parentT1: duration || 300, siblingSegs: contSegs }
      }

      if (isAgentCont) {
        const agentIndex = meta._agentIndex as number
        const contLane = meta._cont_track_index as number ?? 0
        const parentSeg = segs.find(s =>
          !(s.meta as Record<string, unknown>)?._isAgentActionSubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentPoseSubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentContSubtrack &&
          s.trackId === dragState.seg.trackId &&
          (s.meta as Record<string, unknown>)?._agentIndex === agentIndex
        )
        if (!parentSeg) return null
        const contSegs = segs.filter(s => {
          const sm = s.meta as Record<string, unknown>
          return !!sm?._isAgentContSubtrack &&
            (sm._agentIndex as number) === agentIndex &&
            (sm._cont_track_index as number ?? 0) === contLane
        })
        const agentData = bundleRef.current?.annotation?.agents?.[agentIndex]
        const hasExplicitVis = !!(agentData?.visibility_start_timestamp && agentData?.visibility_end_timestamp)
        const pT0 = hasExplicitVis ? parentSeg.t0 : 0
        const pT1 = hasExplicitVis ? parentSeg.t1 : (duration || 300)
        return { parentT0: pT0, parentT1: pT1, siblingSegs: contSegs }
      }

      if (isEgoInfluence) {
        const inflLane = meta._influence_track_index as number ?? 0
        const inflSegs = segs.filter(s => {
          const sm = s.meta as Record<string, unknown>
          return !!sm?._isEgoInfluenceSubtrack &&
            (sm._influence_track_index as number ?? 0) === inflLane
        })
        return { parentT0: 0, parentT1: duration || 300, siblingSegs: inflSegs }
      }

      if (isAgentInfluence) {
        const agentIndex = meta._agentIndex as number
        const inflLane = meta._influence_track_index as number ?? 0
        const parentSeg = segs.find(s =>
          !(s.meta as Record<string, unknown>)?._isAgentActionSubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentPoseSubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentContSubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentInfluenceSubtrack &&
          s.trackId === dragState.seg.trackId &&
          (s.meta as Record<string, unknown>)?._agentIndex === agentIndex
        )
        if (!parentSeg) return null
        const inflSegs = segs.filter(s => {
          const sm = s.meta as Record<string, unknown>
          return !!sm?._isAgentInfluenceSubtrack &&
            (sm._agentIndex as number) === agentIndex &&
            (sm._influence_track_index as number ?? 0) === inflLane
        })
        const agentData = bundleRef.current?.annotation?.agents?.[agentIndex]
        const hasExplicitVis = !!(agentData?.visibility_start_timestamp && agentData?.visibility_end_timestamp)
        const pT0 = hasExplicitVis ? parentSeg.t0 : 0
        const pT1 = hasExplicitVis ? parentSeg.t1 : (duration || 300)
        return { parentT0: pT0, parentT1: pT1, siblingSegs: inflSegs }
      }

      return null
    }

    const onMove = (e: MouseEvent) => {
      if (!container) return
      const rect = container.getBoundingClientRect()
      const mx = e.clientX - rect.left, my = e.clientY - rect.top + verticalScrollRef.current
      const w = container.clientWidth
      const pps = (w * zoomLevel) / Math.max(duration, 1)
      const dt = (mx - dragState.startX) / pps

      const segs = baseSegmentsRef.current

      if (isSubtrackDrag) {
        // Subtrack drag: clamp to parent bounds + avoid sibling overlaps
        const range = getSubtrackClampRange()
        if (!range) return
        const { parentT0, parentT1, siblingSegs } = range

        if (dragState.mode === 'left') {
          const rawT0 = Math.max(parentT0, Math.min(dragState.origT1 - 0.2, dragState.origT0 + dt))
          const clamped = clampToAvoidOverlap(siblingSegs, dragState.seg.trackId, dragState.seg.id, rawT0, dragState.origT1, 'left', dragSubType)
          if (clamped) {
            clamped.t0 = Math.max(clamped.t0, parentT0)
            setDragState(s => s && { ...s, curT0: clamped.t0, curT1: clamped.t1 })
          }
        } else if (dragState.mode === 'right') {
          const rawT1 = Math.min(parentT1, Math.max(dragState.origT0 + 0.2, dragState.origT1 + dt))
          const clamped = clampToAvoidOverlap(siblingSegs, dragState.seg.trackId, dragState.seg.id, dragState.origT0, rawT1, 'right', dragSubType)
          if (clamped) {
            clamped.t1 = Math.min(clamped.t1, parentT1)
            setDragState(s => s && { ...s, curT0: clamped.t0, curT1: clamped.t1 })
          }
        } else {
          const segLen = dragState.origT1 - dragState.origT0
          const rawT0 = Math.max(parentT0, Math.min(parentT1 - segLen, dragState.origT0 + dt))
          const clamped = clampToAvoidOverlap(siblingSegs, dragState.seg.trackId, dragState.seg.id, rawT0, rawT0 + segLen, 'move', dragSubType)
          if (clamped) {
            clamped.t0 = Math.max(clamped.t0, parentT0)
            clamped.t1 = Math.min(clamped.t1, parentT1)
            setDragState(s => s && { ...s, curT0: clamped.t0, curT1: clamped.t1 })
          }
        }
        return
      }

      // Non-condition segments: original logic (isCondition defaults to false)
      if (dragState.mode === 'left') {
        const rawT0 = Math.max(0, Math.min(dragState.origT1 - 0.2, dragState.origT0 + dt))
        const clamped = clampToAvoidOverlap(segs, dragState.seg.trackId, dragState.seg.id, rawT0, dragState.origT1, 'left')
        if (clamped) setDragState(s => s && { ...s, curT0: clamped.t0, curT1: clamped.t1 })
      } else if (dragState.mode === 'right') {
        const rawT1 = Math.min(duration || 999, Math.max(dragState.origT0 + 0.2, dragState.origT1 + dt))
        const clamped = clampToAvoidOverlap(segs, dragState.seg.trackId, dragState.seg.id, dragState.origT0, rawT1, 'right')
        if (clamped) setDragState(s => s && { ...s, curT0: clamped.t0, curT1: clamped.t1 })
      } else {
        const segLen = dragState.origT1 - dragState.origT0
        const rawT0 = Math.max(0, Math.min((duration || 999) - segLen, dragState.origT0 + dt))
        // Cross-track detection
        const tl = trackListRef.current
        let rowIdx = -1
        let cy = HEADER_HEIGHT
        for (let i = 0; i < tl.length; i++) {
          const rh = getTrackHeight(tl[i].group, tl[i].id, lightLaneCounts, tl[i]._collapsed)
          if (my >= cy && my < cy + rh) { rowIdx = i; break }
          cy += rh + getTrackGap(tl, i)
        }
        let targetTrackId = dragState.seg.trackId
        if (rowIdx >= 0 && rowIdx < tl.length) {
          const candidate = tl[rowIdx]
          if (getTrackGroup(dragState.seg.trackId) === getTrackGroup(candidate.id)) {
            targetTrackId = candidate.id
          }
        }
        const clamped = clampToAvoidOverlap(segs, targetTrackId, dragState.seg.id, rawT0, rawT0 + segLen, 'move')
        if (clamped) setDragState(s => s && { ...s, curT0: clamped.t0, curT1: clamped.t1, targetTrackId })
      }
    }
    const onUp = () => {
      const ds = dragRef.current, b = bundleRef.current, clipId = selectedClipIdRef.current
      if (ds && b?.annotation) {
        const timeChanged = Math.abs(ds.curT0 - ds.origT0) > 0.01 || Math.abs(ds.curT1 - ds.origT1) > 0.01
        const trackChanged = ds.targetTrackId && ds.targetTrackId !== ds.seg.trackId
        const targetTrack = ds.targetTrackId || ds.seg.trackId
        const dsMeta = ds.seg.meta as Record<string, unknown>
        const isSubtrackSeg: boolean | 'agent_action' | 'agent_property' | 'agent_pose' | 'agent_cont' | 'obj_cont' | 'light_cont' | 'light_phys_cont' | 'signal_head' | 'ego_influence' | 'ego_property' | 'agent_influence' = dsMeta?._isAgentActionSubtrack ? 'agent_action'
          : dsMeta?._isAgentPropertySubtrack ? 'agent_property'
          : dsMeta?._isAgentPoseSubtrack ? 'agent_pose'
          : dsMeta?._isAgentContSubtrack ? 'agent_cont'
          : dsMeta?._isObjContSubtrack ? 'obj_cont'
          : dsMeta?._isLightContSubtrack ? 'light_cont'
          : dsMeta?._isLightPhysContSubtrack ? 'light_phys_cont'
          : dsMeta?._isSignalHeadSubtrack ? 'signal_head'
          : dsMeta?._isEgoInfluenceSubtrack ? 'ego_influence'
          : dsMeta?._isEgoPropertySubtrack ? 'ego_property'
          : dsMeta?._isAgentInfluenceSubtrack ? 'agent_influence'
          : !!(dsMeta?._isCondSubtrack || dsMeta?._isObjStateSubtrack || dsMeta?._isLightStateSubtrack || dsMeta?._isEgoContSubtrack)
        let finalCheck
        if (isSubtrackDrag) {
          const range = getSubtrackClampRange()
          if (!range) { setDragState(null); return }
          finalCheck = clampToAvoidOverlap(range.siblingSegs, targetTrack, ds.seg.id, ds.curT0, ds.curT1, 'move', isSubtrackSeg)
        } else {
          finalCheck = clampToAvoidOverlap(baseSegmentsRef.current, targetTrack, ds.seg.id, ds.curT0, ds.curT1, 'move', isSubtrackSeg)
        }
        if (!finalCheck) { setDragState(null); return }
        let updated = b.annotation
        if (timeChanged) updated = applySegmentTimeUpdate(updated, ds.seg.id, finalCheck.t0, finalCheck.t1)
        if (trackChanged) updated = moveSegmentToTrack(updated, ds.seg.id, ds.seg.trackId, ds.targetTrackId!)
        if (timeChanged || trackChanged) {
          const nb = { ...b, annotation: updated }
          updateBundle(nb)
          if (clipId) guardedSave(clipId, nb).catch(() => {})
        }
      }
      setDragState(null)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
  }, [dragState, duration, updateBundle, zoomLevel])

  // --- Right-click to create ---
  const handleRightClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      e.preventDefault()
      if (useStore.getState().editsBlocked()) return
      if (bundle?.annotation?.eventful === false) return
      const canvas = canvasRef.current, container = containerRef.current
      if (!canvas || !container) return
      const rect = canvas.getBoundingClientRect()
      const mx = e.clientX - rect.left, myRaw = e.clientY - rect.top
      if (myRaw < HEADER_HEIGHT) return
      const my = myRaw + verticalScroll
      const w = container.clientWidth
      const pps = (w * zoomLevel) / Math.max(duration, 1)
      const liveScrollOffset = getLiveScrollOffset()
      const t = mx / pps + liveScrollOffset
      // Find row with variable heights
      let rowIdx = -1
      let cumY = HEADER_HEIGHT
      for (let i = 0; i < trackList.length; i++) {
        const h = getTrackHeight(trackList[i].group, trackList[i].id, lightLaneCounts, trackList[i]._collapsed)
        if (my >= cumY && my < cumY + h) { rowIdx = i; break }
        cumY += h + getTrackGap(trackList, i)
      }
      if (rowIdx < 0 || rowIdx >= trackList.length) return
      if (trackList[rowIdx]._collapsed) return
      const track = trackList[rowIdx]
      const trackY = getTrackY(trackList, rowIdx, lightLaneCounts)
      const localY = my - trackY

      // Right-click on a keypoint diamond → delete it.
      if (keypointsVisible && !useStore.getState().editsBlocked()) {
        const curBundle = bundleRef.current
        const ann = curBundle?.annotation
        if (ann) {
          type KpTarget =
            | { kind: 'agent' | 'traffic_object' | 'environment'; entityIdx: number; kps: { timestamp: string; x: number; y: number }[] }
            | { kind: 'signal_head'; entityIdx: number; headIdx: number; kps: { timestamp: string; x: number; y: number }[] }
          let target: KpTarget | null = null

          const onMainBar = (track.group === 'Agents' || track.group === 'Objects' || track.group === 'Environments') && localY >= 0 && localY < ENV_MAIN_HEIGHT
          if (onMainBar) {
            for (const seg of tracks.get(track.id) || []) {
              const m = seg.meta as Record<string, unknown>
              if (m._objKind === 'agent' && m._agentIndex != null && t >= seg.t0 && t <= seg.t1) {
                const idx = m._agentIndex as number
                const kps = ann.agents?.[idx]?.keypoints
                if (kps) target = { kind: 'agent', entityIdx: idx, kps }
                break
              }
              if (m._objKind === 'traffic_object' && m._objIndex != null && t >= seg.t0 && t <= seg.t1) {
                const idx = m._objIndex as number
                const kps = ann.traffic_objects?.[idx]?.keypoints
                if (kps) target = { kind: 'traffic_object', entityIdx: idx, kps }
                break
              }
              if (m._objKind === 'environment' && m._envIndex != null && t >= seg.t0 && t <= seg.t1) {
                const idx = m._envIndex as number
                const kps = ann.environments?.[idx]?.keypoints
                if (kps) target = { kind: 'environment', entityIdx: idx, kps }
                break
              }
            }
          } else if (track.group === 'TrafficLights') {
            for (const seg of tracks.get(track.id) || []) {
              const m = seg.meta as Record<string, unknown>
              if (!m?._isSignalHeadSubtrack) continue
              if (!(t >= seg.t0 && t <= seg.t1)) continue
              const li = m._lightIndex as number | undefined
              const hi = m._headIndex as number | undefined
              if (li == null || hi == null) continue
              const light = ann.traffic_lights?.[li]
              if (!light) continue
              const trackPCMax = lightLaneCounts.get(`${track.id}_physcontmax`) ?? 1
              const shTop = getSignalHeadY(light, hi, trackPCMax)
              if (!(localY >= shTop && localY < shTop + COND_SUBTRACK_HEIGHT)) continue
              const kps = light.signal_heads?.[hi]?.keypoints
              if (kps?.length) { target = { kind: 'signal_head', entityIdx: li, headIdx: hi, kps }; break }
            }
          }

          if (target) {
            const HIT_PX = 6
            let hitIdx = -1
            let bestDist = HIT_PX
            for (let i = 0; i < target.kps.length; i++) {
              const kpX = (parseTs(target.kps[i].timestamp) - liveScrollOffset) * pps
              const d = Math.abs(kpX - mx)
              if (d <= bestDist) { bestDist = d; hitIdx = i }
            }
            if (hitIdx >= 0) {
              const updated = JSON.parse(JSON.stringify(ann)) as SilAvAnnotation
              const newList = removeKeypointAt(target.kps, hitIdx)
              if (target.kind === 'agent') updated.agents[target.entityIdx].keypoints = newList
              else if (target.kind === 'traffic_object') updated.traffic_objects[target.entityIdx].keypoints = newList
              else if (target.kind === 'signal_head') updated.traffic_lights[target.entityIdx].signal_heads[target.headIdx].keypoints = newList
              else updated.environments[target.entityIdx].keypoints = newList
              const newBundle = curBundle
                ? { ...curBundle, annotation: updated }
                : { schema_version: '2.0.0', video: { clip_id: selectedClipIdRef.current || '', fps: 30, duration_s: duration || 0 }, annotation: updated, status: 'pending', provenance: { generated_by: 'human' } }
              updateBundle(newBundle)
              const clipId = selectedClipIdRef.current
              if (clipId) guardedSave(clipId, newBundle).catch(() => {})
              return
            }
          }
        }
      }

      const hasSubtrack = track.group === 'Environments' || track.group === 'Objects' || track.group === 'TrafficLights' || track.group === 'Agents' || track.group === 'Ego'
      const _trackSegsForCheck = tracks.get(track.id) || []
      const _sm = (s: TimelineSegment) => s.meta as Record<string, unknown>
      const _hasParentAtT = (track.group === 'Ego')
        ? true
        : (track.group === 'Environments')
          ? _trackSegsForCheck.some(s => !_sm(s)._isCondSubtrack && t >= s.t0 && t <= s.t1)
          : (track.group === 'Objects')
            ? _trackSegsForCheck.some(s => !_sm(s)._isObjStateSubtrack && !_sm(s)._isObjContSubtrack && t >= s.t0 && t <= s.t1)
            : (track.group === 'TrafficLights')
              ? _trackSegsForCheck.some(s => !_sm(s)._isSignalHeadSubtrack && !_sm(s)._isLightStateSubtrack && !_sm(s)._isLightContSubtrack && !_sm(s)._isLightPhysContSubtrack && t >= s.t0 && t <= s.t1)
              : (track.group === 'Agents')
                ? _trackSegsForCheck.some(s => !_sm(s)._isAgentActionSubtrack && !_sm(s)._isAgentPropertySubtrack && !_sm(s)._isAgentPoseSubtrack && !_sm(s)._isAgentContSubtrack && !_sm(s)._isAgentInfluenceSubtrack && t >= s.t0 && t <= s.t1)
                : true
      const inSubtrackArea = hasSubtrack && _hasParentAtT && (
        track.group === 'Ego' ? localY >= EGO_SUBTRACK_BASE : localY >= ENV_MAIN_HEIGHT
      )
      const _dblAgentContRows = track.group === 'Agents' ? (lightLaneCounts.get(track.id) ?? 1) : 0
      const _dblObjContRows = track.group === 'Objects' ? (lightLaneCounts.get(track.id) ?? 1) : 0
      const inContRow = track.group === 'Objects' ? localY >= OBJ_CONT_ROW_Y && localY < getObjStateRowY(_dblObjContRows)
        : track.group === 'Agents' ? localY >= AGENT_SUBTRACK_BASE && localY < AGENT_SUBTRACK_BASE + _dblAgentContRows * COND_SUBTRACK_HEIGHT
        : false
      // TrafficLights use findLightHeadAtY for hit detection instead of inContRow

      // Don't create a new main segment if right-clicked on an existing main segment.
      // In subtrack areas we skip this guard: each row-specific handler uses
      // findAdjacentCondGap filtered to its own subtrack type, so a blunt time-only
      // check here would incorrectly block e.g. containment creation when a state
      // subtrack exists at the same time position.
      const t2x = (tt: number) => (tt - liveScrollOffset) * pps
      if (!inSubtrackArea) {
        if (track.group === 'Ego') return // main bar area is not interactive
        for (const seg of tracks.get(track.id) || []) {
          const isMainSeg = !(seg.meta as Record<string, unknown>)?._isCondSubtrack &&
            !(seg.meta as Record<string, unknown>)?._isObjStateSubtrack &&
            !(seg.meta as Record<string, unknown>)?._isSignalHeadSubtrack &&
            !(seg.meta as Record<string, unknown>)?._isLightStateSubtrack &&
            !(seg.meta as Record<string, unknown>)?._isObjContSubtrack &&
            !(seg.meta as Record<string, unknown>)?._isLightContSubtrack &&
            !(seg.meta as Record<string, unknown>)?._isLightPhysContSubtrack &&
            !(seg.meta as Record<string, unknown>)?._isAgentActionSubtrack &&
            !(seg.meta as Record<string, unknown>)?._isAgentPropertySubtrack &&
            !(seg.meta as Record<string, unknown>)?._isAgentPoseSubtrack &&
            !(seg.meta as Record<string, unknown>)?._isEgoInfluenceSubtrack &&
            !(seg.meta as Record<string, unknown>)?._isEgoPropertySubtrack &&
            !(seg.meta as Record<string, unknown>)?._isAgentInfluenceSubtrack &&
            !(seg.meta as Record<string, unknown>)?._isEgoContSubtrack &&
            !(seg.meta as Record<string, unknown>)?._isAgentContSubtrack
          if (isMainSeg && mx >= t2x(seg.t0) && mx <= t2x(seg.t1)) return
        }
      }

      // Double-click in subtrack area → create condition/state adjacent to existing
      if (inSubtrackArea) {
        const emptyAnn: SilAvAnnotation = { brief_description: '', environments: [], conditions: [], traffic_objects: [], traffic_lights: [], ego_vehicle: { actions: [] }, agents: [] }
        const curBundle = bundleRef.current
        const ann = curBundle?.annotation || emptyAnn
        const allTrackSegs = tracks.get(track.id) || []

        if (track.group === 'Environments') {
          const envSegs = allTrackSegs.filter(s => !(s.meta as Record<string, unknown>)?._isCondSubtrack)
          const parentEnv = envSegs.find(s => t >= s.t0 && t <= s.t1)
          if (!parentEnv) return
          const envId = (parentEnv.meta as Record<string, unknown>)?.id as number
          if (envId == null) return

          const _envCondRows = lightLaneCounts.get(`${track.id}_cond`) ?? 1
          const rowOffset = Math.floor((localY - ENV_MAIN_HEIGHT) / COND_SUBTRACK_HEIGHT)
          const clickedCondRow = Math.max(0, Math.min(_envCondRows - 1, rowOffset))
          const condSegs = allTrackSegs
            .filter(s => {
              const sm = s.meta as Record<string, unknown>
              return !!sm?._isCondSubtrack
                && sm?.env_id === envId
                && (sm._cond_track_index as number ?? 0) === clickedCondRow
            })
            .sort((a, b) => a.t0 - b.t0)

          const placement = findAdjacentCondGap(condSegs, parentEnv.t0, parentEnv.t1, t)
          if (!placement) return

          const updated = addConditionToEnv(ann, envId, placement.t0, placement.t1, clickedCondRow)
          const clipId = selectedClipIdRef.current
          const newBundle = curBundle
            ? { ...curBundle, annotation: updated }
            : { schema_version: '2.0.0', video: { clip_id: clipId || '', fps: 30, duration_s: duration || 0 }, annotation: updated, status: 'pending', provenance: { generated_by: 'human' } }
          updateBundle(newBundle)
          if (clipId) guardedSave(clipId, newBundle).catch(() => {})
        } else if (track.group === 'Objects') {
          const objSegs = allTrackSegs.filter(s => !(s.meta as Record<string, unknown>)?._isObjStateSubtrack && !(s.meta as Record<string, unknown>)?._isObjContSubtrack)
          const parentObj = objSegs.find(s => t >= s.t0 && t <= s.t1)
          if (!parentObj) return
          const objIndex = (parentObj.meta as Record<string, unknown>)?._objIndex as number
          if (objIndex == null) return

          if (inContRow) {
            const clickedContRow = Math.floor((localY - OBJ_CONT_ROW_Y) / COND_SUBTRACK_HEIGHT)
            const contSegs = allTrackSegs
              .filter(s => !!(s.meta as Record<string, unknown>)?._isObjContSubtrack
                && (s.meta as Record<string, unknown>)?._objIndex === objIndex
                && ((s.meta as Record<string, unknown>)?._cont_track_index as number ?? 0) === clickedContRow)
              .sort((a, b) => a.t0 - b.t0)
            const placement = findAdjacentCondGap(contSegs, parentObj.t0, parentObj.t1, t)
            if (!placement) return
            const updated = addContainmentToObject(ann, objIndex, placement.t0, placement.t1, clickedContRow)
            const clipId = selectedClipIdRef.current
            const newBundle = curBundle
              ? { ...curBundle, annotation: updated }
              : { schema_version: '2.0.0', video: { clip_id: clipId || '', fps: 30, duration_s: duration || 0 }, annotation: updated, status: 'pending', provenance: { generated_by: 'human' } }
            updateBundle(newBundle)
            if (clipId) guardedSave(clipId, newBundle).catch(() => {})
          } else {
            const stateSegs = allTrackSegs
              .filter(s => !!(s.meta as Record<string, unknown>)?._isObjStateSubtrack && (s.meta as Record<string, unknown>)?._objIndex === objIndex)
              .sort((a, b) => a.t0 - b.t0)
            const placement = findAdjacentCondGap(stateSegs, parentObj.t0, parentObj.t1, t)
            if (!placement) return
            const updated = addStateToObject(ann, objIndex, placement.t0, placement.t1)
            const clipId = selectedClipIdRef.current
            const newBundle = curBundle
              ? { ...curBundle, annotation: updated }
              : { schema_version: '2.0.0', video: { clip_id: clipId || '', fps: 30, duration_s: duration || 0 }, annotation: updated, status: 'pending', provenance: { generated_by: 'human' } }
            updateBundle(newBundle)
            if (clipId) guardedSave(clipId, newBundle).catch(() => {})
          }
        } else if (track.group === 'TrafficLights') {
          const lightMainSegs = allTrackSegs.filter(s => {
            const sm = s.meta as Record<string, unknown>
            return !sm?._isLightStateSubtrack && !sm?._isLightContSubtrack && !sm?._isLightPhysContSubtrack && !sm?._isSignalHeadSubtrack
          })
          const parentLight = lightMainSegs.find(s => t >= s.t0 && t <= s.t1)
          if (!parentLight) return
          const lightIndex = (parentLight.meta as Record<string, unknown>)?._lightIndex as number
          if (lightIndex == null) return
          const light = ann.traffic_lights?.[lightIndex]
          if (!light) return

          // Physical containment zone: rows between main bar and signal heads
          const trackPCMax = lightLaneCounts.get(`${track.id}_physcontmax`) ?? 1
          const physContZoneEnd = ENV_MAIN_HEIGHT + trackPCMax * COND_SUBTRACK_HEIGHT
          if (localY >= ENV_MAIN_HEIGHT && localY < physContZoneEnd) {
            const clickedRow = Math.floor((localY - ENV_MAIN_HEIGHT) / COND_SUBTRACK_HEIGHT)
            const physContSegs = allTrackSegs
              .filter(s => {
                const sm = s.meta as Record<string, unknown>
                return !!sm?._isLightPhysContSubtrack && sm?._lightIndex === lightIndex && (sm._cont_track_index as number ?? 0) === clickedRow
              })
              .sort((a, b) => a.t0 - b.t0)
            const placement = findAdjacentCondGap(physContSegs, parentLight.t0, parentLight.t1, t)
            if (!placement) return
            const updated = addPhysicalContainmentToLight(ann, lightIndex, placement.t0, placement.t1, clickedRow)
            const clipId = selectedClipIdRef.current
            const newBundle = curBundle
              ? { ...curBundle, annotation: updated }
              : { schema_version: '2.0.0', video: { clip_id: clipId || '', fps: 30, duration_s: duration || 0 }, annotation: updated, status: 'pending', provenance: { generated_by: 'human' } }
            updateBundle(newBundle)
            if (clipId) guardedSave(clipId, newBundle).catch(() => {})
            return
          }

          const hitInfo = findLightHeadAtY(light, localY, trackPCMax)
          if (hitInfo) {
            const { headIdx, inState: inStateDbl, inCont: inContDbl, inSHBar } = hitInfo
            const sh = light.signal_heads?.[headIdx]
            if (!sh && !inSHBar) return
            const shSeg = allTrackSegs.find(s => !!(s.meta as Record<string, unknown>)?._isSignalHeadSubtrack && (s.meta as Record<string, unknown>)?._lightIndex === lightIndex && (s.meta as Record<string, unknown>)?._headIndex === headIdx && t >= s.t0 && t <= s.t1)
            const effectiveInSHBar = inSHBar || (!shSeg && (inContDbl || inStateDbl))
            if (!shSeg && !effectiveInSHBar) return
            const shT0 = shSeg?.t0 ?? parentLight.t0
            const shT1 = shSeg?.t1 ?? parentLight.t1
            if (inContDbl && shSeg) {
              const contSegs = allTrackSegs
                .filter(s => !!(s.meta as Record<string, unknown>)?._isLightContSubtrack && (s.meta as Record<string, unknown>)?._lightIndex === lightIndex && (s.meta as Record<string, unknown>)?._headIndex === headIdx)
                .sort((a, b) => a.t0 - b.t0)
              const placement = findAdjacentCondGap(contSegs, shT0, shT1, t)
              if (!placement) return
              const updated = addContainmentToLight(ann, lightIndex, headIdx, placement.t0, placement.t1)
              const clipId = selectedClipIdRef.current
              const newBundle = curBundle
                ? { ...curBundle, annotation: updated }
                : { schema_version: '2.0.0', video: { clip_id: clipId || '', fps: 30, duration_s: duration || 0 }, annotation: updated, status: 'pending', provenance: { generated_by: 'human' } }
              updateBundle(newBundle)
              if (clipId) guardedSave(clipId, newBundle).catch(() => {})
            } else if (inStateDbl && shSeg) {
              const clickedLane = hitInfo.stateLane
              const stateSegs = allTrackSegs
                .filter(s => !!(s.meta as Record<string, unknown>)?._isLightStateSubtrack && (s.meta as Record<string, unknown>)?._lightIndex === lightIndex && (s.meta as Record<string, unknown>)?._headIndex === headIdx
                  && ((s.meta as Record<string, unknown>)?._state_track_index as number ?? 0) === clickedLane)
                .sort((a, b) => a.t0 - b.t0)
              const placement = findAdjacentCondGap(stateSegs, shT0, shT1, t)
              if (!placement) return
              const updated = addStateToLight(ann, lightIndex, headIdx, placement.t0, placement.t1, clickedLane)
              const clipId = selectedClipIdRef.current
              const newBundle = curBundle
                ? { ...curBundle, annotation: updated }
                : { schema_version: '2.0.0', video: { clip_id: clipId || '', fps: 30, duration_s: duration || 0 }, annotation: updated, status: 'pending', provenance: { generated_by: 'human' } }
              updateBundle(newBundle)
              if (clipId) guardedSave(clipId, newBundle).catch(() => {})
            }
          }
        } else if (track.group === 'Agents') {
          const agentSegs = allTrackSegs.filter(s => !(s.meta as Record<string, unknown>)?._isAgentActionSubtrack && !(s.meta as Record<string, unknown>)?._isAgentPropertySubtrack && !(s.meta as Record<string, unknown>)?._isAgentPoseSubtrack && !(s.meta as Record<string, unknown>)?._isAgentContSubtrack && !(s.meta as Record<string, unknown>)?._isAgentInfluenceSubtrack)
          const parentAgent = agentSegs.find(s => t >= s.t0 && t <= s.t1)
          if (!parentAgent) return
          const agentIndex = (parentAgent.meta as Record<string, unknown>)?._agentIndex as number
          if (agentIndex == null) return

          const _dblContRowsA = lightLaneCounts.get(track.id) ?? 1
          const _dblInflRowsA = lightLaneCounts.get(`${track.id}_infl`) ?? 0
          const _agentPoseBaseA = AGENT_SUBTRACK_BASE + _dblContRowsA * COND_SUBTRACK_HEIGHT
          const _agentInflBaseA = _agentPoseBaseA + COND_SUBTRACK_HEIGHT
          const _agentActionBaseA = _agentInflBaseA + _dblInflRowsA * COND_SUBTRACK_HEIGHT
          const _agentPropertyBaseA = _agentActionBaseA + COND_SUBTRACK_HEIGHT
          const inPoseRowDbl = localY >= _agentPoseBaseA && localY < _agentInflBaseA
          const inInfluenceRowDbl = localY >= _agentInflBaseA && localY < _agentActionBaseA
          const inActionRowDbl = localY >= _agentActionBaseA && localY < _agentPropertyBaseA
          const inPropertyRowDbl = localY >= _agentPropertyBaseA

          if (inInfluenceRowDbl) {
            const clickedInflRow = Math.floor((localY - _agentInflBaseA) / COND_SUBTRACK_HEIGHT)
            const inflSegs = allTrackSegs
              .filter(s => !!(s.meta as Record<string, unknown>)?._isAgentInfluenceSubtrack && (s.meta as Record<string, unknown>)?._agentIndex === agentIndex && ((s.meta as Record<string, unknown>)?._influence_track_index as number ?? 0) === clickedInflRow)
              .sort((a, b) => a.t0 - b.t0)
            const placement = findAdjacentCondGap(inflSegs, parentAgent.t0, parentAgent.t1, t)
            if (!placement) return
            const updated = addInfluenceToAgent(ann, agentIndex, placement.t0, placement.t1, clickedInflRow)
            const clipId = selectedClipIdRef.current
            const newBundle = curBundle
              ? { ...curBundle, annotation: updated }
              : { schema_version: '2.0.0', video: { clip_id: clipId || '', fps: 30, duration_s: duration || 0 }, annotation: updated, status: 'pending', provenance: { generated_by: 'human' } }
            updateBundle(newBundle)
            if (clipId) guardedSave(clipId, newBundle).catch(() => {})
          } else if (inContRow) {
            const clickedContRow = Math.floor((localY - AGENT_SUBTRACK_BASE) / COND_SUBTRACK_HEIGHT)
            const contSegs = allTrackSegs
              .filter(s => !!(s.meta as Record<string, unknown>)?._isAgentContSubtrack && (s.meta as Record<string, unknown>)?._agentIndex === agentIndex && ((s.meta as Record<string, unknown>)?._cont_track_index as number ?? 0) === clickedContRow)
              .sort((a, b) => a.t0 - b.t0)
            const placement = findAdjacentCondGap(contSegs, parentAgent.t0, parentAgent.t1, t)
            if (!placement) return
            const updated = addContainmentToAgent(ann, agentIndex, placement.t0, placement.t1, clickedContRow)
            const clipId = selectedClipIdRef.current
            const newBundle = curBundle
              ? { ...curBundle, annotation: updated }
              : { schema_version: '2.0.0', video: { clip_id: clipId || '', fps: 30, duration_s: duration || 0 }, annotation: updated, status: 'pending', provenance: { generated_by: 'human' } }
            updateBundle(newBundle)
            if (clipId) guardedSave(clipId, newBundle).catch(() => {})
          } else if (inPoseRowDbl) {
            const poseSegs = allTrackSegs
              .filter(s => !!(s.meta as Record<string, unknown>)?._isAgentPoseSubtrack && (s.meta as Record<string, unknown>)?._agentIndex === agentIndex)
              .sort((a, b) => a.t0 - b.t0)
            const placement = findAdjacentCondGap(poseSegs, parentAgent.t0, parentAgent.t1, t)
            if (!placement) return
            const updated = addPoseToAgent(ann, agentIndex, placement.t0, placement.t1)
            const clipId = selectedClipIdRef.current
            const newBundle = curBundle
              ? { ...curBundle, annotation: updated }
              : { schema_version: '2.0.0', video: { clip_id: clipId || '', fps: 30, duration_s: duration || 0 }, annotation: updated, status: 'pending', provenance: { generated_by: 'human' } }
            updateBundle(newBundle)
            if (clipId) guardedSave(clipId, newBundle).catch(() => {})
          } else if (inActionRowDbl) {
            const actionSegs = allTrackSegs
              .filter(s => !!(s.meta as Record<string, unknown>)?._isAgentActionSubtrack && (s.meta as Record<string, unknown>)?._agentIndex === agentIndex)
              .sort((a, b) => a.t0 - b.t0)
            const placement = findAdjacentCondGap(actionSegs, parentAgent.t0, parentAgent.t1, t)
            if (!placement) return
            const updated = addActionToAgent(ann, agentIndex, placement.t0, placement.t1)
            const clipId = selectedClipIdRef.current
            const newBundle = curBundle
              ? { ...curBundle, annotation: updated }
              : { schema_version: '2.0.0', video: { clip_id: clipId || '', fps: 30, duration_s: duration || 0 }, annotation: updated, status: 'pending', provenance: { generated_by: 'human' } }
            updateBundle(newBundle)
            if (clipId) guardedSave(clipId, newBundle).catch(() => {})
          } else if (inPropertyRowDbl) {
            const clickedPropRow = Math.floor((localY - _agentPropertyBaseA) / COND_SUBTRACK_HEIGHT)
            const propSegs = allTrackSegs
              .filter(s => !!(s.meta as Record<string, unknown>)?._isAgentPropertySubtrack && (s.meta as Record<string, unknown>)?._agentIndex === agentIndex && ((s.meta as Record<string, unknown>)?._prop_track_index as number ?? 0) === clickedPropRow)
              .sort((a, b) => a.t0 - b.t0)
            const placement = findAdjacentCondGap(propSegs, parentAgent.t0, parentAgent.t1, t)
            if (!placement) return
            const updated = addPropertyToAgent(ann, agentIndex, placement.t0, placement.t1, clickedPropRow)
            const clipId = selectedClipIdRef.current
            const newBundle = curBundle
              ? { ...curBundle, annotation: updated }
              : { schema_version: '2.0.0', video: { clip_id: clipId || '', fps: 30, duration_s: duration || 0 }, annotation: updated, status: 'pending', provenance: { generated_by: 'human' } }
            updateBundle(newBundle)
            if (clipId) guardedSave(clipId, newBundle).catch(() => {})
          }
        } else if (track.group === 'Ego') {
          const egoContRows = lightLaneCounts.get('ego_act') ?? 1
          const egoInflRows = lightLaneCounts.get('ego_act_infl') ?? 0
          const egoInflBaseY = EGO_SUBTRACK_BASE + egoContRows * COND_SUBTRACK_HEIGHT
          const egoActionBaseY = egoInflBaseY + egoInflRows * COND_SUBTRACK_HEIGHT
          const egoPropertyBaseY = egoActionBaseY + COND_SUBTRACK_HEIGHT
          const inEgoContRow = localY >= EGO_SUBTRACK_BASE && localY < egoInflBaseY
          const inInfluenceRow = localY >= egoInflBaseY && localY < egoActionBaseY
          const inActionRow = localY >= egoActionBaseY && localY < egoPropertyBaseY
          const inPropertyRow = localY >= egoPropertyBaseY

          if (inPropertyRow) {
            const clickedPropRow = Math.floor((localY - egoPropertyBaseY) / COND_SUBTRACK_HEIGHT)
            const propSegs = allTrackSegs
              .filter((s: TimelineSegment) => !!(s.meta as Record<string, unknown>)?._isEgoPropertySubtrack && ((s.meta as Record<string, unknown>)?._prop_track_index as number ?? 0) === clickedPropRow)
              .sort((a: TimelineSegment, b: TimelineSegment) => a.t0 - b.t0)
            const placement = findAdjacentCondGap(propSegs, 0, duration || Infinity, t)
            if (!placement) return
            const updated = addPropertyToEgo(ann, placement.t0, placement.t1, clickedPropRow)
            const clipId = selectedClipIdRef.current
            const newBundle = curBundle
              ? { ...curBundle, annotation: updated }
              : { schema_version: '2.0.0', video: { clip_id: clipId || '', fps: 30, duration_s: duration || 0 }, annotation: updated, status: 'pending', provenance: { generated_by: 'human' } }
            updateBundle(newBundle)
            if (clipId) guardedSave(clipId, newBundle).catch(() => {})
          } else if (inActionRow) {
            const actionSegs = allTrackSegs
              .filter((s: TimelineSegment) => !!(s.meta as Record<string, unknown>)?._isEgoActionSubtrack)
              .sort((a: TimelineSegment, b: TimelineSegment) => a.t0 - b.t0)
            const placement = findAdjacentCondGap(actionSegs, 0, duration || Infinity, t)
            if (!placement) return
            const updated = addSegmentToAnnotation(ann, 'ego_act', 'none', placement.t0, placement.t1)
            const clipId = selectedClipIdRef.current
            const newBundle = curBundle
              ? { ...curBundle, annotation: updated }
              : { schema_version: '2.0.0', video: { clip_id: clipId || '', fps: 30, duration_s: duration || 0 }, annotation: updated, status: 'pending', provenance: { generated_by: 'human' } }
            updateBundle(newBundle)
            if (clipId) guardedSave(clipId, newBundle).catch(() => {})
          } else if (inInfluenceRow) {
            const clickedInflRow = Math.floor((localY - egoInflBaseY) / COND_SUBTRACK_HEIGHT)
            const inflSegs = allTrackSegs
              .filter((s: TimelineSegment) => !!(s.meta as Record<string, unknown>)?._isEgoInfluenceSubtrack && ((s.meta as Record<string, unknown>)?._influence_track_index as number ?? 0) === clickedInflRow)
              .sort((a: TimelineSegment, b: TimelineSegment) => a.t0 - b.t0)
            const placement = findAdjacentCondGap(inflSegs, 0, duration || Infinity, t)
            if (!placement) return
            const updated = addInfluenceToEgo(ann, placement.t0, placement.t1, clickedInflRow)
            const clipId = selectedClipIdRef.current
            const newBundle = curBundle
              ? { ...curBundle, annotation: updated }
              : { schema_version: '2.0.0', video: { clip_id: clipId || '', fps: 30, duration_s: duration || 0 }, annotation: updated, status: 'pending', provenance: { generated_by: 'human' } }
            updateBundle(newBundle)
            if (clipId) guardedSave(clipId, newBundle).catch(() => {})
          } else if (inEgoContRow) {
            const clickedContRow = Math.floor((localY - EGO_SUBTRACK_BASE) / COND_SUBTRACK_HEIGHT)
            const contSegs = allTrackSegs
              .filter((s: TimelineSegment) => !!(s.meta as Record<string, unknown>)?._isEgoContSubtrack && ((s.meta as Record<string, unknown>)?._cont_track_index as number ?? 0) === clickedContRow)
              .sort((a: TimelineSegment, b: TimelineSegment) => a.t0 - b.t0)
            const placement = findAdjacentCondGap(contSegs, 0, duration || Infinity, t)
            if (!placement) return
            const updated = addContainmentToEgo(ann, placement.t0, placement.t1, clickedContRow)
            const clipId = selectedClipIdRef.current
            const newBundle = curBundle
              ? { ...curBundle, annotation: updated }
              : { schema_version: '2.0.0', video: { clip_id: clipId || '', fps: 30, duration_s: duration || 0 }, annotation: updated, status: 'pending', provenance: { generated_by: 'human' } }
            updateBundle(newBundle)
            if (clipId) guardedSave(clipId, newBundle).catch(() => {})
          }
        }
        return
      }

      // Place adjacent to existing segments on this track (no gap)
      const mainSegs = baseSegments
        .filter(s => s.trackId === track.id &&
          !(s.meta as Record<string, unknown>)?._isCondSubtrack &&
          !(s.meta as Record<string, unknown>)?._isObjStateSubtrack &&
          !(s.meta as Record<string, unknown>)?._isSignalHeadSubtrack &&
          !(s.meta as Record<string, unknown>)?._isLightStateSubtrack &&
          !(s.meta as Record<string, unknown>)?._isObjContSubtrack &&
          !(s.meta as Record<string, unknown>)?._isLightContSubtrack &&
          !(s.meta as Record<string, unknown>)?._isLightPhysContSubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentActionSubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentPropertySubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentPoseSubtrack &&
          !(s.meta as Record<string, unknown>)?._isEgoInfluenceSubtrack &&
          !(s.meta as Record<string, unknown>)?._isEgoPropertySubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentInfluenceSubtrack &&
          !(s.meta as Record<string, unknown>)?._isEgoContSubtrack &&
          !(s.meta as Record<string, unknown>)?._isAgentContSubtrack)
        .sort((a, b) => a.t0 - b.t0)
      const placement = findAdjacentCondGap(mainSegs, 0, duration || 20, t)
      if (!placement) return // no space

      const t0 = placement.t0, t1 = placement.t1
      let label = track.name
      if (track.id.startsWith('env_')) label = 'oxd:Road'
      else if (track.id === 'ego_act') label = 'fst:DrivingInLane'
      else if (track.id.startsWith('obj_')) label = 'fst:StopSign'
      else if (track.id.startsWith('light_')) label = 'TrafficLight'
      else if (track.id.startsWith('agent_')) label = 'fst:DrivingInLane'

      const emptyAnn: SilAvAnnotation = { brief_description: '', environments: [], conditions: [], traffic_objects: [], traffic_lights: [], ego_vehicle: { actions: [] }, agents: [] }
      const curBundle = bundleRef.current
      const ann = curBundle?.annotation || emptyAnn
      const updated = addSegmentToAnnotation(ann, track.id, label, t0, t1)
      const clipId = selectedClipIdRef.current
      const newBundle = curBundle
        ? { ...curBundle, annotation: updated }
        : { schema_version: '2.0.0', video: { clip_id: clipId || '', fps: 30, duration_s: duration || 0 }, annotation: updated, status: 'pending', provenance: { generated_by: 'human' } }
      updateBundle(newBundle)
      if (clipId) guardedSave(clipId, newBundle).catch(() => {})
    },
    [duration, getLiveScrollOffset, zoomLevel, trackList, tracks, baseSegments, updateBundle, verticalScroll, lightLaneCounts, keypointsVisible]
  )

  const handleCanvasMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current, container = containerRef.current
    if (!canvas || !container) return
    const rect = canvas.getBoundingClientRect()
    setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top })
  }, [])

  const handleCanvasMouseLeave = useCallback(() => { setMousePos(null) }, [])

  const getCursorAt = useCallback((mx: number, myRaw: number): string => {
    if (dragState) return dragState.mode === 'move' ? 'grabbing' : 'col-resize'
    const container = containerRef.current; if (!container) return 'default'
    const w = container.clientWidth
    const pps = (w * zoomLevel) / Math.max(duration, 1)
    const liveScrollOffset = getLiveScrollOffset()
    const t2x = (t: number) => (t - liveScrollOffset) * pps
    if (myRaw < HEADER_HEIGHT) {
      const phX = t2x(playheadTime)
      if (myRaw < SCRUBBER_LANE_HEIGHT && Math.abs(mx - phX) <= 10) return 'col-resize'
      return 'pointer'
    }
    const my = myRaw + verticalScroll
    let rowIdx = -1
    let cumY = HEADER_HEIGHT
    for (let i = 0; i < trackList.length; i++) {
      const h = getTrackHeight(trackList[i].group, trackList[i].id, lightLaneCounts, trackList[i]._collapsed)
      if (my >= cumY && my < cumY + h) { rowIdx = i; break }
      cumY += h + getTrackGap(trackList, i)
    }
    if (rowIdx < 0 || rowIdx >= trackList.length) return 'default'
    if (trackList[rowIdx]._collapsed) return 'pointer'
    const trackSegs = tracks.get(trackList[rowIdx].id) || []
    const y = getTrackY(trackList, rowIdx, lightLaneCounts)
    const trackH = getTrackHeight(trackList[rowIdx].group, trackList[rowIdx].id, lightLaneCounts)

    // Keypoint diamonds: show the default cursor so users can tell it's a hit target.
    if (keypointsVisible) {
      const grp = trackList[rowIdx].group
      const localY = my - y
      const t = mx / pps + liveScrollOffset
      const HIT_PX = 6
      const onMainBar = (grp === 'Agents' || grp === 'Objects' || grp === 'Environments') && localY >= 0 && localY < ENV_MAIN_HEIGHT
      if (onMainBar) {
        for (const seg of trackSegs) {
          const sm = seg.meta as Record<string, unknown>
          let kps: { timestamp: string; x: number; y: number }[] | undefined
          if (sm?._objKind === 'agent' && sm?._agentIndex != null && t >= seg.t0 && t <= seg.t1) {
            kps = bundle?.annotation?.agents?.[sm._agentIndex as number]?.keypoints
          } else if (sm?._objKind === 'traffic_object' && sm?._objIndex != null && t >= seg.t0 && t <= seg.t1) {
            kps = bundle?.annotation?.traffic_objects?.[sm._objIndex as number]?.keypoints
          } else if (sm?._objKind === 'environment' && sm?._envIndex != null && t >= seg.t0 && t <= seg.t1) {
            kps = bundle?.annotation?.environments?.[sm._envIndex as number]?.keypoints
          }
          if (!kps?.length) continue
          for (const kp of kps) {
            if (Math.abs(t2x(parseTs(kp.timestamp)) - mx) <= HIT_PX) return 'pointer'
          }
        }
      }
      if (grp === 'TrafficLights') {
        for (const seg of trackSegs) {
          const sm = seg.meta as Record<string, unknown>
          if (!sm?._isSignalHeadSubtrack) continue
          if (!(t >= seg.t0 && t <= seg.t1)) continue
          const li = sm._lightIndex as number | undefined
          const hi = sm._headIndex as number | undefined
          if (li == null || hi == null) continue
          const light = bundle?.annotation?.traffic_lights?.[li]
          if (!light) continue
          const trackPCMax = lightLaneCounts.get(`${trackList[rowIdx].id}_physcontmax`) ?? 1
          const shTop = getSignalHeadY(light, hi, trackPCMax)
          if (!(localY >= shTop && localY < shTop + COND_SUBTRACK_HEIGHT)) continue
          const kps = light.signal_heads?.[hi]?.keypoints
          if (!kps?.length) continue
          for (const kp of kps) {
            if (Math.abs(t2x(parseTs(kp.timestamp)) - mx) <= HIT_PX) return 'pointer'
          }
        }
      }
    }
    const EDGE_ZONE = 8
    const grp = trackList[rowIdx].group
    const hasSubtrack = grp === 'Environments' || grp === 'Objects' || grp === 'TrafficLights' || grp === 'Agents' || grp === 'Ego'
    const getSegZoneCursor = (seg: TimelineSegment): 'main' | 'subtrack_pose' | 'subtrack1' | 'subtrack2' => {
      const m = seg.meta as Record<string, unknown>
      if (m?._isEgoActionSubtrack) return 'subtrack1'
      if (m?._isEgoInfluenceSubtrack) return 'subtrack1'
      if (m?._isAgentInfluenceSubtrack) return 'subtrack1'
      if (m?._isEgoContSubtrack) return 'subtrack1'
      if (m?._isAgentContSubtrack) return 'subtrack2'
      if (m?._isObjContSubtrack || m?._isLightContSubtrack || m?._isLightPhysContSubtrack) return 'subtrack2'
      if (m?._isAgentPoseSubtrack) return 'subtrack_pose'
      if (m?._isAgentPropertySubtrack || m?._isEgoPropertySubtrack) return 'subtrack1'
      if (m?._isAgentActionSubtrack) return 'subtrack1'
      if (m?._isCondSubtrack || m?._isObjStateSubtrack || m?._isLightStateSubtrack || m?._isSignalHeadSubtrack) return 'subtrack1'
      return 'main'
    }
    const getSegBoundsCursor = (seg: TimelineSegment): { top: number; bottom: number } => {
      const sm = seg.meta as Record<string, unknown>
      if (sm?._isEgoMain) return { top: y, bottom: y + EGO_SUBTRACK_BASE }
      const zone = getSegZoneCursor(seg)
      if (zone === 'main') return { top: y, bottom: y + (hasSubtrack ? ENV_MAIN_HEIGHT : trackH) }
      if (zone === 'subtrack_pose') {
        const _spCr = lightLaneCounts.get(trackList[rowIdx].id) ?? 1
        const poseY = AGENT_SUBTRACK_BASE + _spCr * COND_SUBTRACK_HEIGHT
        return { top: y + poseY, bottom: y + poseY + COND_SUBTRACK_HEIGHT }
      }
      if (grp === 'TrafficLights' && sm?._isLightPhysContSubtrack) {
        const pcY = y + ENV_MAIN_HEIGHT + (sm._cont_track_index as number ?? 0) * COND_SUBTRACK_HEIGHT
        return { top: pcY, bottom: pcY + COND_SUBTRACK_HEIGHT }
      }
      if (grp === 'TrafficLights' && (sm?._isSignalHeadSubtrack || sm?._isLightStateSubtrack || sm?._isLightContSubtrack)) {
        const lhi = sm._headIndex as number ?? 0
        const ll = bundle?.annotation?.traffic_lights?.[(sm._lightIndex as number) ?? 0]
        const trackPCMax = lightLaneCounts.get(`${trackList[rowIdx].id}_physcontmax`) ?? 1
        const shBaseY = ll ? getSignalHeadY(ll, lhi, trackPCMax) : ENV_MAIN_HEIGHT
        if (sm._isSignalHeadSubtrack) return { top: y + shBaseY, bottom: y + shBaseY + COND_SUBTRACK_HEIGHT }
        if (sm._isLightContSubtrack) return { top: y + shBaseY + COND_SUBTRACK_HEIGHT, bottom: y + shBaseY + 2 * COND_SUBTRACK_HEIGHT }
        if (sm._isLightStateSubtrack) {
          const stIdx = sm._state_track_index as number ?? 0
          const stY = shBaseY + 2 * COND_SUBTRACK_HEIGHT + stIdx * COND_SUBTRACK_HEIGHT
          return { top: y + stY, bottom: y + stY + COND_SUBTRACK_HEIGHT }
        }
        return { top: y + shBaseY, bottom: y + shBaseY + COND_SUBTRACK_HEIGHT }
      }
      if (sm?._isEgoActionSubtrack) {
        const _eaCr = lightLaneCounts.get('ego_act') ?? 1; const _eaIr = lightLaneCounts.get('ego_act_infl') ?? 0
        const actionY = EGO_SUBTRACK_BASE + _eaCr * COND_SUBTRACK_HEIGHT + _eaIr * COND_SUBTRACK_HEIGHT
        return { top: y + actionY, bottom: y + actionY + COND_SUBTRACK_HEIGHT }
      }
      if (sm?._isEgoInfluenceSubtrack) {
        const inflIdx = sm._influence_track_index as number ?? 0
        const _eiCr = lightLaneCounts.get('ego_act') ?? 1
        const inflY = EGO_SUBTRACK_BASE + _eiCr * COND_SUBTRACK_HEIGHT + inflIdx * COND_SUBTRACK_HEIGHT
        return { top: y + inflY, bottom: y + inflY + COND_SUBTRACK_HEIGHT }
      }
      if (sm?._isAgentInfluenceSubtrack) {
        const inflIdx = sm._influence_track_index as number ?? 0
        const _aiCr = lightLaneCounts.get(trackList[rowIdx].id) ?? 1
        const inflY = AGENT_SUBTRACK_BASE + _aiCr * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT + inflIdx * COND_SUBTRACK_HEIGHT
        return { top: y + inflY, bottom: y + inflY + COND_SUBTRACK_HEIGHT }
      }
      if (sm?._isAgentPropertySubtrack) {
        const propIdx = sm._prop_track_index as number ?? 0
        const _apCr2 = lightLaneCounts.get(trackList[rowIdx].id) ?? 1
        const _apIr2 = lightLaneCounts.get(`${trackList[rowIdx].id}_infl`) ?? 0
        const propY = AGENT_SUBTRACK_BASE + _apCr2 * COND_SUBTRACK_HEIGHT + _apIr2 * COND_SUBTRACK_HEIGHT + 2 * COND_SUBTRACK_HEIGHT + propIdx * COND_SUBTRACK_HEIGHT
        return { top: y + propY, bottom: y + propY + COND_SUBTRACK_HEIGHT }
      }
      if (sm?._isEgoPropertySubtrack) {
        const propIdx = sm._prop_track_index as number ?? 0
        const _epCr2 = lightLaneCounts.get('ego_act') ?? 1
        const _epIr2 = lightLaneCounts.get('ego_act_infl') ?? 0
        const propY = EGO_SUBTRACK_BASE + _epCr2 * COND_SUBTRACK_HEIGHT + _epIr2 * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT + propIdx * COND_SUBTRACK_HEIGHT
        return { top: y + propY, bottom: y + propY + COND_SUBTRACK_HEIGHT }
      }
      if (zone === 'subtrack2') {
        if (sm?._isAgentContSubtrack) {
          const contIdx = sm._cont_track_index as number ?? 0
          const contY = AGENT_SUBTRACK_BASE + contIdx * COND_SUBTRACK_HEIGHT
          return { top: y + contY, bottom: y + contY + COND_SUBTRACK_HEIGHT }
        }
        if (sm?._isObjContSubtrack) {
          const contIdx = sm._cont_track_index as number ?? 0
          const contY = OBJ_CONT_ROW_Y + contIdx * COND_SUBTRACK_HEIGHT
          return { top: y + contY, bottom: y + contY + COND_SUBTRACK_HEIGHT }
        }
        const contY = grp === 'Objects' ? OBJ_CONT_ROW_Y : AGENT_SUBTRACK_BASE
        return { top: y + contY, bottom: y + contY + COND_SUBTRACK_HEIGHT }
      }
      if (sm?._isEgoContSubtrack) {
        const contIdx = sm._cont_track_index as number ?? 0
        const contY = EGO_SUBTRACK_BASE + contIdx * COND_SUBTRACK_HEIGHT
        return { top: y + contY, bottom: y + contY + COND_SUBTRACK_HEIGHT }
      }
      if (grp === 'Agents') { const _aaCr = lightLaneCounts.get(trackList[rowIdx].id) ?? 1; const _aaIr = lightLaneCounts.get(`${trackList[rowIdx].id}_infl`) ?? 0; const aaY = AGENT_SUBTRACK_BASE + _aaCr * COND_SUBTRACK_HEIGHT + _aaIr * COND_SUBTRACK_HEIGHT + COND_SUBTRACK_HEIGHT; return { top: y + aaY, bottom: y + aaY + COND_SUBTRACK_HEIGHT } }
      if (grp === 'Objects') { const _ocrY2 = getObjStateRowY(lightLaneCounts.get(trackList[rowIdx].id) ?? 1); return { top: y + _ocrY2, bottom: y + _ocrY2 + COND_SUBTRACK_HEIGHT } }
      if (sm?._isCondSubtrack) {
        const condIdx = sm._cond_track_index as number ?? 0
        const condY = ENV_MAIN_HEIGHT + condIdx * COND_SUBTRACK_HEIGHT
        return { top: y + condY, bottom: y + condY + COND_SUBTRACK_HEIGHT }
      }
      return { top: y + ENV_MAIN_HEIGHT, bottom: y + ENV_MAIN_HEIGHT + COND_SUBTRACK_HEIGHT }
    }
    for (const seg of trackSegs) {
      const { top: segTop, bottom: segBottom } = getSegBoundsCursor(seg)
      const x0 = t2x(seg.t0), x1 = t2x(seg.t1)
      if (mx < x0 - EDGE_ZONE || mx > x1 + EDGE_ZONE || my < segTop || my > segBottom) continue
      if (mx >= x0 - EDGE_ZONE && mx <= x0 + EDGE_ZONE) return 'col-resize'
      if (mx >= x1 - EDGE_ZONE && mx <= x1 + EDGE_ZONE) return 'col-resize'
      if (mx >= x0 && mx <= x1) return 'pointer'
    }
    return 'default'
  }, [duration, getLiveScrollOffset, trackList, tracks, zoomLevel, dragState, verticalScroll, lightLaneCounts, keypointsVisible, bundle, playheadTime])

  const [cursor, setCursor] = useState('default')
  const updateCursor = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.ctrlKey || e.shiftKey) { setCursor('default'); return }
    const canvas = canvasRef.current; if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    setCursor(getCursorAt(e.clientX - rect.left, e.clientY - rect.top))
  }, [getCursorAt])

  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault()
      commitNativeScroll()
      setZoom(zoomLevel + (e.deltaY > 0 ? -0.2 : 0.2))
    }
    else if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
      const el = containerRef.current
      if (!el) return
      e.preventDefault()
      el.scrollBy({ left: e.deltaX, behavior: 'auto' })
      const pps = (el.clientWidth * zoomLevel) / Math.max(duration, 1)
      scrollOffsetRef.current = pps > 0 ? el.scrollLeft / pps : scrollOffsetRef.current
      scheduleNativeScrollCommit()
    }
    else if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
      // Vertical scroll
      e.preventDefault()
      const totalGaps = trackList.reduce((sum, _t, i) => sum + (i > 0 ? getTrackGap(trackList, i - 1) : 0), 0)
      const totalTrackHeight = trackList.reduce((sum, t) => sum + getTrackHeight(t.group, t.id, lightLaneCounts, t._collapsed), 0) + HEADER_HEIGHT + totalGaps + BOTTOM_PADDING
      const viewportHeight = containerRef.current?.clientHeight ?? 400
      const maxVScroll = Math.max(0, totalTrackHeight - viewportHeight)
      setVerticalScroll(prev => Math.max(0, Math.min(maxVScroll, prev + e.deltaY)))
    }
    // Horizontal scroll is handled natively by the scrollable container
  }, [commitNativeScroll, duration, scheduleNativeScrollCommit, zoomLevel, setZoom, trackList, lightLaneCounts])

  const zoomIn = () => { commitNativeScroll(); setZoom(zoomLevel * 1.5) }
  const zoomOut = () => { commitNativeScroll(); setZoom(zoomLevel / 1.5) }
  const zoomFit = () => { setZoom(1); setScroll(0) }

  // Sync native scrollLeft with store scrollOffset
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const viewW = el.clientWidth
    const pps = (viewW * zoomLevel) / Math.max(duration, 1)
    const targetScrollLeft = scrollOffset * pps
    if (Math.abs(el.scrollLeft - targetScrollLeft) > 1) {
      el.scrollLeft = targetScrollLeft
    }
  }, [scrollOffset, zoomLevel, duration])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    el.addEventListener('scrollend', commitNativeScroll)
    return () => el.removeEventListener('scrollend', commitNativeScroll)
  }, [commitNativeScroll])

  // --- Keyboard shortcuts ---
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA') return

      // Ctrl+Z / Cmd+Z — undo
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault()
        undo()
        return
      }

      if (!selectedPath || !bundle?.annotation) return
      if (bundle.annotation.eventful === false) return
      const ann = bundle.annotation
      const seg = segments.find((s) => s.id === selectedPath)
      if (!seg) return

      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault()
        const updated = JSON.parse(JSON.stringify(ann)) as typeof ann
        const segMeta = seg.meta as Record<string, unknown>
        if (segMeta?._isCondSubtrack) {
          const ci = (segMeta._condIndex as number) ?? -1
          if (ci >= 0) {
            const condId = updated.conditions[ci]?.id
            if (condId) cleanupDeletedIds(updated, [condId])
            updated.conditions.splice(ci, 1)
          }
        } else if (seg.trackId.startsWith('env_')) {
          const ei = (segMeta._envIndex as number) ?? -1
          if (ei >= 0) {
            const envId = updated.environments[ei]?.id
            updated.environments.splice(ei, 1)
            if (envId != null) cleanupDeletedEnv(updated, envId)
          }
        } else if (segMeta?._isObjContSubtrack) {
          const oi = (segMeta._objIndex as number) ?? -1
          const ci = (segMeta._contIndex as number) ?? -1
          if (oi >= 0 && ci >= 0) updated.traffic_objects[oi]?.containment?.splice(ci, 1)
        } else if (segMeta?._isLightPhysContSubtrack) {
          const li = (segMeta._lightIndex as number) ?? -1
          const ci = (segMeta._contIndex as number) ?? -1
          if (li >= 0 && ci >= 0) updated.traffic_lights[li]?.containment?.splice(ci, 1)
        } else if (segMeta?._isLightContSubtrack) {
          const li = (segMeta._lightIndex as number) ?? -1
          const hi = (segMeta._headIndex as number) ?? -1
          const ci = (segMeta._contIndex as number) ?? -1
          if (li >= 0 && hi >= 0 && ci >= 0) updated.traffic_lights[li]?.signal_heads?.[hi]?.env_controlled?.splice(ci, 1)
        } else if (segMeta?._isObjStateSubtrack) {
          const oi = (segMeta._objIndex as number) ?? -1
          const si = (segMeta._stateIndex as number) ?? -1
          if (oi >= 0 && si >= 0) {
            const stateId = updated.traffic_objects[oi].state_sequence[si]?.id
            if (stateId) cleanupDeletedIds(updated, [stateId])
            updated.traffic_objects[oi].state_sequence.splice(si, 1)
          }
        } else if (segMeta?._isSignalHeadSubtrack) {
          const li = (segMeta._lightIndex as number) ?? -1
          const hi = (segMeta._headIndex as number) ?? -1
          if (li >= 0 && hi >= 0) {
            const head = updated.traffic_lights[li].signal_heads[hi]
            const headIds = [head.id, ...(head.state_sequence.map(s => s.id).filter(Boolean) as string[])]
            cleanupDeletedIds(updated, headIds)
            updated.traffic_lights[li].signal_heads.splice(hi, 1)
          }
        } else if (segMeta?._isLightStateSubtrack) {
          const li = (segMeta._lightIndex as number) ?? -1
          const hi = (segMeta._headIndex as number) ?? -1
          const si = (segMeta._stateIndex as number) ?? -1
          if (li >= 0 && hi >= 0 && si >= 0) {
            const stateId = updated.traffic_lights[li].signal_heads[hi].state_sequence[si]?.id
            if (stateId) cleanupDeletedIds(updated, [stateId])
            updated.traffic_lights[li].signal_heads[hi].state_sequence.splice(si, 1)
          }
        } else if (seg.trackId.startsWith('light_')) {
          const li = (seg.meta as { _lightIndex?: number })?._lightIndex
          if (li != null && li >= 0) {
            const light = updated.traffic_lights[li]
            const lightIds = [light.id, ...light.signal_heads.flatMap(h => [h.id, ...(h.state_sequence.map(s => s.id).filter(Boolean) as string[])])]
            cleanupDeletedIds(updated, lightIds)
            updated.traffic_lights.splice(li, 1)
          }
        } else if ((seg.meta as Record<string, unknown>)?._isEgoInfluenceSubtrack) {
          const ii = (seg.meta as { _inflIndex?: number })?._inflIndex
          if (ii != null && ii >= 0) updated.ego_vehicle?.influenced_by?.splice(ii, 1)
        } else if ((seg.meta as Record<string, unknown>)?._isAgentInfluenceSubtrack) {
          const ai = (seg.meta as { _agentIndex?: number })?._agentIndex ?? -1
          const ii = (seg.meta as { _inflIndex?: number })?._inflIndex ?? -1
          if (ai >= 0 && ii >= 0) (updated.agents[ai] as unknown as Record<string, unknown[]>)?.influenced_by?.splice(ii, 1)
        } else if ((seg.meta as Record<string, unknown>)?._isEgoContSubtrack) {
          const ci = (seg.meta as { _contIndex?: number })?._contIndex
          if (ci != null && ci >= 0) updated.ego_vehicle?.containment?.splice(ci, 1)
        } else if ((seg.meta as Record<string, unknown>)?._isAgentContSubtrack) {
          const ai = (seg.meta as { _agentIndex?: number })?._agentIndex ?? -1
          const ci = (seg.meta as { _contIndex?: number })?._contIndex ?? -1
          if (ai >= 0 && ci >= 0) updated.agents[ai]?.containment?.splice(ci, 1)
        } else if ((seg.meta as Record<string, unknown>)?._isEgoPropertySubtrack) {
          const pi = (seg.meta as { _propIndex?: number })?._propIndex
          if (pi != null && pi >= 0) {
            const propId = updated.ego_vehicle?.properties?.[pi]?.id
            if (propId) cleanupDeletedIds(updated, [propId])
            updated.ego_vehicle?.properties?.splice(pi, 1)
          }
        } else if ((seg.meta as Record<string, unknown>)?._isAgentPropertySubtrack) {
          const ai = (seg.meta as { _agentIndex?: number })?._agentIndex ?? -1
          const pi = (seg.meta as { _propIndex?: number })?._propIndex ?? -1
          if (ai >= 0 && pi >= 0) {
            const propId = updated.agents[ai]?.properties?.[pi]?.id
            if (propId) cleanupDeletedIds(updated, [propId])
            updated.agents[ai]?.properties?.splice(pi, 1)
          }
        } else if (seg.trackId === 'ego_act') {
          const idx = segments.filter((s) => s.trackId === 'ego_act' && !(s.meta as Record<string, unknown>)?._isEgoContSubtrack && !(s.meta as Record<string, unknown>)?._isEgoInfluenceSubtrack && !(s.meta as Record<string, unknown>)?._isEgoPropertySubtrack).findIndex((s) => s.id === selectedPath)
          const actionId = updated.ego_vehicle.actions[idx]?.id
          if (actionId) cleanupDeletedIds(updated, [actionId])
          updated.ego_vehicle.actions = updated.ego_vehicle.actions.filter((_, i) => i !== idx)
        } else if (seg.trackId.startsWith('obj_')) {
          const meta = seg.meta as { _objIndex?: number; _lightIndex?: number; _objKind?: string }
          if (meta._objKind === 'traffic_light' && meta._lightIndex != null && meta._lightIndex >= 0) {
            const light = updated.traffic_lights[meta._lightIndex]
            const lightIds = [light.id, ...light.signal_heads.flatMap(h => [h.id, ...(h.state_sequence.map(s => s.id).filter(Boolean) as string[])])]
            cleanupDeletedIds(updated, lightIds)
            updated.traffic_lights.splice(meta._lightIndex, 1)
          } else if (meta._objIndex != null && meta._objIndex >= 0) {
            const obj = updated.traffic_objects[meta._objIndex]
            const objIds = [obj.id, ...(obj.state_sequence.map(s => s.id).filter(Boolean) as string[])]
            cleanupDeletedIds(updated, objIds)
            updated.traffic_objects.splice(meta._objIndex, 1)
          }
        } else if (seg.trackId.startsWith('agent_')) {
          const meta = seg.meta as { _agentIndex?: number; _isAgentActionSubtrack?: boolean; _actIndex?: number; _segType?: string } | undefined
          const ai = meta?._agentIndex
          if (ai != null && ai >= 0) {
            if (meta?._segType === 'action') {
              const actIdx = meta?._actIndex ?? -1
              if (actIdx >= 0 && actIdx < (updated.agents[ai]?.actions || []).length) {
                const actionId = updated.agents[ai].actions[actIdx]?.id
                if (actionId) cleanupDeletedIds(updated, [actionId])
                updated.agents[ai].actions.splice(actIdx, 1)
              }
            } else if (meta?._segType === 'property') {
              const pi = (meta as Record<string, unknown>)?._propIndex as number ?? -1
              if (pi >= 0) {
                const propId = updated.agents[ai]?.properties?.[pi]?.id
                if (propId) cleanupDeletedIds(updated, [propId])
                updated.agents[ai]?.properties?.splice(pi, 1)
              }
            } else if (meta?._segType === 'pose') {
              const pi = (meta as Record<string, unknown>)?._poseIndex as number ?? -1
              if (pi >= 0) {
                updated.agents[ai]?.ego_relative_pose?.splice(pi, 1)
              }
            } else {
              const agent = updated.agents[ai]
              const agentIds = [agent.id, ...(agent.actions.map(a => a.id).filter(Boolean) as string[])]
              cleanupDeletedIds(updated, agentIds)
              updated.agents.splice(ai, 1)
            }
          }
        }
        updateBundle({ ...bundle, annotation: updated }); selectPath(null)
      }
    }
    const handleModifier = (e: KeyboardEvent) => {
      if (e.key === 'Control' || e.key === 'Shift') setCursor('default')
    }
    window.addEventListener('keydown', handleKey)
    window.addEventListener('keydown', handleModifier)
    return () => {
      window.removeEventListener('keydown', handleKey)
      window.removeEventListener('keydown', handleModifier)
    }
  }, [selectedPath, bundle, segments, playheadTime, selectPath, updateBundle, undo])

  // --- Render ---
  if (bundle?.annotation?.eventful === false) {
    return <div className="h-full flex items-center justify-center bg-surface-sunken text-text-disabled text-sm">Nominal driving — no annotation needed</div>
  }

  return (
    <div className="h-full flex flex-col">
      <Tooltip.Provider delayDuration={200}>
        <div className="flex-shrink-0 h-9 px-4 flex items-center gap-2 border-b border-border-default bg-surface-sunken">
          <ZoomButton onClick={zoomIn} icon={ZoomIn} label="Zoom in" />
          <ZoomButton onClick={zoomOut} icon={ZoomOut} label="Zoom out" />
          <ZoomButton onClick={zoomFit} icon={Maximize2} label="Fill timeline" />
          <div className="flex items-center gap-4 ml-auto">
            <Kb keys="Del" desc="Delete" />
            <Kb keys={'← →'} desc="Frame step" />
            <Kb keys="Space" desc="Play/Pause" />
            <span className="w-px h-4 bg-border-default mx-1" />
            <Kb keys="Right-click" desc="Create" />
            <Kb keys="Shift+click" desc="Link" />
            <Kb keys="Ctrl+click" desc="Because of" />
          </div>
        </div>
      </Tooltip.Provider>
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Label panel */}
        <div className="flex-shrink-0 border-r border-border-default overflow-hidden" style={{ width: LABEL_PANEL_WIDTH, backgroundColor: canvasColors.bg }} onWheel={handleWheel}>
          <div style={{ height: HEADER_HEIGHT }} className="border-b border-border-default" />
          <div style={{ transform: `translateY(-${verticalScroll}px)` }}>
          {trackList.map(({ id, name, color, group, _collapsed }, i) => {
            const prev = trackList[i - 1]
            const isFirstInGroup = !prev || prev.group !== group
            const isDynamic = group === 'Environments' || group === 'Objects' || group === 'TrafficLights' || group === 'Agents' || group === 'EgoContainment'
            // Extract track index from id for remove
            const trackIdxMatch = id.match(/^(?:env|cond|obj|light|agent|ego_cont)_(\d+)$/)
            const trackIdx = trackIdxMatch ? parseInt(trackIdxMatch[1], 10) : -1

            const gapAbove = i > 0 ? getTrackGap(trackList, i - 1) : 0

            if (_collapsed) {
              return (
                <div
                  key={`collapsed_${group}`}
                  className="flex flex-col border-b border-border-subtle"
                  style={{ height: COLLAPSED_GROUP_HEIGHT, marginTop: gapAbove, backgroundColor: i % 2 === 0 ? canvasColors.trackA : canvasColors.trackB, borderLeft: `3px solid ${color}50` }}
                >
                  <div
                    className="flex items-center gap-1.5 mx-1 mt-1 px-3 py-1 rounded-md cursor-pointer hover:brightness-125 transition-all"
                    style={{}}
                    onClick={() => toggleGroupCollapse(group)}
                    title={`Expand ${group}`}
                  >
                    <ChevronRight className="w-3 h-3 flex-shrink-0" style={{ color }} />
                    <span className="uppercase text-[10px] font-semibold tracking-[0.08em] flex-1" style={{ color }}>{group}</span>
                  </div>
                </div>
              )
            }

            const subLabels = getSubtrackRowLabels(group, id, lightLaneCounts, bundle?.annotation)
            // Header pill for the group lives in the gap above the row,
            // not inside it — keeps the row's top slot clear so the
            // track-name row doesn't collide with the first absolutely-
            // positioned subtrack label ("containment").
            const headerSpaceAbove = Math.max(0, gapAbove - CATEGORY_HEADER_HEIGHT)
            return (
              <Fragment key={id}>
                {isFirstInGroup && (
                  <div
                    className="flex items-center gap-1 px-2"
                    style={{ height: CATEGORY_HEADER_HEIGHT, marginTop: headerSpaceAbove }}
                  >
                    <div
                      className="flex items-center gap-1.5 px-3 py-1 rounded-md cursor-pointer hover:brightness-125 transition-all"
                      style={{}}
                      onClick={() => toggleGroupCollapse(group)}
                      title={`Collapse ${group}`}
                    >
                      <ChevronDown className="w-3 h-3 flex-shrink-0" style={{ color }} />
                      <span className="uppercase text-[10px] font-semibold tracking-[0.08em] truncate" style={{ color }}>{group}</span>
                    </div>
                    <div className="flex-1" />
                    {isDynamic && (
                      <button type="button" onClick={() => handleAddTrack(group as DynamicGroup)}
                        className="w-6 h-6 flex items-center justify-center rounded-md hover:bg-surface-hover text-text-muted hover:text-success transition-colors flex-shrink-0"
                        title={`Add ${group.toLowerCase()} track`}>
                        <Plus className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                )}
                <div
                  className="relative flex flex-col justify-start pt-0.5 gap-0.5 px-2 border-b border-border-subtle"
                  style={{ height: getTrackHeight(group, id, lightLaneCounts), marginTop: isFirstInGroup ? 0 : gapAbove, backgroundColor: i % 2 === 0 ? canvasColors.trackA : canvasColors.trackB, borderLeft: `3px solid ${color}50` }}
                >
                {subLabels.map((sl, si) => sl.label ? (
                  <span
                    key={si}
                    className="absolute right-2 text-[9px] uppercase tracking-wider pointer-events-none select-none"
                    style={{ top: sl.y + 1, height: COND_SUBTRACK_HEIGHT - 2, lineHeight: `${COND_SUBTRACK_HEIGHT - 2}px`, color: `${sl.color}cc` }}
                  >{sl.label}</span>
                ) : null)}
                <div className="flex items-center gap-1">
                  <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
                  <span className="text-[11px] text-text-secondary truncate flex-1">{name}</span>
                  {isDynamic && trackIdx >= 0 && (
                    <button type="button"
                      onClick={() => handleRemoveTrack(group as DynamicGroup, trackIdx)}
                      className="w-6 h-6 flex items-center justify-center rounded-md hover:bg-surface-hover text-text-muted hover:text-danger transition-colors"
                      title={`Remove ${name}`}
                    >
                      <Minus className="w-3 h-3" />
                    </button>
                  )}
                </div>
                </div>
              </Fragment>
            )
          })}
          </div>
        </div>
        {/* Canvas with native horizontal scroll */}
        <div
          ref={containerRef}
          className="flex-1 overflow-x-auto overflow-y-hidden relative"
          onWheel={handleWheel}
          onMouseUp={commitNativeScroll}
          onTouchEnd={commitNativeScroll}
          onPointerCancel={commitNativeScroll}
          onScroll={(e) => {
            const el = e.currentTarget
            const viewW = el.clientWidth
            const pps = (viewW * zoomLevel) / Math.max(duration, 1)
            scrollOffsetRef.current = pps > 0 ? el.scrollLeft / pps : scrollOffsetRef.current
            scheduleNativeScrollCommit()
          }}
        >
          <div style={{ width: `${zoomLevel * 100}%`, height: '100%', position: 'relative' }}>
            <canvas ref={canvasRef} className="block sticky left-0 top-0" style={{ cursor }}
              onMouseDown={handleCanvasMouseDown}
              onContextMenu={handleRightClick}
              onMouseMove={(e) => { handleCanvasMouseMove(e); updateCursor(e) }}
              onMouseLeave={handleCanvasMouseLeave}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
