import { useRef, useEffect, useCallback, useState, useMemo } from 'react'
import { useStore } from '../lib/store'
import { Play, Pause, SkipBack, SkipForward, ChevronsLeft, ChevronsRight, MapPin, Maximize } from 'lucide-react'
import { parseTs, annotationToSegments } from '../lib/timeline-utils'
import { interpolateKeypoint, upsertKeypoint, moveKeypointAt, removeKeypointAt, EXACT_SNAP_SECONDS } from '../lib/keypoint-utils'
import type { SilAvAnnotation, Keypoint } from '../lib/types'

function fmt(s: number) {
  const m = Math.floor(s / 60)
  const sec = s % 60
  return `${m}:${sec.toFixed(1).padStart(4, '0')}`
}

interface VideoRect { x: number; y: number; w: number; h: number }

function getVideoRect(video: HTMLVideoElement, container: HTMLDivElement): VideoRect {
  const vw = video.videoWidth
  const vh = video.videoHeight
  const cw = container.clientWidth
  const ch = container.clientHeight
  if (!vw || !vh) return { x: 0, y: 0, w: cw, h: ch }
  const scale = Math.min(cw / vw, ch / vh)
  const w = vw * scale
  const h = vh * scale
  return { x: (cw - w) / 2, y: (ch - h) / 2, w, h }
}

function normalizedToPixel(nx: number, ny: number, vr: VideoRect) {
  return { px: vr.x + nx * vr.w, py: vr.y + ny * vr.h }
}

function pixelToNormalized(px: number, py: number, vr: VideoRect) {
  if (vr.w <= 0 || vr.h <= 0) return { nx: 0, ny: 0 }
  return { nx: (px - vr.x) / vr.w, ny: (py - vr.y) / vr.h }
}

type EntityKind = 'agent' | 'traffic_object' | 'signal_head' | 'environment'

interface ActiveMarker {
  entityKind: EntityKind
  entityIdx: number
  entitySubIdx: number
  entityId: string
  x: number
  y: number
  interpolated: boolean
  realKpIdx: number | null
}

function entityColor(kind: EntityKind): string {
  if (kind === 'agent') return '#a855f7'
  if (kind === 'traffic_object') return '#f59e0b'
  if (kind === 'signal_head') return '#ef4444'
  return '#22c55e'
}

function kpsFor(ann: SilAvAnnotation, kind: EntityKind, idx: number, subIdx: number): Keypoint[] | undefined {
  if (kind === 'agent') return ann.agents?.[idx]?.keypoints
  if (kind === 'traffic_object') return ann.traffic_objects?.[idx]?.keypoints
  if (kind === 'signal_head') return ann.traffic_lights?.[idx]?.signal_heads?.[subIdx]?.keypoints
  return ann.environments?.[idx]?.keypoints
}

function assignKeypoints(ann: SilAvAnnotation, kind: EntityKind, idx: number, subIdx: number, kps: Keypoint[]): void {
  if (kind === 'agent') ann.agents[idx].keypoints = kps
  else if (kind === 'traffic_object') ann.traffic_objects[idx].keypoints = kps
  else if (kind === 'signal_head') ann.traffic_lights[idx].signal_heads[subIdx].keypoints = kps
  else ann.environments[idx].keypoints = kps
}

function collectActiveMarkers(ann: SilAvAnnotation, time: number): ActiveMarker[] {
  const result: ActiveMarker[] = []
  const add = (kps: Keypoint[] | undefined, kind: EntityKind, idx: number, subIdx: number, entityId: string) => {
    const ip = interpolateKeypoint(kps, time)
    if (!ip) return
    let realIdx: number | null = null
    if (!ip.interpolated && kps) {
      let best = EXACT_SNAP_SECONDS
      for (let i = 0; i < kps.length; i++) {
        const d = Math.abs(parseTs(kps[i].timestamp) - time)
        if (d <= best) { best = d; realIdx = i }
      }
    }
    result.push({ entityKind: kind, entityIdx: idx, entitySubIdx: subIdx, entityId, x: ip.x, y: ip.y, interpolated: ip.interpolated, realKpIdx: realIdx })
  }
  for (let i = 0; i < (ann.agents || []).length; i++) {
    const a = ann.agents[i]
    add(a.keypoints, 'agent', i, -1, a.id || `Agent${i + 1}`)
  }
  for (let i = 0; i < (ann.traffic_objects || []).length; i++) {
    const o = ann.traffic_objects[i]
    add(o.keypoints, 'traffic_object', i, -1, o.id || `Object${i + 1}`)
  }
  for (let i = 0; i < (ann.traffic_lights || []).length; i++) {
    const l = ann.traffic_lights[i]
    for (let j = 0; j < (l.signal_heads || []).length; j++) {
      const sh = l.signal_heads[j]
      add(sh.keypoints, 'signal_head', i, j, sh.id || `SignalHead${i + 1}.${j + 1}`)
    }
  }
  for (let i = 0; i < (ann.environments || []).length; i++) {
    const env = ann.environments[i]
    add(env.keypoints, 'environment', i, -1, env.id || `Env${i + 1}`)
  }
  return result
}

function findEntitySegmentId(ann: SilAvAnnotation, kind: EntityKind, idx: number, subIdx: number): string | null {
  const segs = annotationToSegments(ann)
  for (const s of segs) {
    const m = s.meta as Record<string, unknown> | undefined
    if (!m) continue
    if (kind === 'agent' && m._agentIndex === idx && m._objKind === 'agent') return s.id
    if (kind === 'traffic_object' && m._objIndex === idx && m._objKind === 'traffic_object') return s.id
    if (kind === 'signal_head' && m._isSignalHeadSubtrack && m._lightIndex === idx && m._headIndex === subIdx) return s.id
    if (kind === 'environment' && m._envIndex === idx && m._objKind === 'environment') return s.id
  }
  return null
}

interface Interaction {
  mode: 'idle' | 'moving'
  entityKind: EntityKind
  entityIdx: number
  entitySubIdx: number
  kpIdx: number
  preview: { x: number; y: number } | null
  downX: number
  downY: number
  moved: boolean
}

function idleInteraction(): Interaction {
  return { mode: 'idle', entityKind: 'agent', entityIdx: -1, entitySubIdx: -1, kpIdx: -1, preview: null, downX: 0, downY: 0, moved: false }
}

const HIT_RADIUS_PX = 10
const MARKER_RADIUS_PX = 6

export function VideoPlayer() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const rafRef = useRef<number | null>(null)
  const interRef = useRef<Interaction>(idleInteraction())
  const scrubTargetRef = useRef<number | null>(null)
  const scrubSeekingRef = useRef(false)
  const scrubRafRef = useRef<number | null>(null)
  const scrubWatchdogRef = useRef<number | null>(null)
  const requestScrubFlushRef = useRef<() => void>(() => {})
  const {
    selectedClipId, setDuration, setPlayhead, playheadTime, duration, bundle,
    updateBundle, markDirty, selectedPath, selectPath, keypointsVisible, toggleKeypointsVisible,
    arrowTypes, toggleArrowType,
  } = useStore()

  const wrapperRef = useRef<HTMLDivElement>(null)
  const zoomRef = useRef({ zoom: 1, panX: 0, panY: 0 })
  const dragPanRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null)

  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [zoomLevel, setZoomLevel] = useState(1)
  const [videoError, setVideoError] = useState<string | null>(null)

  const applyZoomTransform = useCallback(() => {
    const w = wrapperRef.current
    if (!w) return
    const { zoom, panX, panY } = zoomRef.current
    w.style.transform = zoom === 1 && panX === 0 && panY === 0
      ? '' : `translate(${panX}px, ${panY}px) scale(${zoom})`
    setZoomLevel(zoom)
  }, [])

  const clampPan = useCallback(() => {
    const container = containerRef.current
    if (!container) return
    const z = zoomRef.current
    if (z.zoom <= 1) { z.panX = 0; z.panY = 0; return }
    const cw = container.clientWidth
    const ch = container.clientHeight
    z.panX = Math.max(cw * (1 - z.zoom), Math.min(0, z.panX))
    z.panY = Math.max(ch * (1 - z.zoom), Math.min(0, z.panY))
  }, [])

  const resetZoom = useCallback(() => {
    zoomRef.current = { zoom: 1, panX: 0, panY: 0 }
    applyZoomTransform()
  }, [applyZoomTransform])

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    setVideoError(null)
    if (!selectedClipId) {
      v.removeAttribute('src')
      v.load()
      return
    }
    v.src = `/api/clips/${encodeURIComponent(selectedClipId)}/video`
  }, [selectedClipId])

  const onVideoError = useCallback(async () => {
    if (!selectedClipId) return
    try {
      const r = await fetch(`/api/clips/${encodeURIComponent(selectedClipId)}/video`)
      if (r.ok) {
        setVideoError('video element failed to decode the response')
        return
      }
      let detail: string
      try {
        const body = await r.json() as { detail?: string }
        detail = body.detail ?? `HTTP ${r.status}`
      } catch {
        detail = `HTTP ${r.status}`
      }
      setVideoError(detail)
    } catch (err) {
      setVideoError(err instanceof Error ? err.message : String(err))
    }
  }, [selectedClipId])

  const highlightedEntity = useMemo((): { kind: EntityKind; idx: number; subIdx: number } | null => {
    if (!selectedPath || !bundle?.annotation) return null
    const segs = annotationToSegments(bundle.annotation)
    const seg = segs.find(s => s.id === selectedPath)
    if (!seg?.meta) return null
    const meta = seg.meta as Record<string, unknown>
    if (meta._agentIndex != null) return { kind: 'agent', idx: meta._agentIndex as number, subIdx: -1 }
    if (meta._objIndex != null && meta._objKind === 'traffic_object') return { kind: 'traffic_object', idx: meta._objIndex as number, subIdx: -1 }
    if (meta._isSignalHeadSubtrack && meta._lightIndex != null && meta._headIndex != null) return { kind: 'signal_head', idx: meta._lightIndex as number, subIdx: meta._headIndex as number }
    if (meta._envIndex != null && meta._objKind === 'environment') return { kind: 'environment', idx: meta._envIndex as number, subIdx: -1 }
    return null
  }, [selectedPath, bundle])

  const onMeta = useCallback(() => {
    if (videoRef.current) setDuration(videoRef.current.duration)
  }, [setDuration])

  useEffect(() => {
    const container = containerRef.current
    const canvas = canvasRef.current
    if (!container || !canvas) return
    const sync = () => {
      const w = container.clientWidth
      const h = container.clientHeight
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w
        canvas.height = h
      }
    }
    sync()
    const ro = new ResizeObserver(sync)
    ro.observe(container)
    return () => ro.disconnect()
  }, [selectedClipId])

  const redrawOverlay = useCallback(() => {
    const canvas = canvasRef.current
    const video = videoRef.current
    const container = containerRef.current
    if (!canvas || !video || !container) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (!keypointsVisible) return

    const vr = getVideoRect(video, container)
    const state = useStore.getState()
    const ann = state.bundle?.annotation
    if (!ann) return

    const inter = interRef.current
    const markers = collectActiveMarkers(ann, state.playheadTime)

    for (let i = 0; i < markers.length; i++) {
      const m = markers[i]
      const isDragging = inter.mode === 'moving' && inter.entityKind === m.entityKind && inter.entityIdx === m.entityIdx && inter.entitySubIdx === m.entitySubIdx && inter.preview != null
      const drawX = isDragging ? inter.preview!.x : m.x
      const drawY = isDragging ? inter.preview!.y : m.y
      const { px, py } = normalizedToPixel(drawX, drawY, vr)
      const color = entityColor(m.entityKind)
      const isSelected = highlightedEntity != null && highlightedEntity.kind === m.entityKind && highlightedEntity.idx === m.entityIdx && highlightedEntity.subIdx === m.entitySubIdx

      if (isSelected) {
        ctx.beginPath()
        ctx.arc(px, py, MARKER_RADIUS_PX + 4, 0, Math.PI * 2)
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.globalAlpha = 0.6
        ctx.stroke()
        ctx.globalAlpha = 1
      }

      ctx.beginPath()
      ctx.arc(px, py, MARKER_RADIUS_PX, 0, Math.PI * 2)
      if (m.interpolated) {
        ctx.setLineDash([3, 3])
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.globalAlpha = 0.65
        ctx.stroke()
        ctx.setLineDash([])
        ctx.globalAlpha = 1
      } else {
        ctx.fillStyle = color
        ctx.globalAlpha = 0.2
        ctx.fill()
        ctx.globalAlpha = 1
        ctx.strokeStyle = color
        ctx.lineWidth = 1.5
        ctx.stroke()
      }

      const label = m.entityId
      ctx.font = '11px monospace'
      const tm = ctx.measureText(label)
      const lx = px + MARKER_RADIUS_PX + 4
      const ly = py - MARKER_RADIUS_PX - 2
      ctx.fillStyle = m.interpolated ? 'rgba(0,0,0,0.5)' : 'rgba(0,0,0,0.7)'
      ctx.fillRect(lx - 2, ly - 11, tm.width + 4, 14)
      ctx.fillStyle = color
      ctx.fillText(label, lx, ly)
    }
  }, [highlightedEntity, keypointsVisible])

  useEffect(() => { redrawOverlay() }, [playheadTime, bundle, keypointsVisible, redrawOverlay])

  useEffect(() => {
    if (!playing) return
    const tick = () => {
      const v = videoRef.current
      if (v && !v.paused) {
        useStore.setState({ playheadTime: v.currentTime })
        redrawOverlay()
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [playing, redrawOverlay])

  useEffect(() => {
    const v = videoRef.current
    if (!v || !selectedClipId) return
    if (scrubSeekingRef.current || scrubTargetRef.current !== null) return
    if (Math.abs(v.currentTime - playheadTime) > 0.15) v.currentTime = playheadTime
  }, [playheadTime, selectedClipId])

  const clearScrubWatchdog = useCallback(() => {
    if (scrubWatchdogRef.current !== null) {
      window.clearTimeout(scrubWatchdogRef.current)
      scrubWatchdogRef.current = null
    }
  }, [])

  const flushScrub = useCallback(() => {
    scrubRafRef.current = null
    const v = videoRef.current
    if (!v) return
    const target = scrubTargetRef.current
    if (target === null) return
    setPlayhead(target)
    if (scrubSeekingRef.current) return
    scrubTargetRef.current = null
    if (Math.abs(target - v.currentTime) < 1e-4) return
    scrubSeekingRef.current = true
    v.currentTime = target
    // Recover if 'seeked' is missed (some browsers drop it on rapid seeks).
    clearScrubWatchdog()
    scrubWatchdogRef.current = window.setTimeout(() => {
      scrubWatchdogRef.current = null
      scrubSeekingRef.current = false
      if (scrubTargetRef.current !== null && scrubRafRef.current === null) requestScrubFlushRef.current()
    }, 500)
  }, [clearScrubWatchdog, setPlayhead])

  const requestScrubFlush = useCallback(() => {
    if (scrubRafRef.current !== null) return
    scrubRafRef.current = requestAnimationFrame(flushScrub)
  }, [flushScrub])

  useEffect(() => {
    requestScrubFlushRef.current = requestScrubFlush
  }, [requestScrubFlush])

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    const onSeeked = () => {
      clearScrubWatchdog()
      scrubSeekingRef.current = false
      if (scrubTargetRef.current !== null) requestScrubFlush()
    }
    v.addEventListener('seeked', onSeeked)
    return () => {
      v.removeEventListener('seeked', onSeeked)
      if (scrubRafRef.current !== null) {
        cancelAnimationFrame(scrubRafRef.current)
        scrubRafRef.current = null
      }
      clearScrubWatchdog()
      scrubTargetRef.current = null
      scrubSeekingRef.current = false
    }
  }, [clearScrubWatchdog, selectedClipId, requestScrubFlush])

  const toggle = useCallback(() => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) {
      void v.play()
    } else {
      v.pause()
    }
  }, [])
  const step = useCallback((dir: number) => {
    const v = videoRef.current
    if (!v) return
    if (!v.paused) v.pause()
    const fps = bundle?.video?.fps || 30
    const dur = v.duration || 0
    const base = scrubTargetRef.current ?? v.currentTime
    scrubTargetRef.current = Math.max(0, Math.min(dur, base + dir / fps))
    requestScrubFlush()
  }, [bundle?.video?.fps, requestScrubFlush])
  const jump = useCallback((dt: number) => {
    const v = videoRef.current
    if (!v) return
    v.pause()
    v.currentTime = Math.max(0, Math.min(v.duration || 0, v.currentTime + dt))
    setPlayhead(v.currentTime)
  }, [setPlayhead])

  useEffect(() => { if (videoRef.current) videoRef.current.playbackRate = speed }, [speed])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement
      const tag = el.tagName
      const isRange = tag === 'INPUT' && (el as HTMLInputElement).type === 'range'
      if ((tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') && !isRange) return

      if (e.key === ' ' || e.code === 'Space') {
        e.preventDefault()
        toggle()
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault()
        step(e.key === 'ArrowRight' ? 1 : -1)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [step, toggle])

  useEffect(() => { resetZoom() }, [selectedClipId, resetZoom])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const z = zoomRef.current
      if (e.ctrlKey || e.metaKey || !e.shiftKey) {
        const rect = container.getBoundingClientRect()
        const mx = e.clientX - rect.left
        const my = e.clientY - rect.top
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
        const newZoom = Math.min(10, Math.max(1, z.zoom * factor))
        z.panX = mx - (mx - z.panX) * (newZoom / z.zoom)
        z.panY = my - (my - z.panY) * (newZoom / z.zoom)
        z.zoom = newZoom
        if (newZoom === 1) { z.panX = 0; z.panY = 0 }
        clampPan()
        applyZoomTransform()
      } else {
        z.panX -= e.deltaX || e.deltaY
        clampPan()
        applyZoomTransform()
      }
    }
    container.addEventListener('wheel', onWheel, { passive: false })
    return () => container.removeEventListener('wheel', onWheel)
  }, [selectedClipId, applyZoomTransform, clampPan])

  // Mouse interactions on canvas: pan when zoomed, otherwise keypoint manipulation.
  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    const video = videoRef.current
    if (!canvas || !container || !video) return

    // Interpolated markers are visual hints only — only real (annotated) keypoints
    // are hit-testable, so clicks on a dashed marker behave like clicks on empty canvas.
    const hitTestMarker = (cx: number, cy: number): ActiveMarker | null => {
      const ann = useStore.getState().bundle?.annotation
      if (!ann) return null
      const markers = collectActiveMarkers(ann, useStore.getState().playheadTime)
      const vr = getVideoRect(video, container)
      let best: ActiveMarker | null = null
      let bestDist = HIT_RADIUS_PX
      for (const m of markers) {
        if (m.interpolated) continue
        const { px, py } = normalizedToPixel(m.x, m.y, vr)
        const d = Math.hypot(px - cx, py - cy)
        if (d <= bestDist) { bestDist = d; best = m }
      }
      return best
    }

    const canvasPointFromEvent = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect()
      const cx = (e.clientX - rect.left) * (canvas.width / rect.width)
      const cy = (e.clientY - rect.top) * (canvas.height / rect.height)
      return { cx, cy }
    }

    const onMouseDown = (e: MouseEvent) => {
      if (e.button === 2) return // right-click handled separately
      if (e.button !== 0) return

      if (zoomRef.current.zoom > 1) {
        e.preventDefault()
        const z = zoomRef.current
        dragPanRef.current = { x: e.clientX, y: e.clientY, panX: z.panX, panY: z.panY }
        canvas.style.cursor = 'grabbing'
        return
      }

      if (!keypointsVisible) return

      const { cx, cy } = canvasPointFromEvent(e)
      const hit = hitTestMarker(cx, cy)
      const inter = interRef.current
      const isReadOnly = useStore.getState().editsBlocked()

      if (hit && hit.realKpIdx != null) {
        const segId = findEntitySegmentId(useStore.getState().bundle!.annotation, hit.entityKind, hit.entityIdx, hit.entitySubIdx)
        if (segId) selectPath(segId)
        if (isReadOnly) { e.preventDefault(); return }
        inter.mode = 'moving'
        inter.entityKind = hit.entityKind
        inter.entityIdx = hit.entityIdx
        inter.entitySubIdx = hit.entitySubIdx
        inter.kpIdx = hit.realKpIdx
        inter.preview = { x: hit.x, y: hit.y }
        inter.downX = cx
        inter.downY = cy
        inter.moved = false
        canvas.style.cursor = 'grabbing'
        e.preventDefault()
        return
      }

      if (isReadOnly) return

      // Empty canvas: upsert a keypoint for the currently-selected entity.
      const highlighted = (() => {
        const ann = useStore.getState().bundle?.annotation
        const path = useStore.getState().selectedPath
        if (!ann || !path) return null
        const segs = annotationToSegments(ann)
        const seg = segs.find(s => s.id === path)
        if (!seg?.meta) return null
        const m = seg.meta as Record<string, unknown>
        if (m._agentIndex != null) return { kind: 'agent' as EntityKind, idx: m._agentIndex as number, subIdx: -1, t0: seg.t0, t1: seg.t1 }
        if (m._objIndex != null && m._objKind === 'traffic_object') return { kind: 'traffic_object' as EntityKind, idx: m._objIndex as number, subIdx: -1, t0: seg.t0, t1: seg.t1 }
        if (m._isSignalHeadSubtrack && m._lightIndex != null && m._headIndex != null) return { kind: 'signal_head' as EntityKind, idx: m._lightIndex as number, subIdx: m._headIndex as number, t0: seg.t0, t1: seg.t1 }
        if (m._envIndex != null && m._objKind === 'environment') return { kind: 'environment' as EntityKind, idx: m._envIndex as number, subIdx: -1, t0: seg.t0, t1: seg.t1 }
        return null
      })()
      if (!highlighted) return

      const vr = getVideoRect(video, container)
      const { nx, ny } = pixelToNormalized(cx, cy, vr)
      if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return

      const currentBundle = useStore.getState().bundle
      if (!currentBundle) return
      const ann = JSON.parse(JSON.stringify(currentBundle.annotation)) as SilAvAnnotation
      const rawT = useStore.getState().playheadTime
      const t = Math.max(highlighted.t0, Math.min(highlighted.t1, rawT))
      if (t !== rawT) setPlayhead(t)
      const existing = kpsFor(ann, highlighted.kind, highlighted.idx, highlighted.subIdx)
      const updated = upsertKeypoint(existing, t, nx, ny)
      assignKeypoints(ann, highlighted.kind, highlighted.idx, highlighted.subIdx, updated)

      const nextBundle = { ...currentBundle, annotation: ann }
      updateBundle(nextBundle)
      markDirty()
      e.preventDefault()
    }

    const onMouseMove = (e: MouseEvent) => {
      if (dragPanRef.current) {
        e.preventDefault()
        zoomRef.current.panX = dragPanRef.current.panX + (e.clientX - dragPanRef.current.x)
        zoomRef.current.panY = dragPanRef.current.panY + (e.clientY - dragPanRef.current.y)
        clampPan()
        applyZoomTransform()
        return
      }

      const inter = interRef.current
      if (inter.mode === 'moving') {
        const { cx, cy } = canvasPointFromEvent(e)
        const vr = getVideoRect(video, container)
        const { nx, ny } = pixelToNormalized(cx, cy, vr)
        inter.preview = { x: Math.max(0, Math.min(1, nx)), y: Math.max(0, Math.min(1, ny)) }
        if (Math.hypot(cx - inter.downX, cy - inter.downY) > 2) inter.moved = true
        redrawOverlay()
        return
      }

      // Idle hover: set cursor based on what's under the pointer.
      if (zoomRef.current.zoom > 1) { canvas.style.cursor = 'grab'; return }
      if (!keypointsVisible) { canvas.style.cursor = 'default'; return }
      const { cx, cy } = canvasPointFromEvent(e)
      const hit = hitTestMarker(cx, cy)
      canvas.style.cursor = hit ? 'move' : 'crosshair'
    }

    const onMouseUp = (e: MouseEvent) => {
      if (dragPanRef.current) {
        dragPanRef.current = null
        canvas.style.cursor = zoomRef.current.zoom > 1 ? 'grab' : 'default'
        return
      }

      const inter = interRef.current
      if (inter.mode !== 'moving') return
      const { preview, entityKind, entityIdx, entitySubIdx, kpIdx, moved } = inter
      interRef.current = idleInteraction()

      if (!moved || !preview) {
        redrawOverlay()
        return
      }

      if (useStore.getState().editsBlocked()) { redrawOverlay(); return }

      const currentBundle = useStore.getState().bundle
      if (!currentBundle) return
      const ann = JSON.parse(JSON.stringify(currentBundle.annotation)) as SilAvAnnotation
      const existing = kpsFor(ann, entityKind, entityIdx, entitySubIdx)
      if (kpIdx < 0 || kpIdx >= (existing?.length ?? 0)) { redrawOverlay(); return }
      const updated = moveKeypointAt(existing, kpIdx, preview.x, preview.y)
      assignKeypoints(ann, entityKind, entityIdx, entitySubIdx, updated)
      const nextBundle = { ...currentBundle, annotation: ann }
      updateBundle(nextBundle)
      markDirty()
      e.preventDefault()
    }

    const onContextMenu = (e: MouseEvent) => {
      if (!keypointsVisible) return
      const { cx, cy } = canvasPointFromEvent(e)
      const hit = hitTestMarker(cx, cy)
      if (!hit || hit.realKpIdx == null) {
        e.preventDefault()
        return
      }
      e.preventDefault()
      if (useStore.getState().editsBlocked()) return
      const currentBundle = useStore.getState().bundle
      if (!currentBundle) return
      const ann = JSON.parse(JSON.stringify(currentBundle.annotation)) as SilAvAnnotation
      const existing = kpsFor(ann, hit.entityKind, hit.entityIdx, hit.entitySubIdx)
      const updated = removeKeypointAt(existing, hit.realKpIdx)
      assignKeypoints(ann, hit.entityKind, hit.entityIdx, hit.entitySubIdx, updated)
      const nextBundle = { ...currentBundle, annotation: ann }
      updateBundle(nextBundle)
      markDirty()
    }

    canvas.addEventListener('mousedown', onMouseDown)
    canvas.addEventListener('contextmenu', onContextMenu)
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    return () => {
      canvas.removeEventListener('mousedown', onMouseDown)
      canvas.removeEventListener('contextmenu', onContextMenu)
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
  }, [selectedClipId, applyZoomTransform, clampPan, keypointsVisible, redrawOverlay, selectPath, setPlayhead, updateBundle, markDirty])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'b' || e.key === 'B') {
        const tag = (e.target as HTMLElement).tagName
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
        e.preventDefault()
        useStore.getState().toggleKeypointsVisible()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  if (!selectedClipId) {
    return <div className="h-full flex items-center justify-center bg-[#0a0a18] text-[#444] text-base">Select a video from the sidebar</div>
  }

  const hasEntitySelected = highlightedEntity != null
  const selectedKeypointCount = (highlightedEntity && bundle?.annotation)
    ? (kpsFor(bundle.annotation, highlightedEntity.kind, highlightedEntity.idx, highlightedEntity.subIdx)?.length ?? 0)
    : 0

  return (
    <div className="h-full flex flex-col bg-black">
      <div ref={containerRef} className="group/video flex-1 flex items-center justify-center overflow-hidden min-h-0 relative">
        <div ref={wrapperRef} className="w-full h-full relative flex items-center justify-center" style={{ transformOrigin: '0 0' }}>
          <video ref={videoRef} className="max-w-full max-h-full object-contain" onLoadedMetadata={onMeta} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onError={onVideoError} playsInline preload="auto" />
          {videoError ? (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="max-w-md mx-4 px-4 py-3 rounded bg-[#1a0a0a]/95 border border-red-900/60 text-sm text-red-200 text-center pointer-events-auto">
                <div className="font-semibold text-red-300 mb-1">Video unavailable</div>
                <div className="text-xs text-red-200/90 whitespace-pre-wrap">{videoError}</div>
              </div>
            </div>
          ) : null}
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full"
            style={{ pointerEvents: 'auto' }}
          />
        </div>
        {zoomLevel > 1 ? (
          <div className="absolute top-2 right-2 flex flex-col items-end gap-1 z-10">
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-[#aaa] bg-[#1a1a2e]/90 rounded font-mono" style={{ padding: '4px 8px' }}>{Math.round(zoomLevel * 100)}%</span>
              <button onClick={resetZoom} title="Reset zoom (fit)"
                className="w-7 h-7 flex items-center justify-center rounded-lg bg-[#1a1a2e]/90 text-[#aaa] hover:text-white hover:bg-[#2a2a4a] transition-all">
                <Maximize className="w-3.5 h-3.5" />
              </button>
            </div>
            <span className="text-[10px] text-[#666] bg-[#1a1a2e]/70 px-2 py-1 rounded opacity-0 group-hover/video:opacity-100 transition-opacity duration-300">Drag to pan</span>
          </div>
        ) : (
          <div className="absolute top-2 right-2 z-10 flex flex-col items-end gap-1 pointer-events-none">
            <span className="text-[10px] text-[#aaa] bg-[#1a1a2e]/95 px-2 py-1 rounded opacity-0 group-hover/video:opacity-100 transition-opacity duration-300">Scroll to zoom</span>
            {keypointsVisible ? (
              <div className="text-[10px] bg-[#1a1a2e]/95 px-2 py-1 rounded flex flex-col items-end gap-0.5 opacity-0 group-hover/video:opacity-100 transition-opacity duration-300">
                <span className="text-[#ccc] font-medium">Keypoints</span>
                {!hasEntitySelected ? (
                  <span className="text-[#bbb]">select an entity in the timeline</span>
                ) : selectedKeypointCount === 0 ? (
                  <span className="text-amber-300">click video to place</span>
                ) : (
                  <>
                    <span className="text-[#bbb]">click to add</span>
                    <span className="text-[#bbb]">drag to move</span>
                    <span className="text-[#bbb]">right-click to delete</span>
                  </>
                )}
              </div>
            ) : null}
          </div>
        )}
      </div>

      <div className="flex-shrink-0 flex items-center gap-3 px-4 py-3 bg-[#111128] border-t border-[#1e1e38]">
        <div className="flex items-center gap-2 flex-1 basis-0 justify-start">
          <button onClick={toggle} className="w-9 h-9 flex items-center justify-center rounded-lg bg-blue-500 text-white hover:bg-blue-400 transition-all shadow-md">
            {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
          </button>
          <div className="flex items-center gap-1 bg-[#1e1e3a] rounded-lg p-1">
            <button onClick={() => jump(-1)} title="Jump back 1s" className="w-8 h-8 flex items-center justify-center rounded-lg text-[#888] hover:text-white hover:bg-[#2a2a4a] transition-all">
              <ChevronsLeft className="w-4 h-4" />
            </button>
            <button onClick={() => step(-1)} title="Step back 1 frame" className="w-8 h-8 flex items-center justify-center rounded-lg text-[#888] hover:text-white hover:bg-[#2a2a4a] transition-all">
              <SkipBack className="w-4 h-4" />
            </button>
            <button onClick={() => step(1)} title="Step forward 1 frame" className="w-8 h-8 flex items-center justify-center rounded-lg text-[#888] hover:text-white hover:bg-[#2a2a4a] transition-all">
              <SkipForward className="w-4 h-4" />
            </button>
            <button onClick={() => jump(1)} title="Jump forward 1s" className="w-8 h-8 flex items-center justify-center rounded-lg text-[#888] hover:text-white hover:bg-[#2a2a4a] transition-all">
              <ChevronsRight className="w-4 h-4" />
            </button>
          </div>
          <div className="flex items-center gap-1 bg-[#1e1e3a] rounded-lg p-1 ml-2">
            {[0.5, 1, 2, 3].map(s => (
              <button key={s} onClick={() => setSpeed(s)}
                className={`w-8 h-8 flex items-center justify-center text-xs font-semibold rounded-lg transition-all tabular-nums ${speed === s ? 'bg-[#2a2a4a] text-white' : 'text-[#888] hover:text-white'}`}>
                {s}x
              </button>
            ))}
          </div>
        </div>
        <div className="flex-1 flex justify-center">
          <span className="text-[#888] text-xs font-mono tabular-nums">{fmt(playheadTime)} / {fmt(duration)}</span>
        </div>
        <div className="flex items-center gap-2 flex-1 basis-0 justify-end">
          <div className="flex items-center gap-1 bg-[#1e1e3a] rounded-lg p-1">
            <button onClick={toggleKeypointsVisible} title={keypointsVisible ? 'Hide keypoints (B)' : 'Show keypoints (B)'}
              className={`w-8 h-8 flex items-center justify-center rounded-lg transition-all ${keypointsVisible ? 'bg-[#2a2a4a] text-blue-400' : 'text-[#888] hover:text-white'}`}>
              <MapPin className="w-4 h-4" />
            </button>
          </div>
          <div className="flex items-center gap-1 bg-[#1e1e3a] rounded-lg p-1">
            {([
              { key: 'becauseOf',    label: 'B', color: '#f97316', title: 'Because Of' },
              { key: 'linkTo',       label: 'L', color: '#60a5fa', title: 'Link To' },
              { key: 'containedIn',  label: 'C', color: '#4ade80', title: 'Contained In' },
              { key: 'influencedBy', label: 'I', color: '#fb923c', title: 'Influenced By' },
              { key: 'actionTarget', label: 'T', color: '#10b981', title: 'Action Target' },
            ] as const).map(({ key, label, color, title }) => (
              <button key={key} onClick={() => toggleArrowType(key)} title={`${arrowTypes[key] ? 'Hide' : 'Show'} ${title} arrows`}
                className={`w-8 h-8 flex items-center justify-center rounded-lg text-[18px] font-bold transition-all leading-none ${arrowTypes[key] ? 'bg-[#2a2a4a]' : 'hover:text-white'}`}
                style={{ color: arrowTypes[key] ? color : '#888' }}>
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
