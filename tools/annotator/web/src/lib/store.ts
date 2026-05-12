import { create } from 'zustand'
import type { AnnotationBundle, UiConfig, SilAvAnnotation } from './types'
import { autoAssignOverlappingTracks, migrateIdsToString } from './timeline-utils'
import type { ClipEntry } from './api'


const MAX_UNDO = 10

// Theme persistence -----------------------------------------------------------
// `theme` is the user's stored preference. `effectiveTheme` is the resolved
// concrete theme used for tokens — equal to `theme` unless `theme === 'system'`,
// in which case it tracks the OS preference. Initial values are seeded from
// `main.tsx`, which hydrates them synchronously before React mounts to avoid
// FOUC. See `src/main.tsx`.
export type Theme = 'system' | 'light' | 'dark'
const THEME_STORAGE_KEY = 'causal-av-annotator.theme'

function resolveEffective(theme: Theme): 'light' | 'dark' {
  if (typeof window === 'undefined') return 'dark'
  if (theme === 'system') {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  }
  return theme
}

function readInitialTheme(): Theme {
  if (typeof window !== 'undefined') {
    const hint = (window as unknown as { __initialTheme?: Theme }).__initialTheme
    if (hint === 'system' || hint === 'light' || hint === 'dark') return hint
    try {
      const v = window.localStorage.getItem(THEME_STORAGE_KEY)
      if (v === 'light' || v === 'dark' || v === 'system') return v
    } catch {
      // ignore
    }
  }
  return 'system'
}


interface AppState {
  // Clip catalog
  clips: ClipEntry[]
  selectedClipId: string | null
  bundle: AnnotationBundle | null
  // Playback / selection
  playheadTime: number
  duration: number
  selectedPath: string | null
  zoomLevel: number
  scrollOffset: number
  // Track counts (derived from bundle on load, persisted via _ui_config)
  envTrackCount: number
  objectTrackCount: number
  lightTrackCount: number
  agentTrackCount: number
  egoContTrackCount: number
  // Lock policy
  // `locked` is the per-clip soft lock that the LockBar toggles. Default true:
  // every clip switch re-locks so accidental edits are impossible.
  locked: boolean
  // `serverReadOnly` mirrors /api/health.read_only — if the server was launched
  // without write access we permanently hide the unlock button.
  serverReadOnly: boolean
  // Dirty tracking for the explicit Save flow (Step 6 wires the save itself).
  dirty: boolean
  saveError: string | null
  // Per-clip reload signal; bumped by selectClip so effects re-fetch.
  clipLoadKey: number
  // Keypoint overlay toggle.
  keypointsVisible: boolean
  arrowTypes: { becauseOf: boolean; linkTo: boolean; containedIn: boolean; influencedBy: boolean; actionTarget: boolean }
  _undoStack: SilAvAnnotation[]
  _isUndoing: boolean
  // Snapshot of the annotation slice at the last known-persisted moment:
  // either (a) the bundle fetched after a clip is loaded (`setBundle`) or
  // (b) the bundle echoed back from a successful save. Undo consults this
  // snapshot so that undoing back to the saved state correctly clears
  // `dirty`. Stored as a JSON string for cheap structural equality.
  lastSavedAnnotationJson: string | null

  // Catalog / selection actions
  setClips: (c: ClipEntry[]) => void
  selectClip: (id: string | null) => void
  setBundle: (b: AnnotationBundle | null) => void
  updateBundle: (b: AnnotationBundle) => void
  undo: () => void
  canUndo: () => boolean
  setPlayhead: (t: number) => void
  setDuration: (d: number) => void
  selectPath: (p: string | null) => void
  setZoom: (z: number) => void
  setScroll: (s: number) => void
  setEnvTrackCount: (n: number) => void
  setObjectTrackCount: (n: number) => void
  setLightTrackCount: (n: number) => void
  setAgentTrackCount: (n: number) => void
  setEgoContTrackCount: (n: number) => void
  // Lock + dirty actions
  lock: () => void
  unlock: () => void
  setServerReadOnly: (v: boolean) => void
  markDirty: () => void
  clearDirty: () => void
  setSaveError: (msg: string | null) => void
  // Captures the current bundle's annotation as the new "last saved"
  // baseline. Call after a successful PUT or after initial bundle load so
  // undo can correctly decide whether to clear `dirty`.
  captureSavedSnapshot: (ann: SilAvAnnotation | null) => void
  // Returns true when edits to the bundle are currently blocked. Identical to
  // `locked` today; kept as a helper because ~35 sites across the timeline /
  // right-panel / video-player call it as `editsBlocked()`.
  editsBlocked: () => boolean
  toggleKeypointsVisible: () => void
  toggleArrowType: (type: keyof AppState['arrowTypes']) => void
  getUiConfig: () => UiConfig
  // Theme
  theme: Theme
  effectiveTheme: 'light' | 'dark'
  setTheme: (t: Theme) => void
}

function deriveTrackCounts(b: AnnotationBundle | null): { env: number; obj: number; light: number; agent: number } {
  if (!b?.annotation) return { env: 1, obj: 1, light: 1, agent: 1 }
  const ann = b.annotation
  let maxEnv = 0
  for (const env of ann.environments || []) maxEnv = Math.max(maxEnv, (env._track_index ?? 0) + 1)
  let maxObj = 0
  for (const obj of ann.traffic_objects || []) maxObj = Math.max(maxObj, (obj._track_index ?? 0) + 1)
  let maxLight = 0
  for (const light of ann.traffic_lights || []) maxLight = Math.max(maxLight, (light._track_index ?? 0) + 1)
  let maxAgent = 0
  for (const agent of ann.agents || []) maxAgent = Math.max(maxAgent, (agent._track_index ?? 0) + 1)
  return { env: Math.max(1, maxEnv), obj: Math.max(1, maxObj), light: Math.max(1, maxLight), agent: Math.max(1, maxAgent) }
}

function applyBundle(b: AnnotationBundle | null, s: AppState, initialLoad = false) {
  if (initialLoad && b?.annotation) {
    migrateIdsToString(b.annotation)
    autoAssignOverlappingTracks(b.annotation)
  }
  const dataCounts = deriveTrackCounts(b)
  const ui = b?._ui_config
  return {
    bundle: b,
    duration: b?.video?.duration_s || s.duration,
    envTrackCount: Math.max(dataCounts.env, ui?.envTrackCount ?? 1),
    objectTrackCount: Math.max(dataCounts.obj, ui?.objectTrackCount ?? 1),
    lightTrackCount: Math.max(dataCounts.light, ui?.lightTrackCount ?? 1),
    agentTrackCount: Math.max(dataCounts.agent, ui?.agentTrackCount ?? 1),
    zoomLevel: ui?.zoomLevel ?? s.zoomLevel,
    scrollOffset: ui?.scrollOffset ?? s.scrollOffset,
  }
}

const __initialTheme: Theme = readInitialTheme()
const __initialEffective: 'light' | 'dark' = (
  typeof window !== 'undefined'
    ? (window as unknown as { __initialEffectiveTheme?: 'light' | 'dark' }).__initialEffectiveTheme
    : undefined
) ?? resolveEffective(__initialTheme)

export const useStore = create<AppState>((set, get) => ({
  clips: [],
  selectedClipId: null,
  bundle: null,
  playheadTime: 0,
  duration: 0,
  selectedPath: null,
  zoomLevel: 1,
  scrollOffset: 0,
  envTrackCount: 1,
  objectTrackCount: 1,
  lightTrackCount: 1,
  agentTrackCount: 1,
  egoContTrackCount: 1,
  locked: true,
  serverReadOnly: false,
  dirty: false,
  saveError: null,
  clipLoadKey: 0,
  keypointsVisible: true,
  arrowTypes: { becauseOf: true, linkTo: true, containedIn: true, influencedBy: true, actionTarget: true },
  _undoStack: [],
  _isUndoing: false,
  lastSavedAnnotationJson: null,

  setClips: (c) => set({ clips: c }),

  selectClip: (id) => set((s) => ({
    selectedClipId: id,
    bundle: null,
    playheadTime: 0,
    selectedPath: null,
    envTrackCount: 1, objectTrackCount: 1, lightTrackCount: 1, agentTrackCount: 1, egoContTrackCount: 1,
    zoomLevel: 1, scrollOffset: 0,
    // Auto-lock on every clip switch.
    locked: true,
    dirty: false,
    saveError: null,
    clipLoadKey: s.clipLoadKey + 1,
    _undoStack: [], _isUndoing: false,
    lastSavedAnnotationJson: null,
  })),

  setBundle: (b) => set((s) => ({
    ...applyBundle(b, s, true),
    _undoStack: [],
    _isUndoing: false,
    // Initial fetch is, by definition, the on-disk truth — record the
    // annotation as the saved baseline.
    lastSavedAnnotationJson: b?.annotation ? JSON.stringify(b.annotation) : null,
  })),

  updateBundle: (b) => set((s) => {
    const stack = [...s._undoStack]
    if (!s._isUndoing && s.bundle?.annotation) {
      stack.push(JSON.parse(JSON.stringify(s.bundle.annotation)))
      if (stack.length > MAX_UNDO) stack.shift()
    }
    return {
      ...applyBundle(b, s),
      _undoStack: stack,
      _isUndoing: false,
      dirty: true,
    }
  }),

  undo: () => set((s) => {
    if (s._undoStack.length === 0 || !s.bundle) return s
    const stack = [...s._undoStack]
    const prev = stack.pop()!
    const restored = { ...s.bundle, annotation: prev }
    // If the restored annotation matches the last persisted snapshot, the
    // undo brought us back to the on-disk state — `dirty` clears. Anything
    // else stays dirty.
    const restoredJson = JSON.stringify(prev)
    const matchesSaved = s.lastSavedAnnotationJson !== null
      && restoredJson === s.lastSavedAnnotationJson
    return {
      ...applyBundle(restored, s),
      _undoStack: stack,
      _isUndoing: true,
      selectedPath: null,
      dirty: !matchesSaved,
      // Saved baseline doesn't move during undo — only setBundle /
      // captureSavedSnapshot mutate it.
    }
  }),

  canUndo: () => get()._undoStack.length > 0,

  setPlayhead: (t) => set((s) => ({ playheadTime: Math.max(0, Math.min(t, s.duration || Infinity)) })),
  setDuration: (d) => set({ duration: d }),
  selectPath: (p) => set({ selectedPath: p }),
  setZoom: (z) => set((s) => {
    const newZoom = Math.max(0.5, Math.min(20, z))
    const maxScroll = Math.max(0, s.duration - s.duration / newZoom)
    return { zoomLevel: newZoom, scrollOffset: Math.min(s.scrollOffset, maxScroll) }
  }),
  setScroll: (s) => set({ scrollOffset: Math.max(0, s) }),
  setEnvTrackCount: (n) => set({ envTrackCount: Math.max(1, n) }),
  setObjectTrackCount: (n) => set({ objectTrackCount: Math.max(1, n) }),
  setLightTrackCount: (n) => set({ lightTrackCount: Math.max(1, n) }),
  setAgentTrackCount: (n) => set({ agentTrackCount: Math.max(1, n) }),
  setEgoContTrackCount: (n) => set({ egoContTrackCount: Math.max(1, n) }),

  lock: () => set({ locked: true }),
  unlock: () => set((s) => (s.serverReadOnly ? s : { locked: false })),
  setServerReadOnly: (v) => set({ serverReadOnly: v }),
  markDirty: () => set({ dirty: true }),
  clearDirty: () => set((s) => ({
    dirty: false,
    saveError: null,
    lastSavedAnnotationJson: s.bundle?.annotation
      ? JSON.stringify(s.bundle.annotation)
      : s.lastSavedAnnotationJson,
  })),
  setSaveError: (msg) => set({ saveError: msg }),
  captureSavedSnapshot: (ann) => set({
    lastSavedAnnotationJson: ann ? JSON.stringify(ann) : null,
  }),

  editsBlocked: () => get().locked,
  toggleKeypointsVisible: () => set((s) => ({ keypointsVisible: !s.keypointsVisible })),
  toggleArrowType: (type) => set((s) => ({ arrowTypes: { ...s.arrowTypes, [type]: !s.arrowTypes[type] } })),

  theme: __initialTheme,
  effectiveTheme: __initialEffective,
  setTheme: (t) => {
    const effective = resolveEffective(t)
    if (typeof window !== 'undefined') {
      try { window.localStorage.setItem(THEME_STORAGE_KEY, t) } catch { /* ignore */ }
      document.documentElement.dataset.theme = effective
    }
    set({ theme: t, effectiveTheme: effective })
  },

  getUiConfig: () => {
    const s = get()
    return {
      envTrackCount: s.envTrackCount,
      objectTrackCount: s.objectTrackCount,
      lightTrackCount: s.lightTrackCount,
      agentTrackCount: s.agentTrackCount,
      zoomLevel: s.zoomLevel,
      scrollOffset: s.scrollOffset,
    }
  },
}))
