import { create } from 'zustand'
import type { AnnotationBundle, UiConfig, VideoListItem, SilAvAnnotation } from './types'
import { autoAssignOverlappingTracks, migrateIdsToString } from './timeline-utils'
import type { AuthUser } from './auth'
import type { WorkTask, FeedbackEntry, TrackReviewEntry, ReviewDraftState } from './api'


const MAX_UNDO = 10


interface AppState {
  // Auth
  currentUser: AuthUser | null
  // Work manager
  currentTask: WorkTask | null
  myTasks: WorkTask[]
  // Existing state
  videos: VideoListItem[]
  selectedClipId: string | null
  bundle: AnnotationBundle | null
  playheadTime: number
  duration: number
  selectedPath: string | null
  zoomLevel: number
  scrollOffset: number
  envTrackCount: number
  objectTrackCount: number
  lightTrackCount: number
  agentTrackCount: number
  egoContTrackCount: number
  feedback: FeedbackEntry[]
  trackReviews: TrackReviewEntry[]
  // Shared review-draft state for the current (annotation, round). The top
  // Save button in RightPanel hits this when the reviewer is mid-review;
  // ReviewSection mirrors it into UI controls. ``reviewDraftVersion === 0``
  // means "no server-side draft yet for this round".
  reviewDraft: ReviewDraftState
  reviewDraftVersion: number
  readOnly: boolean
  viewingUserId: string | null
  // When true, reviewer/admin/owner is editing the annotation they're reviewing.
  // Saves are routed to the annotator's row via the target_user_id query param.
  // Resets to false on clip / viewing-user change so the toggle never leaks
  // across contexts.
  reviewEditMode: boolean
  clipLoadKey: number
  keypointsVisible: boolean
  arrowTypes: { becauseOf: boolean; linkTo: boolean; containedIn: boolean; influencedBy: boolean; actionTarget: boolean }
  _undoStack: SilAvAnnotation[]
  _isUndoing: boolean

  // Auth actions
  setCurrentUser: (u: AuthUser | null) => void
  // Work manager actions
  setCurrentTask: (t: WorkTask | null) => void
  setMyTasks: (tasks: WorkTask[]) => void
  // Existing actions
  setVideos: (v: VideoListItem[]) => void
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
  setFeedback: (f: FeedbackEntry[]) => void
  setTrackReviews: (r: TrackReviewEntry[]) => void
  setReviewDraft: (d: ReviewDraftState) => void
  setReviewDraftVersion: (v: number) => void
  resetReviewDraft: () => void
  setReadOnly: (v: boolean) => void
  setViewingUserId: (id: string | null) => void
  setReviewEditMode: (v: boolean) => void
  editsBlocked: () => boolean
  toggleKeypointsVisible: () => void
  toggleArrowType: (type: keyof AppState['arrowTypes']) => void
  getUiConfig: () => UiConfig
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

function stripLegacyBboxes(ann: SilAvAnnotation): void {
  const drop = (e: Record<string, unknown>) => { delete e.bounding_boxes }
  for (const a of ann.agents || []) drop(a as unknown as Record<string, unknown>)
  for (const o of ann.traffic_objects || []) drop(o as unknown as Record<string, unknown>)
  for (const l of ann.traffic_lights || []) drop(l as unknown as Record<string, unknown>)
  delete (ann as unknown as Record<string, unknown>).bounding_box_sequence
}

function applyBundle(b: AnnotationBundle | null, s: AppState, initialLoad = false) {
  if (initialLoad && b?.annotation) {
    migrateIdsToString(b.annotation)
    stripLegacyBboxes(b.annotation)
    autoAssignOverlappingTracks(b.annotation)
  }
  const dataCounts = deriveTrackCounts(b)
  const ui = b?._ui_config
  // Track counts must reflect the bundle being loaded — never carry over from
  // the previous bundle's count. Otherwise switching between annotations on
  // the same clip (e.g., reviewer viewing different annotators) leaves phantom
  // empty rows from the prior, "richer" bundle.
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

export const useStore = create<AppState>((set, get) => ({
  currentUser: null,
  currentTask: null,
  myTasks: [],
  videos: [],
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
  feedback: [],
  trackReviews: [],
  reviewDraft: { rows: {}, general_comment: '', missing_per_group: {} },
  reviewDraftVersion: 0,
  clipLoadKey: 0,
  readOnly: false,
  viewingUserId: null,
  reviewEditMode: false,
  keypointsVisible: true,
  arrowTypes: { becauseOf: true, linkTo: true, containedIn: true, influencedBy: true, actionTarget: true },
  _undoStack: [],
  _isUndoing: false,

  setCurrentUser: (u) => set({ currentUser: u }),
  setCurrentTask: (t) => set({ currentTask: t }),
  setMyTasks: (tasks) => set((s) => {
    let currentTask = s.currentTask
    if (currentTask) {
      const fresh = tasks.find(t => t.id === currentTask!.id)
      currentTask = fresh && fresh.status === 'in_progress' ? fresh : null
    }
    return { myTasks: tasks, currentTask }
  }),
  setVideos: (v) => set({ videos: v }),

  selectClip: (id) => set((s) => ({
    selectedClipId: id, bundle: null, playheadTime: 0, selectedPath: null,
    envTrackCount: 1, objectTrackCount: 1, lightTrackCount: 1, agentTrackCount: 1, egoContTrackCount: 1,
    zoomLevel: 1, scrollOffset: 0,
    feedback: [], trackReviews: [],
    reviewDraft: { rows: {}, general_comment: '', missing_per_group: {} }, reviewDraftVersion: 0,
    clipLoadKey: s.clipLoadKey + 1, readOnly: false, viewingUserId: null, reviewEditMode: false,
    _undoStack: [], _isUndoing: false,
  })),

  setBundle: (b) => set((s) => ({
    ...applyBundle(b, s, true),
    _undoStack: [],
    _isUndoing: false,
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
    }
  }),

  undo: () => set((s) => {
    if (s._undoStack.length === 0 || !s.bundle) return s
    const stack = [...s._undoStack]
    const prev = stack.pop()!
    const restored = { ...s.bundle, annotation: prev }
    return {
      ...applyBundle(restored, s),
      _undoStack: stack,
      _isUndoing: true,
      selectedPath: null,
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
  setFeedback: (f) => set({ feedback: f }),
  setTrackReviews: (r) => set({ trackReviews: r }),
  setReviewDraft: (d) => set({ reviewDraft: d }),
  setReviewDraftVersion: (v) => set({ reviewDraftVersion: v }),
  resetReviewDraft: () => set({ reviewDraft: { rows: {}, general_comment: '', missing_per_group: {} }, reviewDraftVersion: 0 }),
  setReadOnly: (v) => set({ readOnly: v }),
  setViewingUserId: (id) => set((s) => ({
    viewingUserId: id,
    reviewEditMode: id !== s.viewingUserId ? false : s.reviewEditMode,
  })),
  setReviewEditMode: (v) => set({ reviewEditMode: v }),

  // Convenience: returns true when edits to the bundle are currently blocked.
  // Edits are allowed when not readOnly, OR when readOnly but the reviewer
  // has flipped Edit Mode on (saves route to the annotator's row via
  // target_user_id).
  editsBlocked: () => {
    const s = get()
    return s.readOnly && !s.reviewEditMode
  },
  toggleKeypointsVisible: () => set((s) => ({ keypointsVisible: !s.keypointsVisible })),
  toggleArrowType: (type) => set((s) => ({ arrowTypes: { ...s.arrowTypes, [type]: !s.arrowTypes[type] } })),

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
