import type { AnnotationBundle, VideoListItem } from './types'
import { useStore } from './store'
import { getToken, clearToken } from './auth'
import type { AuthUser } from './auth'
import { timeTracker } from './time-tracker'

const BASE = '/api'

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {}
  const token = getToken()
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  if (body) {
    headers['Content-Type'] = 'application/json'
  }
  const opts: RequestInit = { method, headers }
  if (body) {
    opts.body = JSON.stringify(body)
  }
  const res = await fetch(`${BASE}${path}`, opts)
  if (res.status === 401) {
    clearToken()
    window.location.reload()
    throw new Error('Session expired')
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    const detail = err.detail
    // detail may be a plain string or a structured {code, message} object.
    const message = typeof detail === 'object' && detail !== null
      ? (detail.message || res.statusText)
      : (detail || res.statusText)
    const e = new Error(message) as Error & { code?: string }
    if (typeof detail === 'object' && detail !== null && typeof detail.code === 'string') {
      e.code = detail.code
    }
    throw e
  }
  return res.json()
}

export const api = {
  // Auth
  login: (email: string, password: string) =>
    request<{ token: string; user: AuthUser }>('POST', '/auth/login', { email, password }),
  register: (email: string, password: string, display_name: string = '') =>
    request<{ id: string; message: string }>('POST', '/auth/register', { email, password, display_name }),
  me: () => request<AuthUser>('GET', '/auth/me'),
  listUsers: () => request<AuthUser[]>('GET', '/auth/users'),
  updateUser: (userId: string, updates: { is_approved?: boolean; role?: string }) =>
    request<AuthUser>('PUT', `/auth/users/${userId}`, updates),

  // Videos
  listVideos: () => request<VideoListItem[]>('GET', '/videos'),
  streamUrl: (id: string) => {
    const token = getToken()
    return token ? `${BASE}/videos/${id}/stream?token=${encodeURIComponent(token)}` : `${BASE}/videos/${id}/stream`
  },
  videoInfo: (id: string) => request<Record<string, unknown>>('GET', `/videos/${id}/info`),

  // Annotations
  triggerAnnotation: (id: string) => request<{ status: string }>('POST', `/videos/${id}/annotate`),
  annotationStatus: (id: string) => request<{ job_status: string }>('GET', `/videos/${id}/annotate/status`),
  getAnnotations: (id: string, userId?: string) => {
    const params = userId ? `?user_id=${encodeURIComponent(userId)}` : ''
    return request<AnnotationBundle>('GET', `/videos/${id}/annotations${params}`)
  },
  listClipAnnotations: (clipId: string) =>
    request<ClipAnnotationEntry[]>('GET', `/videos/${clipId}/annotations/list`),

  saveAnnotations: async (id: string, bundle: unknown, targetUserId?: string | null) => {
    const s = useStore.getState()
    const uiConfig = s.getUiConfig()
    const { ms: activeMs, rollback } = timeTracker.flush()
    const withUi = { ...(bundle as Record<string, unknown>), _ui_config: uiConfig, _active_ms: activeMs }
    // When the reviewer is in Edit Mode, every save (including incidental
    // ones from Timeline/VideoPlayer) must target the annotator's row, not
    // the reviewer's own. Callers may still pass an explicit targetUserId.
    const effectiveTarget = targetUserId ?? (s.reviewEditMode && s.viewingUserId ? s.viewingUserId : null)
    const path = effectiveTarget
      ? `/videos/${id}/annotations?target_user_id=${encodeURIComponent(effectiveTarget)}`
      : `/videos/${id}/annotations`
    try {
      const result = await request<AnnotationBundle & { _track_reviews?: TrackReviewEntry[] }>('PUT', path, withUi)
      // Sync track reviews from server (handles auto-invalidation of changed approvals)
      if (result._track_reviews) {
        useStore.getState().setTrackReviews(result._track_reviews)
        delete (result as unknown as Record<string, unknown>)._track_reviews
      }
      return result
    } catch (e) {
      rollback()
      throw e
    }
  },

  // Legacy approve/disapprove removed — use track review workflow instead

  // Feedback
  getFeedback: (clipId: string, userId: string) =>
    request<FeedbackEntry[]>('GET', `/videos/${clipId}/feedback?user_id=${encodeURIComponent(userId)}`),

  // Time logs
  getTimeBreakdown: (clipId: string, userId: string) =>
    request<TimeBreakdown>('GET', `/videos/${clipId}/time-logs?user_id=${encodeURIComponent(userId)}`),

  // Track reviews — submitted (immutable) per-row decisions
  getTrackReviews: (clipId: string, userId: string) =>
    request<TrackReviewEntry[]>('GET', `/videos/${clipId}/track-reviews?user_id=${encodeURIComponent(userId)}`),

  // Review draft — shared in-progress review state for the current round
  getReviewDraft: (clipId: string, userId: string) =>
    request<ReviewDraftResponse>('GET', `/videos/${clipId}/review-draft?user_id=${encodeURIComponent(userId)}`),
  putReviewDraft: async (clipId: string, userId: string, state: ReviewDraftState, expectedVersion: number) => {
    const { ms, rollback } = timeTracker.flush()
    try {
      return await request<{ version: number }>('PUT', `/videos/${clipId}/review-draft`, {
        user_id: userId,
        state,
        expected_version: expectedVersion,
        active_ms: ms,
      })
    } catch (e) {
      rollback()
      throw e
    }
  },

  // Submit consumes the current-round draft and seals the round.
  submitReview: async (clipId: string, userId: string, expectedVersion: number) => {
    const { ms, rollback } = timeTracker.flush()
    try {
      return await request<TrackReviewEntry[]>('POST', `/videos/${clipId}/review-submit`, {
        user_id: userId,
        expected_version: expectedVersion,
        active_ms: ms,
      })
    } catch (e) {
      rollback()
      throw e
    }
  },

  // Append-only commentary, allowed before or after submit. track_id NULL = general.
  postReviewComment: async (clipId: string, userId: string, text: string, trackId?: string | null) => {
    const { ms, rollback } = timeTracker.flush()
    try {
      return await request<FeedbackEntry>('POST', `/videos/${clipId}/review-comment`, {
        user_id: userId,
        text,
        track_id: trackId ?? null,
        active_ms: ms,
      })
    } catch (e) {
      rollback()
      throw e
    }
  },
  exportAnnotations: (id: string) => request<AnnotationBundle>('GET', `/videos/${id}/export`),
  downloadGoldenSet: () => request<unknown>('POST', '/videos/download-golden-set'),
  downloadStatus: () => request<DownloadProgress>('GET', '/videos/download-status'),

  // Work manager
  getMyTasks: () => request<WorkTask[]>('GET', '/work/my-tasks'),
  getNextTask: () => request<{ task: WorkTask | null; message?: string }>('POST', '/work/next'),
  startTask: (taskId: string) => request<WorkTask>('POST', `/work/tasks/${taskId}/start`),
  switchTask: (taskId: string) => request<WorkTask>('POST', `/work/tasks/${taskId}/switch`),
  submitTask: (taskId: string) => request<WorkTask>('POST', `/work/tasks/${taskId}/submit`),
  releaseTask: (taskId: string) => request<WorkTask>('POST', `/work/tasks/${taskId}/release`),

  // Admin work manager
  createBatch: (data: { name: string; description?: string; tasks: BatchTaskInput[] }) =>
    request<{ id: string }>('POST', '/work/batches', data),
  getBatches: () => request<BatchSummary[]>('GET', '/work/batches'),
  getBatchDetail: (batchId: string) => request<BatchDetail>('GET', `/work/batches/${batchId}`),
  getWorkStats: () => request<WorkStats>('GET', '/work/stats'),

  // Batch annotators
  addBatchAnnotator: (batchId: string, userId: string) =>
    request<{ batch_id: string; user_id: string }>('POST', `/work/batches/${batchId}/annotators`, { user_id: userId }),
  removeBatchAnnotator: (batchId: string, userId: string) =>
    request<{ batch_id: string; user_id: string }>('DELETE', `/work/batches/${batchId}/annotators/${userId}`),
  getBatchAnnotators: (batchId: string) =>
    request<BatchAnnotator[]>('GET', `/work/batches/${batchId}/annotators`),
}

// Work manager types
export interface WorkTask {
  id: string
  batch_id: string
  clip_id: string
  task_type: string
  task_definition: string
  priority: number
  status: string
  annotation_status: string | null
  assigned_to: string | null
  deadline: string | null
}

export interface BatchTaskInput {
  clip_id: string
  task_type: string
  task_definition: string
  priority?: number
  deadline?: string
}

export interface BatchAnnotator {
  id: string
  display_name: string
  email: string
}

export interface BatchSummary {
  id: string
  name: string
  description: string
  status: string
  total: number
  accepted: number
  in_progress: number
  needs_revision: number
  annotators: BatchAnnotator[]
}

export interface BatchDetail extends BatchSummary {
  tasks: WorkTask[]
}

export interface WorkStats {
  batch_count: number
  total_tasks: number
  accepted: number
  in_progress: number
  needs_revision: number
  acceptance_rate: number
}

export interface ClipAnnotationEntry {
  user_id: string
  display_name: string
  status: string
}

export interface FeedbackEntry {
  id: string
  reviewer_name: string
  text: string
  track_id: string | null  // NULL = general / annotation-level commentary
  round: number
  created_at: string
}

export type TrackGroup = 'environments' | 'ego' | 'objects' | 'agents'
export type TrackReviewStatus = 'approved' | 'needs_revision'

export interface SubtrackDecision {
  kind: string
  index: number
  status: TrackReviewStatus
}

export interface TrackReviewEntry {
  id: string
  track_group: TrackGroup
  track_id: string | null
  status: TrackReviewStatus
  subtrack_decisions: SubtrackDecision[] | null
  reviewer_name: string
  round: number
  created_at: string
}

export interface ReviewDraftRow {
  track_group: TrackGroup
  status: TrackReviewStatus | null
  subtrack_decisions: SubtrackDecision[]
  comment: string
}

export interface ReviewDraftState {
  rows: Record<string, ReviewDraftRow>
  general_comment: string
  // Per-display-group free-text "what's missing" flags. Materialized into
  // general feedback rows on submit, prefixed with the group label.
  missing_per_group?: Record<string, string>
}

export type ReviewRowHint = 'changed' | 'new'

export interface ReviewDraftResponse {
  state: ReviewDraftState
  version: number
  updated_by_name: string
  updated_at: string
  row_hints: Record<string, ReviewRowHint>
}

export interface TimeBreakdownEntry {
  kind: 'label' | 'review'
  round: number
  active_ms: number
  entry_count: number
}

export interface TimeBreakdown {
  clip_id: string
  user_id: string
  total_label_ms: number
  total_review_ms: number
  breakdown: TimeBreakdownEntry[]
}

export interface DownloadProgress {
  active: boolean
  total: number
  completed: number
  current_clip_id: string | null
  failed: string[]
  done: boolean
  completed_clips: string[]
}
