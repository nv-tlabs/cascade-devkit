import { useEffect, useState } from 'react'
import { useStore } from '../lib/store'
import { api } from '../lib/api'
import type { DownloadProgress } from '../lib/api'
import { Download, Search, Film, CheckCircle, XCircle, Clock, Loader2, Star, User as UserIcon, AlertTriangle } from 'lucide-react'
import type { ClipAnnotationEntry } from '../lib/api'

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'approved':
      return <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
    case 'disapproved':
      return <XCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
    case 'needs_revision':
      return <AlertTriangle className="w-4 h-4 text-orange-400 flex-shrink-0" />
    case 'annotating':
      return <Loader2 className="w-4 h-4 text-blue-500 animate-spin flex-shrink-0" />
    default:
      return <Clock className="w-4 h-4 text-amber-500 flex-shrink-0" />
  }
}

export function Sidebar({ reviewerMode = false }: { reviewerMode?: boolean }) {
  const { videos, setVideos, selectClip, selectedClipId, currentUser, setReadOnly, setViewingUserId, viewingUserId, feedback } = useStore()
  const [filter, setFilter] = useState('')
  const [downloading, setDownloading] = useState(false)
  const [progress, setProgress] = useState<DownloadProgress | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const [clipAnnotations, setClipAnnotations] = useState<Record<string, ClipAnnotationEntry[]>>({})
  const isReviewer = reviewerMode || currentUser?.role === 'reviewer'

  useEffect(() => {
    api.listVideos().then(setVideos).catch(() => setVideos([]))
  }, [setVideos, refreshKey])

  useEffect(() => {
    const interval = setInterval(() => setRefreshKey(k => k + 1), 5000)
    return () => clearInterval(interval)
  }, [])

  // Poll download progress while active
  useEffect(() => {
    if (!downloading) { setProgress(null); return }
    let cancelled = false
    const poll = async () => {
      while (!cancelled) {
        try {
          const status = await api.downloadStatus()
          if (cancelled) break
          setProgress(status)
          if (status.done) {
            setDownloading(false)
            setRefreshKey(k => k + 1)
            break
          }
        } catch { /* ignore */ }
        await new Promise(r => setTimeout(r, 2000))
      }
    }
    poll()
    return () => { cancelled = true }
  }, [downloading])

  useEffect(() => {
    if (!isReviewer || !selectedClipId) return
    api.listClipAnnotations(selectedClipId)
      .then(entries => setClipAnnotations({ [selectedClipId]: entries }))
      .catch(() => {})
  }, [isReviewer, selectedClipId, feedback.length])

  const q = filter.trim().toLowerCase()
  const filtered = q
    ? videos.filter((v) => v.filename.toLowerCase().includes(q) || v.clip_id.toLowerCase().includes(q))
    : videos

  const handleDownloadGoldenSet = async () => {
    setDownloading(true)
    try {
      await api.downloadGoldenSet()
    } catch (e) {
      alert('Download failed: ' + (e instanceof Error ? e.message : String(e)))
      setDownloading(false)
    }
  }

  return (
    <aside className="flex flex-col h-full bg-[#13132a] border-r border-[#2a2a45] overflow-x-hidden overflow-y-hidden">
      <header className="shrink-0 p-3 border-b border-[#2a2a45]">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-[#e0e0e0] font-semibold text-sm">SIL-AV Annotator</h1>
            <p className="text-[10px] text-[#8888aa] mt-0.5">Golden Set</p>
          </div>
        </div>
        {downloading && progress ? (
          <div className="w-full mt-3 p-3 bg-[#1e1e3a] border border-[#2a2a45] rounded text-xs">
            <div className="flex items-center justify-between text-[#e0e0e0] mb-2">
              <span className="flex items-center gap-1.5">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Downloading...
              </span>
              <span className="text-[#8888aa]">{progress.completed}/{progress.total}</span>
            </div>
            <div className="w-full h-1.5 bg-[#0a0a1a] rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-500"
                style={{ width: `${progress.total ? (progress.completed / progress.total) * 100 : 0}%` }}
              />
            </div>
            {progress.current_clip_id && (
              <p className="mt-1.5 text-[10px] text-[#8888aa] truncate">
                {progress.current_clip_id.slice(0, 8)}...
              </p>
            )}
            {progress.failed.length > 0 && (
              <p className="mt-1 text-[10px] text-red-400">
                {progress.failed.length} failed
              </p>
            )}
          </div>
        ) : (
          <button
            onClick={handleDownloadGoldenSet}
            disabled={downloading}
            className="w-full mt-3 py-2 px-3 flex items-center justify-center gap-2 text-xs bg-[#1e1e3a] text-[#e0e0e0] border border-[#2a2a45] rounded hover:bg-[#252550] disabled:opacity-50 transition-colors"
          >
            <Download className="w-4 h-4" />
            Download All
          </button>
        )}
      </header>

      <div className="px-3 py-2 border-b border-[#2a2a45]">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[#666] pointer-events-none" />
          <input
            type="text"
            placeholder="Search clips..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="w-full py-1.5 pl-8 pr-2 text-xs bg-[#1e1e3a] border border-[#2a2a45] rounded text-[#e0e0e0] placeholder:text-[#666] focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
      </div>

      <ul className="flex-1 overflow-y-auto overflow-x-hidden p-2 space-y-0.5">
        {filtered.map((v) => {
          const selected = v.clip_id === selectedClipId
          return (
            <li key={v.clip_id}>
              <button
                onClick={() => selectClip(v.clip_id)}
                className={`w-full text-left p-2 rounded transition-colors flex items-center gap-2.5 ${
                  selected
                    ? 'bg-[#2a2a50] border-l-[3px] border-l-blue-500'
                    : 'bg-transparent border-l-[3px] border-l-transparent hover:bg-[#222245]'
                }`}
              >
                <Film className="w-4 h-4 text-[#666] flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <code className="block text-[11px] font-mono text-[#e0e0e0] truncate">
                    {v.clip_id.slice(0, 8)}
                  </code>
                  {!isReviewer && v.has_annotation && (
                    <span className="text-[10px] text-[#8888aa] mt-0.5 inline-block">
                      {v.env_count} env · {v.agent_count} agents{v.ego_action_count > 0 ? ` · ${v.ego_action_count} ego` : ''}
                    </span>
                  )}
                </div>
                {!isReviewer && <StatusIcon status={v.status} />}
              </button>
              {selected && isReviewer && clipAnnotations[v.clip_id] && (
                <div className="ml-6 mt-1 mb-1 space-y-0.5">
                  <button
                    onClick={() => {
                      setViewingUserId(null)
                      setReadOnly(false)
                    }}
                    className={`w-full text-left px-2 py-1 rounded text-[10px] flex items-center gap-1.5 ${
                      !viewingUserId
                        ? 'bg-yellow-900/40 text-yellow-300'
                        : 'bg-yellow-900/20 text-yellow-400/70 hover:bg-yellow-900/30'
                    }`}
                  >
                    <Star className="w-3 h-3" />
                    Golden (you)
                  </button>
                  {clipAnnotations[v.clip_id]
                    .filter(a => a.user_id !== currentUser?.id)
                    .map(a => (
                      <button
                        key={a.user_id}
                        onClick={() => {
                          setViewingUserId(a.user_id)
                          setReadOnly(true)
                        }}
                        className={`w-full text-left px-2 py-1 rounded text-[10px] flex items-center gap-1.5 ${
                          viewingUserId === a.user_id
                            ? 'bg-[#2a2a50] text-[#e0e0e0]'
                            : 'text-[#8888aa] hover:bg-[#222245]'
                        }`}
                      >
                        <UserIcon className="w-3 h-3" />
                        {a.display_name}
                        <span className={`ml-auto text-[9px] ${
                          a.status === 'needs_revision' ? 'text-orange-400' :
                          a.status === 'approved' ? 'text-green-400' :
                          a.status === 'submitted' ? 'text-blue-400' :
                          'opacity-60'
                        }`}>{a.status === 'needs_revision' ? 'revision requested' : a.status}</span>
                      </button>
                    ))}
                </div>
              )}
            </li>
          )
        })}
      </ul>
    </aside>
  )
}
