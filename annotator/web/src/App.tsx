import { useEffect, useState, useCallback, useRef } from 'react'
import { useStore } from './lib/store'
import { getBundle, getHealth, listClips } from './lib/api'
import { saveCurrentBundle } from './lib/save'
import { Sidebar } from './components/Sidebar'
import { Timeline } from './components/Timeline'
import { RightPanel } from './components/RightPanel'
import { LockBar } from './components/LockBar'

/**
 * Top-level layout: sidebar (clips) | center (LockBar + video + timeline) | right (segment editor).
 * Step 4 wires clip selection + bundle fetch; lock policy lives in LockBar (Step 5).
 */
export default function App() {
  const setClips = useStore(s => s.setClips)
  const setBundle = useStore(s => s.setBundle)
  const setServerReadOnly = useStore(s => s.setServerReadOnly)
  const selectedClipId = useStore(s => s.selectedClipId)
  const clipLoadKey = useStore(s => s.clipLoadKey)

  // Resizable video/timeline splitter.
  const [videoHeight, setVideoHeight] = useState(38) // percentage
  const draggingRef = useRef(false)
  const containerElRef = useRef<HTMLDivElement>(null)

  const onSplitterMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    draggingRef.current = true
    document.body.style.cursor = 'row-resize'
    document.body.style.userSelect = 'none'
    const onMove = (ev: MouseEvent) => {
      if (!draggingRef.current || !containerElRef.current) return
      const rect = containerElRef.current.getBoundingClientRect()
      const pct = ((ev.clientY - rect.top) / rect.height) * 100
      setVideoHeight(Math.max(15, Math.min(70, pct)))
    }
    const onUp = () => {
      draggingRef.current = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [])

  // Initial load: server health + clip list.
  useEffect(() => {
    getHealth()
      .then(h => setServerReadOnly(!!h.read_only))
      .catch(() => setServerReadOnly(false))
    listClips()
      .then(setClips)
      .catch(() => setClips([]))
  }, [setClips, setServerReadOnly])

  // Load the selected clip's bundle whenever the selection (or reload key) changes.
  useEffect(() => {
    if (!selectedClipId) return
    let cancelled = false
    getBundle(selectedClipId)
      .then(b => { if (!cancelled) setBundle(b) })
      .catch(() => { if (!cancelled) setBundle(null) })
    return () => { cancelled = true }
  }, [selectedClipId, clipLoadKey, setBundle])

  // Cmd/Ctrl+S → save (subject to dirty + unlocked + server-writable).
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const isSave = (e.metaKey || e.ctrlKey) && (e.key === 's' || e.key === 'S')
      if (!isSave) return
      const s = useStore.getState()
      if (!s.dirty || s.locked || s.serverReadOnly) return
      e.preventDefault()
      saveCurrentBundle()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  // beforeunload → browser native "unsaved changes" prompt when dirty.
  const dirty = useStore(s => s.dirty)
  useEffect(() => {
    if (!dirty) return
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault()
      // Older Chromium requires a string return; modern browsers ignore it.
      e.returnValue = ''
      return ''
    }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [dirty])

  return (
    <div className="h-screen w-screen flex overflow-hidden bg-[#0c0c1a]">
      <div className="w-[240px] flex-shrink-0 flex flex-col">
        <Sidebar />
      </div>

      <div ref={containerElRef} className="flex-1 flex flex-col min-w-0 overflow-hidden border-x border-[#1e1e38]">
        <LockBar />
        <div
          className="flex-shrink-0"
          style={{ height: `${videoHeight}%`, minHeight: '180px', maxHeight: '70vh' }}
        >
          {/* Phase 3 will mount VideoPlayer here. */}
          <div className="h-full flex items-center justify-center bg-[#0a0a18] text-[#444] text-base">
            {selectedClipId ? 'Video — Phase 3' : 'Select a clip from the sidebar'}
          </div>
        </div>
        <div
          onMouseDown={onSplitterMouseDown}
          className="flex-shrink-0 cursor-row-resize"
          style={{ height: 6, background: '#1e1e38' }}
          onMouseEnter={e => (e.currentTarget.style.background = 'rgba(59,130,246,0.4)')}
          onMouseLeave={e => (e.currentTarget.style.background = '#1e1e38')}
        />
        <div className="flex-1 min-h-[180px] overflow-hidden">
          <Timeline />
        </div>
      </div>

      <div className="w-[320px] flex-shrink-0">
        <RightPanel />
      </div>
    </div>
  )
}
