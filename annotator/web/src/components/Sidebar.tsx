import { useState } from 'react'
import * as Dialog from '@radix-ui/react-dialog'
import { useStore } from '../lib/store'
import { saveCurrentBundle } from '../lib/save'
import { Search, Film, FileText } from 'lucide-react'

/**
 * Clip list. One entry per clip from /api/clips, with a kind badge
 * (`annotated` / `unlabelled`). Clicking a clip selects it; if the current
 * bundle is dirty, we prompt before switching.
 *
 * The dialog Save button is disabled while the bundle is locked (you can't
 * save without unlocking first); user gets only Cancel / Discard in that
 * state, matching the Step 6 contract.
 */
export function Sidebar() {
  const clips = useStore(s => s.clips)
  const selectedClipId = useStore(s => s.selectedClipId)
  const selectClip = useStore(s => s.selectClip)
  const dirty = useStore(s => s.dirty)
  const locked = useStore(s => s.locked)
  const serverReadOnly = useStore(s => s.serverReadOnly)
  const saveError = useStore(s => s.saveError)
  const currentClipId = useStore(s => s.selectedClipId)

  const [filter, setFilter] = useState('')
  const [pendingClipId, setPendingClipId] = useState<string | null>(null)
  const [savingInDialog, setSavingInDialog] = useState(false)

  const saveDisabled = locked || serverReadOnly

  const q = filter.trim().toLowerCase()
  const filtered = q
    ? clips.filter(c => c.clip_id.toLowerCase().includes(q))
    : clips

  const requestSelect = (clipId: string) => {
    if (clipId === selectedClipId) return
    if (dirty) {
      setPendingClipId(clipId)
      return
    }
    selectClip(clipId)
  }

  const confirmDiscardAndSwitch = () => {
    if (pendingClipId) selectClip(pendingClipId)
    setPendingClipId(null)
  }

  const confirmSaveAndSwitch = async () => {
    if (!pendingClipId) return
    setSavingInDialog(true)
    const ok = await saveCurrentBundle()
    setSavingInDialog(false)
    if (!ok) return  // saveError already populated; keep the dialog open.
    selectClip(pendingClipId)
    setPendingClipId(null)
  }

  return (
    <aside className="flex flex-col h-full bg-surface-raised border-r border-border-default overflow-x-hidden overflow-y-hidden">
      <header className="shrink-0 p-3 border-b border-border-default">
        <h1 className="text-text-primary font-semibold text-sm">causal-av-annotator</h1>
        <p className="text-[10px] text-text-muted mt-0.5">{clips.length} clip{clips.length === 1 ? '' : 's'}</p>
      </header>

      <div className="px-3 py-2 border-b border-border-default">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-text-muted pointer-events-none" />
          <input
            type="text"
            placeholder="Search clips..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="w-full py-1.5 pl-8 pr-2 text-xs bg-surface-overlay border border-border-default rounded text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
      </div>

      <ul className="flex-1 overflow-y-auto overflow-x-hidden p-2 space-y-0.5">
        {filtered.map((c) => {
          const selected = c.clip_id === selectedClipId
          const Icon = c.kind === 'annotated' ? FileText : Film
          const badge = c.kind === 'annotated'
            ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30'
            : 'text-amber-400 bg-amber-500/10 border-amber-500/30'
          return (
            <li key={c.clip_id}>
              <button
                onClick={() => requestSelect(c.clip_id)}
                className={`w-full text-left p-2 rounded transition-colors flex items-center gap-2.5 ${
                  selected
                    ? 'bg-surface-hover border-l-[3px] border-l-blue-500'
                    : 'bg-transparent border-l-[3px] border-l-transparent hover:bg-surface-hover'
                }`}
              >
                <Icon className="w-4 h-4 text-text-muted flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <code className="block text-[11px] font-mono text-text-primary truncate">
                    {c.clip_id.length > 16 ? c.clip_id.slice(0, 8) + '…' : c.clip_id}
                  </code>
                </div>
                <span className={`text-[9px] px-1.5 py-0.5 rounded border ${badge}`}>{c.kind}</span>
              </button>
            </li>
          )
        })}
        {filtered.length === 0 && (
          <li className="text-[11px] text-text-muted italic px-2 py-3">No clips match.</li>
        )}
      </ul>

      <Dialog.Root open={pendingClipId !== null} onOpenChange={(open) => { if (!open) setPendingClipId(null) }}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/50 z-40" />
          <Dialog.Content className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-[400px] max-w-[90vw] rounded-2xl border border-border-default bg-surface-raised p-5 shadow-2xl">
            <Dialog.Title className="text-sm font-semibold text-text-primary">Unsaved changes</Dialog.Title>
            <Dialog.Description className="mt-2 text-xs text-text-muted">
              Save changes to <code className="text-text-primary">{currentClipId ?? '...'}</code> before switching?
            </Dialog.Description>
            {saveError ? (
              <p className="mt-3 text-[11px] text-red-300/90 italic">
                Save failed — see banner above.
              </p>
            ) : null}
            <div className="mt-5 flex justify-end gap-2">
              <button
                onClick={() => setPendingClipId(null)}
                className="px-3 py-1.5 rounded-lg text-xs bg-surface-overlay text-text-muted border border-border-default hover:text-text-primary"
              >
                Cancel
              </button>
              <button
                onClick={confirmDiscardAndSwitch}
                className="px-3 py-1.5 rounded-lg text-xs bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20"
              >
                Discard
              </button>
              <button
                onClick={confirmSaveAndSwitch}
                disabled={saveDisabled || savingInDialog}
                title={saveDisabled ? 'Unlock the bundle first to enable Save' : 'Save then switch'}
                className="px-3 py-1.5 rounded-lg text-xs bg-accent/20 text-blue-300 border border-blue-500/40 hover:bg-accent/30 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {savingInDialog ? 'Saving…' : 'Save'}
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </aside>
  )
}
