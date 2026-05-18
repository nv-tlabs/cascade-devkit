// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { useState } from 'react'
import * as Dialog from '@radix-ui/react-dialog'
import { useStore } from '../lib/store'
import { saveCurrentBundle } from '../lib/save'
import { Search, Film, FileText, CheckCircle2, Circle } from 'lucide-react'

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
      <header className="shrink-0 px-5 py-4 border-b border-border-default">
        <h1 className="text-text-primary font-semibold text-sm">cascade-annotator</h1>
        <p className="text-xs font-medium text-text-muted mt-0.5">{clips.length} clip{clips.length === 1 ? '' : 's'}</p>
      </header>

      <div className="px-5 py-3 border-b border-border-default">
        <div className="relative">
          <Search
            className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none"
            width={16}
            height={16}
          />
          <input
            type="text"
            placeholder="Search clips..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="w-full h-10 pl-10 pr-3 text-sm bg-surface-overlay border border-border-default rounded-md text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-accent focus:border-transparent"
          />
        </div>
      </div>

      <ul className="flex-1 overflow-y-auto overflow-x-hidden p-3 space-y-1.5">
        {filtered.map((c) => {
          const selected = c.clip_id === selectedClipId
          const Icon = c.kind === 'annotated' ? FileText : Film
          // Status badge precedence: `kind === 'unlabelled'` always wins
          // over any stale status string. For annotated clips, the
          // bundle's `status` decides between "complete" and "in progress".
          // Legacy bundles without `status` fall through to "in progress".
          let badgeKind: 'unlabelled' | 'in-progress' | 'complete' = 'in-progress'
          if (c.kind === 'unlabelled') badgeKind = 'unlabelled'
          else if (c.status === 'complete') badgeKind = 'complete'
          const badgeCls =
            badgeKind === 'complete'
              ? 'text-success bg-success-bg border-success/30'
              : badgeKind === 'in-progress'
                ? 'text-warning bg-warning-bg border-warning/30'
                : 'text-text-muted bg-surface-overlay border-border-default'
          const badgeLabel =
            badgeKind === 'complete'
              ? 'complete'
              : badgeKind === 'in-progress'
                ? 'in progress'
                : 'unlabelled'
          const BadgeIcon =
            badgeKind === 'complete' ? CheckCircle2 : Circle
          const displayId = c.clip_id.length > 20 ? c.clip_id.slice(0, 12) + '…' : c.clip_id
          return (
            <li key={c.clip_id}>
              <button
                onClick={() => requestSelect(c.clip_id)}
                className={`w-full text-left px-3 py-2.5 rounded-md transition-colors flex items-center gap-3 ${
                  selected
                    ? 'bg-surface-selected border-l-2 border-l-accent'
                    : 'bg-transparent border-l-2 border-l-transparent hover:bg-surface-hover'
                }`}
              >
                <Icon className="w-4 h-4 text-text-muted flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <code className="block text-xs font-mono text-text-primary truncate pr-1">
                    {displayId}
                  </code>
                  <span className={`mt-1 inline-flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wide px-3 py-1 rounded-full border ${badgeCls}`}>
                    <BadgeIcon className="w-2.5 h-2.5" />
                    {badgeLabel}
                  </span>
                </div>
              </button>
            </li>
          )
        })}
        {filtered.length === 0 && (
          <li className="text-xs text-text-muted italic px-2 py-3">No clips match.</li>
        )}
      </ul>

      <Dialog.Root open={pendingClipId !== null} onOpenChange={(open) => { if (!open) setPendingClipId(null) }}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/50 z-40" />
          <Dialog.Content className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-[400px] max-w-[90vw] rounded-lg border border-border-default bg-surface-raised p-5 shadow-2xl">
            <Dialog.Title className="text-sm font-semibold text-text-primary">Unsaved changes</Dialog.Title>
            <Dialog.Description className="mt-2 text-xs text-text-muted">
              Save changes to <code className="font-mono text-[12px] text-text-primary">{currentClipId ?? '...'}</code> before switching?
            </Dialog.Description>
            {saveError ? (
              <p className="mt-3 text-[11px] text-danger italic">
                Save failed — see banner above.
              </p>
            ) : null}
            <div className="mt-5 flex justify-end gap-2">
              <button
                onClick={() => setPendingClipId(null)}
                className="h-9 px-4 inline-flex items-center rounded-md text-sm font-medium text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmDiscardAndSwitch}
                className="h-9 px-4 inline-flex items-center rounded-md text-sm font-medium bg-danger-bg text-danger border border-danger/30 hover:bg-danger/15 transition-colors"
              >
                Discard
              </button>
              <button
                onClick={confirmSaveAndSwitch}
                disabled={saveDisabled || savingInDialog}
                title={saveDisabled ? 'Unlock the bundle first to enable Save' : 'Save then switch'}
                className="h-9 px-4 inline-flex items-center rounded-md text-sm font-semibold bg-accent text-accent-fg hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
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
