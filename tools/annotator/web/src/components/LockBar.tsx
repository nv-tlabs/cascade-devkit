// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { useState } from 'react'
import { useStore } from '../lib/store'
import { Lock, LockOpen, AlertCircle, Monitor, Sun, Moon, X } from 'lucide-react'

/**
 * Top bar across the editor pane. Drives the lock policy:
 *   - locked + server writable     → "Unlock to edit" + inline confirm
 *   - locked + server read-only    → permanently locked, no unlock button
 *   - unlocked                     → "Lock" button + edits enabled
 *   - dirty                        → amber pill + "Unsaved changes"
 *   - save error                   → separate banner row below the lock bar
 *
 * The save button itself lives in RightPanel; this bar only owns the lock
 * toggle + dirty indicator + theme toggle.
 */
export function LockBar() {
  const locked = useStore(s => s.locked)
  const serverReadOnly = useStore(s => s.serverReadOnly)
  const dirty = useStore(s => s.dirty)
  const saveError = useStore(s => s.saveError)
  const selectedClipId = useStore(s => s.selectedClipId)
  const bundle = useStore(s => s.bundle)
  const lock = useStore(s => s.lock)
  const unlock = useStore(s => s.unlock)
  const setSaveError = useStore(s => s.setSaveError)

  const [confirming, setConfirming] = useState(false)

  const filePath = bundle?.video?.source ?? selectedClipId ?? null

  // --- Confirmation row (locked, user clicked Unlock to edit) ---
  if (confirming && locked && !serverReadOnly) {
    return (
      <div className="flex-shrink-0 px-4 py-2.5 flex items-center gap-3 border-b border-warning/30 bg-warning-bg text-warning text-xs">
        <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
        <span className="flex-1">
          Unlocking allows you to modify or delete annotation entries. Continue?
        </span>
        <button
          onClick={() => setConfirming(false)}
          className="h-8 px-3 inline-flex items-center rounded-md text-xs font-medium bg-surface-overlay text-text-secondary border border-border-default hover:text-text-primary transition-colors"
        >
          Cancel
        </button>
        <button
          onClick={() => { unlock(); setConfirming(false) }}
          className="h-8 px-3 inline-flex items-center rounded-md text-xs font-semibold bg-warning text-text-inverse border border-warning hover:opacity-90 transition-opacity"
        >
          Unlock anyway
        </button>
      </div>
    )
  }

  return (
    <>
      <LockBarRow
        locked={locked}
        serverReadOnly={serverReadOnly}
        dirty={dirty}
        filePath={filePath}
        canUnlock={!!selectedClipId}
        onLock={lock}
        onUnlockRequest={() => setConfirming(true)}
      />
      {saveError && (
        <SaveErrorBanner message={saveError} onDismiss={() => setSaveError(null)} />
      )}
    </>
  )
}

function LockBarRow({
  locked, serverReadOnly, dirty, filePath, canUnlock, onLock, onUnlockRequest,
}: {
  locked: boolean
  serverReadOnly: boolean
  dirty: boolean
  filePath: string | null
  canUnlock: boolean
  onLock: () => void
  onUnlockRequest: () => void
}) {
  const Icon = locked ? Lock : LockOpen
  const statusText = serverReadOnly
    ? 'Locked — server is read-only'
    : locked
      ? 'Locked — read-only'
      : 'Editing enabled'
  const statusCls = !locked ? 'text-success' : 'text-text-secondary'

  return (
    <div className="flex-shrink-0 h-11 px-4 flex items-center gap-3 border-b border-border-subtle bg-surface-raised">
      <Icon className={`w-3.5 h-3.5 ${statusCls}`} />
      <span className={`text-xs font-medium ${statusCls}`}>{statusText}</span>
      {filePath && (
        <>
          <span className="w-px h-3 bg-border-default" />
          <code className="text-[11px] font-mono text-text-muted truncate max-w-[280px] pr-1" title={filePath}>{filePath}</code>
        </>
      )}
      <div className="flex-1" />
      {dirty && <DirtyPill />}
      {!serverReadOnly && locked && (
        <button
          onClick={onUnlockRequest}
          disabled={!canUnlock}
          className="h-8 px-4 inline-flex items-center rounded-md text-xs font-semibold bg-accent-soft-bg text-accent-soft-fg border border-accent-soft-border hover:bg-accent hover:text-accent-fg transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Unlock to edit
        </button>
      )}
      {!serverReadOnly && !locked && (
        <button
          onClick={onLock}
          className="h-8 px-4 inline-flex items-center rounded-md text-xs font-semibold bg-surface-overlay text-text-secondary border border-border-default hover:bg-surface-hover hover:text-text-primary transition-colors"
        >
          Lock
        </button>
      )}
      <ThemeToggle />
    </div>
  )
}

function ThemeToggle() {
  const theme = useStore(s => s.theme)
  const setTheme = useStore(s => s.setTheme)
  const next = theme === 'system' ? 'light' : theme === 'light' ? 'dark' : 'system'
  const Icon = theme === 'system' ? Monitor : theme === 'light' ? Sun : Moon
  const label = `Theme: ${theme} (click to switch to ${next})`
  return (
    <button
      onClick={() => setTheme(next)}
      title={label}
      aria-label={label}
      className="h-8 w-8 inline-flex items-center justify-center rounded-md text-text-muted hover:bg-surface-hover hover:text-text-primary transition-colors"
    >
      <Icon className="w-3.5 h-3.5" />
    </button>
  )
}

function DirtyPill() {
  return (
    <span className="inline-flex items-center gap-1.5 h-6 px-3 rounded-full bg-warning-bg text-warning text-[11px] font-medium">
      <span className="w-1.5 h-1.5 rounded-full bg-warning" />
      Unsaved changes
    </span>
  )
}

function SaveErrorBanner({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div className="flex-shrink-0 h-9 px-4 flex items-center gap-2 bg-danger-bg text-danger border-b border-danger/40">
      <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
      <span className="text-xs font-medium">Save failed:</span>
      <span className="text-xs truncate pr-1" title={message}>{message}</span>
      <div className="flex-1" />
      <button
        onClick={onDismiss}
        aria-label="Dismiss save error"
        className="h-7 w-7 inline-flex items-center justify-center rounded-md hover:bg-danger/15 transition-colors"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  )
}
