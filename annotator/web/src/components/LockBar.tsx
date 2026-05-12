import { useState } from 'react'
import { useStore } from '../lib/store'
import { Lock, LockOpen, AlertCircle, Monitor, Sun, Moon } from 'lucide-react'

/**
 * Top bar across the editor pane. Drives the lock policy:
 *   - locked + server writable     → "Unlock to edit" + inline confirm
 *   - locked + server read-only    → permanently locked, no unlock button
 *   - unlocked                     → "Lock" button + edits enabled
 *   - dirty                        → red dot + "Unsaved changes"
 *   - save error                   → banner with message
 *
 * The save button itself lives in RightPanel; this bar only owns the lock
 * toggle + dirty indicator.
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
      <div className="flex-shrink-0 px-3 py-1.5 flex items-center gap-3 border-b border-amber-700/30 bg-amber-900/30 text-amber-200 text-[11px]">
        <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
        <span className="flex-1">
          Unlocking allows you to modify or delete annotation entries. Continue?
        </span>
        <button
          onClick={() => setConfirming(false)}
          className="px-2 py-0.5 rounded text-[10px] bg-[#1a1a35] text-[#aaa] border border-[#2a2a50] hover:text-white transition-colors"
        >
          Cancel
        </button>
        <button
          onClick={() => { unlock(); setConfirming(false) }}
          className="px-2 py-0.5 rounded text-[10px] bg-amber-700/40 text-amber-200 border border-amber-600/50 hover:bg-amber-700/60 transition-colors font-semibold"
        >
          Unlock anyway
        </button>
      </div>
    )
  }

  // --- Server read-only ---
  if (serverReadOnly) {
    return (
      <div className="flex-shrink-0 h-9 px-3 flex items-center gap-2 border-b border-[#1e1e38] bg-[#13132a] text-[#8888aa]">
        <Lock className="w-3.5 h-3.5" />
        <span className="text-[11px] font-medium">Locked — server is read-only</span>
        {filePath && (
          <code className="text-[10px] font-mono text-[#666] truncate ml-1">· {filePath}</code>
        )}
        {dirty && <DirtyIndicator />}
        <div className="flex-1" />
        {saveError && <SaveError message={saveError} onDismiss={() => setSaveError(null)} />}
        <ThemeToggle />
      </div>
    )
  }

  // --- Locked (default) ---
  if (locked) {
    return (
      <div className="flex-shrink-0 h-9 px-3 flex items-center gap-2 border-b border-[#1e1e38] bg-[#13132a] text-[#8888aa]">
        <Lock className="w-3.5 h-3.5" />
        <span className="text-[11px] font-medium">Locked — read-only</span>
        {filePath && (
          <code className="text-[10px] font-mono text-[#666] truncate ml-1">· {filePath}</code>
        )}
        {dirty && <DirtyIndicator />}
        <div className="flex-1" />
        {saveError && <SaveError message={saveError} onDismiss={() => setSaveError(null)} />}
        <button
          onClick={() => setConfirming(true)}
          disabled={!selectedClipId}
          className="px-2.5 py-0.5 rounded text-[10px] bg-blue-600/20 text-blue-300 border border-blue-600/40 hover:bg-blue-600/30 transition-colors font-semibold disabled:opacity-40"
        >
          Unlock to edit
        </button>
        <ThemeToggle />
      </div>
    )
  }

  // --- Unlocked ---
  return (
    <div className="flex-shrink-0 h-9 px-3 flex items-center gap-2 border-b border-emerald-700/30 bg-emerald-900/15 text-emerald-300">
      <LockOpen className="w-3.5 h-3.5" />
      <span className="text-[11px] font-medium">Editing enabled</span>
      {filePath && (
        <code className="text-[10px] font-mono text-emerald-400/60 truncate ml-1">· {filePath}</code>
      )}
      {dirty && <DirtyIndicator />}
      <div className="flex-1" />
      {saveError && <SaveError message={saveError} onDismiss={() => setSaveError(null)} />}
      <button
        onClick={() => lock()}
        className="px-2.5 py-0.5 rounded text-[10px] bg-[#1a1a35] text-[#aaa] border border-[#2a2a50] hover:bg-[#252550] hover:text-white transition-colors font-semibold"
      >
        Lock
      </button>
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
      <Icon className="w-4 h-4" />
    </button>
  )
}

function DirtyIndicator() {
  return (
    <span className="inline-flex items-center gap-1 text-[10px] text-orange-300 ml-1">
      <span className="w-1.5 h-1.5 rounded-full bg-orange-400" />
      Unsaved changes
    </span>
  )
}

function SaveError({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[10px] text-red-300 bg-red-900/40 border border-red-700/40 px-2 py-0.5 rounded mr-2">
      <AlertCircle className="w-3 h-3" />
      <span className="max-w-[280px] truncate" title={message}>Save failed: {message}</span>
      <button onClick={onDismiss} className="hover:text-white">×</button>
    </span>
  )
}
