import { useStore } from '../lib/store'
import { Lock } from 'lucide-react'

/**
 * Step 4 stub. Step 5 replaces this with the full lock-policy UI
 * (Unlock to edit / inline confirm / Lock button / dirty indicator /
 *  server-read-only state).
 *
 * For now we render a fixed-height bar so the layout matches the final shell.
 */
export function LockBar() {
  const selectedClipId = useStore(s => s.selectedClipId)
  return (
    <div className="flex-shrink-0 h-9 px-3 flex items-center gap-2 border-b border-[#1e1e38] bg-[#13132a] text-[#8888aa]">
      <Lock className="w-3.5 h-3.5" />
      <span className="text-[11px]">Locked — read-only</span>
      {selectedClipId && (
        <code className="text-[10px] font-mono text-[#666] truncate">· {selectedClipId}</code>
      )}
    </div>
  )
}
