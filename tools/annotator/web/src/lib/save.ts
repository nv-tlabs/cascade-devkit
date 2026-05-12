// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { useStore } from './store'
import { saveBundle } from './api'

/**
 * The save action used by every entry point (Save button, Cmd/Ctrl+S
 * keyboard shortcut, clip-switch dialog). Single source of truth so the
 * dirty / saveError / lock-policy gating is identical across surfaces.
 *
 * Returns true on success, false on failure (in which case `store.saveError`
 * has been populated).
 */
export async function saveCurrentBundle(): Promise<boolean> {
  const s = useStore.getState()
  if (!s.selectedClipId || !s.bundle) return false
  if (!s.dirty) return true
  if (s.locked || s.serverReadOnly) return false
  try {
    const saved = await saveBundle(s.selectedClipId, s.bundle)
    useStore.setState({
      bundle: saved,
      dirty: false,
      saveError: null,
      lastSavedAnnotationJson: saved.annotation ? JSON.stringify(saved.annotation) : null,
    })
    return true
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e)
    useStore.getState().setSaveError(msg)
    return false
  }
}
