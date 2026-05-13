// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'

// --- Synchronous theme hydration -------------------------------------------------
// Read the stored theme preference and apply the resolved theme to
// <html data-theme> BEFORE React mounts, so the first paint matches the
// user's preference (no flash-of-unstyled-content).
const THEME_STORAGE_KEY = 'cascade-annotator.theme'
type StoredTheme = 'system' | 'light' | 'dark'

function readStoredTheme(): StoredTheme {
  try {
    const v = localStorage.getItem(THEME_STORAGE_KEY)
    if (v === 'light' || v === 'dark' || v === 'system') return v
  } catch {
    // Storage unavailable — fall through to default.
  }
  return 'system'
}

function resolveEffective(theme: StoredTheme): 'light' | 'dark' {
  if (theme === 'system') {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  }
  return theme
}

const __initialStoredTheme = readStoredTheme()
const __initialEffectiveTheme = resolveEffective(__initialStoredTheme)
document.documentElement.dataset.theme = __initialEffectiveTheme

// Expose for the store to read during initialization. Cast-only — kept off
// the public Window typings since this is a one-shot handoff.
;(window as unknown as { __initialTheme?: StoredTheme; __initialEffectiveTheme?: 'light' | 'dark' }).__initialTheme = __initialStoredTheme
;(window as unknown as { __initialTheme?: StoredTheme; __initialEffectiveTheme?: 'light' | 'dark' }).__initialEffectiveTheme = __initialEffectiveTheme

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
