// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    rules: {
      // Allow leading-underscore identifiers to signal "intentionally
      // unused" — the standard convention for destructure / function-arg
      // placeholders. Without this, every `const { foo: _foo } = …` or
      // `(_clipId, _data) => …` trips the unused-vars rule, masking real
      // unused-import noise behind a wall of false positives.
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
          destructuredArrayIgnorePattern: '^_',
        },
      ],
      // React Compiler signal: "I could auto-memoize this, but your
      // manual useCallback/useMemo deps prevent me." That's information,
      // not a correctness defect — the manual version still works.
      // Some sites in Timeline.tsx omit deps deliberately (per-frame
      // values that would cascade re-renders); auto-memo migration is
      // tracked in #60. Demote from error to warn so `make lint` is
      // gated on real defects, not perf hints.
      'react-hooks/preserve-manual-memoization': 'warn',
    },
  },
])
