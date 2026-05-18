// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { defineConfig } from 'vitest/config'

// Vitest config kept separate from `vite.config.ts` so the dev server's
// React + Tailwind plugins don't load during pure-JS unit tests (faster
// startup, no JSX in the test files anyway). When a future test needs
// the DOM or React rendering, add `environment: "jsdom"` here.
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
    reporters: 'default',
  },
})
