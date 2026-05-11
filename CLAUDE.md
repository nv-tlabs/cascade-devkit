# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This repo is at its initial commit and contains only `LICENSE` and `README.md` — no source code, build system, language choice, or test harness has been established yet. Treat the next substantive change as a bootstrap: the language, package manager, and directory layout are still open decisions, so confirm them with the user before scaffolding rather than assuming a stack.

## Purpose

Per `README.md`: tooling to access the AV Causal Dataset. "AV" most likely refers to autonomous vehicles given the "causal dataset" framing, but the dataset itself is not described in-repo — ask the user for the dataset's location, access protocol, and schema before writing access code.

## When this file goes stale

Once a stack is chosen and real code lands, replace these sections with: the actual build/test/lint commands, the architectural pieces that span multiple files, and any dataset-access conventions that aren't obvious from reading the code.
