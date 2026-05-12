# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# av-causal-dataset-tools — make targets used by humans and agents.
#
# AGENTS.md cites these verbs as the canonical workflow surface. If you
# rename a target, update AGENTS.md and any matching `.claude/skills/`
# in the same PR — the Makefile is the indirection layer that lets the
# implementation change without churning every doc that quotes a
# command.

.PHONY: help install test lint fmt annotator-dev annotator-build

help:  ## Show this help and exit.
	@echo "Usage: make <target>"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-22s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install:  ## Install Python deps (all extras + groups) and the annotator's npm deps.
	uv sync --all-extras
	cd tools/annotator/web && npm install

test:  ## Run the full pytest suite.
	uv run pytest

lint:  ## Lint Python (ruff) and the annotator frontend (eslint).
	uv run ruff check .
	cd tools/annotator/web && npm run lint

fmt:  ## Format Python with ruff.
	uv run ruff format .

annotator-dev:  ## Launch the annotator backend. Pass DATA=<path>. Optional: PORT=<n>.
	@if [ -z "$(DATA)" ]; then \
	  echo "error: set DATA=<path>. example: make annotator-dev DATA=~/01_json_annotations"; \
	  exit 2; \
	fi
	uv run causal-av-annotate $(DATA) $(if $(PORT),--port $(PORT))

annotator-build:  ## Build the annotator frontend bundle (tools/annotator/web/dist).
	cd tools/annotator/web && npm run build
