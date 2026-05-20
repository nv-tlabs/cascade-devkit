# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# av-causal-dataset-tools — make targets used by humans and agents.
#
# AGENTS.md cites these verbs as the canonical workflow surface. If you
# rename a target, update AGENTS.md and any matching `.claude/skills/`
# in the same PR — the Makefile is the indirection layer that lets the
# implementation change without churning every doc that quotes a
# command.

.PHONY: help install test lint fmt annotator-dev annotator-build notebooks

help:  ## Show this help and exit.
	@echo "Usage: make <target>"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-22s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install:  ## Install Python deps (all extras + groups) and the annotator's npm deps.
	uv sync --all-extras
	cd tools/annotator/web && npm install

test:  ## Run Python (pytest) and annotator frontend (vitest) test suites.
	uv run pytest
	cd tools/annotator/web && npm test

lint:  ## Lint Python (ruff) and the annotator frontend (eslint).
	uv run ruff check .
	cd tools/annotator/web && npm run lint

fmt:  ## Format Python with ruff.
	uv run ruff format .

annotator-dev: annotator-build  ## Build the annotator frontend bundle, then launch the backend. Optional: DATA=<path> (falls back to CASCADE_AV_DATASET_ROOT from .env), PORT=<n>. Always rebuilds — stale bundles cause confusing "my fix didn't land" debugging sessions; the 2-3s rebuild cost is worth it.
	@DATA="$(DATA)"; \
	if [ -z "$$DATA" ] && [ -f .env ]; then \
	  set -a; . ./.env; set +a; \
	  DATA="$$CASCADE_AV_DATASET_ROOT"; \
	fi; \
	if [ -z "$$DATA" ]; then \
	  echo "error: no DATA path. Pass DATA=<path>, or set CASCADE_AV_DATASET_ROOT in .env (see .env.example)" >&2; \
	  exit 2; \
	fi; \
	uv run cascade-annotate "$$DATA" $(if $(PORT),--port $(PORT))

annotator-build:  ## Build the annotator frontend bundle (tools/annotator/web/dist). Always rebuilds.
	cd tools/annotator/web && npm run build

notebooks:  ## Launch JupyterLab against ./notebooks. Optional: DATA=<path> (falls back to CASCADE_AV_DATASET_ROOT from .env).
	@DATA="$(DATA)"; \
	if [ -z "$$DATA" ] && [ -f .env ]; then \
	  set -a; . ./.env; set +a; \
	  DATA="$$CASCADE_AV_DATASET_ROOT"; \
	fi; \
	if [ -z "$$DATA" ]; then \
	  echo "warning: no dataset root (DATA= unset, CASCADE_AV_DATASET_ROOT unset in .env) — notebooks that iterate the corpus will fail. Continuing anyway." >&2; \
	else \
	  export CASCADE_AV_DATASET_ROOT="$$DATA"; \
	fi; \
	uv run --all-extras --group notebooks jupyter lab notebooks/
