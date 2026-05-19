# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# av-causal-dataset-tools — make targets used by humans and agents.
#
# AGENTS.md cites these verbs as the canonical workflow surface. If you
# rename a target, update AGENTS.md and any matching `.claude/skills/`
# in the same PR — the Makefile is the indirection layer that lets the
# implementation change without churning every doc that quotes a
# command.

.PHONY: help install test lint fmt annotator-dev annotator-build migrate notebooks

# Sentinel file emitted by `npm run build` (vite). Used as a prerequisite
# of `annotator-dev` so the bundle is built on first launch; the explicit
# `annotator-build` verb remains the always-rebuild path.
WEB_DIST_INDEX := tools/annotator/web/dist/index.html

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

annotator-dev: $(WEB_DIST_INDEX)  ## Launch the annotator backend. Optional: DATA=<path> (falls back to CASCADE_AV_DATASET_ROOT from .env), PORT=<n>. Builds the frontend bundle automatically when missing.
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

# File-target that runs the build only when the bundle is missing — this
# is what gives `annotator-dev` its "first launch builds, subsequent
# launches skip" behavior. The explicit `annotator-build` verb above is
# the always-rebuild path users reach for after editing UI source.
$(WEB_DIST_INDEX):
	@echo "==> Frontend bundle missing — running 'npm run build' first..."
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

migrate:  ## Migrate annotations to the current schema. Pass INPUT=<path>. Optional OUTPUT=<path>.
	@if [ -z "$(INPUT)" ]; then \
	  echo "error: set INPUT=<path>. example: make migrate INPUT=/home/horde/02json OUTPUT=/tmp/migrated"; \
	  exit 2; \
	fi
	uv run cascade-migrate $(INPUT) $(if $(OUTPUT),--output-dir $(OUTPUT)) -v
