#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# scripts/install.sh — bootstrap a fresh Ubuntu/Debian host for the CASCADE
# DevKit. Idempotent: re-running is safe; steps that are already complete
# no-op.
#
# Run from the repo root after cloning:
#
#     ./scripts/install.sh
#
# Designed to mirror the manual recipe in README's Prerequisites
# section. A future curl-pipe one-liner can wrap this script unchanged.
#
# Steps:
#   1. Sanity: confirm we're in the repo root.
#   2. OS check: Ubuntu/Debian only for now.
#   3. apt: install make, build-essential, ffmpeg, curl.
#   4. Node ≥ 18 (LTS): install via NodeSource if missing.
#   5. uv: install via the official Astral installer if missing.
#   6. Project: run `make install` (uv sync --all-extras + annotator npm).
#   7. Smoke: import cascade_av in the project venv to confirm the
#      Python side works end-to-end.

set -euo pipefail

# ---------------------------------------------------------------------------
# Pretty output. Single-source the color codes so a tty-less run (e.g.
# piped to less / tee / CI logs) still reads cleanly.
#
# Use `$'\033…'` (bash ANSI-C quoting) rather than `"\033…"` (regular
# double-quoting) so the variables hold the actual ESC byte, not the four
# literal characters `\`, `0`, `3`, `3`. `printf` interprets the latter
# for us, but the closing `cat <<EOF` heredoc does not — without this,
# the "Next steps" block prints `\033[1m1.\033[0m` instead of a bold "1."
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
    YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi

step() { printf "${BOLD}${BLUE}==>${RESET} ${BOLD}%s${RESET}\n" "$*"; }
info() { printf "${DIM}    %s${RESET}\n" "$*"; }
warn() { printf "${YELLOW}    %s${RESET}\n" "$*"; }
ok()   { printf "${GREEN}    ✓ %s${RESET}\n" "$*"; }
die()  { printf "${RED}✗ %s${RESET}\n" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# sudo helper. Root users skip sudo; everyone else gets the standard prompt
# the first time apt is invoked. We intentionally do NOT cache sudo
# credentials silently — the user should see each privileged step.
# ---------------------------------------------------------------------------
if [ "${EUID:-$(id -u)}" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

# ---------------------------------------------------------------------------
# Repo coordinates. Kept up here so the curl-pipe path and the post-clone
# path point at the same upstream.
# ---------------------------------------------------------------------------
REPO_URL="${CASCADE_REPO_URL:-https://github.com/nv-tlabs/cascade-devkit.git}"
REPO_DIRNAME="${CASCADE_REPO_DIR:-cascade-devkit}"

is_repo_root() {
    [ -f "pyproject.toml" ] && [ -f "Makefile" ] && [ -d "src/cascade_av" ]
}

# ---------------------------------------------------------------------------
# 1. OS check. Anything other than Ubuntu/Debian falls out with a clear
# pointer to the manual recipe. Runs before any clone so we don't fetch a
# repo we can't actually install on.
# ---------------------------------------------------------------------------
step "OS check"
if [ ! -f /etc/os-release ]; then
    die "cannot read /etc/os-release; this script supports Ubuntu/Debian only. See the manual recipe in README.md → Prerequisites."
fi
. /etc/os-release
case "${ID:-}:${ID_LIKE:-}" in
    ubuntu:*|debian:*|*:*ubuntu*|*:*debian*) ok "${PRETTY_NAME:-$ID}";;
    *) die "this script supports Ubuntu/Debian; detected ID=${ID}, ID_LIKE=${ID_LIKE:-}. See the manual recipe in README.md → Prerequisites." ;;
esac

# ---------------------------------------------------------------------------
# 2. Locate (or clone) the DevKit. Three branches:
#   (a) cwd is already the repo root → just continue.
#   (b) ./$REPO_DIRNAME exists and looks like the repo → cd into it.
#   (c) otherwise → install git if missing, clone, cd in.
# Branch (c) is what makes the curl-pipe one-liner self-contained:
# `curl … | bash` runs in a fresh cwd with no checkout, so the script
# fetches the repo it's about to install.
# ---------------------------------------------------------------------------
step "Locating the CASCADE DevKit"
if is_repo_root; then
    ok "running from repo root ($(pwd))"
elif [ -d "$REPO_DIRNAME" ] && (cd "$REPO_DIRNAME" && is_repo_root); then
    cd "$REPO_DIRNAME"
    ok "using existing clone at $(pwd)"
else
    info "no DevKit checkout here; cloning $REPO_URL into ./$REPO_DIRNAME"
    if ! command -v git >/dev/null 2>&1; then
        info "git not on PATH; installing via apt"
        $SUDO apt-get update -qq
        $SUDO apt-get install -y --no-install-recommends git ca-certificates
    fi
    git clone --depth 1 "$REPO_URL" "$REPO_DIRNAME"
    cd "$REPO_DIRNAME"
    is_repo_root || die "cloned $REPO_URL but it doesn't look like the CASCADE DevKit; aborting"
    ok "cloned and entered $(pwd)"
fi

# ---------------------------------------------------------------------------
# 3. apt packages. `-y` for non-interactive; we suppress recommended-but-
# optional packages to keep the install lean. `apt-get install` is
# idempotent on already-present packages.
# ---------------------------------------------------------------------------
step "Installing system packages (make, build-essential, ffmpeg, curl)"
info "may prompt for sudo password"
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends \
    make build-essential ffmpeg curl ca-certificates
ok "apt packages installed"

# ---------------------------------------------------------------------------
# 4. Node ≥ 18. If `node --version` reports a major version ≥ 18, leave
# the existing install alone; otherwise pull the NodeSource setup script
# and install Node 20 LTS. Done this way so the script doesn't trample
# a working Node 18/22/etc. that the user already has.
# ---------------------------------------------------------------------------
step "Node (≥ 18 LTS)"
node_ok=0
if command -v node >/dev/null 2>&1; then
    node_major=$(node --version | sed 's/^v\([0-9]*\)\..*/\1/')
    if [ "$node_major" -ge 18 ] 2>/dev/null; then
        ok "node $(node --version) already on PATH"
        node_ok=1
    else
        warn "node $(node --version) found, but < 18; will install Node 20 LTS via NodeSource"
    fi
fi
if [ "$node_ok" -eq 0 ]; then
    info "fetching NodeSource setup script"
    curl -fsSL https://deb.nodesource.com/setup_20.x | $SUDO -E bash -
    $SUDO apt-get install -y nodejs
    ok "node $(node --version) installed"
fi

# ---------------------------------------------------------------------------
# 5. uv. The Astral installer detects existing installs and is itself
# idempotent — re-running just leaves the binary in place.
# ---------------------------------------------------------------------------
step "uv (Python env / dep manager)"
if command -v uv >/dev/null 2>&1; then
    ok "uv $(uv --version | awk '{print $2}') already on PATH"
else
    info "running the official Astral installer"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer drops uv into ~/.local/bin and prints a "source this"
    # hint; make uv discoverable for the rest of THIS script regardless.
    if [ -f "$HOME/.local/bin/env" ]; then
        # shellcheck disable=SC1091
        . "$HOME/.local/bin/env"
    elif [ -d "$HOME/.local/bin" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH; close this shell and re-run"
    ok "uv $(uv --version | awk '{print $2}') installed"
fi

# ---------------------------------------------------------------------------
# 6. Project: `make install` does `uv sync --all-extras` plus the
# annotator's `npm install`. Everything is local to the repo (./.venv,
# ./tools/annotator/web/node_modules) so no system-level state changes.
# ---------------------------------------------------------------------------
step "Installing project (uv sync --all-extras + annotator npm deps)"
make install
ok "project installed"

# ---------------------------------------------------------------------------
# 7. Smoke test: import cascade_av in the project venv to confirm the
# Python side actually loads. Skips the pytest suite (which needs the
# corpus or a CASCADE_AV_DATASET_ROOT) because the user may not have
# the data yet — the point here is "the install completed."
# ---------------------------------------------------------------------------
step "Smoke test"
uv run python -c "
import cascade_av
from cascade_av.dataset import CascadeDataset
from cascade_av.query import parse_timestamp
print(f'cascade_av {getattr(cascade_av, \"__version__\", \"(dev)\")}: import OK')
print(f'CascadeDataset, parse_timestamp resolve')
"
ok "Python side imports cleanly"

# ---------------------------------------------------------------------------
# Done — point the user at the next concrete things to try.
# ---------------------------------------------------------------------------
echo
step "Install complete"
# The repo path the user lands in is wherever the cwd ended up after the
# "locate or clone" step — either the existing checkout they invoked us
# from, or the freshly-cloned ./$REPO_DIRNAME. Show it explicitly so a
# curl-pipe user knows which directory to cd into.
INSTALL_DIR="$(pwd)"
cat <<EOF
${DIM}Installed to:${RESET} $INSTALL_DIR

${DIM}Next steps:${RESET}

  ${BOLD}1.${RESET} Enter the DevKit directory (if you're not already there):
       ${DIM}cd $INSTALL_DIR${RESET}

  ${BOLD}2.${RESET} Point the DevKit at your annotation JSONs:
       ${DIM}export CASCADE_AV_DATASET_ROOT=/path/to/json_annotations${RESET}

  ${BOLD}3.${RESET} Run the test suite (does not need the corpus for unit tests):
       ${DIM}make test${RESET}

  ${BOLD}4.${RESET} Launch the annotator against a directory of clips:
       ${DIM}make annotator-dev DATA=/path/to/clips${RESET}

  ${BOLD}5.${RESET} Read README.md → Getting the data for where annotation
     bundles come from, or open one of the notebooks under notebooks/.
EOF
