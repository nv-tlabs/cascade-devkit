# AGENTS.md — repo guide for coding agents

> A README for coding agents. **All AI coding agents working in this
> repository must read this file before making any changes.** The
> general spec is at <https://agents.md>. Tools that read this natively
> include OpenAI Codex CLI, GitHub Copilot, Cursor, Gemini CLI, Jules,
> Factory, Devin, and Windsurf. Claude Code reads `CLAUDE.md`, which
> imports this file via `@AGENTS.md` — both stay in sync.

## What this repo is

`cascade_av` is the **DevKit for the CASCADE dataset** (*Causal
Spatio-Temporal Analysis of Driving Environments*) — causal and
spatio-temporal action annotations on top of NVIDIA's *Physical AI AV
Dataset*. It parses the annotation JSON into a typed Pydantic tree and
exposes a small query language for searching the corpus by entity,
attribute, time, and cause. User-facing docs live in
[`README.md`](README.md). The annotation tool — a local FastAPI + React
app for editing the JSON bundles — ships under
[`tools/annotator/`](tools/annotator/).

## Start here

```bash
# Fresh Ubuntu / Debian host? One command installs system prereqs
# (apt + Node LTS + uv), runs `make install`, and smoke-tests the
# Python side. Idempotent. See README.md → Quick install for what it
# does step-by-step.
./scripts/install.sh

# Already have the prereqs (uv, Node, ffmpeg)? Skip the bootstrap
# and install directly:
make install         # uv sync --all-extras + annotator npm deps

# Smoke-test: run the pytest suite. Unit tests do not need the corpus.
make test
```

### Running in a devcontainer

A [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json)
ships with the repo. Open the folder in any devcontainer-compatible
host (VS Code Dev Containers, GitHub Codespaces, Copilot Workspace,
Devin, Jules, Cursor, etc.) and rebuild in container — Python 3.11,
`uv`, Node LTS, and `ffmpeg` are installed automatically and
`make install` runs as part of `postCreateCommand`. Port `8765`
(annotator default) is auto-forwarded. No GPU passthrough.

### Reading order

If you are new to the repo, read in this order:

1. This file.
2. [`docs/dev/contrib.md`](docs/dev/contrib.md) — branch-first
   workflow, worktree layout, atomic commits, PR conventions, `gh`
   CLI usage. **Mandatory before any non-trivial change.**
3. [`docs/dev/architecture.md`](docs/dev/architecture.md) — design
   decisions; **read before any structural change**.
4. [`README.md`](README.md) — user-facing overview of the dataset and
   query DSL.

## Repo layout

| Path | What lives here |
|------|-----------------|
| `src/cascade_av/` | Python DevKit: schema, I/O, dataset, query DSL |
| `src/cascade_av/spec/` | Pydantic models for the annotation JSON (schema `2.0.0`) |
| `src/cascade_av/query/` | DSL lexer + parser + evaluator |
| `src/cascade_av/viz/` | Visualization surface: `render_frame`, `render_timeline`, `ClipPlayer`, carousel. Lives behind the optional `[viz]` extra. |
| `tools/` | User-facing utilities layered on the DevKit |
| `tools/annotator/` | Local FastAPI + React annotation tool (uv workspace member) |
| `tools/annotator/web/` | Vite + React 19 + Tailwind v4 frontend |
| `tests/` | pytest suite |
| `examples/` | Runnable Python scripts; see README "Examples and notebooks" |
| `notebooks/` | Jupyter notebooks (built from `scripts/build_notebooks.py`) |
| `scripts/` | One-off Python utilities (notebook builder, vocabulary scan) |
| `docs/user/` | User-facing reference docs (`query_language.md`, `visualization.md`, `annotator.md`) |
| `docs/dev/` | Living dev docs (`contrib.md`, `architecture.md`, `changelog.md`) |
| `meta/` | **Gitignored.** Working research and planning notes. |

## Common workflows

All targets run from the repo root. The Makefile is the source of
truth; this table mirrors `make help`. If you add a new verb, update
**both** the Makefile and this table — and the relevant skill under
`.claude/skills/` — in the same PR.

| Verb | What it does |
|------|--------------|
| `make install` | `uv sync --all-extras` + `npm install` in `tools/annotator/web` |
| `make test` | Run Python (pytest) and annotator frontend (vitest) test suites |
| `make lint` | `ruff check` on Python + `npm run lint` on the annotator frontend |
| `make fmt` | `ruff format` on Python |
| `make annotator-dev [DATA=<path>] [PORT=<n>]` | Build the frontend bundle, then launch the annotator backend pointed at a directory of clips or videos. `DATA` defaults to `CASCADE_AV_DATASET_ROOT` from `.env`; `PORT` defaults to `8765`. Always rebuilds the bundle (~2-3s) — stale `dist/` no longer silently serves an old UI. |
| `make annotator-build` | Build the annotator frontend bundle (`tools/annotator/web/dist`) without launching the server. Same recipe `annotator-dev` runs as its first step; reach for it when you want to verify the bundle compiles without booting the backend. |
| `make notebooks [DATA=<path>]` | Launch JupyterLab against `./notebooks/`. `CASCADE_AV_DATASET_ROOT` is sourced from `.env` so notebooks that iterate the corpus "just work"; pass `DATA=<path>` to override. |
| `make migrate INPUT=<path> [OUTPUT=<path>]` | Run `cascade-migrate` on a file or directory; in-place when `OUTPUT` is omitted |
| `make help` | Print every target with its description |

## Running Python

Prefer `uv run <cmd>` over bare `python` so the workspace environment
(all extras + dev/notebooks groups, the `tools/annotator/` workspace
member, and the project's pinned interpreter) is used. The Makefile
targets already do this; reach for `uv run` directly for ad-hoc scripts,
notebook builds, or one-off REPLs.

## Testing

`make test` runs `uv run pytest` and then the annotator frontend's
vitest suite (`cd tools/annotator/web && npm test`). The annotator's
video tests mock `subprocess.run` / `shutil.which`, so real `ffmpeg` is
never invoked. Tests that need the corpus expect
`CASCADE_AV_DATASET_ROOT` to point at a directory of annotation JSONs;
unit tests do not require it. Frontend tests live next to the modules
they exercise as `*.test.ts` and run under Node (no DOM); add an
`environment: "jsdom"` line in `vitest.config.ts` if a future test
needs the browser DOM.

## Skills

Procedural workflows that agents repeat live as Agent Skills under
`.claude/skills/`. Each is a folder with a `SKILL.md` whose frontmatter
has `name` + `description`; the body loads on demand. See the open
standard at <https://agentskills.io>.

Current project skills:

| Skill | Purpose |
|-------|---------|
| [`run-tests`](.claude/skills/run-tests/SKILL.md) | Run the pytest suite via `make test`; diagnose common failure modes. |
| [`annotator-dev`](.claude/skills/annotator-dev/SKILL.md) | Launch the annotator backend; diagnose blank-page / port / video issues. |
| [`add-annotator-cli-flag`](.claude/skills/add-annotator-cli-flag/SKILL.md) | Add a new flag to the `cascade-annotate` CLI; conventions + test pattern. |

More skills land as repeated patterns emerge. Personal/experimental
skills live in `~/.claude/skills/`; only project-wide procedures
belong here.

## Agent expectations

The full workflow is in [`docs/dev/contrib.md`](docs/dev/contrib.md);
the short version for agents:

1. **Use a worktree.** Branch off `origin/main` with
   `git worktree add -b <type>/<desc>
   ../av-causal-dataset-tools-<type>-<desc> origin/main`.
   Never commit to `main`. Prefixes: `feat/`, `fix/`, `refactor/`,
   `docs/`, `chore/`.
2. **Atomic commits.** One logical change per commit, message format
   `<type>(<scope>): <summary>` followed by a body explaining what +
   why + any trade-offs considered.
3. **Small PRs.** Reviewable in <30 min. Open with `gh pr create`.
   Update `docs/dev/changelog.md` **in the same PR** with one line in
   the form `- YYYY-MM-DD <type>(<scope>): <summary> (#<PR>)`.
4. **Surface issues with `gh issue create`** — don't silently work
   around defects you discover. Search `gh issue list --search "..."`
   before filing duplicates.
5. **No force-push to `main`. No secrets. Don't auto-merge your own
   PRs.**
6. **Architecture first.** Read `docs/dev/architecture.md` before
   structural changes.

### Do not touch

| Path | Why |
|------|-----|
| `meta/` | Gitignored working notes — local-only by convention. |
| `.venv/`, `node_modules/`, `tools/annotator/web/dist/` | Generated; rebuild with `make install` / `make annotator-build`. |
| `*.bak` | Annotator first-save backups of user data. |
| `pyproject.toml` `[build-system]` | uv build-system pin — discuss before changing. |

## Security & data handling

### Environment variables

The DevKit reads these from the environment. Copy
[`.env.example`](.env.example) to `.env` and fill in real values;
`.env` is gitignored.

| Variable | Purpose | Required? |
|----------|---------|-----------|
| `CASCADE_AV_DATASET_ROOT` | Path to the annotation-JSON directory | Yes for examples / notebooks / tests that iterate the real corpus |
| `CASCADE_AV_VERBOSE` | Set to `1` for per-file detail in dataset-scan warnings | No (default `0`) |
| `HF_TOKEN` | HuggingFace Hub token; required for `cascade-annotate --video-source=hf` (and `auto`'s HF fallback) against gated repos like `nvidia/PhysicalAI-Autonomous-Vehicles`. `huggingface-cli login` is an equivalent alternative. | Yes for HF-backed video |

Setting options: shell `export` (or per-command
`VAR=... make test`); a `.env` file at the repo root (auto-loaded by
the annotator itself — `cascade-annotate` walks up from cwd at
startup, `override=False` so shell exports win — plus IDE test
runners, Docker Compose, and `dotenv-cli`, but **not** plain shell or
the other `uv run` / `make` targets); or on the Horde DGXC VM via
`hgx secrets`, which auto-inject across sessions.

### Never commit

- `.env` (real values; gitignored).
- `*.bak` — annotator first-save backups of user data.
- `~/.cache/cascade-annotator/` — transcode cache.
- Anything under `meta/` — gitignored working/research notes.
- Real corpus paths in source. Use `/path/to/json_annotations` in
  examples and docs.

### Never log

The dataset root, real clip IDs, sample annotation content, or any
auth token (HuggingFace, parent dataset) in commit messages, PR
descriptions, issue bodies, or externally uploaded stack traces.

### Data invariants

- **The annotator writes user data** via atomic `os.replace`; the
  first save for a clip in a server session stamps `<file>.bak`.
  Both invariants are tested; do not change either without explicit
  approval.

## When in doubt

- **Stuck on workflow?** Re-read `docs/dev/contrib.md`.
- **About to make a structural change?** Read
  `docs/dev/architecture.md` first.
- **Found a defect?** File a `gh issue` before patching around it.
