# AGENTS.md — repo guide for coding agents

> A README for coding agents. **All AI coding agents working in this
> repository must read this file before making any changes.** The
> general spec is at <https://agents.md>. Tools that read this natively
> include OpenAI Codex CLI, GitHub Copilot, Cursor, Gemini CLI, Jules,
> Factory, Devin, and Windsurf. Claude Code reads `CLAUDE.md`, which
> imports this file via `@AGENTS.md` — both stay in sync.

## What this repo is

`causal_ai_av` is the **DevKit for the AV Causal Dataset** — causal and
spatio-temporal action annotations on top of NVIDIA's *Physical AI AV
Dataset*. It parses the annotation JSON into a typed Pydantic tree and
exposes a small query language for searching the corpus by entity,
attribute, time, and cause. User-facing docs live in
[`README.md`](README.md). The annotation tool — a local FastAPI + React
app for editing the JSON bundles — ships under
[`tools/annotator/`](tools/annotator/).

## Start here

```bash
# Install Python deps (all extras + dev/notebooks groups) and the
# annotator's npm deps.
make install

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
| `src/causal_ai_av/` | Python DevKit: schema, I/O, dataset, query DSL |
| `src/causal_ai_av/spec/` | Pydantic models for the annotation JSON (schema `2.0.0`) |
| `src/causal_ai_av/query/` | DSL lexer + parser + evaluator |
| `src/causal_ai_av/viz/` | Visualization surface: `render_frame`, `render_timeline`, `ClipPlayer`, carousel. Lives behind the optional `[viz]` extra. |
| `tools/` | User-facing utilities layered on the DevKit |
| `tools/annotator/` | Local FastAPI + React annotation tool (uv workspace member) |
| `tools/annotator/web/` | Vite + React 19 + Tailwind v4 frontend |
| `tests/` | pytest suite |
| `examples/` | Runnable Python scripts; see README "Examples and notebooks" |
| `notebooks/` | Jupyter notebooks (built from `scripts/build_notebooks.py`) |
| `scripts/` | One-off Python utilities (notebook builder, vocabulary scan) |
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
| `make test` | Run the full pytest suite |
| `make lint` | `ruff check` on Python + `npm run lint` on the annotator frontend |
| `make fmt` | `ruff format` on Python |
| `make annotator-dev DATA=<path>` | Launch the annotator backend pointed at a directory of clips or videos |
| `make annotator-build` | Build the annotator frontend bundle (`tools/annotator/web/dist`) |
| `make help` | Print every target with its description |

## Testing

`make test` runs `uv run pytest`. The annotator's video tests mock
`subprocess.run` / `shutil.which`, so real `ffmpeg` is never invoked.
Tests that need the corpus expect `CAUSAL_AV_DATASET_ROOT` to point at
a directory of annotation JSONs; unit tests do not require it.

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
| [`add-annotator-cli-flag`](.claude/skills/add-annotator-cli-flag/SKILL.md) | Add a new flag to the `causal-av-annotate` CLI; conventions + test pattern. |

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
| `CAUSAL_AV_DATASET_ROOT` | Path to the annotation-JSON directory | Yes for examples / notebooks / tests that iterate the real corpus |
| `CAUSAL_AV_VERBOSE` | Set to `1` for per-file detail in dataset-scan warnings | No (default `0`) |

Setting options: shell `export` (or per-command
`VAR=... make test`); a `.env` file at the repo root (auto-loaded by
IDE test runners, Docker Compose, and `dotenv-cli`, but **not** by
plain shell or `uv run`); or on the Horde DGXC VM via `hgx secrets`,
which auto-inject across sessions.

### Never commit

- `.env` (real values; gitignored).
- `*.bak` — annotator first-save backups of user data.
- `~/.cache/causal-av-annotator/` — transcode cache.
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
