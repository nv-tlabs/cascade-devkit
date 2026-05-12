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
[`annotator/`](annotator/).

## Start here

```bash
# Install Python deps (all extras + dev/notebooks groups) and the
# annotator's npm deps.
make install

# Smoke-test: run the pytest suite. Unit tests do not need the corpus.
make test
```

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
| `annotator/` | Local FastAPI + React annotation tool (separate workspace member) |
| `annotator/web/` | Vite + React 19 + Tailwind v4 frontend |
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
| `make install` | `uv sync --all-extras` + `npm install` in `annotator/web` |
| `make test` | Run the full pytest suite |
| `make lint` | `ruff check` on Python + `npm run lint` on the annotator frontend |
| `make fmt` | `ruff format` on Python |
| `make annotator-dev DATA=<path>` | Launch the annotator backend pointed at a directory of clips or videos |
| `make annotator-build` | Build the annotator frontend bundle (`annotator/web/dist`) |
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

*Bootstrap pending — initial skills land in a follow-up PR.*

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
| `.venv/`, `node_modules/`, `annotator/web/dist/` | Generated; rebuild with `make install` / `make annotator-build`. |
| `*.bak` | Annotator first-save backups of user data. |
| `pyproject.toml` `[build-system]` | uv build-system pin — discuss before changing. |

## Security & data handling

- **No secrets in the repo.** No `.env`, no credentials, no API keys.
  See [`docs/dev/contrib.md`](docs/dev/contrib.md) §AI Agent Rules.
- **Dataset root** is configured via the `CAUSAL_AV_DATASET_ROOT`
  environment variable. Do not hard-code paths to the corpus.
- **Do not log the dataset root, clip IDs, or any sample annotation
  content** in commit messages, PR descriptions, or issue bodies.
  Sample paths in examples should remain placeholders
  (`/path/to/json_annotations`).
- **The annotator writes user data.** Saves go through atomic
  `os.replace`; the first save for a clip in a session stamps
  `<file>.bak`. Do not change either invariant without explicit
  approval.

## When in doubt

- **Stuck on workflow?** Re-read `docs/dev/contrib.md`.
- **About to make a structural change?** Read
  `docs/dev/architecture.md` first.
- **Found a defect?** File a `gh issue` before patching around it.
