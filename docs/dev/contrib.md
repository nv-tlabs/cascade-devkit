# Contributing to this project

This document defines the workflow for all contributors — human and AI agents alike (Claude Code, Cursor, Copilot, etc.). Read it before making any changes.

---

## Branch & Commit Workflow

### Branching

- `main` is the development branch. **Never commit directly to `main`.**
- All work happens on feature branches cut from `main`:
  ```
  git checkout main && git pull
  git checkout -b <type>/<short-description>
  ```
- Branch naming convention:
  - `feat/<description>` — new feature
  - `fix/<description>` — bug fix
  - `refactor/<description>` — restructuring without behavior change
  - `docs/<description>` — documentation only
  - `chore/<description>` — tooling, config, deps

### Worktrees

We prefer [`git worktree`](https://git-scm.com/docs/git-worktree) over swapping branches in a single checkout. Worktrees give every branch its own working directory backed by the same `.git` store, which means:

- Multiple agents (or you + an agent) can edit different branches in parallel without stepping on each other.
- Long-running tooling (dev servers, test watchers, GPU jobs, indexers) stays untouched when you switch tasks — no `git stash`, no rebuilds.
- The main checkout stays clean for reviewing PRs or running `main`.

**Layout convention.** Keep worktrees as sibling directories of the main checkout, named after the branch:

```
~/Projects/
  av-causal-dataset-tools/                # main checkout, tracks `main`
  av-causal-dataset-tools-feat-loader/    # worktree on `feat/loader`
  av-causal-dataset-tools-fix-nan-ts/     # worktree on `fix/nan-timestamps`
```

**Create a worktree for a new branch.** From the main checkout:

```bash
git fetch origin
# Create a new branch off origin/main AND a worktree for it in one step
git worktree add -b feat/loader ../av-causal-dataset-tools-feat-loader origin/main
cd ../av-causal-dataset-tools-feat-loader
```

**Create a worktree for an existing branch** (e.g. to review someone else's PR locally):

```bash
git fetch origin
git worktree add ../av-causal-dataset-tools-pr-42 origin/their-branch
```

**List and inspect worktrees:**

```bash
git worktree list            # show all worktrees + their HEADs
git worktree list --porcelain
```

**Remove a worktree** once the branch is merged or abandoned:

```bash
# From any worktree (typically the main checkout):
git worktree remove ../av-causal-dataset-tools-feat-loader
git branch -d feat/loader     # delete the branch if merged (use -D to force)

# If the directory was deleted manually, prune stale metadata:
git worktree prune
```

**Rules and gotchas:**

- One branch per worktree — git refuses to check the same branch out in two worktrees. That is the feature, not a bug.
- Don't nest worktrees inside the main checkout (`./worktrees/...`) — tooling that walks the tree (linters, formatters, IDE indexers) will double-process files.
- Each worktree has its own untracked files, build artifacts, and `node_modules`/`venv`. Re-install deps after `git worktree add` if needed.
- `git worktree remove` refuses if the worktree has uncommitted changes; resolve those first rather than passing `--force`.

### Commits

Each commit must:
- Represent a **single logical change** — do not bundle unrelated changes.
- Have a clear message structured as:

  ```
  <type>(<scope>): <short summary>

  <body — what changed and why. Mention any trade-offs or alternatives considered.>
  ```

  Example:
  ```
  feat(auth): add JWT refresh token rotation

  Access tokens now expire in 15 min. Refresh tokens are single-use and
  rotated on each use to limit the blast radius of a stolen token.
  Chose rotation over sliding window to meet the security requirements
  in docs/ARCHITECTURE.md#authentication.
  ```

- Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

### Pull Requests

- PRs are opened from a feature branch into `main`.
- **Keep PRs small and focused.** A good PR can be reviewed in under 30 minutes. If a branch grows large, split it.
- PR title follows the same `<type>(<scope>): <summary>` format as commits.
- PR description must include:
  - What the change does
  - Why it is being made
  - How to test or verify it
  - Any open questions or follow-up issues
- Link related GitHub issues when applicable.
- **Update `docs/dev/changelog.md` in the same PR** with a one-line entry
  describing the change. The PR number is not known until `gh pr create`
  returns it — either predict the next number with
  `gh pr list --state all --limit 1 --json number`, or commit with a
  placeholder, open the PR, amend the commit with the real `(#<N>)`, and
  force-push the feature branch (force-push is forbidden on `main` but
  permitted on your own feature branch).
- At least one approval is required before merging.

---

## docs/dev Directory

The `docs/dev` directory is the project's living memory. All contributors must keep it up to date.

| File | Purpose |
|------|---------|
| `docs/dev/architecture.md` | Design decisions, system design, rationale. **Check this before starting any non-trivial work.** |
| `docs/dev/contrib.md` | This document, explaining how to do contributions to the project. |
| `docs/dev/changelog.md` | One-line entry per merged PR (newest first). **Update in the same PR that makes the change** — see [Pull Requests](#pull-requests). |

### Rules for agents

- **Read `docs/dev/architecture.md` first** before proposing or implementing any structural change.
- Record newly discovered issues as GitHub issues using `gh` rather than silently working around them (see [GitHub Issues](#github-issues) below).
- Look for open GitHub issues since they could reveal existing problems.
- Add a one-line entry to `docs/dev/changelog.md` **as part of the same PR** that makes the change (not a follow-up). See [Pull Requests](#pull-requests) for how to handle the PR-number reference.

---

## GitHub Issues

We track work and surface defects through GitHub Issues, managed from the terminal with the [`gh` CLI](https://cli.github.com/). Authenticate once with `gh auth login` (or rely on `GH_TOKEN` if it is set in your environment).

### Check for open issues before starting work

Before opening a new issue or starting a non-trivial change, scan what is already tracked — your bug or idea may already be known.

```bash
# List open issues in this repo (default: open, no filter)
gh issue list

# Filter by label, assignee, or author
gh issue list --label bug
gh issue list --label "good first issue"
gh issue list --assignee @me
gh issue list --author @me

# Full-text search across titles and bodies
gh issue list --search "dataset loader"
gh issue list --search "panic in:title"

# Include closed issues when researching prior decisions
gh issue list --state all --search "schema"

# View a specific issue in the terminal
gh issue view 42
gh issue view 42 --comments     # include the discussion thread
gh issue view 42 --web          # open in the browser instead
```

### File a new issue

Keep one issue per problem. A good issue is reproducible, scoped, and actionable.

```bash
# Interactive: prompts for title, body (in $EDITOR), labels, assignees
gh issue create

# Non-interactive: fully specified on the command line
gh issue create \
  --title "fix(loader): NaN timestamps crash CSV ingest" \
  --body "Reproduce by running ./scripts/ingest.py on fixtures/bad_ts.csv. Expected: row skipped with a warning. Actual: ValueError on line 42." \
  --label bug

# Body from a file (useful for longer reports or templates)
gh issue create --title "feat(api): paginated dataset listing" --body-file .github/ISSUE_TEMPLATE/feature.md

# Assign and label at creation time
gh issue create --title "..." --body "..." --label bug --label "needs-triage" --assignee @me
```

Recommended title format mirrors our commit style: `<type>(<scope>): <short summary>`.

A useful issue body includes:
- **Context** — what you were doing and why.
- **Steps to reproduce** — exact commands, inputs, or files.
- **Expected vs. actual** — what should happen, what does happen.
- **Environment** — branch/commit (`git rev-parse --short HEAD`), OS, relevant tool versions.
- **Workaround** (if any) — so others hitting it aren't blocked.

### Update, link, and close issues

```bash
# Comment on an existing issue (e.g. to link a PR or add findings)
gh issue comment 42 --body "Reproduced on main@$(git rev-parse --short HEAD); root cause looks like timestamp parsing."

# Add or remove labels after triage
gh issue edit 42 --add-label "priority:high" --remove-label "needs-triage"

# Close manually (PR merges that mention "Closes #42" will close it automatically)
gh issue close 42 --reason completed
gh issue close 42 --reason "not planned" --comment "Superseded by #57."
```

To auto-close an issue when a PR merges, include `Closes #<n>` (or `Fixes #<n>`) in the PR description.

---

## AI Agent Rules

The following rules apply to all AI coding agents (Claude Code, Cursor, GitHub Copilot, etc.):

0. **Use git worktree** so multiple agents can work on the repository in parallel without stepping on each other. See [Worktrees](#worktrees) for the layout convention and commands.
1. **Branch first.** Never modify source files on `main`. Always create a branch.
2. **Commit atomically.** One logical change per commit with a clear message explaining what and why.
3. **No large PRs.** If a task requires many changes, split into sequential PRs.
4. **Architecture first.** Read `docs/dev/architecture.md` before making structural decisions.
5. **Surface issues.** Add discovered problems to GitHub issues — don't silently patch around them.
6. **No force-push to `main`.** Ever.
7. **Do not commit secrets, credentials, or environment files** (`.env`, key files, etc.).
8. **Do not auto-merge your own PRs** without a human or designated reviewer approval.

---

## Getting Started

First time on the project:

```bash
git clone <repo-url>
cd av-causal-dataset-tools
```

Starting a new piece of work (preferred — uses a worktree so your main checkout stays free):

```bash
git fetch origin
git worktree add -b feat/your-feature ../av-causal-dataset-tools-feat-your-feature origin/main
cd ../av-causal-dataset-tools-feat-your-feature

# Install deps into this worktree (fresh worktrees have no .venv or
# node_modules). Cheap if everything is already cached.
make install

# make changes
git commit -m "feat(scope): what and why"
git push -u origin feat/your-feature
gh pr create                  # open a PR from the terminal
```

When the PR is merged, clean up:

```bash
cd ../av-causal-dataset-tools
git worktree remove ../av-causal-dataset-tools-feat-your-feature
git branch -d feat/your-feature
```

For questions or process issues, file a GitHub issue.