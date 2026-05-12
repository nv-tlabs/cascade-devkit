# GitHub Copilot — repository instructions

See [`AGENTS.md`](../AGENTS.md) at the repository root for the full
set of repository-wide rules that apply when working in this codebase
(branch-first workflow, atomic commits, the canonical `make`-target
verb namespace, security and data-handling rules, etc.).

Subproject-specific rules live in nested `AGENTS.md` files — e.g.
[`tools/annotator/AGENTS.md`](../tools/annotator/AGENTS.md) — and
Copilot loads them automatically when working under that path.

> GitHub Copilot reads `AGENTS.md` natively per
> <https://docs.github.com/en/copilot/customizing-copilot/adding-repository-custom-instructions-for-github-copilot>.
> This file is the conventional `.github/copilot-instructions.md`
> path; both load, and they say the same thing.
