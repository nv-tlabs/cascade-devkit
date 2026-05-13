@AGENTS.md

# Claude Code — additions on top of AGENTS.md

The shared rules for every coding agent are in
[`AGENTS.md`](AGENTS.md), imported above. The notes below apply only
to Claude Code.

- **Use plan mode** for changes that touch more than ~3 files or that
  cross subproject boundaries (root ↔ `tools/annotator/`).
- **Personal skills live in `~/.claude/skills/`.** Project skills
  under `.claude/skills/` are committed and shared with the team.
