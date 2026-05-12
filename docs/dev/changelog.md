# Changelog

One-line entry per merged PR. Newest first. Format:

```
- YYYY-MM-DD <type>(<scope>): <summary> (#<PR>)
```

Types match the commit convention in [`contrib.md`](./contrib.md): `feat`,
`fix`, `refactor`, `docs`, `test`, `chore`.

## Entries

<!-- Add new entries above this line -->
- 2026-05-12 feat(viz): rich timeline — labels, sub-lanes, arrowheads, filter kwargs (#29)
- 2026-05-12 fix(dataset): Sequence handles missing egomotion gracefully (#28)
- 2026-05-12 fix(viz): post-merge cleanup — video-window clamping, carousel boundary guards (#27)
- 2026-05-12 feat(viz): Sequence.visualize() polymorphic dispatcher + docs + notebook (#26)
- 2026-05-12 feat(viz): MatchSet.visualize() carousel (#25)
- 2026-05-12 feat(viz): ClipPlayer interactive widget (#24)
- 2026-05-12 feat(viz): headless render_timeline Plotly figure (#23)
- 2026-05-12 feat(viz): package skeleton + headless render_frame (#22)
- 2026-05-12 feat(query): MatchSet.sequences() iterator helper (#20)
- 2026-05-12 docs(user): drop legacy "old" form from scenario examples in query_language.md (#19)
- 2026-05-12 chore(license): add SPDX Apache-2.0 headers to all current source files (#18)
- 2026-05-12 feat(query): MatchSet retains a weakref back to its source dataset (closes #13) (#17)
- 2026-05-12 docs(user): lead query_language.md with worked examples, demote spec sections (#16)
- 2026-05-12 fix(annotator): remove guardedSave stub from RightPanel.tsx (closes #12) (#15)
- 2026-05-12 fix(annotator): remove unused LabelPalette component (closes #11) (#14)
- 2026-05-12 docs(user): create docs/user/ — move query_language.md, add annotator.md (#10)
- 2026-05-12 docs(security): .env.example + expand Security & data handling in AGENTS.md (#9)
- 2026-05-12 chore(devcontainer): add .devcontainer/devcontainer.json (Python 3.11 + uv + Node LTS + ffmpeg) (#8)
- 2026-05-12 docs(agents): thin tool pointers — .github/copilot-instructions.md, .cursor/rules/agents.mdc (#7)
- 2026-05-12 docs(skills): bootstrap .claude/skills/ — run-tests, annotator-dev, add-annotator-cli-flag (#6)
- 2026-05-12 docs(annotator): nested AGENTS.md scoping Tailwind/Zustand/save+lock rules (#5)
- 2026-05-12 refactor(layout): move annotator/ under tools/annotator/ (#4)
- 2026-05-12 docs(agents): bootstrap agentic-readiness — AGENTS.md, CLAUDE.md (@AGENTS.md), Makefile (#3)
- 2026-05-12 docs(dev): update changelog policy — entry lands in the same PR (#2)
- 2026-05-12 docs(dev): seed contribution workflow + architecture/changelog scaffolds (#1)
