# Changelog

One-line entry per merged PR. Newest first. Format:

```
- YYYY-MM-DD <type>(<scope>): <summary> (#<PR>)
```

Types match the commit convention in [`contrib.md`](./contrib.md): `feat`,
`fix`, `refactor`, `docs`, `test`, `chore`.

## Entries

<!-- Add new entries above this line -->
- 2026-05-13 feat(viz): `families=[...]` whitelist on visualize / render_timeline / carousel — render only selected annotation families, with parent rows auto-retained for surviving entities (#48)
- 2026-05-13 feat(viz): bold parent tick labels + drop "what's coming in v1" forward-talk from notebook 6 (#47)
- 2026-05-13 fix(viz): carousel uses short entity label (`Agent(id, type)`) instead of full Pydantic repr (#46)
- 2026-05-13 fix(viz): monospace tickfont + image sizing="stretch" + pinned default fig width — actually deliver #44's three asks (#45)
- 2026-05-13 feat(viz): widget video extends into label margin + aspect-fit pixel height + autosize horizontally (#44)
- 2026-05-13 feat(viz): widen tick-label pad column so parent centering visibly distinct from sub-row left-alignment (#43)
- 2026-05-13 feat(viz): center parent tick labels + left-align sub-rows + tighten widget video↔timeline gap (#42)
- 2026-05-13 feat(viz): light theme (`plotly_white`) + left-aligned y-tick labels via nbsp padding (#41)
- 2026-05-13 feat(viz): clustered entity blocks — soft background spans labels+bars, colored tick labels, dark gap between entities (#40)
- 2026-05-13 feat(viz): per-entity background row bands + short-form sub-row tick labels (#39)
- 2026-05-13 fix(spec): Condition.type accepts a bare string (coerced to single-element list) (#38)
- 2026-05-13 feat(viz): family sub-rows + per-entity bands + per-family colors + proportional label suppression (#37)
- 2026-05-12 chore(annotator): rename "Relevancy" section heading to "Relevance" (#36)
- 2026-05-12 fix(annotator): tighten timeline toolbar so "Because of" hint fits in narrow viewports (#35)
- 2026-05-12 fix(annotator): timeline zoom toolbar no longer wraps to two lines on narrow viewports (#34)
- 2026-05-12 fix(viz): JPEG-encode frame transport — Play perf from ~0.1 fps to ~5 fps (#33)
- 2026-05-12 fix(annotator): timeline category header no longer overlaps first track name (#32)
- 2026-05-12 chore(make): annotator-dev target forwards optional PORT= variable (#31)
- 2026-05-12 fix(viz): denser layout + readable labels + visible arrowheads + faster Play (#30)
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
