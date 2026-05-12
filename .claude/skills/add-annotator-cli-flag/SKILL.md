---
name: add-annotator-cli-flag
description: Add a new flag to the `causal-av-annotate` CLI. Extend `_build_parser`, thread the value through `main`, propagate to `create_app` / `VideoResolver`, document in the annotator README, and test.
---

# Add a flag to the annotator CLI

The CLI is a flat argparse parser (no sub-commands). All flags live
in `_build_parser()` inside
`tools/annotator/src/annotator/server/cli.py`. The flow is:
`main()` → `args = parser.parse_args(...)` → `create_app(...)` +
`VideoResolver(...)` → `uvicorn.run(...)`.

## Procedure

1. **Add the flag in `_build_parser()`.** Match the conventions of
   neighbouring flags: lowercase-hyphen name, typed `default=`, a
   one-sentence `help=` text. Example:
   ```python
   parser.add_argument(
       "--max-clips",
       type=int,
       default=None,
       help="Cap the sidebar at N clips (debug aid; default: no cap).",
   )
   ```
2. **Pull the value out of `args` in `main()`** and thread it to
   whatever consumes it. `create_app(...)` and `VideoResolver(...)`
   are the two main sinks; both take keyword args.
3. **If the flag changes server behavior**, plumb it through
   `create_app` and into the FastAPI app state — never rely on a
   module-global. Tests construct the app via `create_app`, so
   making the flag a `create_app` parameter keeps it testable.
4. **Update `tools/annotator/README.md`** ("Options" section) with
   one line per new flag.
5. **Add a test under `tools/annotator/tests/`** that exercises the
   new flag if it has user-visible behavior. Reuse the patterns in
   `test_server.py`. Mock `VideoResolver` if the flag doesn't touch
   video.

## Conventions

- **Defaults must be safe.** Use `None` for opt-in flags;
  the boring/safe value for behavior toggles.
- **Boolean flags use `action="store_true"`.** Reserve `--no-X`
  forms only when the default is true and the user needs to disable.
- **No mutually exclusive groups unless truly necessary.** Argparse
  groups complicate `--help`; prefer documenting the conflict in the
  `help=` text.
- **Don't add subcommands.** The annotator is a one-shot daemon CLI;
  if the flag surface grows past ~12 flags, file a `gh issue` before
  bolting on a subparser.

## Verify

```bash
uv run causal-av-annotate --help          # new flag is listed
make test                                  # the new test passes
```
