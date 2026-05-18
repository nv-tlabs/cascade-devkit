# Schema history

One row per released CASCADE annotation-schema version, newest last. The
machine-readable registry lives in
[`src/cascade_av/spec/versions.py`](../../src/cascade_av/spec/versions.py)
(`SCHEMA_HISTORY` tuple) and must be updated in the same PR that changes
this table.

| Version | Released   | Breaking? | Summary                                                                                                                                                                              | Migration |
|---------|------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| 2.0.0   | 2026-05-12 | —         | Initial CASCADE schema. `bounding_boxes` typed on `Agent` / `TrafficObject` / `TrafficLight`.                                                                                        | —         |
| 2.1.0   | 2026-05-18 | yes       | Removed `bounding_boxes` from the base schema. Deprecated data moves to a sibling `<stem>.extra.json` sidecar under `extensions["bbox/1.0"]`; the producer is no longer emitting it. | `cascade-migrate <corpus-dir>` (ships in PR 3) |

## Reading the registry from code

```python
from cascade_av.spec import (
    CURRENT_SCHEMA_VERSION,
    SCHEMA_HISTORY,
    SUPPORTED_SCHEMA_VERSIONS,
    changelog_for,
    is_known,
)

CURRENT_SCHEMA_VERSION      # "2.1.0"
SUPPORTED_SCHEMA_VERSIONS   # frozenset({"2.0.0", "2.1.0"})
changelog_for("2.1.0").summary
```

## Soft-version contract

`schema_version` is a plain `str`. The library does not hard-reject older
versions: a 2.0.0 file loads into a 2.1.0-aware `AnnotationBundle`, with the
typed `bounding_boxes` fields gone and the inline payload riding through
Pydantic's `extra="allow"` machinery into `__pydantic_extra__`. The validator
emits one `DeprecationWarning` per non-current version per process pointing
the user at `cascade-migrate`.

Hard-rejecting an old version would break the migration tool itself (it
needs to load 2.0.0 files in order to migrate them) and would force users
to run a one-shot migration before being able to open any clip in the
annotator. We are not making that trade today.

## Extension surface

When a future schema-extension claims one of the sidecar keys listed in
the table above, it should:

1. Register itself with `cascade_av.extensions.register(...)` or via the
   `cascade_av.extensions` entry-point group.
2. Implement `load(bundle, ext_data) -> None` to attach a typed view onto
   `bundle._extensions[key]`.
3. Implement `dump(bundle) -> dict | None` so its data round-trips back
   to the sidecar.

Sidecar payload that has no registered extension is preserved verbatim
in `bundle._sidecar_raw` and re-written on save, with one `UserWarning`
per unknown key per process.

See [`cascade_av.extensions`](../../src/cascade_av/extensions/__init__.py).
