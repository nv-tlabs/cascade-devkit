# Schema history

One row per released CASCADE annotation-schema version, newest last. The
machine-readable registry lives in
[`src/cascade_av/spec/versions.py`](../../src/cascade_av/spec/versions.py)
(`SCHEMA_HISTORY` tuple) and must be updated in the same PR that changes
this table.

| Version | Released   | Breaking? | Summary                                                                                                                                                                              | Migration |
|---------|------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| 2.0.0   | 2026-05-12 | —         | Initial CASCADE schema. `bounding_boxes` typed on `Agent` / `TrafficObject` / `TrafficLight`.                                                                                        | —         |
| 2.1.0   | 2026-05-18 | yes       | Removed `bounding_boxes` from the base schema. Deprecated data moves to a sibling `<stem>.extra.json` sidecar under `extensions["bbox/1.0"]`; the producer is no longer emitting it. | `cascade-migrate <corpus-dir>` |
| 2.2.0   | 2026-05-18 | yes       | Removed annotator timeline-layout indices (`_track_index`, `_cond_track_index`, `_cont_track_index`, `_state_track_index`, `_influence_track_index`, `_prop_track_index`) from every typed model. Indices move to the sibling sidecar under `extensions["ui/1.0"]`, keyed by entity id. The first-party `UiExtension` ships in-tree (`cascade_av.extensions.ui`). | `cascade-migrate <corpus-dir>` (handles 2.0.0 → 2.2.0 in one pass via BFS) |

## Reading the registry from code

```python
from cascade_av.spec import (
    CURRENT_SCHEMA_VERSION,
    SCHEMA_HISTORY,
    SUPPORTED_SCHEMA_VERSIONS,
    changelog_for,
    is_known,
)

CURRENT_SCHEMA_VERSION      # "2.2.0"
SUPPORTED_SCHEMA_VERSIONS   # frozenset({"2.0.0", "2.1.0", "2.2.0"})
changelog_for("2.2.0").summary
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

A schema extension claims one sidecar key. To add one:

1. Register itself with `cascade_av.extensions.register(...)` or via the
   `cascade_av.extensions` entry-point group.
2. Implement `load(bundle, ext_data) -> None` to attach a typed view onto
   `bundle._extensions[key]`.
3. Implement `dump(bundle) -> dict | None` so its data round-trips back
   to the sidecar. `dump` should walk live entities and drop indices for
   ids no longer present — the stale-key sweep is load-bearing for
   per-entity extensions like `ui/1.0`.

The first-party `UiExtension` (`"ui/1.0"`) ships in-tree; third-party
extensions are loaded via the entry-point group. Sidecar payload with no
registered extension is preserved verbatim in `bundle._sidecar_raw` and
re-written on save, with one `UserWarning` per unknown key per process.

See [`cascade_av.extensions`](../../src/cascade_av/extensions/__init__.py).
