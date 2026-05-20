# Schema history

One row per released CASCADE annotation-schema version, newest last. The
machine-readable registry lives in
[`src/cascade_av/spec/versions.py`](../../src/cascade_av/spec/versions.py)
(`SCHEMA_HISTORY` tuple) and must be updated in the same PR that changes
this table.

| Version | Released   | Breaking? | Summary                                                                                              | Migration |
|---------|------------|-----------|------------------------------------------------------------------------------------------------------|-----------|
| 2.0.0   | 2026-05-20 | —         | Initial CASCADE schema aligned with `sil-dense-annotation-tool 0.4.5`. Re-stamp of the pre-2.0.0 history; corpus is being reconverted upstream. | —         |

## Reading the registry from code

```python
from cascade_av.spec import (
    CURRENT_SCHEMA_VERSION,
    SCHEMA_HISTORY,
    SUPPORTED_SCHEMA_VERSIONS,
    changelog_for,
    is_known,
)

CURRENT_SCHEMA_VERSION      # "2.0.0"
SUPPORTED_SCHEMA_VERSIONS   # frozenset({"2.0.0"})
changelog_for("2.0.0").summary
```

## Soft-version contract

`schema_version` is a plain `str`. The library does not hard-reject older
versions: a bundle stamped with an unknown version still loads, with anything
the typed schema doesn't recognise riding through Pydantic's `extra="allow"`
machinery into `__pydantic_extra__`. The validator emits one
`DeprecationWarning` per non-current version per process.

There is no in-tree migration path from pre-reboot bundles — the corpus is
being reconverted upstream against the 2.0.0 shape directly, and the
`cascade-migrate` CLI was retired in the same reboot PR.

## Extension surface

A schema extension claims one sidecar key. To add one:

1. Register itself with `cascade_av.extensions.register(...)` or via the
   `cascade_av.extensions` entry-point group.
2. Implement `load(bundle, ext_data) -> None` to attach a typed view onto
   `bundle._extensions[key]`.
3. Implement `dump(bundle) -> dict | None` so its data round-trips back
   to the sidecar.

No first-party extensions ship in-tree as of 2.0.0. Sidecar payload with no
registered extension is preserved verbatim in `bundle._sidecar_raw` and
re-written on save, with one `UserWarning` per unknown key per process —
this is what keeps third-party keys like `bbox/1.0` round-tripping.

See [`cascade_av.extensions`](../../src/cascade_av/extensions/__init__.py).
