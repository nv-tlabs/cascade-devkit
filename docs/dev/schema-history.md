# Schema history

One row per released CASCADE annotation-schema version, newest last. The
machine-readable registry lives in
[`src/cascade_av/spec/versions.py`](../../src/cascade_av/spec/versions.py)
(`SCHEMA_HISTORY` tuple) and must be updated in the same PR that changes
this table.

| Version | Released   | Breaking? | Summary                                                                                              | Migration |
|---------|------------|-----------|------------------------------------------------------------------------------------------------------|-----------|
| 2.0.0   | 2026-05-20 | —         | Initial CASCADE schema aligned with `sil-dense-annotation-tool 0.4.5`. Re-stamp of the pre-2.0.0 history; corpus is being reconverted upstream. | —         |

## In-place 0.6.x drift (no version bump)

After 2.0.0 shipped, the annotator advanced through internal versions
0.5.0, 0.6.0, and 0.6.1 **without bumping the bundle's
`schema_version`** — files in the wild still declare `"2.0.0"` while
their content drifted ahead of the original 2.0.0 shape. Rather than
introduce a new schema version for a stream of additive,
backward-compatible vocab/field changes, the DevKit **expanded the
2.0.0 spec in place** to recognise the observed values. No version row
is added here and `CURRENT_SCHEMA_VERSION` stays `"2.0.0"`; the
soft-version contract below already loads these bundles, and the
typed schema now names what they carry instead of leaving it in
`__pydantic_extra__`.

Newly recognised values and fields:

- **Properties:** `Stopped` (on both agent and ego properties).
- **Actions:** a standalone `Jaywalk` base verb, distinct from the
  older `oxd:Walk (jaywalk)` / `oxd:Run (jaywalk[, erratic])` suffix
  forms.
- **Environments:** `fst:SpeedBump`, `fst:LightRailLane`.
- **Signaling intent:** `Left Indicator`, `Right Indicator`.
- **Traffic objects:** `Boom gate`.
- **Clip fields:** `eventful_reason` (categorical rationale for
  `eventful`; advisory vocab `ego_adapts` / `special_environment` /
  `agent_adapts` / `other`) plus a free-text sibling
  `eventful_reason_other`.

Values the drift retired are **kept resolvable** (loaded, queryable)
but marked deprecated upstream: ego action `oxd:NotMove` (now
`oxd:Stop`), signaling intent `Turn` (now Left/Right Indicator), the
agent-amount density tiers (`Light/Medium/Heavy traffic`), and the
combined `(jaywalk[, erratic])` walk/run suffix forms (now standalone
`Jaywalk` + an `Erratic` property). See
[`docs/user/query_language.md`](../user/query_language.md) for the
query-side aliases.

### Environments and containment are optional

`SilAvAnnotation.environments` and `conditions`, and every `containment`
list (on the ego, agents, traffic objects, and lights), default to empty
and are **not guaranteed to be populated**. How densely they are
annotated depends on the annotation's *scope*: batches annotated in an
**ego-centric** mode focus on ego (and agent) behavior and may carry few
or no environments and little containment. In the ego-centric sample
audited, environments averaged ~0.4 per clip (almost all `oxd:Road`),
the ego carried **no** containment at all, and agent containment — where
present — did not reference an environment (`env_id` was blank).
Consumers must treat the environment and containment layers as optional
and must not assume per-clip spatial grounding.

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
