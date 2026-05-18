# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``cascade-migrate`` — convert old CASCADE annotation JSONs to the current schema.

Each adjacent-pair migration is a function in :data:`MIGRATIONS` keyed by
``(from_version, to_version)`` that takes a raw main JSON dict and returns
``(new_main_dict, sidecar_extensions_dict)``. The CLI walks the migration
chain from ``bundle["schema_version"]`` to
:data:`cascade_av.spec.CURRENT_SCHEMA_VERSION` and pair-writes the result.

The CLI operates on raw JSON dicts rather than parsed ``AnnotationBundle``
objects so it can manipulate fields that were removed from the typed schema
(e.g. the 2.0.0 ``bounding_boxes`` arrays vanish from
``cascade_av.spec.schema`` in 2.1.0 but still need to be moved to a sidecar).

Atomicity: the sidecar is written first, then the main file. A crash
between leaves the main at its previous version and the sidecar refreshed
— re-running the migration is safe because it is idempotent.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Callable

from cascade_av.io.local import _atomic_write_text, _sidecar_path_for
from cascade_av.spec import CURRENT_SCHEMA_VERSION

_log = logging.getLogger("cascade_av.cli.migrate")

# ---------------------------------------------------------------------------
# Migration registry — adjacent-pair callables
# ---------------------------------------------------------------------------

#: ``(from, to) -> migrator``. Each callable receives the raw main-JSON dict
#: and returns ``(new_main, sidecar_extensions)`` where the second element
#: maps extension key (e.g. ``"bbox/1.0"``) to its payload.
Migrator = Callable[[dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]]


def _migrate_2_0_0_to_2_1_0(
    main: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """2.0.0 → 2.1.0: pull inline ``bounding_boxes`` out of agents,
    traffic_objects, and traffic_lights into the sidecar's ``bbox/1.0`` key.
    """
    ann = main.get("annotation") or {}
    bbox_groups: dict[str, dict[str, Any]] = {
        "agents": {},
        "traffic_objects": {},
        "traffic_lights": {},
    }
    for group_key in bbox_groups:
        for entity in ann.get(group_key) or []:
            if not isinstance(entity, dict):
                continue
            boxes = entity.pop("bounding_boxes", None)
            if not boxes:
                continue
            entity_id = entity.get("id") or ""
            if not entity_id:
                # Fall back to a synthetic key — better than dropping the data.
                entity_id = f"_unnamed_{len(bbox_groups[group_key])}"
            bbox_groups[group_key][entity_id] = boxes
    populated = {k: v for k, v in bbox_groups.items() if v}
    main["schema_version"] = "2.1.0"
    sidecar = {"bbox/1.0": populated} if populated else {}
    return main, sidecar


MIGRATIONS: dict[tuple[str, str], Migrator] = {
    ("2.0.0", "2.1.0"): _migrate_2_0_0_to_2_1_0,
}


def _migration_path(from_v: str, to_v: str) -> list[tuple[str, str]] | None:
    """Build a chain of adjacent ``(from, to)`` pairs from ``from_v`` to ``to_v``.

    Returns an empty list when the versions are equal, or ``None`` when no
    chain exists in :data:`MIGRATIONS`.
    """
    if from_v == to_v:
        return []
    adjacency: dict[str, list[str]] = {}
    for f, t in MIGRATIONS:
        adjacency.setdefault(f, []).append(t)
    queue: deque[tuple[str, list[tuple[str, str]]]] = deque([(from_v, [])])
    visited: set[str] = {from_v}
    while queue:
        cur, path = queue.popleft()
        for nxt in adjacency.get(cur, []):
            if nxt in visited:
                continue
            new_path = path + [(cur, nxt)]
            if nxt == to_v:
                return new_path
            visited.add(nxt)
            queue.append((nxt, new_path))
    return None


# ---------------------------------------------------------------------------
# Per-file migration
# ---------------------------------------------------------------------------


class MigrationResult:
    """Outcome of attempting to migrate one main JSON."""

    __slots__ = ("path", "status", "from_version", "to_version", "sidecar_keys", "reason")

    def __init__(
        self,
        path: Path,
        status: str,
        *,
        from_version: str = "",
        to_version: str = "",
        sidecar_keys: tuple[str, ...] = (),
        reason: str = "",
    ) -> None:
        self.path = path
        self.status = status  # "migrated" | "skipped" | "error"
        self.from_version = from_version
        self.to_version = to_version
        self.sidecar_keys = sidecar_keys
        self.reason = reason


def _read_main(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def _read_existing_sidecar(sidecar_path: Path) -> dict[str, Any]:
    if not sidecar_path.exists():
        return {}
    try:
        data = json.loads(sidecar_path.read_bytes())
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    extensions = data.get("extensions")
    return extensions if isinstance(extensions, dict) else {}


def _merge_sidecar(
    *,
    existing: dict[str, Any],
    migration_payload: dict[str, Any],
    force: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Combine ``existing`` sidecar extension data with this run's payload.

    Returns ``(merged, conflicts)``. ``conflicts`` lists extension keys
    that the migration would overwrite without ``--force``.
    """
    merged = dict(existing)
    conflicts: list[str] = []
    for key, payload in migration_payload.items():
        if key in merged and not force:
            conflicts.append(key)
            continue
        merged[key] = payload
    return merged, conflicts


def _migrate_one(
    main_path: Path,
    *,
    output_dir: Path | None,
    dry_run: bool,
    force: bool,
) -> MigrationResult:
    main = _read_main(main_path)
    from_v = str(main.get("schema_version") or "")
    if not from_v:
        return MigrationResult(
            main_path, "error", reason="no schema_version field in main JSON",
        )
    path_pairs = _migration_path(from_v, CURRENT_SCHEMA_VERSION)
    if path_pairs is None:
        return MigrationResult(
            main_path,
            "error",
            from_version=from_v,
            to_version=CURRENT_SCHEMA_VERSION,
            reason=f"no migration chain from {from_v!r} to {CURRENT_SCHEMA_VERSION!r}",
        )
    if not path_pairs:
        return MigrationResult(
            main_path,
            "skipped",
            from_version=from_v,
            to_version=from_v,
            reason="already current",
        )

    # Run the chain of migrators, accumulating sidecar payload.
    accumulated_sidecar: dict[str, Any] = {}
    for f, t in path_pairs:
        migrator = MIGRATIONS[(f, t)]
        main, new_sidecar = migrator(main)
        for k, v in new_sidecar.items():
            accumulated_sidecar[k] = v  # later migrators win

    # Pick destinations.
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        main_dest = output_dir / main_path.name
        sidecar_dest = _sidecar_path_for(main_dest)
        existing_sidecar_path = _sidecar_path_for(main_path)
        existing_sidecar = _read_existing_sidecar(existing_sidecar_path)
    else:
        main_dest = main_path
        sidecar_dest = _sidecar_path_for(main_path)
        existing_sidecar = _read_existing_sidecar(sidecar_dest)

    merged_extensions, conflicts = _merge_sidecar(
        existing=existing_sidecar,
        migration_payload=accumulated_sidecar,
        force=force,
    )
    if conflicts:
        return MigrationResult(
            main_path,
            "error",
            from_version=from_v,
            to_version=CURRENT_SCHEMA_VERSION,
            reason=(
                f"refusing to overwrite existing sidecar key(s) "
                f"{conflicts!r}; rerun with --force to replace them."
            ),
        )

    if dry_run:
        return MigrationResult(
            main_path,
            "migrated",
            from_version=from_v,
            to_version=CURRENT_SCHEMA_VERSION,
            sidecar_keys=tuple(sorted(merged_extensions)),
            reason="(dry-run)",
        )

    # Sidecar first, main second.
    if merged_extensions:
        sidecar_payload = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "main_file": main_dest.name,
            "extensions": merged_extensions,
        }
        _atomic_write_text(
            sidecar_dest,
            json.dumps(sidecar_payload, indent=2, sort_keys=True),
        )
    _atomic_write_text(main_dest, json.dumps(main, indent=2))
    return MigrationResult(
        main_path,
        "migrated",
        from_version=from_v,
        to_version=CURRENT_SCHEMA_VERSION,
        sidecar_keys=tuple(sorted(merged_extensions)),
    )


# ---------------------------------------------------------------------------
# Input enumeration + CLI
# ---------------------------------------------------------------------------


def _iter_inputs(target: Path) -> Iterator[Path]:
    """Yield main annotation JSONs under ``target`` (or just it if it's a file)."""
    if target.is_file():
        if target.suffix == ".json" and not target.name.endswith(".extra.json"):
            yield target
        return
    if target.is_dir():
        for p in sorted(target.glob("*.json")):
            if p.name.endswith(".extra.json"):
                continue
            yield p


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cascade-migrate",
        description=(
            "Migrate CASCADE annotation JSON files to the current schema "
            f"version ({CURRENT_SCHEMA_VERSION}). Deprecated payload moves "
            "to a sibling <stem>.extra.json sidecar."
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Annotation JSON file, or directory containing *.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Write migrated files to this directory instead of in-place. "
            "Sidecar(s) land alongside the main file. Default: in-place."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be migrated without writing.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "CI mode: exit non-zero if any input file needs migration. "
            "Implies --dry-run."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing sidecar keys that this migration would "
            "produce. Sidecar keys we do not touch are always preserved."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print one line per processed file.",
    )
    return parser


def _format_result(r: MigrationResult) -> str:
    if r.status == "migrated":
        keys = f" (sidecar: {', '.join(r.sidecar_keys)})" if r.sidecar_keys else ""
        return f"MIGRATED {r.path} -> {r.from_version} → {r.to_version}{keys}"
    if r.status == "skipped":
        return f"SKIPPED  {r.path} ({r.reason})"
    return f"ERROR    {r.path} ({r.reason})"


def _emit(message: str) -> None:
    print(message, flush=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    inputs = list(_iter_inputs(args.input))
    if not inputs:
        _emit(f"no annotation JSON files found under {args.input}")
        return 1

    dry_run = args.dry_run or args.check
    counts = {"migrated": 0, "skipped": 0, "error": 0}
    needs_migration = 0

    for path in inputs:
        try:
            result = _migrate_one(
                path,
                output_dir=args.output_dir,
                dry_run=dry_run,
                force=args.force,
            )
        except Exception as exc:  # noqa: BLE001 — surface any unexpected failure
            result = MigrationResult(
                path, "error", reason=f"unexpected failure: {exc!r}"
            )
        counts[result.status] += 1
        if result.status == "migrated" and result.from_version != result.to_version:
            needs_migration += 1
        if args.verbose or result.status == "error":
            _emit(_format_result(result))

    summary = (
        f"summary: {counts['migrated']} migrated, "
        f"{counts['skipped']} skipped, {counts['error']} error"
    )
    _emit(summary)

    if counts["error"]:
        return 2
    if args.check and needs_migration:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
