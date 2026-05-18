# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``ui/1.0`` schema extension — annotator-tool timeline layout indices.

Background: in schema versions ``2.0.0`` and ``2.1.0`` the annotator's
per-entity row/lane assignments lived as typed fields on every model
(``_track_index``, ``_cond_track_index``, ``_cont_track_index``,
``_state_track_index``, ``_influence_track_index``, ``_prop_track_index``).
They were render concerns leaking into the data model — confusing for
downstream consumers and inconsistent with how bboxes were already split
out in ``2.1.0``.

Schema ``2.2.0`` removes them from the typed schema entirely. They round-trip
in the sidecar ``<stem>.extra.json`` under ``extensions["ui/1.0"]``, keyed by
entity id, in this shape::

    {
      "track_index":           {"<id>": <int>, ...},
      "cond_track_index":      {"<id>": <int>, ...},
      "cont_track_index":      {"<id>": <int>, ...},
      "state_track_index":     {"<id>": <int>, ...},
      "influence_track_index": {"<id>": <int>, ...},
      "prop_track_index":      {"<id>": <int>, ...}
    }

In-memory access goes through ``bundle.ext("ui/1.0")``, which yields a
mutable :class:`UiIndexes` view. Stale-key sweep on save: :meth:`UiExtension.dump`
walks the live entity tree and drops any sidecar key whose entity no longer
exists, so a delete in the annotator cannot orphan indices forever.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cascade_av.extensions.base import Extension

if TYPE_CHECKING:
    from cascade_av.spec import AnnotationBundle


#: The six index names stored in the ``ui/1.0`` payload. Keep in sync with
#: the ``_*_track_index`` aliases that existed on the typed schema through
#: 2.1.0; the leading underscore is dropped here because they're no longer
#: Pydantic field names with on-disk aliases — they're just dictionary keys
#: in the sidecar.
UI_INDEX_KEYS: tuple[str, ...] = (
    "track_index",
    "cond_track_index",
    "cont_track_index",
    "state_track_index",
    "influence_track_index",
    "prop_track_index",
)


class UiIndexes:
    """Mutable view over the ``ui/1.0`` sidecar payload.

    Stored as ``bundle._extensions["ui/1.0"]`` after a successful load (or
    after :meth:`get_or_create`). Consumers query via :meth:`get` and mutate
    via :meth:`set`; the extension's :meth:`UiExtension.dump` collects the
    state into the sidecar payload on save.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, dict[str, int]] | None = None) -> None:
        self._data: dict[str, dict[str, int]] = {}
        if not data:
            return
        for k in UI_INDEX_KEYS:
            sub = data.get(k)
            if not isinstance(sub, dict):
                continue
            cleaned: dict[str, int] = {}
            for entity_id, value in sub.items():
                # Defensive cast: a sidecar written by an older or buggy
                # producer might have stringified ints. We accept and coerce
                # rather than crash on load.
                if not isinstance(entity_id, str) or not entity_id:
                    continue
                try:
                    cleaned[entity_id] = int(value)
                except (TypeError, ValueError):
                    continue
            if cleaned:
                self._data[k] = cleaned

    def get(self, entity_id: str, key: str) -> int | None:
        """Return the stored index for ``(entity_id, key)``, or ``None``."""
        return self._data.get(key, {}).get(entity_id)

    def set(self, entity_id: str, key: str, value: int | None) -> None:
        """Store an index. ``value=None`` removes the entry."""
        if key not in UI_INDEX_KEYS:
            raise ValueError(
                f"unknown ui index key {key!r}; expected one of {UI_INDEX_KEYS}"
            )
        if not entity_id:
            raise ValueError("entity_id must be a non-empty string")
        if value is None:
            self._data.get(key, {}).pop(entity_id, None)
            if key in self._data and not self._data[key]:
                del self._data[key]
            return
        self._data.setdefault(key, {})[entity_id] = int(value)

    def to_payload(self, *, keep_ids: set[str] | None = None) -> dict[str, dict[str, int]]:
        """Return the serialisable payload.

        With ``keep_ids`` set, drop any entry whose id is not in the set —
        this is the stale-key sweep the extension's :meth:`UiExtension.dump`
        relies on. Without ``keep_ids``, return everything verbatim.
        """
        out: dict[str, dict[str, int]] = {}
        for k, sub in self._data.items():
            if keep_ids is None:
                kept = dict(sub)
            else:
                kept = {eid: v for eid, v in sub.items() if eid in keep_ids}
            if kept:
                out[k] = kept
        return out

    @classmethod
    def get_or_create(cls, bundle: "AnnotationBundle") -> "UiIndexes":
        """Return the bundle's :class:`UiIndexes`, creating an empty one if absent.

        Use this from consumers (the annotator frontend's wire-adapter, the
        Python viz layer, the migration CLI) when they need to read or
        mutate indices and don't care whether a sidecar was loaded.
        """
        existing = bundle._extensions.get(UiExtension.key)
        if isinstance(existing, cls):
            return existing
        instance = cls()
        bundle._extensions[UiExtension.key] = instance
        return instance


class UiExtension(Extension):
    """Round-trip ``ui/1.0`` payloads through :class:`UiIndexes`."""

    key = "ui/1.0"
    schema_versions = ("2.2.0",)

    def load(self, bundle: "AnnotationBundle", ext_data: dict[str, Any]) -> None:
        bundle._extensions[self.key] = UiIndexes(ext_data)

    def dump(self, bundle: "AnnotationBundle") -> dict[str, Any] | None:
        indexes = bundle._extensions.get(self.key)
        if not isinstance(indexes, UiIndexes):
            return None
        live = _collect_entity_ids(bundle)
        payload = indexes.to_payload(keep_ids=live)
        return payload or None


def _collect_entity_ids(bundle: "AnnotationBundle") -> set[str]:
    """Walk the live annotation tree and return the set of all entity ids.

    Used as the basis for the dump-time stale-key sweep: any id in
    ``bundle._extensions["ui/1.0"]`` that is no longer in this set has been
    deleted from the bundle, and its sidecar entry should be pruned.
    """
    ids: set[str] = set()
    ann = getattr(bundle, "annotation", None)
    if ann is None:
        return ids

    def _add(entity_id: object) -> None:
        if isinstance(entity_id, str) and entity_id:
            ids.add(entity_id)

    for env in getattr(ann, "environments", None) or ():
        _add(getattr(env, "id", None))
    for cond in getattr(ann, "conditions", None) or ():
        _add(getattr(cond, "id", None))
    for obj in getattr(ann, "traffic_objects", None) or ():
        _add(getattr(obj, "id", None))
        for cont in getattr(obj, "containment", None) or ():
            _add(getattr(cont, "id", None))
    for light in getattr(ann, "traffic_lights", None) or ():
        _add(getattr(light, "id", None))
        for cont in getattr(light, "containment", None) or ():
            _add(getattr(cont, "id", None))
        for head in getattr(light, "signal_heads", None) or ():
            for cont in getattr(head, "env_controlled", None) or ():
                _add(getattr(cont, "id", None))
            for state in getattr(head, "state_sequence", None) or ():
                _add(getattr(state, "id", None))
    ego = getattr(ann, "ego_vehicle", None)
    if ego is not None:
        for prop in getattr(ego, "properties", None) or ():
            _add(getattr(prop, "id", None))
        for cont in getattr(ego, "containment", None) or ():
            _add(getattr(cont, "id", None))
        for infl in getattr(ego, "influenced_by", None) or ():
            _add(getattr(infl, "id", None))
    for agent in getattr(ann, "agents", None) or ():
        _add(getattr(agent, "id", None))
        for prop in getattr(agent, "properties", None) or ():
            _add(getattr(prop, "id", None))
        for cont in getattr(agent, "containment", None) or ():
            _add(getattr(cont, "id", None))
        for infl in getattr(agent, "influenced_by", None) or ():
            _add(getattr(infl, "id", None))
    return ids
