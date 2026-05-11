"""By-ID lookup of every addressable entity in an `AnnotationBundle`.

Causal links (`because_of`, `influencers`, `link_to`, `action_target`) carry
string IDs that point at entities elsewhere in the same bundle. `IdIndex`
resolves them in one pass.

The synthetic id `"Ego"` is reserved for the ego vehicle itself; it does not
appear on the `EgoVehicle` model but is the conventional referent when other
entities point at the ego.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from causal_ai_av.spec import AnnotationBundle

EGO_ID = "Ego"


@dataclass(frozen=True, slots=True)
class Subject:
    """A typed reference to an entity inside an `AnnotationBundle`."""

    kind: str  # see KINDS below
    id: str
    obj: Any  # the Pydantic model instance


# Stable strings used as `Subject.kind`. Anything outside this set indicates
# a future-extension entity carried through `extra="allow"`.
KINDS: tuple[str, ...] = (
    "agent",
    "agent_action",
    "ego",
    "ego_action",
    "environment",
    "condition",
    "traffic_object",
    "traffic_light",
    "signal_head",
    "light_state",
    "object_state",
    "containment",
    "property",
    "influence",
)


class IdIndex:
    """One-pass index of every addressable entity in a bundle."""

    def __init__(self, bundle: AnnotationBundle) -> None:
        self.bundle = bundle
        self._by_id: dict[str, Subject] = {}
        self._build()

    # -- public lookup --------------------------------------------------------

    def get(self, id_: str) -> Subject | None:
        """Return the entity referenced by `id_`, or `None` if it dangles."""
        return self._by_id.get(id_)

    def __contains__(self, id_: str) -> bool:
        return id_ in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def of_kind(self, kind: str) -> list[Subject]:
        """All entities of a given `kind`."""
        return [s for s in self._by_id.values() if s.kind == kind]

    # -- build ----------------------------------------------------------------

    def _add(self, kind: str, id_: str, obj: Any) -> None:
        if id_:
            # Last writer wins on duplicate IDs; the corpus does not enforce
            # uniqueness across all entity classes, and surfacing the second
            # one matches the order the producer wrote it.
            self._by_id[id_] = Subject(kind, id_, obj)

    def _build(self) -> None:
        ann = self.bundle.annotation
        ego = ann.ego_vehicle

        # Synthetic ego anchor — referenced by `link_to: ["Ego"]`, etc.
        self._add("ego", EGO_ID, ego)

        for env in ann.environments:
            self._add("environment", env.id, env)
        for cond in ann.conditions:
            self._add("condition", cond.id, cond)
        for obj in ann.traffic_objects:
            self._add("traffic_object", obj.id, obj)
            for cont in obj.containment:
                self._add("containment", cont.id, cont)
            for state in obj.state_sequence:
                self._add("object_state", state.id, state)
        for light in ann.traffic_lights:
            self._add("traffic_light", light.id, light)
            for head in light.signal_heads:
                self._add("signal_head", head.id, head)
                for state in head.state_sequence:
                    self._add("light_state", state.id, state)

        # Ego sub-entities
        for action in ego.actions:
            self._add("ego_action", action.id, action)
        for prop in ego.properties:
            self._add("property", prop.id, prop)
        for cont in ego.containment:
            self._add("containment", cont.id, cont)
        for infl in ego.influenced_by:
            self._add("influence", infl.id, infl)

        # Agents and their sub-entities
        for agent in ann.agents:
            self._add("agent", agent.id, agent)
            for action in agent.actions:
                self._add("agent_action", action.id, action)
            for prop in agent.properties:
                self._add("property", prop.id, prop)
            for cont in agent.containment:
                self._add("containment", cont.id, cont)
            for infl in agent.influenced_by:
                self._add("influence", infl.id, infl)
