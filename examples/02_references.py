"""How entries compose — ``__ref__`` markers, resolution, and the four refusals.

Run it directly::

    python examples/02_references.py

Example 01 built one entry at a time. This one wires two together: a team whose
entry point *points at* an agent entry rather than carrying a copy of it. Every
outcome is asserted, so an API change turns the test suite red instead of leaving
the walkthrough quietly wrong. The narrative half is ``02-references.md``.

Everything here happens inside a single namespace. ``__namespace__`` addresses an
entry in another one; that is a separate topic and not this example's.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from akgentic.catalog import (
    UNSET_NAMESPACE,
    Catalog,
    CatalogValidationError,
    Entry,
    EntryNotFoundError,
    YamlEntryRepository,
)

REQUIRES: tuple[str, ...] = ()
"""Importable module names this example needs beyond the base install — none.

The test harness calls ``pytest.importorskip`` on every name here before invoking
``main()``. An example that ever needs an optional package must import it *inside*
``main()``: the harness reads this declaration off the already-imported module, so
a module-level import raises before the tuple can be read.
"""

TEAM_ID = "research-team"
LEAD_ID = "lead-agent"
AGENT_CARD_TYPE = "akgentic.core.AgentCard"

# The marker: a pure pointer. Its keys are exactly ``__ref__`` and, optionally,
# ``__type__`` / ``__namespace__`` — nothing else is allowed beside them.
LEAD_REF: dict[str, Any] = {"__ref__": LEAD_ID, "__type__": AGENT_CARD_TYPE}

# A team entry anchors its namespace, so its card starts inline: at that moment
# there is nothing in the namespace to point at.
PLACEHOLDER_CARD: dict[str, Any] = {
    "description": "Placeholder while the namespace bootstraps",
    "skills": [],
    "agent_class": "akgentic.agent.BaseAgent",
    "config": {"name": "@Bootstrap", "role": "Bootstrap"},
}

# Deliberately different from the placeholder, so an assertion on the resolved
# team can only pass if the ref really was followed.
LEAD_CARD: dict[str, Any] = {
    "description": "Coordinates the research team",
    "skills": ["coordination", "planning"],
    "agent_class": "akgentic.agent.BaseAgent",
    "config": {"name": "@Lead", "role": "Lead"},
}


def main() -> None:
    """Wire two entries together against a throwaway YAML catalog, asserting each step."""
    with tempfile.TemporaryDirectory() as tmpdir:
        catalog = Catalog(YamlEntryRepository(Path(tmpdir)))

        # 1. Anchor the namespace with an inline card, then give it an agent entry
        #    of its own and point the team's entry point at that entry instead.
        namespace = catalog.create(_team_entry()).namespace
        catalog.create(_agent_entry(namespace, LEAD_ID, LEAD_CARD))
        _set_entry_point_card(catalog, namespace, LEAD_REF)
        _assert_marker_survives_storage(catalog, namespace)
        _assert_read_side_is_populated(catalog, namespace)
        print("1. entry_point.card resolves through the ref; the marker is still stored")

        # 2. __type__ is a pin, and a mismatch names both sides.
        _assert_type_mismatch_is_refused(catalog, namespace)
        print("2. a __type__ that disagrees with the target is refused")

        # 3. A marker takes no other keys.
        _assert_sibling_key_is_refused(catalog, namespace)
        print("3. a key beside __ref__ is refused, and the message names the idiom")

        # 4. A cycle is accepted on the way in and refused on the way out.
        _assert_cycle_surfaces_on_resolve(catalog, namespace)
        print("4. the loop-closing write was accepted; the next resolve caught the cycle")

        # 5. An entry cannot be deleted while something points at it.
        _assert_delete_is_guarded(catalog, namespace)
        print("5. delete refused while referenced, then accepted once the ref was gone")


def _team_entry() -> Entry:
    """Build the anchor team entry, asking the catalog to mint its namespace."""
    return Entry(
        id=TEAM_ID,
        kind="team",  # plain string: EntryKind is a Literal alias, not an enum
        namespace=UNSET_NAMESPACE,  # replaced by a fresh UUID on create
        user_id="u1",
        model_type="akgentic.team.models.TeamCard",
        payload={
            "name": "Research Team",
            "entry_point": {"card": PLACEHOLDER_CARD, "headcount": 1, "members": []},
            "members": [],
        },
    )


def _agent_entry(namespace: str, entry_id: str, card: dict[str, Any]) -> Entry:
    """Build an ``AgentCard`` entry in ``namespace``."""
    return Entry(
        id=entry_id,
        kind="agent",
        namespace=namespace,
        user_id="u1",  # sub-entries must match the anchor entry's owner
        model_type=AGENT_CARD_TYPE,
        payload=dict(card),
    )


def _set_entry_point_card(catalog: Catalog, namespace: str, card: dict[str, Any]) -> Entry:
    """Rewrite the team's ``entry_point.card`` to ``card`` and store it.

    The marker sits at ``card``, not at ``entry_point``: ``entry_point`` is a
    ``TeamCardMember`` — a card plus a headcount plus subordinates — and only its
    ``card`` field is an ``AgentCard``. Raises on refusal, which is exactly what the
    refusal helpers below rely on.
    """
    team = catalog.get(namespace, TEAM_ID)
    entry_point = {**team.payload["entry_point"], "card": card}
    return catalog.update(
        team.model_copy(update={"payload": {**team.payload, "entry_point": entry_point}})
    )


def _assert_marker_survives_storage(catalog: Catalog, namespace: str) -> None:
    """Require the stored payload to still carry the marker, not the resolved card.

    Resolution happens on the way out. What is written down stays a pointer, which is
    why editing the target updates every entry pointing at it.
    """
    stored = catalog.get(namespace, TEAM_ID).payload["entry_point"]["card"]
    assert stored == LEAD_REF, f"the stored payload was flattened: {stored}"


def _assert_read_side_is_populated(catalog: Catalog, namespace: str) -> None:
    """Require the resolved team to carry the target's values, not the placeholder's.

    ``load_team`` is typed ``-> TeamCard``, so nothing needs narrowing here.
    ``resolve_by_id`` returns a bare ``BaseModel`` and would.
    """
    card = catalog.load_team(namespace).entry_point.card
    assert card.description == LEAD_CARD["description"], card.description
    assert card.skills == LEAD_CARD["skills"], card.skills
    assert card.config.role == "Lead", card.config.role


def _assert_type_mismatch_is_refused(catalog: Catalog, namespace: str) -> None:
    """Require a ``__type__`` disagreeing with the target's ``model_type`` to be refused.

    ``__type__`` is optional. Declaring it buys an error where the ref is *written*
    rather than where it is eventually *used*.
    """
    mismatched = {"__ref__": LEAD_ID, "__type__": "akgentic.llm.PromptTemplate"}
    try:
        _set_entry_point_card(catalog, namespace, mismatched)
    except CatalogValidationError as exc:
        expected = f"Ref '{LEAD_ID}' expected akgentic.llm.PromptTemplate, got {AGENT_CARD_TYPE}"
        assert expected in str(exc), exc
    else:
        raise AssertionError("a __type__ pin that disagreed with the target was accepted")

    # A refused write changes nothing: the good marker is still the stored one.
    assert catalog.get(namespace, TEAM_ID).payload["entry_point"]["card"] == LEAD_REF


def _assert_sibling_key_is_refused(catalog: Catalog, namespace: str) -> None:
    """Require an ordinary key sitting beside ``__ref__`` to be refused by name.

    A marker has no interior to override. A consumer that needs to vary something
    inlines its own payload and refs only the shared part; sharing a bare value goes
    through a ``NativeValue`` entry, which example 03 covers.
    """
    with_sibling = {"__ref__": LEAD_ID, "description": "a locally overridden description"}
    try:
        _set_entry_point_card(catalog, namespace, with_sibling)
    except CatalogValidationError as exc:
        expected = (
            f"ref marker to '{LEAD_ID}' carries key 'description' — a ref marker is a "
            f"pure pointer and takes no other keys. Inline the payload and reference "
            f"shared values via a NativeValue entry."
        )
        assert expected in str(exc), exc
    else:
        raise AssertionError("a ref marker carrying an ordinary key was accepted")


def _assert_cycle_surfaces_on_resolve(catalog: Catalog, namespace: str) -> None:
    """Build a two-entry loop and require ``resolve`` — not the write — to refuse it.

    The ordering is the lesson. Each write resolves against the *stored* state, and at
    the moment the second one runs the loop does not exist yet, so it is accepted. The
    cycle surfaces on the next read.
    """
    catalog.create(_agent_entry(namespace, "agent-a", LEAD_CARD))
    catalog.create(_agent_entry(namespace, "agent-b", LEAD_CARD))
    _point_metadata_at(catalog, namespace, "agent-b", "agent-a")
    closing = _point_metadata_at(catalog, namespace, "agent-a", "agent-b")
    # The write that closes the loop is accepted, and stored. Assert that rather
    # than merely comment it: the timing is this section's whole lesson.
    assert closing.payload["metadata"]["peer"] == {"__ref__": "agent-b"}, closing.payload

    try:
        # The return value is never dereferenced, so a bare BaseModel is fine here.
        catalog.resolve_by_id(namespace, "agent-a")
    except CatalogValidationError as exc:
        assert f"Reference cycle detected at ({namespace}, agent-b)" in str(exc), exc
    else:
        raise AssertionError("a reference cycle resolved instead of raising")


def _point_metadata_at(catalog: Catalog, namespace: str, entry_id: str, target_id: str) -> Entry:
    """Put a ref marker to ``target_id`` in ``entry_id``'s ``metadata`` dict.

    ``AgentCard.metadata`` is a ``dict[str, Any]``, so a marker is legal there — the
    smallest cycle this API can express without reaching for ``NativeValue``.
    """
    entry = catalog.get(namespace, entry_id)
    metadata = {"peer": {"__ref__": target_id}}
    return catalog.update(
        entry.model_copy(update={"payload": {**entry.payload, "metadata": metadata}})
    )


def _assert_delete_is_guarded(catalog: Catalog, namespace: str) -> None:
    """Require a referenced entry to be undeletable, and the legal order to work.

    The question can be asked before the attempt: ``find_references`` returns the
    referring entries, so a caller can show them rather than discovering them in an
    exception.
    """
    referrers = catalog.find_references(namespace, LEAD_ID)
    assert [e.id for e in referrers] == [TEAM_ID], referrers

    try:
        catalog.delete(namespace, LEAD_ID)
    except CatalogValidationError as exc:
        expected = (
            f"Entry '{TEAM_ID}' (kind=team) in namespace '{namespace}' references '{LEAD_ID}'"
        )
        assert expected in str(exc), exc
    else:
        raise AssertionError("a referenced entry was deleted out from under its referrer")

    # The legal order: drop the ref first — here by inlining a card again — then delete.
    _set_entry_point_card(catalog, namespace, PLACEHOLDER_CARD)
    assert catalog.find_references(namespace, LEAD_ID) == []
    catalog.delete(namespace, LEAD_ID)

    try:
        catalog.get(namespace, LEAD_ID)
    except EntryNotFoundError:
        pass  # gone, as it should be
    else:
        raise AssertionError("the entry survived a delete that reported success")


if __name__ == "__main__":
    main()
