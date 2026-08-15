"""The shape of an entry — its key, its ``model_type``, and the whole CRUD surface.

Run it directly::

    python examples/01_first_entry.py

``00_hello_catalog`` is the 30-second tour. This one is the shape of the thing it
toured: what addresses an entry, what constrains its payload class, and what the
catalog refuses. Every outcome is asserted, so an API change turns the test suite
red instead of leaving the walkthrough quietly wrong. The narrative half of this
example is ``01-first-entry.md``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from akgentic.core import AgentCard
from pydantic import ValidationError

from akgentic.catalog import (
    UNSET_NAMESPACE,
    Catalog,
    CatalogValidationError,
    Entry,
    EntryQuery,
    YamlEntryRepository,
)

REQUIRES: tuple[str, ...] = ()
"""Importable module names this example needs beyond the base install — none.

The test harness calls ``pytest.importorskip`` on every name here before invoking
``main()``. An example that ever needs an optional package must import it *inside*
``main()``: the harness reads this declaration off the already-imported module, so
a module-level import raises before the tuple can be read.
"""

# The anchor entry's card is written inline because a team entry bootstraps its
# namespace — there is nothing to point at yet. Example 02 starts pointing.
LEAD_CARD: dict[str, Any] = {
    "description": "Coordinates the team",
    "skills": ["coordination"],
    "agent_class": "akgentic.agent.BaseAgent",
    "config": {"name": "@Lead", "role": "Lead"},
}


def main() -> None:
    """Walk the entry lifecycle against a throwaway YAML catalog, asserting each step."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)  # Path, not str — the repository stores what it is given.
        catalog = Catalog(YamlEntryRepository(root))

        # 1. Setup, recapped from 00 in one line: the sentinel namespace is minted.
        namespace = catalog.create(_team_entry()).namespace
        assert namespace != UNSET_NAMESPACE, "the sentinel namespace was not replaced"
        print(f"1. minted namespace {namespace}")

        # 2. A sub-entry of its own kind, and the four facts about how it is addressed.
        catalog.create(_agent_entry(namespace))
        _assert_key_facts(catalog, root, namespace)
        print("2. kind partitions, the namespace scopes ids, get needs no kind")

        # 3. model_type is checked when the Entry is built, before any import.
        _assert_allowlist_is_checked_at_construction()
        print("3. a model_type outside the allowlist never becomes an Entry")

        # 4. A namespace takes sub-entries only once something anchors it.
        _assert_anchor_is_required(catalog)
        print("4. an unanchored namespace refuses a sub-entry")

        # 5. update: change what you name, keep everything you did not.
        _assert_update_is_surgical(catalog, namespace)
        print("5. updated entry re-read; the fields it never mentioned survived")

        # 6. Listing: the whole namespace, then one filtered query.
        _assert_listing(catalog, namespace)
        print("6. listed the namespace, then filtered it by kind")

        # 7. A misprint is refused by its dotted path; an absent key is not a misprint.
        _assert_nested_misprint_is_refused(catalog, namespace)
        _assert_absent_key_is_accepted(catalog, namespace)
        print("7. 'config.temperatur' refused; the same payload without it accepted")

        # 8. delete: back down to the anchor.
        catalog.delete(namespace, "tuned-agent")
        catalog.delete(namespace, "lead-agent")
        left = [e.id for e in catalog.list_by_namespace(namespace)]
        assert left == ["research-team"], f"only the anchor should remain, found {left}"
        print("8. deleted both sub-entries; only the anchor is left")


def _team_entry() -> Entry:
    """Build the anchor team entry, asking the catalog to mint its namespace."""
    return Entry(
        id="research-team",
        kind="team",  # plain string: EntryKind is a Literal alias, not an enum
        namespace=UNSET_NAMESPACE,  # replaced by a fresh UUID on create
        user_id="u1",
        model_type="akgentic.team.models.TeamCard",
        payload={
            "name": "Research Team",
            "entry_point": {"card": LEAD_CARD, "headcount": 1, "members": []},
            "members": [],
        },
    )


def _agent_entry(namespace: str) -> Entry:
    """Build an agent entry belonging to ``namespace``."""
    return Entry(
        id="lead-agent",
        kind="agent",
        namespace=namespace,
        user_id="u1",  # sub-entries must match the anchor entry's owner
        model_type="akgentic.core.AgentCard",
        payload=dict(LEAD_CARD),
    )


def _assert_key_facts(catalog: Catalog, root: Path, namespace: str) -> None:
    """Pin the four facts the phrase "(kind, namespace, id) compound key" hides.

    All three parts address an entry, but they do not all scope its id: ``kind``
    partitions and filters, while the *namespace alone* is what makes an id unique.
    """
    # (a) kind picks the on-disk partition: <root>/<namespace>/<kind>/<id>.yaml.
    assert (root / namespace / "agent" / "lead-agent.yaml").is_file()

    # (b) an id is unique per NAMESPACE, not per (namespace, kind) — a model entry
    #     cannot take an id an agent entry already holds.
    clash = Entry(
        id="lead-agent",
        kind="model",
        namespace=namespace,
        user_id="u1",
        model_type="akgentic.llm.ModelConfig",
        payload={"provider": "openai", "model": "gpt-4o"},
    )
    try:
        catalog.create(clash)
    except CatalogValidationError as exc:
        assert f"Entry ({namespace}, lead-agent) already exists" in str(exc), exc
    else:
        raise AssertionError("two entries of different kinds shared an id in one namespace")

    # (c) get takes no kind — one id, one entry, whatever kind it turns out to be.
    assert catalog.get(namespace, "lead-agent").kind == "agent"

    # (d) the namespace is the boundary, so the same id elsewhere is a different entry.
    other = catalog.create(_team_entry()).namespace
    assert other != namespace, "each team entry mints its own namespace"
    twin = catalog.create(_agent_entry(other))
    assert twin.namespace == other, "the twin belongs to the second namespace"


def _assert_allowlist_is_checked_at_construction() -> None:
    """Require a non-allowlisted ``model_type`` to fail at ``Entry(...)``, not at create.

    No ``Catalog`` is involved — that is the point. The check sits on the field, so a
    class path the deployment never allowed cannot reach a repository, and nothing is
    imported while finding out.
    """
    try:
        Entry(
            id="widget",
            kind="tool",
            namespace="acme",
            user_id="u1",
            model_type="acme.core.models.Widget",
            payload={},
        )
    except ValidationError as exc:
        assert "outside allowlist" in str(exc), exc
    else:
        raise AssertionError("a model_type outside the allowlist was accepted")


def _assert_anchor_is_required(catalog: Catalog) -> None:
    """Require a sub-entry written into an unanchored namespace to be refused."""
    orphan = Entry(
        id="stray-agent",
        kind="agent",
        namespace="unanchored",
        user_id="u1",
        model_type="akgentic.core.AgentCard",
        payload=dict(LEAD_CARD),
    )
    try:
        catalog.create(orphan)
    except CatalogValidationError as exc:
        assert "Namespace 'unanchored' has no team entry and no meta entry" in str(exc), exc
    else:
        raise AssertionError("a sub-entry created a namespace with nothing anchoring it")


def _assert_update_is_surgical(catalog: Catalog, namespace: str) -> None:
    """Update two things through ``model_copy`` and require the rest to survive.

    ``update`` re-runs the whole write pipeline on the entry it is given, so the entry
    is *derived* from the stored one rather than rebuilt field by field — anything left
    out of a hand-written reconstruction would be silently dropped.
    """
    stored = catalog.get(namespace, "lead-agent")
    catalog.update(
        stored.model_copy(
            update={
                "description": "Coordinates the research team",
                "payload": {**stored.payload, "skills": ["coordination", "planning"]},
            }
        )
    )
    fresh = catalog.get(namespace, "lead-agent")
    assert fresh.description == "Coordinates the research team"
    assert fresh.payload["skills"] == ["coordination", "planning"]
    # Never named by the update, and still exactly as they were.
    assert fresh.payload["config"]["role"] == "Lead"
    assert fresh.payload["agent_class"] == "akgentic.agent.BaseAgent"


def _assert_listing(catalog: Catalog, namespace: str) -> None:
    """Read the namespace whole, then through one ``EntryQuery`` filter."""
    everything = sorted(e.id for e in catalog.list_by_namespace(namespace))
    assert everything == ["lead-agent", "research-team"], everything

    agents = catalog.list(EntryQuery(namespace=namespace, kind="agent"))
    assert [e.id for e in agents] == ["lead-agent"], agents


def _assert_nested_misprint_is_refused(catalog: Catalog, namespace: str) -> None:
    """Require a misprint nested inside ``config`` to be reported by its dotted path.

    ``00_hello_catalog`` shows a flat misprint. This one is a level down, and the
    message locates it — ``config.temperatur`` rather than merely "somewhere in this
    AgentCard". The model named is the entry's *declared* ``model_type``, not the
    nested class that owns the field.
    """
    misprinted = _tuned_agent_entry(
        namespace, {"name": "@Tuned", "role": "Tuned", "temperatur": 0.2}
    )
    try:
        catalog.create(misprinted)
    except CatalogValidationError as exc:
        assert ("unknown key 'config.temperatur' — not a field of akgentic.core.AgentCard") in str(
            exc
        ), exc
    else:
        raise AssertionError("a misprinted nested payload key was accepted")


def _assert_absent_key_is_accepted(catalog: Catalog, namespace: str) -> None:
    """Require the same payload, minus the offending key, to be accepted.

    The check separates the two cases by what the author *wrote*: a key the model never
    accepted is an error, and leaving a key out is not. Omitting a declared field
    remains the supported way to let it take its default — ``squad_id`` is absent from
    what is stored, and comes back as its declared ``None`` when the entry is resolved.
    """
    created = catalog.create(_tuned_agent_entry(namespace, {"name": "@Tuned", "role": "Tuned"}))
    assert created.payload["config"] == {"name": "@Tuned", "role": "Tuned"}

    resolved = catalog.resolve_by_id(namespace, "tuned-agent")
    assert isinstance(resolved, AgentCard), type(resolved)  # resolve_by_id returns BaseModel
    assert resolved.config.squad_id is None, "an omitted field takes its declared default"


def _tuned_agent_entry(namespace: str, config: dict[str, Any]) -> Entry:
    """Build a second agent entry whose ``config`` sub-dict is the caller's to choose."""
    return Entry(
        id="tuned-agent",
        kind="agent",
        namespace=namespace,
        user_id="u1",
        model_type="akgentic.core.AgentCard",
        payload={**LEAD_CARD, "config": config},
    )


if __name__ == "__main__":
    main()
