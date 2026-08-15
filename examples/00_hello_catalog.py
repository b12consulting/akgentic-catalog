"""Hello, catalog — mint a namespace, read it back, and watch a misprint get refused.

Run it directly::

    python examples/00_hello_catalog.py

Every outcome this script demonstrates is asserted, so an API change that breaks the
walkthrough turns the test suite red instead of printing something wrong. The narrative
half of this example is ``00-hello-catalog.md``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

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
``main()``. An empty tuple keeps this example on the happy path while exercising the
mechanism a service-backed example will need, and it keeps the example runnable
outside pytest: the harness skips, the example never calls ``pytest.skip``.
"""

# A team entry anchors its namespace, so it has to be self-contained: the entry_point
# card is written inline because there is nothing to point at yet. Later examples
# replace this with a ``__ref__`` marker.
LEAD_CARD = {
    "description": "Coordinates the team",
    "skills": ["coordination"],
    "agent_class": "akgentic.agent.BaseAgent",
    "config": {"name": "@Lead", "role": "Lead"},
}


def main() -> None:
    """Run the walkthrough against a throwaway YAML catalog, asserting each step."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # A repository is a directory of YAML files; the Catalog service is the
        # only thing that talks to it. Note Path, not str — the constructor stores
        # what it is given.
        catalog = Catalog(YamlEntryRepository(Path(tmpdir)))

        # 1. Creating a team entry with the sentinel namespace mints a fresh one.
        team = catalog.create(_team_entry())
        assert team.namespace != UNSET_NAMESPACE, "the sentinel namespace was not replaced"
        assert team.namespace, "a minted namespace must be non-empty"
        namespace = team.namespace
        print(f"1. minted namespace {namespace}")

        # 2. Read it back: the payload round-trips through Pydantic unchanged.
        stored = catalog.get(namespace, "research-team")
        assert stored.id == "research-team"
        assert stored.kind == "team"
        assert stored.payload["name"] == "Research Team"
        card = stored.payload["entry_point"]["card"]
        assert card["description"] == LEAD_CARD["description"]
        assert card["skills"] == LEAD_CARD["skills"]
        assert card["config"]["role"] == "Lead"
        print("2. payload round-tripped intact")

        # 3. A key the model never accepted is an error, not a silent drop.
        _assert_misprint_is_refused(catalog, namespace)
        print("3. misprinted payload key refused")

        # 4. Delete, and the entry is gone — get raises rather than returning None.
        catalog.delete(namespace, "research-team")
        try:
            catalog.get(namespace, "research-team")
        except EntryNotFoundError:
            print("4. deleted entry is gone")
        else:
            raise AssertionError("get() returned a deleted entry instead of raising")


def _team_entry() -> Entry:
    """Build the team entry, asking the catalog to mint its namespace."""
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


def _assert_misprint_is_refused(catalog: Catalog, namespace: str) -> None:
    """Write an agent entry carrying a misprinted key; require it to be rejected.

    ``skils`` is not a field of ``AgentCard``. A plain Pydantic model would drop it
    without comment and the author would never learn the skills list was ignored;
    the catalog refuses the write instead.
    """
    misprinted = Entry(
        id="lead-agent",
        kind="agent",
        namespace=namespace,
        user_id="u1",  # sub-entries must match the team entry's owner
        model_type="akgentic.core.AgentCard",
        payload={**LEAD_CARD, "skils": ["coordination"]},
    )
    try:
        catalog.create(misprinted)
    except CatalogValidationError as exc:
        assert "skils" in str(exc), f"the error should name the misprinted key: {exc}"
    else:
        raise AssertionError("a misprinted payload key was accepted")


if __name__ == "__main__":
    main()
