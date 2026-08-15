"""A whole namespace as one document — export, import, and the round-trip guarantee.

Run it directly::

    python examples/04_namespace_bundles.py

Every example so far moved one entry at a time. A *bundle* is the unit in which a whole
namespace leaves and re-enters the catalog, and the guarantee that makes it usable —
export, import elsewhere, export again, and get the **same document** — is asserted here
rather than claimed.

Two temporary directories are used on purpose: a round trip through a catalog that
shares a repository with the source proves nothing.

Everything stays inside a single namespace. Cross-namespace targets get their own
``<ns>.<id>`` sections in a bundle; that is a separate topic and not this example's.

The narrative half is ``04-namespace-bundles.md``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import yaml

from akgentic.catalog import (
    UNSET_NAMESPACE,
    Catalog,
    CatalogValidationError,
    Entry,
    YamlEntryRepository,
)

REQUIRES: tuple[str, ...] = ()
"""Importable module names this example needs beyond the base install — none.

The test harness calls ``pytest.importorskip`` on every name here before invoking
``main()``. An example that ever needs an optional package must import it *inside*
``main()``: the harness reads this declaration off the already-imported module, so
a module-level import raises before the tuple can be read.
"""

TEAM_ID = "greeting-team"
AGENT_ID = "greeter"
SHARED_ID = "shared-greeting"
META_ID = "_meta"
OWNER = "u1"

# The three entries that appear under ``entries:``. ``_meta`` is deliberately absent —
# it is hoisted into the bundle header, not duplicated as an entry.
LOCAL_IDS = (TEAM_ID, AGENT_ID, SHARED_ID)

# The eight top-level keys of the bundle wire format, as a closed set.
BUNDLE_ROOT_KEYS = {
    "namespace",
    "user_id",
    "name",
    "description",
    "properties",
    "shareable",
    "public",
    "entries",
}

# The namespace metadata the header is projected from. A header is emitted only when
# something forces it — a non-empty name, description or properties, or a True boolean —
# so an all-default meta would export the legacy three-key shape and there would be no
# header to inspect.
META_PAYLOAD: dict[str, Any] = {
    "name": "Greeting Team",
    "description": "A one-agent namespace used to demonstrate bundles",
    "properties": {"tier": "demo", "owner": "platform"},
    "shareable": False,
    "public": True,
}

# The marker carried by the agent, pointing at the NativeValue entry. Its survival is
# the interesting part of the round trip: a bundle that lost it would still be a valid
# document, just a different one.
GREETING_REF: dict[str, Any] = {"__ref__": SHARED_ID}

# Both closed key sets, refused by name. ``sharable`` is the motivating misprint: it
# reads as correct and would leave the namespace silently un-shareable.
EXPECTED_MISPRINTS = {
    (
        "bundle root has unknown key 'sharable' — expected one of: description, entries, "
        "name, namespace, properties, public, shareable, user_id"
    ),
    (
        f"entry '{AGENT_ID}' has unknown key 'descriptin' — expected one of: description, "
        f"kind, model_type, payload"
    ),
}

ANCHOR_CARD: dict[str, Any] = {
    "description": "Placeholder while the namespace bootstraps",
    "skills": [],
    "agent_class": "akgentic.agent.BaseAgent",
    "config": {"name": "@Bootstrap", "role": "Bootstrap"},
}


def main() -> None:
    """Export a namespace, round-trip it through a second catalog, and refuse two misprints."""
    with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as fresh_dir:
        catalog = Catalog(YamlEntryRepository(Path(source_dir)))
        namespace = _build_namespace(catalog)

        document = catalog.export_namespace_yaml(namespace)
        _assert_header_is_hoisted(catalog, namespace, document)
        print("1. the header carries the _meta values, and _meta is not under entries:")

        fresh = Catalog(YamlEntryRepository(Path(fresh_dir)))
        _assert_round_trip_is_identical(catalog, fresh, namespace, document)
        print("2. export -> import into a fresh catalog -> export produced the same document")

        bad_document = _with_two_misprints(document)
        _assert_both_misprints_refused(catalog, bad_document)
        print("3. an unknown root key and an unknown entry key were both reported in one pass")

        _assert_dry_run_reports_without_writing(catalog, namespace, document, bad_document)
        print("4. the dry run reported without raising, wrote nothing, and passes a good bundle")


# --- Building the namespace -------------------------------------------------------


def _build_namespace(catalog: Catalog) -> str:
    """Create a namespace worth exporting, and return its identifier.

    Order matters: the team entry mints the namespace, so everything else is created
    against the identifier it returns.
    """
    namespace = catalog.create(_team_entry()).namespace
    catalog.create(_meta_entry(namespace))
    catalog.create(_greeting_entry(namespace))
    catalog.create(_agent_entry(namespace))
    return namespace


def _team_entry() -> Entry:
    """Build the anchor team entry, asking the catalog to mint its namespace."""
    return Entry(
        id=TEAM_ID,
        kind="team",  # plain string: EntryKind is a Literal alias, not an enum
        namespace=UNSET_NAMESPACE,  # replaced by a fresh UUID on create
        user_id=OWNER,
        model_type="akgentic.team.models.TeamCard",
        payload={
            "name": "Greeting Team",
            "entry_point": {"card": ANCHOR_CARD, "headcount": 1, "members": []},
            "members": [],
        },
    )


def _meta_entry(namespace: str) -> Entry:
    """Build the namespace's ``_meta`` entry — the entry the bundle header comes from.

    A namespace may be anchored by a team **or** a meta; this one has both. The id
    ``_meta`` is canonical and there is at most one per namespace.
    """
    return Entry(
        id=META_ID,
        kind="meta",
        namespace=namespace,
        user_id=OWNER,
        model_type="akgentic.catalog.models.namespace_meta.NamespaceMeta",
        description="Namespace metadata",
        payload=dict(META_PAYLOAD),
    )


def _greeting_entry(namespace: str) -> Entry:
    """Build the ``NativeValue`` entry the agent points at (example 03's idiom)."""
    return Entry(
        id=SHARED_ID,
        kind="prompt",
        namespace=namespace,
        user_id=OWNER,
        model_type="akgentic.catalog.NativeValue",
        description="Shared greeting body",
        payload={"value": "You are {role}. Greet the user warmly."},
    )


def _agent_entry(namespace: str) -> Entry:
    """Build the one agent, carrying a marker so the bundle contains a ref to preserve."""
    return Entry(
        id=AGENT_ID,
        kind="agent",
        namespace=namespace,
        user_id=OWNER,  # sub-entries must match the anchor entry's owner
        model_type="akgentic.core.AgentCard",
        description="Greets the user",
        payload={
            "description": "Greets the user",
            "agent_class": "akgentic.agent.BaseAgent",
            "skills": [],
            "config": {
                "name": "@Greeter",
                "role": "Greeter",
                "prompt": {
                    "template": dict(GREETING_REF),
                    "params": {"role": "a greeter"},
                },
            },
        },
    )


# --- Assertions -------------------------------------------------------------------


def _marker_at(document: str) -> Any:
    """Return the ref marker stored at the agent's ``config.prompt.template``."""
    root = yaml.safe_load(document)
    return root["entries"][AGENT_ID]["payload"]["config"]["prompt"]["template"]


def _assert_header_is_hoisted(catalog: Catalog, namespace: str, document: str) -> None:
    """Require the header to be a projection of the ``_meta`` entry, structurally.

    Parsed and compared as data, not as substrings: a substring check would pass on a
    document that happened to mention the right words in the wrong places.

    The expected values are read back off the stored ``_meta`` entry rather than off this
    module's constants, so what is being checked is *the export projects the meta entry*
    — not *the export echoes a literal this file also wrote*.
    """
    root = yaml.safe_load(document)
    assert set(root) == BUNDLE_ROOT_KEYS, sorted(root)
    assert root["namespace"] == namespace, root["namespace"]

    stored_meta = catalog.get(namespace, META_ID).payload
    for field in ("name", "description", "properties", "shareable", "public"):
        assert root[field] == stored_meta[field], (field, root[field], stored_meta[field])

    # The meta entry is hoisted into the header, never duplicated as an entry.
    assert META_ID not in root["entries"], sorted(root["entries"])
    assert set(root["entries"]) == set(LOCAL_IDS), sorted(root["entries"])


def _assert_round_trip_is_identical(
    catalog: Catalog,
    fresh: Catalog,
    namespace: str,
    document: str,
) -> None:
    """Export, import into a fresh catalog over its own directory, export again, compare.

    The equality is the guarantee, but an equality alone is a weak assertion — two empty
    documents are equal, and so are two identically broken ones. So the document is first
    pinned down: non-empty, carrying the ids the namespace holds, and carrying the ref
    marker intact. Only then does string identity mean what it looks like it means.

    The final check covers the half the string comparison cannot localise: ``_meta`` never
    travels as an entry, so its reconstruction in the fresh catalog is proved directly.
    """
    assert document.strip(), "the export produced an empty document"
    assert _marker_at(document) == GREETING_REF, _marker_at(document)

    imported = fresh.import_namespace_yaml(document)
    # ``_meta`` is an atomic side effect of the import, not one of the bundle's entries.
    assert {e.id for e in imported} == set(LOCAL_IDS), sorted(e.id for e in imported)

    second = fresh.export_namespace_yaml(namespace)
    assert _marker_at(second) == GREETING_REF, _marker_at(second)
    assert second == document, "the round trip changed the document"

    rebuilt_meta = fresh.get(namespace, META_ID)
    assert rebuilt_meta.payload == catalog.get(namespace, META_ID).payload, rebuilt_meta.payload


def _with_two_misprints(document: str) -> str:
    """Return ``document`` with one unknown bundle-root key and one unknown entry key."""
    root = yaml.safe_load(document)
    root["sharable"] = True  # the misprint that reads as correct
    root["entries"][AGENT_ID]["descriptin"] = "a misprinted description"
    return yaml.safe_dump(root, sort_keys=False, allow_unicode=True)


def _assert_both_misprints_refused(catalog: Catalog, bad_document: str) -> None:
    """Require both closed key sets to refuse, and both findings to arrive in one pass.

    Asserted on ``exc.errors`` — the list — rather than on ``str(exc)``, because "both
    reported together" is the claim. A parser that stopped at the first unknown key would
    satisfy a substring check on the message of two separate attempts.
    """
    try:
        catalog.import_namespace_yaml(bad_document)
    except CatalogValidationError as exc:
        assert EXPECTED_MISPRINTS <= set(exc.errors), exc.errors
    else:
        raise AssertionError("a bundle carrying two unknown keys was imported")


def _assert_dry_run_reports_without_writing(
    catalog: Catalog,
    namespace: str,
    document: str,
    bad_document: str,
) -> None:
    """Require the dry run to report rather than raise, to write nothing, and to discriminate.

    ``validate_namespace_yaml`` never raises — the findings come back in a report, so a
    caller can render all of them instead of surfacing one exception at a time. That makes
    ``report.ok`` the thing to assert, and a report only ever seen failing would not show
    that it can tell a good bundle from a bad one; hence the second, passing run.
    """
    report = catalog.validate_namespace_yaml(bad_document)
    assert report.ok is False, report
    assert report.namespace is None, report.namespace
    assert EXPECTED_MISPRINTS <= set(report.global_errors), report.global_errors

    # Neither the refused import nor the dry run touched the namespace.
    assert catalog.export_namespace_yaml(namespace) == document, "the namespace was modified"

    good = catalog.validate_namespace_yaml(document)
    assert good.ok is True, (good.global_errors, good.entry_issues)


if __name__ == "__main__":
    main()
