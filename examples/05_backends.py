"""One protocol, three backends — the same walkthrough against every one that is present.

Run it directly::

    python examples/05_backends.py

Examples ``00``-``04`` all used ``YamlEntryRepository``. The package ships three
backends behind one ``EntryRepository`` protocol, and the README's claim that *which
database is behind it is a deployment choice* is what this example makes executable:
**one** walkthrough function, run unchanged against every backend present on the
machine, and its observations compared for equality.

The identity is the lesson. A walkthrough rewritten per backend would prove nothing —
only that three pieces of code can each be made to pass.

YAML always runs. Mongo runs when ``pymongo`` and ``mongomock`` import. Postgres runs
when ``nagra`` and ``psycopg`` import, ``DB_CONN_STRING_PERSISTENCE`` names a reachable
database, and the schema bootstrap succeeds. Every arm that does not run prints why.

The narrative half is ``05-backends.md``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from akgentic.core import AgentCard
from pydantic import BaseModel

from akgentic.catalog import (
    Catalog,
    CatalogValidationError,
    Entry,
    EntryRepository,
    YamlEntryRepository,
)

REQUIRES: tuple[str, ...] = ()
"""Importable module names this example needs beyond the base install — none, on purpose.

The harness calls ``pytest.importorskip`` on every name here **before** invoking
``main()``, so a non-empty tuple would skip the *whole* example on a machine without
``pymongo`` — taking the YAML walkthrough, which has to run everywhere, down with it.
This example degrades one backend at a time inside ``main()`` instead.

The rule the tuple exists for still binds. The harness reads this declaration off the
already-imported module, so every optional import below happens **inside** ``main()``
or a helper it calls. ``akgentic.catalog`` re-exports ``MongoEntryRepository`` /
``PostgresEntryRepository`` behind ``try: … except ImportError: pass``, so on an install
without the extra those names do not exist and a module-level import of one would raise
at load time — exactly the failure this rule prevents.
"""

YAML = "yaml"
MONGO = "mongo"
POSTGRES = "postgres"

POSTGRES_DSN_ENV = "DB_CONN_STRING_PERSISTENCE"
"""The supply channel the README documents and the one CI sets. Optional, always."""

NAMESPACE = "example-05-backends"
"""One fixed namespace string, shared by every backend — deliberately not minted.

The delete-guard blocker message embeds the namespace, so two backends that minted
their own UUID would report different messages for a reason that is not a parity
failure. Minting is example ``01``'s lesson and is not re-taught here.
"""

OWNER = "u1"
META_ID = "_meta"
SHARED_ID = "shared-batch-size"
REFERRER_IDS = ("analyst", "reviewer")
ALL_IDS = (META_ID, SHARED_ID, *REFERRER_IDS)

SHARED_BATCH_SIZE = 200

AGENT_CLASS = "akgentic.core.agent.Akgent"
"""Resolved at write time by ``AgentCard``, so it must name an importable class.

Deliberately a class from ``akgentic-core``, a hard dependency of this package:
examples ``00``-``04`` name ``akgentic.agent.BaseAgent`` and so quietly need the
``dev`` extra. This one does not.
"""

BATCH_SIZE_REF: dict[str, Any] = {"__ref__": SHARED_ID}

EXPECTED_BLOCKERS = frozenset(
    f"Entry '{referrer_id}' (kind=agent) in namespace '{NAMESPACE}' references '{SHARED_ID}'"
    for referrer_id in REFERRER_IDS
)


class _Observations(BaseModel):
    """What one walkthrough saw, in a shape two backends' results can be compared in.

    A Pydantic model rather than a ``dict[str, Any]``: this crosses the boundary between
    the walkthrough and the parity check, and the whole point of the check is that the
    two sides agree about every field it holds.

    Every collection field is an order-independent **sorted tuple**.
    ``PostgresEntryRepository.list`` emits no ``ORDER BY``, so comparing lists would turn
    this example red for a reason that has nothing to do with the contract.
    """

    spliced_batch_size: int
    referrer_ids: tuple[str, ...]
    blocker_messages: tuple[str, ...]
    namespace_ids: tuple[str, ...]


def main() -> None:
    """Run one walkthrough per available backend and require the reports to agree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backends, skipped = _available_backends(Path(tmpdir))
        for reason in skipped:
            print(f"   (not run — {reason})")

        reports = {label: _run_arm(label, repository) for label, repository in backends}
        print(f"1. the same walkthrough ran against: {', '.join(reports)}")

        _assert_reports_agree(reports)
        print("2. every backend that ran reported exactly the same observations")


# --- Which backends are here ------------------------------------------------------


def _available_backends(yaml_root: Path) -> tuple[list[tuple[str, EntryRepository]], list[str]]:
    """Return every backend present, plus one skip reason per backend that is not.

    The annotation is the conformance proof. ``EntryRepository`` is a structural
    protocol with **no** ``@runtime_checkable`` decorator, so ``isinstance(repo,
    EntryRepository)`` raises ``TypeError`` and must never be written. ``mypy --strict``
    — which CI runs over this directory — checks the shape; the assertions below check
    the semantics.

    The probes are forgiving on purpose: a missing driver or an unreachable database is
    a skip, not a failure. The walkthrough itself is never wrapped, so a genuine contract
    break stays red.
    """
    backends: list[tuple[str, EntryRepository]] = [(YAML, YamlEntryRepository(yaml_root))]
    skipped: list[str] = []
    for label, probe in ((MONGO, _mongo_repository), (POSTGRES, _postgres_repository)):
        repository, reason = probe()
        if repository is None:
            skipped.append(f"{label}: {reason}")
        else:
            backends.append((label, repository))
    return backends, skipped


def _mongo_repository() -> tuple[EntryRepository | None, str]:
    """Build a Mongo repository over an in-memory ``mongomock`` collection.

    Both imports are local — see ``REQUIRES``. ``mongomock`` is the mechanism the
    package's own suite already uses (``tests/v2/conftest.py``), it needs no Docker and
    no server, and CI has no Mongo service. The trade is worth stating plainly: **the
    repository code exercised here is the real one, the server is not.**

    ``MongoEntryRepository`` takes a live collection, not a config and not a client — it
    owns none of that lifecycle. In a deployment the two lines this replaces are::

        cfg = MongoCatalogConfig(connection_string="mongodb://localhost:27017",
                                 database="akgentic")
        client = cfg.create_client()
        collection = cfg.get_collection(client, cfg.catalog_entries_collection)
    """
    try:
        import mongomock

        from akgentic.catalog import MongoEntryRepository
    except ImportError as exc:
        return None, f"not importable — {exc}"
    # Annotated ``Any`` because a mongomock client is generic over its document type and
    # mypy cannot infer the parameter here. The repository takes the collection as it is.
    collection: Any = mongomock.MongoClient()["akgentic_examples"]["catalog_entries"]
    return MongoEntryRepository(collection), ""


def _postgres_repository() -> tuple[EntryRepository | None, str]:
    """Build a Postgres repository when the environment names a reachable database.

    ``PostgresEntryRepository`` takes a **bare DSN string**, not a config object;
    ``PostgresCatalogConfig`` exists to validate the DSN and to feed ``init_db``, which
    creates the table and is idempotent. The repository constructor never issues DDL.

    The broad ``except`` is the forgiving half of the probe: a DSN pointing at nothing is
    an arm that does not run, not a failed example.
    """
    dsn = os.environ.get(POSTGRES_DSN_ENV, "")
    if not dsn:
        return None, f"{POSTGRES_DSN_ENV} is not set"
    try:
        from akgentic.catalog import PostgresCatalogConfig, PostgresEntryRepository, init_db

        init_db(PostgresCatalogConfig(connection_string=dsn))
        return PostgresEntryRepository(dsn), ""
    except Exception as exc:
        return None, f"{POSTGRES_DSN_ENV} is set but unusable — {type(exc).__name__}: {exc}"


def _run_arm(label: str, repository: EntryRepository) -> _Observations:
    """Run one backend's walkthrough, leaving its store as it was found.

    Only Postgres needs the sweep. YAML and Mongo each got a private, throwaway store,
    while the Postgres arm writes into the one ``catalog_entries`` table the whole suite
    shares — so it clears its own namespace before it starts (an earlier run may have
    been interrupted) and again when it finishes.

    The sweep goes through ``repository.delete`` and **not** ``catalog.delete``: the
    delete guard would rightly refuse to remove a referenced entry. That is the division
    this example is about — the guard is a ``Catalog`` policy, and the repository
    underneath is a plain store.
    """
    if label != POSTGRES:
        return _walkthrough(repository)
    _clear_namespace(repository)
    try:
        return _walkthrough(repository)
    finally:
        _clear_namespace(repository)


def _clear_namespace(repository: EntryRepository) -> None:
    """Remove every entry this example owns, straight through the repository."""
    for entry in repository.list_by_namespace(NAMESPACE):
        repository.delete(entry.namespace, entry.id)


# --- The one walkthrough ----------------------------------------------------------


def _walkthrough(repository: EntryRepository) -> _Observations:
    """Build the namespace, exercise the contract, and report what was observed.

    Takes the **repository** rather than a ``Catalog`` and builds its own service around
    it. That is the point: the repository is the only thing that varies, and every line
    above it is the same code on every backend.

    ``EntryQuery(description_contains=...)`` is deliberately absent — it is
    case-*insensitive* on Postgres (``ILIKE``) and case-*sensitive* on YAML and Mongo.
    Asserting parity over it would make this example red for a live divergence this story
    is not allowed to fix. See ``05-backends.md``.
    """
    catalog = Catalog(repository)
    catalog.create(_meta_entry())
    catalog.create(_shared_entry())
    for referrer_id in REFERRER_IDS:
        catalog.create(_referrer_entry(referrer_id))

    referrers = catalog.find_references(NAMESPACE, SHARED_ID)
    blockers = _refused_delete(catalog)

    # A refused delete changes nothing — the shared entry is still there, and still
    # holds the wrapper rather than the bare scalar.
    survivor = catalog.get(NAMESPACE, SHARED_ID)
    assert survivor.payload == {"value": SHARED_BATCH_SIZE}, survivor.payload

    return _Observations(
        spliced_batch_size=_spliced_batch_size(catalog, REFERRER_IDS[0]),
        referrer_ids=tuple(sorted(e.id for e in referrers)),
        blocker_messages=tuple(sorted(blockers)),
        namespace_ids=tuple(sorted(e.id for e in catalog.list_by_namespace(NAMESPACE))),
    )


def _spliced_batch_size(catalog: Catalog, entry_id: str) -> int:
    """Resolve ``entry_id`` and return the value its marker pulled in from elsewhere.

    Both assertions earn their place. ``resolve_by_id`` is declared ``-> BaseModel``, so
    the first is what lets the attribute be read at all under ``mypy --strict`` — and it
    is also a claim: the catalog built an ``AgentCard``, not a dict. The second is not
    shadowed by any annotation, because ``AgentCard.metadata`` is ``dict[str, Any]``: an
    unwrapped ``{"value": 200}`` would sit there quite happily.
    """
    resolved = catalog.resolve_by_id(NAMESPACE, entry_id)
    assert isinstance(resolved, AgentCard), f"resolved to {type(resolved).__name__}"
    spliced = resolved.metadata["limits"]["batch_size"]
    assert isinstance(spliced, int), f"expected the bare int, got {type(spliced).__name__}"
    return spliced


def _refused_delete(catalog: Catalog) -> list[str]:
    """Attempt the delete the guard must refuse, and return every blocker it named.

    ``try / except / else`` rather than ``pytest.raises``: an example has to run with no
    pytest in sight. One attempt reports *all* the referrers, not one per attempt.
    """
    try:
        catalog.delete(NAMESPACE, SHARED_ID)
    except CatalogValidationError as exc:
        blockers = list(exc.errors)
    else:
        raise AssertionError("the shared entry was deleted out from under its referrers")
    return blockers


# --- Building the namespace -------------------------------------------------------


def _meta_entry() -> Entry:
    """Build the ``_meta`` anchor.

    Nothing else may be created in a namespace until an anchor exists, and a
    ``kind="meta"`` entry is one — no team card, no placeholder, nothing that would
    differ between backends. ``shareable`` stays ``False`` so the delete guard below is
    the namespace-local one; a shareable namespace widens it to a cross-namespace scan,
    which is a different lesson.
    """
    return Entry(
        id=META_ID,
        kind="meta",
        namespace=NAMESPACE,
        user_id=OWNER,
        model_type="akgentic.catalog.models.namespace_meta.NamespaceMeta",
        description="Namespace metadata",
        payload={
            "name": "Backend parity walkthrough",
            "description": "One namespace, rebuilt identically on every backend",
            "properties": {},
            "shareable": False,
            "public": False,
        },
    )


def _shared_entry() -> Entry:
    """Build the one entry both referrers point at — example ``03``'s ``NativeValue``."""
    return Entry(
        id=SHARED_ID,
        kind="model",
        namespace=NAMESPACE,
        user_id=OWNER,
        model_type="akgentic.catalog.NativeValue",
        description="Batch size shared by every agent in the namespace",
        payload={"value": SHARED_BATCH_SIZE},
    )


def _referrer_entry(entry_id: str) -> Entry:
    """Build one agent whose payload points at the shared entry, three levels down.

    The depth is deliberate. A marker sitting at the payload root would be found by a
    walker that never recursed, and the recursion is precisely the half of
    ``find_references`` all three backends share — Mongo and Postgres import the very
    walker the YAML backend defines. What differs between them is the *fetch* underneath,
    and that is what this example actually tests.
    """
    return Entry(
        id=entry_id,
        kind="agent",
        namespace=NAMESPACE,
        user_id=OWNER,  # sub-entries must match the anchor entry's owner
        model_type="akgentic.core.AgentCard",
        description=f"Reads the shared batch size ({entry_id})",
        payload={
            "description": f"Reads the shared batch size ({entry_id})",
            "agent_class": AGENT_CLASS,
            "skills": [],
            "config": {"name": f"@{entry_id}", "role": entry_id},
            "metadata": {"limits": {"batch_size": dict(BATCH_SIZE_REF)}},
        },
    )


# --- The parity assertion, and its honest limits ----------------------------------


def _assert_reports_agree(reports: dict[str, _Observations]) -> None:
    """Pin the YAML report down, then require every other report to equal it.

    An equality on its own is a weak claim — three identical *empty* reports satisfy one
    — so the YAML report is checked for substance first: the value really came through
    the marker, both referrers really were found, and the delete really was refused
    naming each of them.

    **The honest limit.** On a machine with neither optional backend the final comparison
    is over a single report and proves nothing there: it compares YAML with itself. That
    is why the substance checks above are not optional, and why CI matters — it installs
    ``[dev,api,cli,mongo,postgres]`` and runs a Postgres service, so all three arms
    execute on every push.

    The comparison is whole-model equality on ``_Observations``, not a field-by-field
    walk: a field added to the model later is compared without anyone remembering to
    add it here.
    """
    assert YAML in reports, sorted(reports)

    baseline = reports[YAML]
    assert baseline.spliced_batch_size == SHARED_BATCH_SIZE, baseline.spliced_batch_size
    assert set(baseline.referrer_ids) == set(REFERRER_IDS), baseline.referrer_ids
    assert set(baseline.blocker_messages) == EXPECTED_BLOCKERS, baseline.blocker_messages
    assert set(baseline.namespace_ids) == set(ALL_IDS), baseline.namespace_ids

    divergent = {label: report for label, report in reports.items() if report != baseline}
    assert not divergent, divergent


if __name__ == "__main__":
    main()
