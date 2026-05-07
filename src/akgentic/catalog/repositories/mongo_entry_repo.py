"""MongoDB-backed v2 ``EntryRepository`` storing every entry in one collection.

Layout: one ``catalog_entries`` collection keyed by a compound ``_id`` of the
form ``{"namespace": <ns>, "id": <id>}``. ``(namespace, id)`` is the v2
uniqueness key; ``kind`` is metadata. Mongo's primary-key uniqueness on the
compound ``_id`` structurally prevents duplicate ids within a single namespace,
so the duplicate-id reconciliation the YAML backend performs is unnecessary
here.

The repository is intent-preserving: ``put`` writes
``entry.model_dump(mode="json")`` verbatim, so author-written ref markers
(``{"__ref__": ...}``, ``{"__type__": ...}``) round-trip byte-for-byte without
any resolver-layer expansion. The ``Catalog`` service (Story 15.5) runs
``prepare_for_write`` before reaching this repository; the repository itself
never calls ``populate_refs`` / ``reconcile_refs`` / ``prepare_for_write``.

The module name ``mongo_entry_repo`` is temporary: the v1 package
``repositories/mongo/`` still lives at the same directory level and a Python
``repositories/mongo.py`` module would collide with it. Epic 19 Story 19.2
renames this module to ``mongo.py`` after v1 deletion. The module intentionally
lives at ``repositories/`` level (not inside ``repositories/mongo/``) to match
the placement of ``yaml_entry_repo.py`` so the rename is symmetric across
backends.

``pymongo`` is NOT imported at module top level — the only typing surface that
references ``pymongo`` lives under ``if TYPE_CHECKING:``. Runtime code imports
``pymongo.ASCENDING`` inside the method bodies that need it, so importing this
module without ``pymongo`` installed succeeds. The caller only pays the
``pymongo`` tax when they actually hand a live ``Collection`` to the
constructor. The ``repositories/mongo/`` package's own import guard (``try:
import pymongo; except ImportError: raise ImportError("pip install
akgentic-catalog[mongo]")``) is orthogonal — ``mongo_entry_repo.py`` is one
directory level up.

Secondary indexes (``(namespace, kind)`` and ``(namespace, parent_id)``) are
created lazily on the first ``put``. Read-only repositories never touch the
index surface. Under ``mongomock``, ``create_index`` is a no-op that still
records the call so tests can verify the production code path.

Implements the storage layer described in architecture shard 04
(``_bmad-output/akgentic-catalog/architecture/04-repositories.md``) under the
unified-entry contract of ADR-007.
"""

from __future__ import annotations

import builtins
import logging
import re
from typing import TYPE_CHECKING, Any

from akgentic.catalog.models.entry import Entry, EntryKind
from akgentic.catalog.models.queries import EntryQuery
from akgentic.catalog.repositories.yaml_entry_repo import _payload_has_ref

if TYPE_CHECKING:
    import pymongo.collection

__all__ = ["MongoEntryRepository"]

logger = logging.getLogger(__name__)

_list = builtins.list  # Alias: the repository's list() method shadows the built-in


class MongoEntryRepository:
    """Single-collection, compound-``_id`` v2 ``EntryRepository`` on MongoDB.

    Satisfies the ``akgentic.catalog.repositories.base.EntryRepository``
    structural protocol. Construction stores the passed ``Collection`` and does
    not perform any I/O — indexes are created lazily on the first ``put``.

    The document shape is:

    ``{
        "_id": {"namespace": <ns>, "id": <id>},
        "namespace": <ns>,
        "id": <id>,
        "kind": <kind>,
        "user_id": <str|None>,
        "parent_namespace": <str|None>,
        "parent_id": <str|None>,
        "model_type": <akgentic.* dotted path>,
        "description": <str>,
        "payload": {...},
    }``

    ``namespace`` and ``id`` are duplicated at the top level so secondary index
    queries do not need to destructure ``_id`` and so
    ``Entry.model_validate(doc_minus_underscore_id)`` reconstructs the entry
    without having to re-inject ``id`` from ``_id``. This diverges from the v1
    ``repositories/mongo/_helpers.py::to_document`` /``from_document`` helpers
    (which store a plain-string ``_id`` equal to ``entry.id``); v2 uses its own
    shape conversion and does NOT reuse those helpers.

    The repository is intent-preserving — see module docstring. Callers that
    want ref markers expanded must call the resolver pipeline in
    ``akgentic.catalog.resolver`` before / after the repository round-trip;
    this class does not mutate payload content.
    """

    def __init__(self, collection: pymongo.collection.Collection) -> None:  # type: ignore[type-arg]
        """Store the collection; do not touch the server.

        Args:
            collection: A ``pymongo.collection.Collection`` (or mongomock
                equivalent) holding v2 ``Entry`` documents. The repository
                does NOT own the client, the database, or the collection
                lifecycle — the caller wires the ``client → db → collection``
                chain and hands the collection to the constructor.
        """
        self._collection = collection
        self._indexes_created: bool = False
        logger.info("MongoEntryRepository initialized")

    # --- Compound-id and document-shape helpers ---

    def _compound_id(self, namespace: str, id: str) -> dict[str, str]:
        """Return the compound-``_id`` dict ``{"namespace": ns, "id": id}``.

        This is the only form of ``_id`` the v2 collection recognises. The
        dict is constructed fresh on every call — Mongo compares ``_id``
        values by deep equality, so a dict literal is safe to pass to
        ``find_one`` / ``delete_one`` / ``replace_one``.
        """
        return {"namespace": namespace, "id": id}

    def _to_document(self, entry: Entry) -> dict[str, Any]:
        """Serialise ``entry`` to the v2 compound-``_id`` document shape.

        Starts from ``entry.model_dump(mode="json")`` (which includes every
        field of ``Entry`` — ``id``, ``namespace``, ``kind``, … and ``payload``
        verbatim) and adds a compound ``_id`` composed of ``namespace`` + ``id``.
        The top-level ``id`` / ``namespace`` fields are kept so a full read can
        call ``Entry.model_validate`` after popping ``_id`` without having to
        destructure the compound key.

        Diverges from v1 ``_helpers.py::to_document`` — that helper pops ``id``
        to a plain-string ``_id``; v2 keeps ``id`` at the top level AND nests
        it inside the compound ``_id``. The two shapes are incompatible and
        share no code.
        """
        doc = entry.model_dump(mode="json")
        doc["_id"] = self._compound_id(entry.namespace, entry.id)
        return doc

    def _from_document(self, doc: dict[str, Any]) -> Entry:
        """Reconstruct an ``Entry`` from a v2 compound-``_id`` document.

        Takes a shallow copy so the caller's dict is not mutated (some driver
        versions reuse the document object passed to ``replace_one`` or mutate
        cursor results). Pops ``_id`` — the top-level ``namespace`` + ``id``
        fields already carry the same values — and delegates the remaining
        fields to ``Entry.model_validate``.

        Diverges from v1 ``_helpers.py::from_document`` — that helper pops
        ``_id`` into the ``id`` field; v2 discards ``_id`` outright since the
        top-level ``id`` field is authoritative.
        """
        data = dict(doc)
        data.pop("_id", None)
        return Entry.model_validate(data)

    # --- Lazy index creation ---

    def _ensure_indexes(self) -> None:
        """Create the two v2 secondary indexes on the first call; no-op after.

        Declared indexes:

        * ``(namespace, kind)`` — the standard listing path
          (``list_by_namespace`` + ``get_by_kind``).
        * ``(namespace, parent_id)`` — lineage traversal (reserved for future
          "find all clones of X" queries; not directly used in this story).

        No index is created on ``_id`` — Mongo enforces primary-key uniqueness
        on ``_id`` by default. No index is created for the payload walk
        ``find_references`` uses (v2 acceptance — no wildcard payload index).

        The one-shot ``self._indexes_created`` flag keeps subsequent ``put``
        calls cheap. The ``ASCENDING`` constant is imported inside this method
        so the module's top-level imports stay free of ``pymongo``.
        """
        if self._indexes_created:
            return
        from pymongo import ASCENDING

        self._collection.create_index([("namespace", ASCENDING), ("kind", ASCENDING)])
        self._collection.create_index([("namespace", ASCENDING), ("parent_id", ASCENDING)])
        self._indexes_created = True

    # --- Write operations ---

    def put(self, entry: Entry) -> Entry:
        """Upsert ``entry`` keyed by ``(namespace, id)``; return the stored entry.

        Creates the two secondary indexes on the first call (see
        ``_ensure_indexes``) and issues ``replace_one(filter, doc,
        upsert=True)`` with ``filter = {"_id": {"namespace": ..., "id": ...}}``.
        The "second write wins" contract (the v2 upsert semantics pinned by
        the acceptance criteria) means the collection count does not grow on
        re-writes of the same ``(namespace, id)`` pair — including when
        ``kind`` changes between writes, since ``kind`` is metadata and not
        part of the uniqueness key.

        Does NOT use ``insert_one`` + ``DuplicateKeyError`` recovery — the
        contract is unambiguous upsert and ``replace_one`` + ``upsert=True``
        is the direct path. Does NOT call ``populate_refs`` / ``reconcile_refs``
        / ``prepare_for_write`` / ``entry.model_dump(exclude_unset=True)`` —
        the repository is intent-preserving (writes what it was handed).

        Args:
            entry: The entry to persist. The ``(namespace, id)`` pair is the
                uniqueness key.

        Returns:
            The ``entry`` argument unchanged — callers treat it as the stored
            form for chaining or assertion.
        """
        self._ensure_indexes()
        doc = self._to_document(entry)
        self._collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        logger.debug("Upserted entry ns=%s id=%s kind=%s", entry.namespace, entry.id, entry.kind)
        return entry

    def delete(self, namespace: str, id: str) -> None:
        """Remove the entry at ``(namespace, id)``; no-op when absent.

        Issues ``delete_one({"_id": {"namespace": ..., "id": ...}})`` and
        ignores ``DeleteResult.deleted_count`` — the repository accepts
        idempotency. The service layer's ``validate_delete`` is the
        authoritative guard against deleting missing entries.
        """
        self._collection.delete_one({"_id": self._compound_id(namespace, id)})
        logger.debug("Deleted entry ns=%s id=%s (idempotent on miss)", namespace, id)

    # --- Read operations ---

    def get(self, namespace: str, id: str) -> Entry | None:
        """Return the entry at ``(namespace, id)`` or ``None`` if absent.

        Issues ``find_one({"_id": {"namespace": ..., "id": ...}})``. Because
        the filter uses the compound primary key, namespace isolation is
        enforced by construction — a matching id in a different namespace
        cannot bleed across.
        """
        doc = self._collection.find_one({"_id": self._compound_id(namespace, id)})
        if doc is None:
            return None
        return self._from_document(doc)

    def list(self, query: EntryQuery) -> _list[Entry]:
        """Return entries matching ``query`` with AND semantics over set fields.

        Builds a single server-side filter dict via ``_build_filter`` and
        issues one ``find``. Every filter supported by ``EntryQuery`` is
        translated to an exact-match Mongo clause, except
        ``description_contains`` which becomes a case-sensitive
        ``{"$regex": re.escape(value)}`` clause (no ``$options: "i"`` —
        parity with the YAML backend's case-sensitive substring check).

        When ``query.namespace`` is ``None``, the filter omits the
        ``namespace`` clause and the scan spans every document subject to
        the remaining filters.
        """
        mongo_filter = self._build_filter(query)
        cursor = self._collection.find(mongo_filter)
        return [self._from_document(doc) for doc in cursor]

    def _build_filter(self, query: EntryQuery) -> dict[str, Any]:
        """Translate an ``EntryQuery`` into a server-side Mongo filter dict.

        Each non-``None`` field contributes exactly one clause. ``user_id_set``
        is tri-state (``None`` = ignore, ``True`` = ``user_id != null``,
        ``False`` = ``user_id is null``). ``user_id`` and ``user_id_set``
        combine with AND semantics — if both are set, the emitted ``user_id``
        clause honours both constraints jointly so the Mongo filter matches
        the YAML backend's ``_matches`` evaluation (conjunctive over both
        fields). ``description_contains`` uses ``re.escape`` to defuse regex
        metacharacters in user input.
        """
        mongo_filter: dict[str, Any] = {}
        if query.namespace is not None:
            mongo_filter["namespace"] = query.namespace
        if query.kind is not None:
            mongo_filter["kind"] = query.kind
        if query.id is not None:
            mongo_filter["id"] = query.id
        self._apply_user_id_clauses(query, mongo_filter)
        if query.parent_namespace is not None:
            mongo_filter["parent_namespace"] = query.parent_namespace
        if query.parent_id is not None:
            mongo_filter["parent_id"] = query.parent_id
        if query.description_contains is not None:
            mongo_filter["description"] = {"$regex": re.escape(query.description_contains)}
        return mongo_filter

    def _apply_user_id_clauses(
        self,
        query: EntryQuery,
        mongo_filter: dict[str, Any],
    ) -> None:
        """Write a conjunctive ``user_id`` clause into ``mongo_filter``.

        ``user_id`` and ``user_id_set`` are orthogonal knobs that combine
        with AND semantics (parity with YAML's ``_matches``). Table of
        behaviours for the four relevant combinations:

        * both unset → no ``user_id`` key is added.
        * ``user_id`` only → exact match (``{"user_id": value}``).
        * ``user_id_set`` only → presence filter
          (``{"$ne": None}`` / ``None``).
        * both set → the joint constraint: if ``user_id_set=True`` the exact
          value already guarantees non-``None``, so the exact match is
          emitted; if ``user_id_set=False`` the combination is logically
          impossible (a non-``None`` value cannot also be ``None``), so an
          unsatisfiable clause (``{"$in": []}``) is emitted so the query
          returns zero documents — which is exactly what the YAML backend
          returns for the same contradictory pair.
        """
        if query.user_id is None and query.user_id_set is None:
            return
        if query.user_id_set is None:
            mongo_filter["user_id"] = query.user_id
            return
        if query.user_id is None:
            mongo_filter["user_id"] = {"$ne": None} if query.user_id_set else None
            return
        if query.user_id_set:
            # Exact value is already non-None; the exact match satisfies both.
            mongo_filter["user_id"] = query.user_id
            return
        # user_id set but user_id_set is False → contradiction; match nothing.
        mongo_filter["user_id"] = {"$in": []}

    def list_by_namespace(self, namespace: str) -> _list[Entry]:
        """Return every entry in ``namespace`` in a single ``find``.

        Issues ``find({"namespace": namespace})`` — exactly one round-trip, no
        per-kind fan-out. Missing namespace (no documents with that
        ``namespace`` value) returns ``[]`` without raising.
        """
        cursor = self._collection.find({"namespace": namespace})
        return [self._from_document(doc) for doc in cursor]

    def get_by_kind(self, namespace: str, kind: EntryKind) -> Entry | None:
        """Return a single entry of ``kind`` in ``namespace`` or ``None``.

        Issues ``find_one({"namespace": namespace, "kind": kind},
        sort=[("_id.id", ASCENDING)])`` so that, in the corruption case where
        two documents share ``(namespace, kind)`` (which the service-layer
        singleton invariant forbids but the repository tolerates), the
        deterministic alphabetically-first result by ``_id.id`` is returned.
        The repository does NOT raise on duplicates — the singleton invariant
        is owned by ``Catalog.create`` (Story 15.5).
        """
        from pymongo import ASCENDING

        doc = self._collection.find_one(
            {"namespace": namespace, "kind": kind},
            sort=[("_id.id", ASCENDING)],
        )
        if doc is None:
            return None
        return self._from_document(doc)

    def find_references(self, namespace: str, target_id: str) -> _list[Entry]:
        """Return entries in ``namespace`` whose payload references ``target_id``.

        Loads every entry in ``namespace`` via ``list_by_namespace`` (one
        server-side query), then walks each payload in memory using the
        ``_payload_has_ref`` helper shared with the YAML backend. No
        server-side wildcard payload index is used — v2 acceptance
        (``no wildcard payload index``) — the in-memory walk is cheap at
        namespace scale and keeps the two backends in lockstep with a single
        shared walker.

        Namespace isolation is strict: refs in other namespaces are not
        inspected.
        """
        return [
            entry
            for entry in self.list_by_namespace(namespace)
            if _payload_has_ref(entry.payload, target_id)
        ]
