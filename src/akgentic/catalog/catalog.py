"""Unified v2 ``Catalog`` service — CRUD, clone, resolve, and load_team.

This module owns the single public service that composes an ``EntryRepository``
with the resolver pipeline (:mod:`akgentic.catalog.resolver`) to gate CRUD
semantics, own the ``clone`` deep-copy primitive, and expose ``resolve`` /
``resolve_by_id`` / ``load_team``.

Placement is deliberate — ``catalog.py`` sits at the top level of the package
(next to :mod:`resolver`, :mod:`env`) per the shard 10 package-structure plan.
The v1 ``services/`` directory (home of ``TemplateCatalog``, ``ToolCatalog``,
``AgentCatalog``, ``TeamCatalog``) is retired by Epic 19.

Invariants enforced by the service (not the repository):

* **Namespace bootstrap** — non-team entries in a namespace require a pre-existing
  team entry in the same namespace (``_check_bootstrap``).
* **Ownership** — sub-entries' ``user_id`` MUST equal the team entry's
  ``user_id`` (``_check_ownership``).
* **Namespace minting** — a team entry whose ``namespace`` equals the sentinel
  :data:`UNSET_NAMESPACE` has its namespace replaced by a fresh ``uuid.uuid4()``
  string before any other pipeline step runs.
* **Clone atomicity** — ``clone`` collects every intended write in memory, then
  emits them in a single pass; partial failures leave the destination namespace
  untouched.
* **Clone root-only lineage** — ``parent_namespace`` and ``parent_id`` are set
  only on the top-level cloned entry; sub-entries have ``parent_*=None``.
* **``load_team`` single-query** — exactly one ``list_by_namespace`` call reaches
  the repository; ref resolution is served by an in-memory wrapper.

The service never catches and re-wraps :class:`CatalogValidationError` from
``prepare_for_write`` — those propagate unchanged.
"""

from __future__ import annotations

import builtins
import uuid
from typing import Any, Final

from pydantic import BaseModel

from akgentic.catalog.models.entry import Entry
from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.queries import EntryQuery
from akgentic.catalog.repositories.base import EntryRepository
from akgentic.catalog.resolver import REF_KEY, prepare_for_write, validate_delete
from akgentic.catalog.resolver import resolve as _resolve
from akgentic.team.models import TeamCard

__all__ = ["UNSET_NAMESPACE", "Catalog"]

_list = builtins.list  # Alias: Catalog.list shadows the built-in inside the class.


UNSET_NAMESPACE: Final[str] = "__MINT__"
"""Sentinel ``namespace`` value signalling "mint a fresh UUID on create".

The shard-06 "empty string" convention is rejected at the Pydantic layer
(``Entry.namespace`` is ``NonEmptyStr``), so the service owns the magic value.
Callers that want a newly-minted namespace construct
``Entry(namespace=UNSET_NAMESPACE, kind="team", ...)``; :meth:`Catalog.create`
substitutes a fresh ``uuid.uuid4()`` string before running the rest of the
write pipeline.
"""


class Catalog:
    """Unified catalog service — CRUD + clone + resolve + load_team.

    Constructed with a single ``EntryRepository`` dependency; holds no other
    state; performs no I/O in ``__init__``. All semantic concerns — namespace
    bootstrap, ownership propagation, clone dedup, ref reconciliation, delete
    guards — live here; the repository stays a narrow data plane.

    Example:
        >>> repo = YamlEntryRepository(tmp_path)
        >>> catalog = Catalog(repo)
        >>> team = Entry(id="team", kind="team", namespace=UNSET_NAMESPACE,
        ...              model_type="akgentic.team.models.TeamCard",
        ...              payload={...})
        >>> stored = catalog.create(team)  # namespace now a fresh UUID
        >>> stored.namespace
        '3fa85f64-5717-4562-b3fc-2c963f66afa6'
    """

    def __init__(self, repository: EntryRepository) -> None:
        """Store ``repository`` on ``self._repository``; no I/O.

        Args:
            repository: Any concrete ``EntryRepository`` implementation.
        """
        self._repository: EntryRepository = repository

    # --- Read -----------------------------------------------------------------

    def get(self, namespace: str, id: str) -> Entry:
        """Return the entry at ``(namespace, id)`` or raise ``EntryNotFoundError``.

        Args:
            namespace: The namespace of the entry.
            id: The id within ``namespace``.

        Returns:
            The stored ``Entry``.

        Raises:
            EntryNotFoundError: If the repository returns ``None`` for the
                ``(namespace, id)`` pair.
        """
        entry = self._repository.get(namespace, id)
        if entry is None:
            raise EntryNotFoundError(f"Entry ({namespace}, {id}) not found")
        return entry

    def list(self, query: EntryQuery) -> _list[Entry]:
        """Return entries matching ``query`` — repository pass-through."""
        return self._repository.list(query)

    def list_by_namespace(self, namespace: str) -> _list[Entry]:
        """Return every entry in ``namespace`` — repository pass-through."""
        return self._repository.list_by_namespace(namespace)

    def find_references(self, namespace: str, target_id: str) -> _list[Entry]:
        """Return entries in ``namespace`` referencing ``target_id`` — pass-through."""
        return self._repository.find_references(namespace, target_id)

    # --- Write ----------------------------------------------------------------

    def create(self, entry: Entry) -> Entry:
        """Persist a new entry with gating, minting, and ref reconciliation.

        Pipeline:

        1. If ``entry.kind == "team"`` and ``entry.namespace == UNSET_NAMESPACE``:
           mint a fresh ``uuid.uuid4()`` and substitute it.
        2. Reject duplicates via ``_check_duplicate`` — raises
           ``CatalogValidationError`` if ``(namespace, id)`` already exists.
        3. Non-team entries: run ``_check_bootstrap`` then ``_check_ownership``.
           Team entries skip both (a team entry IS the bootstrap; it is the
           ``user_id`` authority for its own namespace).
        4. Run ``prepare_for_write`` — ref resolution, class load, Pydantic
           validation, dump, reconcile. ``CatalogValidationError`` propagates
           unchanged.
        5. Call ``repository.put`` with the prepared entry; return it.

        Args:
            entry: Candidate entry to create. For team entries with a sentinel
                namespace, :attr:`entry.namespace` MUST equal
                :data:`UNSET_NAMESPACE`.

        Returns:
            The persisted ``Entry``, carrying the minted namespace if applicable
            and the reconciled payload from ``prepare_for_write``.

        Raises:
            CatalogValidationError: On duplicate, bootstrap, ownership, or
                ``prepare_for_write`` failure.
        """
        if entry.kind == "team" and entry.namespace == UNSET_NAMESPACE:
            entry = self._mint_team_namespace(entry)

        self._check_duplicate(entry.namespace, entry.id)

        if entry.kind != "team":
            self._check_bootstrap(entry.namespace)
            self._check_ownership(entry)

        prepared = prepare_for_write(entry, self._repository)
        self._repository.put(prepared)
        return prepared

    def update(self, entry: Entry) -> Entry:
        """Update an existing entry; re-run ref-reconciliation and ownership.

        Pipeline:

        1. Existence check: ``repository.get`` MUST return non-``None``.
        2. Run ``prepare_for_write`` (may raise ``CatalogValidationError``).
        3. For non-team entries, re-run ``_check_ownership`` on the prepared
           shape so validator normalisations are honoured. Team entries are
           authoritative for their own ``user_id`` and skip this check —
           changing a team's ``user_id`` is a deliberate ownership transfer
           that leaves sub-entries inconsistent until a caller-side migration
           follows up (not this service's concern).
        4. ``repository.put`` + return.

        ``update`` NEVER mints a namespace; the empty-string / sentinel path
        is exclusive to :meth:`create`.

        Args:
            entry: The candidate entry; its ``(namespace, id)`` MUST already
                exist.

        Returns:
            The prepared, persisted ``Entry``.

        Raises:
            EntryNotFoundError: If no entry exists at ``(entry.namespace, entry.id)``.
            CatalogValidationError: From ``prepare_for_write`` or
                ``_check_ownership``.
        """
        if self._repository.get(entry.namespace, entry.id) is None:
            raise EntryNotFoundError(f"Entry ({entry.namespace}, {entry.id}) not found")
        prepared = prepare_for_write(entry, self._repository)
        if prepared.kind != "team":
            self._check_ownership(prepared)
        self._repository.put(prepared)
        return prepared

    def delete(self, namespace: str, id: str) -> None:
        """Delete ``(namespace, id)``, guarded by ``validate_delete``.

        ``EntryNotFoundError`` fires for missing targets (distinguished from
        inbound-ref blockers). ``CatalogValidationError`` fires when inbound
        refs exist and carries each referring entry's id in the error message.

        Args:
            namespace: The namespace of the entry to delete.
            id: The id within ``namespace``.

        Raises:
            EntryNotFoundError: If the target does not exist.
            CatalogValidationError: If any inbound refs would be broken.
        """
        if self._repository.get(namespace, id) is None:
            raise EntryNotFoundError(f"Entry ({namespace}, {id}) not found")
        errors = validate_delete(namespace, id, self._repository)
        if errors:
            raise CatalogValidationError(errors)
        self._repository.delete(namespace, id)

    # --- Clone ----------------------------------------------------------------

    def clone(
        self,
        src_namespace: str,
        src_id: str,
        dst_namespace: str,
        dst_user_id: str | None,
    ) -> Entry:
        """Deep-copy an entry tree into ``dst_namespace`` with ref rewrite and dedup.

        Semantics (pinned by ADR-07 clone):

        * Id preservation — when ``dst_namespace != src_namespace`` the source
          id is reused; when equal, a numeric suffix (``-2``, ``-3``, …) is
          appended until no collision exists in the repository or in the
          intra-call dedup map.
        * Root-only lineage — the top-level cloned entry carries
          ``parent_namespace=src_namespace`` and ``parent_id=src_id``;
          sub-entries have ``parent_*=None``.
        * Deduplication — an intra-call ``cloned`` map keyed by the source
          ``(namespace, id)`` pair guarantees every source sub-entry is cloned
          exactly once per ``clone`` call.
        * Atomicity — every ``put`` is deferred until the entire recursive
          resolution completes. A mid-resolution failure leaves
          ``dst_namespace`` untouched.
        * No ``prepare_for_write`` — source entries were validated when they
          were created; cloning is a structural copy + ref rewrite.
        * Ownership — satisfied by construction (every cloned entry's
          ``user_id`` is stamped to ``dst_user_id``), so ``_check_ownership``
          is not re-run per-write.

        Args:
            src_namespace: Source namespace containing the entry to clone.
            src_id: Id of the source entry within ``src_namespace``.
            dst_namespace: Destination namespace receiving the cloned entries.
            dst_user_id: ``user_id`` to stamp on every cloned entry; ``None``
                for enterprise-scoped clones.

        Returns:
            The top-level cloned entry, as freshly re-read from the repository.

        Raises:
            EntryNotFoundError: If the source entry does not exist.
            CatalogValidationError: If the source graph references a missing
                entry (atomicity guarantees zero destination writes on this
                path).
        """
        if self._repository.get(src_namespace, src_id) is None:
            raise EntryNotFoundError(f"Source entry ({src_namespace}, {src_id}) not found")
        cloned: dict[tuple[str, str], str] = {}
        pending_writes: _list[Entry] = []
        top_new_id = self._clone_one(
            src_namespace=src_namespace,
            src_id=src_id,
            is_top_level=True,
            cloned=cloned,
            pending_writes=pending_writes,
            dst_namespace=dst_namespace,
            dst_user_id=dst_user_id,
        )
        for entry in pending_writes:
            self._repository.put(entry)
        top = self._repository.get(dst_namespace, top_new_id)
        if top is None:  # pragma: no cover — defensive; put just stored it
            raise CatalogValidationError(
                [f"Clone post-write lookup failed for ({dst_namespace}, {top_new_id})"]
            )
        return top

    # --- Resolve --------------------------------------------------------------

    def resolve(self, entry: Entry) -> BaseModel:
        """Hydrate ``entry`` into a runtime Pydantic instance — delegates to resolver."""
        return _resolve(entry, self._repository)

    def resolve_by_id(self, namespace: str, id: str) -> BaseModel:
        """Convenience: ``self.resolve(self.get(namespace, id))``."""
        return self.resolve(self.get(namespace, id))

    def load_team(self, namespace: str) -> TeamCard:
        """Load and resolve the ``kind="team"`` entry in ``namespace`` into a ``TeamCard``.

        Issues exactly ONE ``list_by_namespace`` call against the real
        repository, then builds a pre-loaded in-memory wrapper to short-circuit
        every per-ref ``get`` call ``populate_refs`` would otherwise make. A
        defensive runtime ``isinstance(result, TeamCard)`` check catches
        misconfigured team entries (those whose ``model_type`` points at a
        non-``TeamCard`` class) before they leak out.

        Args:
            namespace: The namespace to load.

        Returns:
            The resolved :class:`TeamCard` for ``namespace``.

        Raises:
            CatalogValidationError: If no team entry exists in ``namespace``
                or if the team entry's ``model_type`` resolves to a class
                other than :class:`TeamCard`.
        """
        entries = self._repository.list_by_namespace(namespace)
        team_entries = [e for e in entries if e.kind == "team"]
        if not team_entries:
            raise CatalogValidationError([f"Namespace '{namespace}' has no team entry"])
        team_entry = team_entries[0]
        in_memory = _InMemoryEntryRepository(entries)
        result = _resolve(team_entry, in_memory)
        if not isinstance(result, TeamCard):
            raise CatalogValidationError(
                [f"Team entry's model_type resolved to {type(result).__name__}, expected TeamCard"]
            )
        return result

    # --- Private helpers ------------------------------------------------------

    def _mint_team_namespace(self, entry: Entry) -> Entry:
        """Return a copy of ``entry`` with ``namespace`` set to a fresh UUID string."""
        return entry.model_copy(update={"namespace": str(uuid.uuid4())})

    def _check_duplicate(self, namespace: str, id: str) -> None:
        """Raise ``CatalogValidationError`` if ``(namespace, id)`` already exists."""
        if self._repository.get(namespace, id) is not None:
            raise CatalogValidationError([f"Entry ({namespace}, {id}) already exists"])

    def _check_bootstrap(self, namespace: str) -> None:
        """Ensure a team entry exists in ``namespace``; raise otherwise."""
        if self._repository.get_by_kind(namespace, "team") is None:
            raise CatalogValidationError(
                [
                    f"Namespace '{namespace}' has no team entry — create the team "
                    f"entry first (bootstrap invariant)"
                ]
            )

    def _check_ownership(self, entry: Entry) -> None:
        """Ensure ``entry.user_id == team_entry.user_id`` within ``entry.namespace``."""
        team = self._repository.get_by_kind(entry.namespace, "team")
        if team is None:
            # Bootstrap already guards this for create(); update() uses the same
            # helper, and an update to a non-team entry that left the team
            # behind is a corrupted state — reject defensively.
            raise CatalogValidationError(
                [
                    f"Namespace '{entry.namespace}' has no team entry — cannot "
                    f"verify ownership for '{entry.id}'"
                ]
            )
        if entry.user_id != team.user_id:
            raise CatalogValidationError(
                [
                    f"Ownership mismatch in namespace '{entry.namespace}': "
                    f"entry '{entry.id}' has user_id={entry.user_id!r} but "
                    f"team has user_id={team.user_id!r}"
                ]
            )

    def _clone_one(
        self,
        src_namespace: str,
        src_id: str,
        is_top_level: bool,
        cloned: dict[tuple[str, str], str],
        pending_writes: _list[Entry],
        dst_namespace: str,
        dst_user_id: str | None,
    ) -> str:
        """Clone a single source entry, recursing into its payload refs."""
        key = (src_namespace, src_id)
        if key in cloned:
            return cloned[key]

        src = self._repository.get(src_namespace, src_id)
        if src is None:
            raise CatalogValidationError(
                [f"Clone source ({src_namespace}, {src_id}) not found during resolution"]
            )

        new_id = self._mint_dst_id(dst_namespace, src_namespace, src_id, cloned)
        # Record BEFORE recursing so a back-ref to the same source returns
        # the same dst id (prevents infinite recursion on cycles-by-construction).
        cloned[key] = new_id

        def _callback(target_id: str) -> str:
            return self._clone_one(
                src_namespace=src_namespace,
                src_id=target_id,
                is_top_level=False,
                cloned=cloned,
                pending_writes=pending_writes,
                dst_namespace=dst_namespace,
                dst_user_id=dst_user_id,
            )

        new_payload = _rewrite_refs(src.payload, _callback)
        new_entry = src.model_copy(
            update={
                "id": new_id,
                "namespace": dst_namespace,
                "user_id": dst_user_id,
                "parent_namespace": src_namespace if is_top_level else None,
                "parent_id": src_id if is_top_level else None,
                "payload": new_payload,
            }
        )
        pending_writes.append(new_entry)
        return new_id

    def _mint_dst_id(
        self,
        dst_namespace: str,
        src_namespace: str,
        src_id: str,
        cloned: dict[tuple[str, str], str],
    ) -> str:
        """Return the destination id for a clone.

        Cross-namespace: reuse ``src_id``. Same-namespace: append a numeric
        suffix starting at ``-2`` and increment until the candidate does not
        collide with either a stored entry or an id already planned by the
        current clone operation.
        """
        if dst_namespace != src_namespace:
            return src_id
        planned: set[str] = set(cloned.values())
        suffix = 2
        while True:
            candidate = f"{src_id}-{suffix}"
            if self._repository.get(dst_namespace, candidate) is None and candidate not in planned:
                return candidate
            suffix += 1


def _rewrite_refs(node: Any, clone_target: Any) -> Any:
    """Recursively copy ``node``, replacing ref targets via ``clone_target``.

    Dicts carrying a ``REF_KEY`` entry have their target id replaced by
    ``clone_target(target_id)`` — the callback is typically a recursive clone
    invocation that also clones the target entry. ``TYPE_KEY`` (if present) is
    preserved verbatim. Non-ref dicts and lists recurse structurally; leaves
    pass through unchanged.

    Args:
        node: Arbitrary payload subtree.
        clone_target: Callback mapping a source target id to the corresponding
            destination target id (with side effect of cloning the target).

    Returns:
        A new payload subtree with every ref marker pointing at the newly
        minted destination ids.
    """
    if isinstance(node, dict):
        if REF_KEY in node:
            new: dict[str, Any] = dict(node)
            new[REF_KEY] = clone_target(node[REF_KEY])
            return new
        return {k: _rewrite_refs(v, clone_target) for k, v in node.items()}
    if isinstance(node, list):
        return [_rewrite_refs(v, clone_target) for v in node]
    return node


class _InMemoryEntryRepository:
    """Pre-loaded ``EntryRepository`` wrapper serving ``get`` from a list.

    Used exclusively by :meth:`Catalog.load_team` to short-circuit the per-ref
    ``get`` calls :func:`~akgentic.catalog.resolver.populate_refs` would
    otherwise issue against the namespace that was just loaded wholesale.
    Every non-``get`` method raises :class:`NotImplementedError` — this wrapper
    is intentionally a degraded shape, not a drop-in replacement for a real
    repository.
    """

    def __init__(self, entries: list[Entry]) -> None:
        """Index ``entries`` by ``(namespace, id)`` for O(1) ``get`` lookups."""
        self._by_key: dict[tuple[str, str], Entry] = {(e.namespace, e.id): e for e in entries}

    def get(self, namespace: str, id: str) -> Entry | None:
        """Return the pre-loaded entry or ``None`` if absent."""
        return self._by_key.get((namespace, id))

    def put(self, entry: Entry) -> Entry:  # noqa: ARG002
        raise NotImplementedError(
            "InMemoryEntryRepository supports only .get(); use the real repository "
            "for other operations"
        )

    def delete(self, namespace: str, id: str) -> None:  # noqa: ARG002
        raise NotImplementedError(
            "InMemoryEntryRepository supports only .get(); use the real repository "
            "for other operations"
        )

    def list(self, query: EntryQuery) -> _list[Entry]:  # noqa: ARG002
        raise NotImplementedError(
            "InMemoryEntryRepository supports only .get(); use the real repository "
            "for other operations"
        )

    def list_by_namespace(self, namespace: str) -> _list[Entry]:  # noqa: ARG002
        raise NotImplementedError(
            "InMemoryEntryRepository supports only .get(); use the real repository "
            "for other operations"
        )

    def get_by_kind(self, namespace: str, kind: Any) -> Entry | None:  # noqa: ARG002
        raise NotImplementedError(
            "InMemoryEntryRepository supports only .get(); use the real repository "
            "for other operations"
        )

    def find_references(self, namespace: str, target_id: str) -> _list[Entry]:  # noqa: ARG002
        raise NotImplementedError(
            "InMemoryEntryRepository supports only .get(); use the real repository "
            "for other operations"
        )
