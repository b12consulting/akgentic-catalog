"""Tests for ``Catalog`` CRUD semantics — Story 15.5 ACs 5–20, 37, 38, 40.

Every behavioural test runs against both backends via the ``catalog_factory``
fixture (yaml + mongo, parametrised with explicit ids). Spy / invocation
counting tests use the backend-agnostic ``counting_catalog`` fixture built on
``CountingEntryRepository`` around the ``FakeEntryRepository`` test double.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from akgentic.catalog.catalog import UNSET_NAMESPACE, Catalog
from akgentic.catalog.models.entry import Entry
from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.namespace_meta import NamespaceMeta
from akgentic.catalog.models.queries import EntryQuery
from akgentic.catalog.repositories.base import EntryRepository

from ..conftest import team_payload
from .conftest import (
    CatalogFactory,
    CountingEntryRepository,
    register_akgentic_test_module,
)

_TEAM_TYPE = "akgentic.team.models.TeamCard"


def _team_payload() -> dict[str, Any]:
    """``team_payload`` with the card description this module has always used."""
    return team_payload(card_description="entry")


def _seed_team(
    catalog: Catalog,
    namespace: str,
    user_id: str = "anonymous",
    team_id: str = "team",
) -> Entry:
    """Seed a team entry in ``namespace`` and return the persisted entry."""
    entry = Entry(
        id=team_id,
        kind="team",
        namespace=namespace,
        user_id=user_id,
        model_type=_TEAM_TYPE,
        payload=_team_payload(),
    )
    return catalog.create(entry)


class _LeafPayloadModel(BaseModel):
    """Leaf referent for ref-based sub-entry payload validation."""

    provider: str = "openai"
    temperature: float = 0.0


class _AgentPayloadModel(BaseModel):
    """Minimal payload model for non-team test entries (avoids real AgentCard deps)."""

    provider: str = "openai"
    temperature: float = 0.0
    model_cfg: _LeafPayloadModel | None = None


def _register_agent_model(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register the ``_AgentPayloadModel`` under a fake ``akgentic.*`` path."""
    module_name = register_akgentic_test_module(
        monkeypatch,
        "tests_fixture_15_5_crud_agent",
        _AgentPayloadModel=_AgentPayloadModel,
        _LeafPayloadModel=_LeafPayloadModel,
    )
    return f"{module_name}._AgentPayloadModel"


def _register_leaf_model(monkeypatch: pytest.MonkeyPatch) -> str:
    """Return the fully-qualified path for the leaf model (same registration)."""
    module_name = register_akgentic_test_module(
        monkeypatch,
        "tests_fixture_15_5_crud_agent",
        _AgentPayloadModel=_AgentPayloadModel,
        _LeafPayloadModel=_LeafPayloadModel,
    )
    return f"{module_name}._LeafPayloadModel"


# --- AC5 — pass-throughs via counting_catalog -------------------------------------


class TestPassThroughs:
    """AC5 — list, list_by_namespace, find_references are repository pass-throughs."""

    def test_list_delegates_once(
        self, counting_catalog: tuple[Catalog, CountingEntryRepository]
    ) -> None:
        catalog, counting = counting_catalog
        counting.reset()
        query = EntryQuery(namespace="ns-x")
        result = catalog.list(query)
        assert result == []
        assert counting.count("list") == 1

    def test_list_by_namespace_delegates_once(
        self, counting_catalog: tuple[Catalog, CountingEntryRepository]
    ) -> None:
        catalog, counting = counting_catalog
        counting.reset()
        result = catalog.list_by_namespace("ns-x")
        assert result == []
        assert counting.count("list_by_namespace") == 1

    def test_find_references_delegates_once(
        self, counting_catalog: tuple[Catalog, CountingEntryRepository]
    ) -> None:
        catalog, counting = counting_catalog
        counting.reset()
        result = catalog.find_references("ns-x", "anything")
        assert result == []
        assert counting.count("find_references") == 1


# --- AC6 — get -----------------------------------------------------------------


class TestGet:
    """AC6 — hit returns entry; miss raises ``EntryNotFoundError``."""

    def test_get_hit_returns_entry(self, catalog_factory: CatalogFactory) -> None:
        catalog, _ = catalog_factory()
        team = _seed_team(catalog, namespace="ns-get")
        result = catalog.get("ns-get", team.id)
        assert result.id == team.id
        assert result.namespace == "ns-get"

    def test_get_miss_raises(self, catalog_factory: CatalogFactory) -> None:
        catalog, _ = catalog_factory()
        with pytest.raises(EntryNotFoundError) as exc:
            catalog.get("ns-missing", "no-such")
        msg = str(exc.value)
        assert "not found" in msg
        assert "ns-missing" in msg
        assert "no-such" in msg


# --- AC7, AC8 — namespace minting and rejection ---------------------------------


class TestCreateNamespaceMint:
    """AC7 — team + UNSET_NAMESPACE mints a UUID; AC8 — non-team empty-ns rejected."""

    def test_team_unset_namespace_mints_uuid(self, catalog_factory: CatalogFactory) -> None:
        catalog, repo = catalog_factory()
        entry = Entry(
            id="team",
            kind="team",
            namespace=UNSET_NAMESPACE,
            model_type=_TEAM_TYPE,
            payload=_team_payload(),
        )
        stored = catalog.create(entry)
        assert stored.namespace != UNSET_NAMESPACE
        assert stored.namespace != ""
        # uuid4() strings are 36 chars: 8-4-4-4-12.
        assert len(stored.namespace) == 36
        # Repository actually received the stored namespace, not the sentinel.
        assert repo.get(stored.namespace, "team") is not None
        assert repo.get(UNSET_NAMESPACE, "team") is None

    def test_two_mints_yield_distinct_uuids(self, catalog_factory: CatalogFactory) -> None:
        catalog, _ = catalog_factory()
        a = catalog.create(
            Entry(
                id="team",
                kind="team",
                namespace=UNSET_NAMESPACE,
                model_type=_TEAM_TYPE,
                payload=_team_payload(),
            )
        )
        b = catalog.create(
            Entry(
                id="team",
                kind="team",
                namespace=UNSET_NAMESPACE,
                model_type=_TEAM_TYPE,
                payload=_team_payload(),
            )
        )
        assert a.namespace != b.namespace

    def test_team_with_concrete_namespace_is_honoured(
        self, catalog_factory: CatalogFactory
    ) -> None:
        catalog, _ = catalog_factory()
        stored = _seed_team(catalog, namespace="chosen-ns")
        assert stored.namespace == "chosen-ns"

    def test_empty_string_namespace_rejected_at_entry_construction(self) -> None:
        # AC8 — NonEmptyStr rejects empty strings at Pydantic layer.
        with pytest.raises(ValidationError):
            Entry(
                id="assistant",
                kind="agent",
                namespace="",
                model_type=_TEAM_TYPE,
                payload={},
            )


# --- AC9 — duplicate -------------------------------------------------------------


class TestCreateDuplicate:
    """AC9 — create rejects duplicate (namespace, id)."""

    def test_duplicate_raises_and_does_not_write(
        self, counting_catalog: tuple[Catalog, CountingEntryRepository]
    ) -> None:
        catalog, counting = counting_catalog
        _seed_team(catalog, namespace="ns-dup")
        counting.reset()
        with pytest.raises(CatalogValidationError) as exc:
            _seed_team(catalog, namespace="ns-dup")
        msg = str(exc.value)
        assert "already exists" in msg
        assert "ns-dup" in msg
        assert "team" in msg  # id appears in the error
        # No put during the failing call.
        assert counting.count("put") == 0


# --- AC10 — bootstrap ------------------------------------------------------------


class TestCreateBootstrap:
    """AC10 — non-team entry in a fresh namespace requires a team entry first."""

    def test_agent_without_team_fails(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        catalog, _ = catalog_factory()
        agent_type = _register_agent_model(monkeypatch)
        agent = Entry(
            id="assistant",
            kind="agent",
            namespace="fresh-ns",
            user_id="alice",
            model_type=agent_type,
            payload={"provider": "openai"},
        )
        with pytest.raises(CatalogValidationError) as exc:
            catalog.create(agent)
        msg = str(exc.value)
        assert "has no team entry and no meta entry" in msg
        assert "fresh-ns" in msg


# --- Story 17.10 — create in meta-only namespace --------------------------------


_NAMESPACE_META_TYPE = "akgentic.catalog.models.namespace_meta.NamespaceMeta"


def _seed_meta(
    catalog: Catalog,
    namespace: str,
    user_id: str = "anonymous",
) -> Entry:
    """Seed a meta entry in ``namespace`` and return the persisted entry."""
    return catalog.create(
        Entry(
            id="_meta",
            kind="meta",
            namespace=namespace,
            user_id=user_id,
            model_type=_NAMESPACE_META_TYPE,
            description="",
            payload={"name": namespace, "description": "", "properties": {}, "shareable": False},
        )
    )


class TestCreateInMetaOnlyNamespace:
    """Story 17.10 — meta entry bootstraps a namespace; ownership anchors on meta."""

    def test_create_meta_in_fresh_namespace_succeeds(self, catalog_factory: CatalogFactory) -> None:
        """Create _meta in a fresh namespace — succeeds, user_id == "anonymous"."""
        catalog, _ = catalog_factory()
        meta = _seed_meta(catalog, "meta-only-ns", user_id="anonymous")
        assert meta.kind == "meta"
        assert meta.namespace == "meta-only-ns"
        assert meta.user_id == "anonymous"

    def test_create_model_in_meta_only_namespace_succeeds(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Create a kind="model" entry in a meta-only namespace — ownership anchor is meta."""
        catalog, _ = catalog_factory()
        _seed_meta(catalog, "meta-only-ns", user_id="anonymous")
        agent_type = _register_agent_model(monkeypatch)
        model = Entry(
            id="shared-model",
            kind="model",
            namespace="meta-only-ns",
            user_id="anonymous",
            model_type=agent_type,
            payload={},
        )
        stored = catalog.create(model)
        assert stored.id == "shared-model"
        assert stored.namespace == "meta-only-ns"

    def test_create_model_with_mismatched_user_id_fails(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Create a kind="model" with user_id != meta.user_id — fails with "anchor (meta)"."""
        catalog, _ = catalog_factory()
        _seed_meta(catalog, "meta-only-ns", user_id="anonymous")
        agent_type = _register_agent_model(monkeypatch)
        model = Entry(
            id="rogue-model",
            kind="model",
            namespace="meta-only-ns",
            user_id="eve",
            model_type=agent_type,
            payload={},
        )
        with pytest.raises(CatalogValidationError) as exc:
            catalog.create(model)
        msg = str(exc.value)
        assert "Ownership mismatch" in msg
        assert "anchor (meta)" in msg

    def test_create_non_anchor_in_empty_namespace_fails(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Create any non-team, non-meta entry in a fresh empty namespace — fails."""
        catalog, _ = catalog_factory()
        agent_type = _register_agent_model(monkeypatch)
        model = Entry(
            id="orphan-model",
            kind="model",
            namespace="empty-ns",
            user_id="alice",
            model_type=agent_type,
            payload={},
        )
        with pytest.raises(CatalogValidationError) as exc:
            catalog.create(model)
        msg = str(exc.value)
        assert "has no team entry and no meta entry" in msg


# --- AC11 + AC40 — ownership ----------------------------------------------------


class TestCreateOwnership:
    """AC11 + AC40 — user_id on sub-entries must equal the team's user_id."""

    def test_matching_user_id_accepted(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        catalog, _ = catalog_factory()
        _seed_team(catalog, namespace="ns-own", user_id="alice")
        agent_type = _register_agent_model(monkeypatch)
        agent = Entry(
            id="assistant",
            kind="agent",
            namespace="ns-own",
            user_id="alice",
            model_type=agent_type,
            payload={},
        )
        stored = catalog.create(agent)
        assert stored.user_id == "alice"

    def test_mismatched_user_ids_rejected(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        catalog, _ = catalog_factory()
        _seed_team(catalog, namespace="ns-own", user_id="alice")
        agent_type = _register_agent_model(monkeypatch)
        agent = Entry(
            id="assistant",
            kind="agent",
            namespace="ns-own",
            user_id="bob",
            model_type=agent_type,
            payload={},
        )
        with pytest.raises(CatalogValidationError) as exc:
            catalog.create(agent)
        msg = str(exc.value)
        assert "bob" in msg
        assert "alice" in msg
        assert "ns-own" in msg

    def test_none_sub_entry_when_team_is_user_rejected(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        catalog, _ = catalog_factory()
        _seed_team(catalog, namespace="ns-own", user_id="alice")
        agent_type = _register_agent_model(monkeypatch)
        agent = Entry(
            id="assistant",
            kind="agent",
            namespace="ns-own",
            user_id="anonymous",
            model_type=agent_type,
            payload={},
        )
        with pytest.raises(CatalogValidationError) as exc:
            catalog.create(agent)
        msg = str(exc.value)
        assert "alice" in msg
        assert "anonymous" in msg

    def test_enterprise_none_none_accepted(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        catalog, _ = catalog_factory()
        _seed_team(catalog, namespace="ns-ent", user_id="anonymous")
        agent_type = _register_agent_model(monkeypatch)
        agent = Entry(
            id="assistant",
            kind="agent",
            namespace="ns-ent",
            user_id="anonymous",
            model_type=agent_type,
            payload={},
        )
        stored = catalog.create(agent)
        assert stored.user_id == "anonymous"


# --- AC12 — prepare_for_write is invoked ---------------------------------------


class TestCreateRunsPrepareForWrite:
    """AC12 — ``prepare_for_write`` is invoked once before ``repository.put``."""

    def test_prepare_for_write_called(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        catalog, _ = catalog_factory()
        _seed_team(catalog, namespace="ns-pre", user_id="alice")
        agent_type = _register_agent_model(monkeypatch)

        calls: list[tuple[Entry, EntryRepository]] = []

        import akgentic.catalog.catalog as catalog_module

        real = catalog_module.prepare_for_write

        def _spy(entry: Entry, repository: EntryRepository, **kwargs: Any) -> Entry:
            calls.append((entry, repository))
            return real(entry, repository, **kwargs)

        monkeypatch.setattr(catalog_module, "prepare_for_write", _spy)

        agent = Entry(
            id="assistant",
            kind="agent",
            namespace="ns-pre",
            user_id="alice",
            model_type=agent_type,
            payload={"provider": "openai"},
        )
        catalog.create(agent)
        assert len(calls) == 1
        assert calls[0][0].id == "assistant"


# --- AC13 — team skips bootstrap + ownership ----------------------------------


class TestCreateTeamSkipsInvariants:
    """AC13 — team entry in a fresh namespace with user_id="anonymous" is accepted."""

    def test_enterprise_team_in_fresh_namespace_accepted(
        self, catalog_factory: CatalogFactory
    ) -> None:
        catalog, repo = catalog_factory()
        stored = _seed_team(catalog, namespace="fresh-ent", user_id="anonymous")
        assert stored.user_id == "anonymous"
        assert repo.get("fresh-ent", stored.id) is not None


# --- AC14 — create returns stored shape ----------------------------------------


class TestCreateReturnsStored:
    """AC14 — return value is the persisted entry (minted namespace + reconciled payload)."""

    def test_returned_entry_is_persisted_shape(self, catalog_factory: CatalogFactory) -> None:
        catalog, repo = catalog_factory()
        stored = _seed_team(catalog, namespace="ns-ret")
        round_trip = repo.get("ns-ret", stored.id)
        assert round_trip is not None
        assert round_trip == stored


# --- AC15 — update missing target ---------------------------------------------


class TestUpdateMissing:
    """AC15 — update on a non-existent (namespace, id) raises ``EntryNotFoundError``."""

    def test_update_missing_raises(self, catalog_factory: CatalogFactory) -> None:
        catalog, _ = catalog_factory()
        entry = Entry(
            id="team",
            kind="team",
            namespace="ns-missing",
            model_type=_TEAM_TYPE,
            payload=_team_payload(),
        )
        with pytest.raises(EntryNotFoundError) as exc:
            catalog.update(entry)
        msg = str(exc.value)
        assert "not found" in msg
        assert "ns-missing" in msg
        assert "team" in msg


# --- AC16 — update ownership re-check / team self-transfer --------------------


class TestUpdateOwnership:
    """AC16 — update re-runs the ownership check, team entries included."""

    def test_update_sub_entry_user_id_mismatch_rejected(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        catalog, _ = catalog_factory()
        _seed_team(catalog, namespace="ns-upd", user_id="alice")
        agent_type = _register_agent_model(monkeypatch)
        agent = Entry(
            id="assistant",
            kind="agent",
            namespace="ns-upd",
            user_id="alice",
            model_type=agent_type,
            payload={"provider": "openai"},
        )
        catalog.create(agent)
        bad = agent.model_copy(update={"user_id": "bob"})
        with pytest.raises(CatalogValidationError) as exc:
            catalog.update(bad)
        msg = str(exc.value)
        assert "bob" in msg and "alice" in msg

    def test_update_team_user_id_transfer_rejected(self, catalog_factory: CatalogFactory) -> None:
        """A namespace with a real owner cannot change hands through a write.

        The team entry used to be exempt from the ownership check on the
        premise that rewriting its ``user_id`` was a deliberate transfer. Under
        ADR-023 the persisted owner is preserved on every write, so the premise
        is gone and the exemption with it — the check now runs for team entries
        exactly as it does for every other kind.
        """
        catalog, repo = catalog_factory()
        team = _seed_team(catalog, namespace="ns-team-transfer", user_id="alice")
        new = team.model_copy(update={"user_id": "bob"})
        with pytest.raises(CatalogValidationError) as exc:
            catalog.update(new)
        msg = str(exc.value)
        assert "bob" in msg and "alice" in msg
        # The persisted team is untouched.
        assert repo.get("ns-team-transfer", team.id).user_id == "alice"  # type: ignore[union-attr]


# --- AC17 — update never mints --------------------------------------------------


class TestUpdateDoesNotMint:
    """AC17 — update never mints; sentinel namespace path is missing-target."""

    def test_update_sentinel_namespace_raises_not_found(
        self, catalog_factory: CatalogFactory
    ) -> None:
        catalog, _ = catalog_factory()
        entry = Entry(
            id="team",
            kind="team",
            namespace=UNSET_NAMESPACE,
            model_type=_TEAM_TYPE,
            payload=_team_payload(),
        )
        with pytest.raises(EntryNotFoundError):
            catalog.update(entry)


# --- AC18, AC19, AC20 — delete -------------------------------------------------


class TestDeleteInboundRefs:
    """AC18 — delete is blocked by inbound refs and names every referrer."""

    def test_delete_blocked_lists_all_referrers(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        catalog, _ = catalog_factory()
        _seed_team(catalog, namespace="ns-del", user_id="anonymous")
        # Register agent and leaf models.
        agent_type = _register_agent_model(monkeypatch)
        leaf_type = _register_leaf_model(monkeypatch)
        leaf = Entry(
            id="id_gpt_41",
            kind="model",
            namespace="ns-del",
            model_type=leaf_type,
            payload={"provider": "openai"},
        )
        catalog.create(leaf)
        for aid in ("agent-a", "agent-b"):
            catalog.create(
                Entry(
                    id=aid,
                    kind="agent",
                    namespace="ns-del",
                    model_type=agent_type,
                    payload={"model_cfg": {"__ref__": "id_gpt_41"}},
                )
            )
        with pytest.raises(CatalogValidationError) as exc:
            catalog.delete("ns-del", "id_gpt_41")
        msg = str(exc.value)
        assert "agent-a" in msg
        assert "agent-b" in msg


class TestDeleteMissing:
    """AC19 — delete of a non-existent target raises ``EntryNotFoundError``."""

    def test_delete_missing_raises(self, catalog_factory: CatalogFactory) -> None:
        catalog, _ = catalog_factory()
        with pytest.raises(EntryNotFoundError) as exc:
            catalog.delete("ns-nope", "who")
        msg = str(exc.value)
        assert "not found" in msg
        assert "ns-nope" in msg
        assert "who" in msg


class TestDeleteClean:
    """AC20 — delete with no referrers removes the entry."""

    def test_delete_removes_entry(self, catalog_factory: CatalogFactory) -> None:
        catalog, repo = catalog_factory()
        team = _seed_team(catalog, namespace="ns-clean", user_id="anonymous")
        # The team entry has no inbound refs (no other entries exist).
        catalog.delete("ns-clean", team.id)
        assert repo.get("ns-clean", team.id) is None


# --- AC5 / Story 17.2 — meta singleton on create ---------------------------


_NAMESPACE_META_TYPE = "akgentic.catalog.models.namespace_meta.NamespaceMeta"


def _meta_entry(namespace: str, user_id: str, entry_id: str = "_meta") -> Entry:
    """Build a valid ``kind="meta"`` ``Entry`` for the singleton tests."""
    return Entry(
        id=entry_id,
        kind="meta",
        namespace=namespace,
        user_id=user_id,
        model_type=_NAMESPACE_META_TYPE,
        description="namespace metadata",
        payload={"name": namespace, "description": "", "properties": {}},
    )


class TestMetaSingletonOnCreate:
    """Story 17.2 AC5 — at most one ``kind="meta"`` entry per namespace."""

    def test_first_meta_create_succeeds(self, catalog_factory: CatalogFactory) -> None:
        """First meta entry in a namespace persists like any other entry."""
        catalog, _ = catalog_factory()
        team = _seed_team(catalog, namespace="tenant-42", user_id="alice")
        meta = _meta_entry(team.namespace, user_id="alice")
        stored = catalog.create(meta)
        assert stored.kind == "meta"
        assert stored.id == "_meta"
        assert stored.namespace == "tenant-42"
        # Round-trip: list_by_namespace returns it.
        rows = catalog.list_by_namespace("tenant-42")
        assert any(e.kind == "meta" and e.id == "_meta" for e in rows)

    def test_second_meta_create_rejected(self, catalog_factory: CatalogFactory) -> None:
        """Second meta entry (different id, same namespace) raises with pinned msg."""
        catalog, _ = catalog_factory()
        _seed_team(catalog, namespace="tenant-42", user_id="alice")
        catalog.create(_meta_entry("tenant-42", user_id="alice"))
        # A second meta entry with a different id must still be rejected.
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.create(_meta_entry("tenant-42", user_id="alice", entry_id="meta-extra"))
        assert "already has a meta entry" in exc_info.value.errors[0]
        # Namespace is interpolated in the message.
        assert "'tenant-42'" in exc_info.value.errors[0]

    def test_duplicate_id_check_wins_over_meta_singleton(
        self, catalog_factory: CatalogFactory
    ) -> None:
        """When the second meta create collides on (namespace, id), duplicate-id wins."""
        catalog, _ = catalog_factory()
        _seed_team(catalog, namespace="tenant-42", user_id="alice")
        catalog.create(_meta_entry("tenant-42", user_id="alice"))
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.create(_meta_entry("tenant-42", user_id="alice"))  # same id
        # The duplicate-id check is more specific and runs first.
        msg = exc_info.value.errors[0]
        assert "already exists" in msg

    def test_meta_singleton_does_not_block_other_namespaces(
        self, catalog_factory: CatalogFactory
    ) -> None:
        """A meta entry in namespace A does not prevent meta in namespace B."""
        catalog, _ = catalog_factory()
        _seed_team(catalog, namespace="tenant-a", user_id="alice")
        _seed_team(catalog, namespace="tenant-b", user_id="bob")
        catalog.create(_meta_entry("tenant-a", user_id="alice"))
        # Different namespace — should succeed.
        stored = catalog.create(_meta_entry("tenant-b", user_id="bob"))
        assert stored.namespace == "tenant-b"

    def test_meta_singleton_does_not_block_other_kinds(
        self, catalog_factory: CatalogFactory
    ) -> None:
        """The singleton check is skipped for kind != meta — non-meta paths unaffected."""
        catalog, _ = catalog_factory()
        team = _seed_team(catalog, namespace="tenant-42", user_id="alice")
        # Should succeed — kind=team is not the meta path.
        assert team.kind == "team"


class _MetaWithExtraField(NamespaceMeta):
    """A namespace-meta model carrying a field the writer has never heard of."""

    extra_field: str = "sentinel"


class TestNamespaceMetaWriter:
    """``Catalog.put_namespace_meta`` — the single writer for a ``_meta`` entry."""

    def test_a_namespace_meta_field_the_writer_never_names_reaches_the_stored_payload(
        self, catalog_factory: CatalogFactory
    ) -> None:
        """The payload is copied off the model, never rebuilt field by field.

        A field list here would be correct the day it is written and would
        silently drop whatever is added to ``NamespaceMeta`` afterwards — no
        error, no failed test, the value simply gone. Only a field the writer
        cannot possibly know about proves the copy (Golden Rule #12).

        The entry *construction* is what is under guard, so this exercises
        ``_meta_entry`` directly: the full write path validates the payload
        against the declared ``model_type`` and would refuse the unknown key
        before it could reach the entry.
        """
        catalog, _ = catalog_factory()
        entry = catalog._meta_entry(
            "ns-guard", _MetaWithExtraField(name="n", description="d"), "anonymous"
        )
        assert entry.payload["extra_field"] == "sentinel"
        assert entry.description == "d"

    def test_the_stored_meta_entry_carries_the_description(
        self, catalog_factory: CatalogFactory
    ) -> None:
        """The picker reads ``Entry.description``, so the writer must set it."""
        catalog, repo = catalog_factory()
        _seed_team(catalog, "ns-desc")
        stored, created = catalog.put_namespace_meta(
            "ns-desc", NamespaceMeta(name="N", description="a described namespace")
        )
        assert created is True
        assert stored.description == "a described namespace"
        persisted = repo.get("ns-desc", "_meta")
        assert persisted is not None
        assert persisted.description == "a described namespace"

    def test_a_second_write_updates_rather_than_creates(
        self, catalog_factory: CatalogFactory
    ) -> None:
        catalog, repo = catalog_factory()
        _seed_team(catalog, "ns-twice")
        catalog.put_namespace_meta("ns-twice", NamespaceMeta(name="first"))
        stored, created = catalog.put_namespace_meta("ns-twice", NamespaceMeta(name="second"))
        assert created is False
        assert stored.payload["name"] == "second"
        assert len([e for e in repo.list_by_namespace("ns-twice") if e.kind == "meta"]) == 1

    def test_the_team_owns_the_meta_entry_when_no_owner_is_given(
        self, catalog_factory: CatalogFactory
    ) -> None:
        catalog, _ = catalog_factory()
        _seed_team(catalog, "ns-owned", user_id="alice")
        stored, _created = catalog.put_namespace_meta("ns-owned", NamespaceMeta(name="N"))
        assert stored.user_id == "alice"

    def test_a_namespace_with_nothing_in_it_yet_falls_back_to_anonymous(
        self, catalog_factory: CatalogFactory
    ) -> None:
        """No team, no prior meta — the first write bootstraps the namespace."""
        catalog, _ = catalog_factory()
        stored, created = catalog.put_namespace_meta("ns-fresh", NamespaceMeta(name="N"))
        assert created is True
        assert stored.user_id == "anonymous"
