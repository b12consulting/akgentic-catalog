"""Tests for the catalog write path's owner resolution.

ADR-023 supersedes ADR-028 §Decision 7's unconditional stamp. The rule under
test:

    On write, the owner is the **persisted** owner when one exists. When it
    does not — the resource is new, or its persisted owner is
    ``ANONYMOUS_USER_ID`` — the **authenticated caller** is stamped. The body /
    YAML ``user_id`` is never authoritative in either case.

So an admin editing a namespace owned by someone else no longer takes it over,
while a namespace with no real owner is still claimed by whoever saves it.

When no caller is set (community tier, ``_caller_user_id`` contextvar is
``None``), behaviour is byte-unchanged: ``create`` / ``update`` keep the body
``user_id``; ``clone`` uses ``dst_user_id`` (default ``"anonymous"``);
``import_namespace_yaml`` keeps the YAML ``user_id``.

Every branch of the resolution is exercised — persisted owner present,
persisted owner anonymous/absent, and no caller at all — across YAML + Mongo
backends via the ``catalog_factory`` fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from akgentic.catalog.catalog import Catalog
from akgentic.catalog.models.entry import ANONYMOUS_USER_ID, Entry
from akgentic.catalog.repositories.base import EntryRepository
from akgentic.catalog.repositories.yaml import YamlEntryRepository

from ..conftest import team_payload
from .conftest import (
    CatalogFactory,
    CountingEntryRepository,
    make_meta_entry,
    register_akgentic_test_module,
)

_TEAM_TYPE = "akgentic.team.models.TeamCard"


class _Leaf(BaseModel):
    """Permissive leaf payload used for stamping tests."""

    provider: str = "openai"


@pytest.fixture
def leaf_type(monkeypatch: pytest.MonkeyPatch) -> str:
    module_name = register_akgentic_test_module(
        monkeypatch,
        "tests_fixture_27_2_caller_owner_stamping",
        Leaf=_Leaf,
    )
    return f"{module_name}.Leaf"


def _seed_team(catalog: Catalog, namespace: str, *, user_id: str = "anonymous") -> Entry:
    return catalog.create(
        Entry(
            id="team",
            kind="team",
            namespace=namespace,
            user_id=user_id,
            model_type=_TEAM_TYPE,
            payload=team_payload(),
        )
    )


def _leaf_entry(namespace: str, id: str, leaf_type: str, *, user_id: str = "anonymous") -> Entry:
    return Entry(
        id=id,
        kind="prompt",
        namespace=namespace,
        user_id=user_id,
        model_type=leaf_type,
        payload={"provider": "x"},
    )


class _ExplodingRepository:
    """Repository double that fails the test on ANY attribute access.

    The owner resolution takes the persisted owner as an argument, so it must
    never reach for the repository itself. Anything the helper touches here —
    a read, a write, even an attribute lookup — surfaces as a failure naming
    what it reached for.
    """

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"_stamp_owner touched the repository: .{name}")


# --- _stamp_owner ---------------------------------------------------------


class TestOwnerResolution:
    """The helper itself: persisted owner in, resolved owner out, no I/O."""

    def _catalog(self) -> Catalog:
        exploding: Any = _ExplodingRepository()
        return Catalog(exploding)

    def test_a_persisted_owner_beats_both_the_caller_and_the_body(self, leaf_type: str) -> None:
        catalog = self._catalog()
        entry = _leaf_entry("ns", "leaf", leaf_type, user_id="mallory")
        with Catalog.as_caller("admin"):
            resolved = catalog._stamp_owner(entry, "alice")
        assert resolved.user_id == "alice"

    @pytest.mark.parametrize("persisted", [None, ANONYMOUS_USER_ID])
    def test_the_caller_is_stamped_when_there_is_no_owner_to_preserve(
        self, leaf_type: str, persisted: str | None
    ) -> None:
        catalog = self._catalog()
        entry = _leaf_entry("ns", "leaf", leaf_type, user_id="mallory")
        with Catalog.as_caller("admin"):
            resolved = catalog._stamp_owner(entry, persisted)
        assert resolved.user_id == "admin"

    @pytest.mark.parametrize("persisted", [None, ANONYMOUS_USER_ID, "alice"])
    @pytest.mark.parametrize("incoming", [ANONYMOUS_USER_ID, "mallory"])
    def test_community_tier_preserves_the_incoming_user_id(
        self, leaf_type: str, persisted: str | None, incoming: str
    ) -> None:
        catalog = self._catalog()
        entry = _leaf_entry("ns", "leaf", leaf_type, user_id=incoming)
        resolved = catalog._stamp_owner(entry, persisted)
        assert resolved.user_id == incoming


# --- create ---------------------------------------------------------------


class TestStampOnCreate:
    def test_create_under_caller_stamps_owner(
        self, catalog_factory: CatalogFactory, leaf_type: str
    ) -> None:
        catalog, repo = catalog_factory()
        with Catalog.as_caller("gpiroux"):
            _seed_team(catalog, "ns")
            catalog.create(_leaf_entry("ns", "leaf", leaf_type, user_id="anonymous"))
        persisted = repo.get("ns", "leaf")
        assert persisted is not None
        assert persisted.user_id == "gpiroux"
        # Team entry is also stamped (AC6) — keeps the ownership anchor uniform.
        team = repo.get("ns", "team")
        assert team is not None
        assert team.user_id == "gpiroux"

    def test_create_no_caller_is_byte_unchanged(
        self, catalog_factory: CatalogFactory, leaf_type: str
    ) -> None:
        catalog, repo = catalog_factory()
        _seed_team(catalog, "ns")
        catalog.create(_leaf_entry("ns", "leaf", leaf_type, user_id="anonymous"))
        persisted = repo.get("ns", "leaf")
        assert persisted is not None
        assert persisted.user_id == "anonymous"

    def test_create_into_a_fresh_namespace_is_owned_by_the_caller(
        self, catalog_factory: CatalogFactory, leaf_type: str
    ) -> None:
        """Nothing is persisted yet, so there is no owner to preserve."""
        catalog, repo = catalog_factory()
        with Catalog.as_caller("admin"):
            team = _seed_team(catalog, "ns-fresh", user_id="mallory")
        assert team.user_id == "admin"
        persisted = repo.get("ns-fresh", "team")
        assert persisted is not None
        assert persisted.user_id == "admin"

    def test_create_check_ownership_passes_under_caller(
        self, catalog_factory: CatalogFactory, leaf_type: str
    ) -> None:
        # A non-team/non-meta entry whose incoming user_id differs from the
        # caller still succeeds because the stamp fires BEFORE _check_ownership.
        catalog, repo = catalog_factory()
        with Catalog.as_caller("gpiroux"):
            _seed_team(catalog, "ns")
            catalog.create(_leaf_entry("ns", "leaf", leaf_type, user_id="someone-else"))
        persisted = repo.get("ns", "leaf")
        assert persisted is not None
        assert persisted.user_id == "gpiroux"


# --- update ---------------------------------------------------------------


class TestStampOnUpdate:
    def test_update_under_caller_ignores_the_body_user_id(
        self, catalog_factory: CatalogFactory, leaf_type: str
    ) -> None:
        catalog, repo = catalog_factory()
        with Catalog.as_caller("gpiroux"):
            _seed_team(catalog, "ns")
            catalog.create(_leaf_entry("ns", "leaf", leaf_type, user_id="anonymous"))
            # Candidate carries a foreign user_id; the persisted owner
            # (gpiroux, stamped on create) survives it.
            candidate = _leaf_entry("ns", "leaf", leaf_type, user_id="someone-else")
            candidate = candidate.model_copy(update={"description": "updated"})
            catalog.update(candidate)
        persisted = repo.get("ns", "leaf")
        assert persisted is not None
        assert persisted.user_id == "gpiroux"

    def test_update_preserves_another_users_ownership(
        self, catalog_factory: CatalogFactory, leaf_type: str
    ) -> None:
        """An admin editing alice's entry leaves alice as the owner.

        The body carries a third name so a green result cannot come from the
        body being honoured — only the persisted owner produces ``alice``.
        """
        catalog, repo = catalog_factory()
        _seed_team(catalog, "ns-owned", user_id="alice")
        catalog.create(_leaf_entry("ns-owned", "leaf", leaf_type, user_id="alice"))
        candidate = _leaf_entry("ns-owned", "leaf", leaf_type, user_id="mallory").model_copy(
            update={"description": "edited by an admin"}
        )
        with Catalog.as_caller("admin"):
            stored = catalog.update(candidate)
        assert stored.user_id == "alice"
        persisted = repo.get("ns-owned", "leaf")
        assert persisted is not None
        assert persisted.user_id == "alice"
        assert persisted.description == "edited by an admin"

    def test_update_claims_an_anonymous_owned_entry_for_the_caller(
        self, catalog_factory: CatalogFactory, leaf_type: str
    ) -> None:
        """An unowned entry is claimed by whoever next saves it (ADR-023 §D3).

        The leaf is seeded through the repository rather than ``create``: a
        namespace anchored to ``admin`` would reject an anonymous-owned
        sub-entry at create time, and the anonymous leaf under a real anchor
        is exactly the pre-fix state this fallback exists to repair.
        """
        catalog, repo = catalog_factory()
        _seed_team(catalog, "ns-anon", user_id="admin")
        repo.put(_leaf_entry("ns-anon", "leaf", leaf_type, user_id=ANONYMOUS_USER_ID))
        candidate = _leaf_entry("ns-anon", "leaf", leaf_type, user_id="mallory").model_copy(
            update={"description": "claimed"}
        )
        with Catalog.as_caller("admin"):
            stored = catalog.update(candidate)
        assert stored.user_id == "admin"
        persisted = repo.get("ns-anon", "leaf")
        assert persisted is not None
        assert persisted.user_id == "admin"

    def test_update_no_caller_keeps_candidate_user_id(
        self, catalog_factory: CatalogFactory, leaf_type: str
    ) -> None:
        catalog, repo = catalog_factory()
        # Seed a team owned by "owner" and a sub-entry owned by "owner".
        _seed_team(catalog, "ns", user_id="owner")
        catalog.create(_leaf_entry("ns", "leaf", leaf_type, user_id="owner"))
        candidate = _leaf_entry("ns", "leaf", leaf_type, user_id="owner").model_copy(
            update={"description": "updated"}
        )
        catalog.update(candidate)
        persisted = repo.get("ns", "leaf")
        assert persisted is not None
        assert persisted.user_id == "owner"


# --- clone ----------------------------------------------------------------


class TestStampOnClone:
    def test_clone_under_caller_supersedes_dst_user_id(
        self, catalog_factory: CatalogFactory, leaf_type: str
    ) -> None:
        catalog, repo = catalog_factory()
        # Source namespace owned by "gpiroux" so the caller can see it.
        with Catalog.as_caller("gpiroux"):
            _seed_team(catalog, "src")
            catalog.create(_leaf_entry("src", "leaf", leaf_type))
            _seed_team(catalog, "dst")
            # dst_user_id explicitly set to something else — caller supersedes it.
            catalog.clone("src", "leaf", "dst", dst_user_id="ignored-owner")
        cloned = repo.get("dst", "leaf")
        assert cloned is not None
        assert cloned.user_id == "gpiroux"

    def test_clone_no_caller_uses_dst_user_id(
        self, catalog_factory: CatalogFactory, leaf_type: str
    ) -> None:
        catalog, repo = catalog_factory()
        _seed_team(catalog, "src")
        catalog.create(_leaf_entry("src", "leaf", leaf_type))
        _seed_team(catalog, "dst")
        catalog.clone("src", "leaf", "dst", dst_user_id="bob")
        cloned = repo.get("dst", "leaf")
        assert cloned is not None
        assert cloned.user_id == "bob"

    def test_clone_no_caller_default_anonymous(
        self, catalog_factory: CatalogFactory, leaf_type: str
    ) -> None:
        catalog, repo = catalog_factory()
        _seed_team(catalog, "src")
        catalog.create(_leaf_entry("src", "leaf", leaf_type))
        _seed_team(catalog, "dst")
        catalog.clone("src", "leaf", "dst")
        cloned = repo.get("dst", "leaf")
        assert cloned is not None
        assert cloned.user_id == "anonymous"


# --- import_namespace_yaml ------------------------------------------------


class TestStampOnImport:
    def _seed_anonymous_namespace(self, catalog: Catalog, namespace: str, leaf_type: str) -> None:
        _seed_team(catalog, namespace, user_id="anonymous")
        catalog.create(make_meta_entry(namespace, shareable=False, public=False))
        catalog.create(_leaf_entry(namespace, "leaf", leaf_type, user_id="anonymous"))

    def _seed_owned_namespace(
        self, catalog: Catalog, namespace: str, leaf_type: str, *, user_id: str
    ) -> None:
        """Seed a NON-public namespace owned by ``user_id``.

        ``public: false`` is load-bearing, not incidental. The ownership anchor
        must be read straight off the repository: a public namespace passes the
        visibility filter, so a filtered read would find the anchor anyway and
        the defect this file guards would stay invisible.
        """
        _seed_team(catalog, namespace, user_id=user_id)
        catalog.create(make_meta_entry(namespace, shareable=False, public=False, user_id=user_id))
        catalog.create(_leaf_entry(namespace, "leaf", leaf_type, user_id=user_id))

    def test_import_over_another_users_namespace_preserves_every_owner(
        self, catalog_factory: CatalogFactory, leaf_type: str
    ) -> None:
        """The admin "Save namespace" action no longer takes the namespace over.

        The bundle's own ``user_id`` is rewritten to a third name so a green
        result cannot come from the YAML being honoured — only the persisted
        anchor produces ``alice``.
        """
        catalog, repo = catalog_factory()
        self._seed_owned_namespace(catalog, "ns-alice", leaf_type, user_id="alice")
        bundle = catalog.export_namespace_yaml("ns-alice").replace(
            "user_id: alice", "user_id: mallory", 1
        )
        assert "mallory" in bundle

        with Catalog.as_caller("admin"):
            persisted = catalog.import_namespace_yaml(bundle)

        for entry in persisted:
            assert entry.user_id == "alice", f"{entry.id} lost its owner"
        for entry_id in ("team", "leaf", "_meta"):
            stored = repo.get("ns-alice", entry_id)
            assert stored is not None, entry_id
            assert stored.user_id == "alice", f"{entry_id} lost its owner"

    def test_import_into_an_absent_namespace_stamps_the_caller(
        self, catalog_factory: CatalogFactory, leaf_type: str
    ) -> None:
        """No persisted anchor at all — the caller owns what they create."""
        catalog, repo = catalog_factory()
        self._seed_owned_namespace(catalog, "ns-gone", leaf_type, user_id="alice")
        bundle = catalog.export_namespace_yaml("ns-gone")
        catalog.delete_namespace("ns-gone")

        with Catalog.as_caller("admin"):
            persisted = catalog.import_namespace_yaml(bundle)

        for entry in persisted:
            assert entry.user_id == "admin", f"{entry.id} not stamped"
        for entry_id in ("team", "leaf", "_meta"):
            stored = repo.get("ns-gone", entry_id)
            assert stored is not None, entry_id
            assert stored.user_id == "admin", f"{entry_id} not stamped"

    def test_the_anchor_is_read_before_the_swap_touches_anything(
        self, tmp_path: Path, leaf_type: str
    ) -> None:
        """The swap destroys the state the anchor is read from.

        Reading it afterwards — or per entry, mid-swap — answers from state the
        import has already rewritten. This pins the read ahead of the first
        write of the whole import.
        """
        counting = CountingEntryRepository(YamlEntryRepository(tmp_path))
        repo: EntryRepository = counting
        catalog = Catalog(repo)
        self._seed_owned_namespace(catalog, "ns-order", leaf_type, user_id="alice")
        bundle = catalog.export_namespace_yaml("ns-order")
        counting.reset()

        with Catalog.as_caller("admin"):
            catalog.import_namespace_yaml(bundle)

        anchor_reads = [
            i
            for i, (name, args, _) in enumerate(counting.calls)
            if name == "get_by_kind" and args == ("ns-order", "team")
        ]
        writes = [i for i, (name, _, _) in enumerate(counting.calls) if name in ("put", "delete")]
        assert anchor_reads, "the ownership anchor was never read"
        assert writes, "the import wrote nothing"
        assert anchor_reads[0] < writes[0]

    def test_import_under_caller_stamps_every_entry_and_meta(
        self, catalog_factory: CatalogFactory, leaf_type: str
    ) -> None:
        catalog, repo = catalog_factory()
        # Build a bundle from an anonymous-owned namespace (the reported bug).
        self._seed_anonymous_namespace(catalog, "src", leaf_type)
        bundle = catalog.export_namespace_yaml("src")

        # Re-import the same bundle (same namespace) under a caller.
        with Catalog.as_caller("gpiroux"):
            persisted = catalog.import_namespace_yaml(bundle)

        for entry in persisted:
            assert entry.user_id == "gpiroux", f"{entry.id} not stamped"
        # The hoisted _meta entry (upserted) is also caller-owned.
        meta = repo.get("src", "_meta")
        assert meta is not None
        assert meta.user_id == "gpiroux"
        # And the persisted team / leaf in the repository.
        team = repo.get("src", "team")
        leaf = repo.get("src", "leaf")
        assert team is not None and team.user_id == "gpiroux"
        assert leaf is not None and leaf.user_id == "gpiroux"

    def test_import_no_caller_keeps_yaml_user_id(
        self, catalog_factory: CatalogFactory, leaf_type: str
    ) -> None:
        catalog, repo = catalog_factory()
        self._seed_anonymous_namespace(catalog, "src", leaf_type)
        bundle = catalog.export_namespace_yaml("src")
        persisted = catalog.import_namespace_yaml(bundle)
        for entry in persisted:
            assert entry.user_id == "anonymous"
        meta = repo.get("src", "_meta")
        assert meta is not None
        assert meta.user_id == "anonymous"

    def test_import_bundle_uniformity_preserved_under_caller(
        self, catalog_factory: CatalogFactory, leaf_type: str
    ) -> None:
        # A multi-entry bundle whose YAML entries carry mixed/anonymous owners
        # still passes _validate_bundle_invariants under a caller (all stamped
        # to the caller first), and persists successfully.
        catalog, repo = catalog_factory()
        _seed_team(catalog, "src", user_id="anonymous")
        catalog.create(make_meta_entry("src", shareable=False, public=False))
        catalog.create(_leaf_entry("src", "leaf-a", leaf_type, user_id="anonymous"))
        catalog.create(_leaf_entry("src", "leaf-b", leaf_type, user_id="anonymous"))
        bundle = catalog.export_namespace_yaml("src")

        with Catalog.as_caller("gpiroux"):
            persisted = catalog.import_namespace_yaml(bundle)

        ids = {e.id for e in persisted}
        assert {"team", "leaf-a", "leaf-b"} <= ids
        for e in persisted:
            assert e.user_id == "gpiroux"
