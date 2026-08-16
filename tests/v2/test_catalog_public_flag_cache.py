"""Tests for Story 18.4 — the public flag, its caching, and its invalidation.

Mirrors ``test_catalog_shared_flag_cache.py`` for the ``public`` flag.
``_is_namespace_public`` reads the namespace's ``_meta`` entry with
strict-bool semantics — only a real ``True`` opts the namespace in; the
parsed metadata is cached per-Catalog-instance and dropped on any meta-entry
mutation via ``_invalidate_meta_caches``, which serves both flags.
"""

from __future__ import annotations

from typing import Any

from akgentic.catalog.catalog import Catalog
from akgentic.catalog.models.entry import Entry
from akgentic.catalog.models.namespace_meta import NamespaceMeta

from ..conftest import team_payload
from .conftest import (
    CountingEntryRepository,
    FakeEntryRepository,
    make_meta_entry,
)

_TEAM_TYPE = "akgentic.team.models.TeamCard"
_NAMESPACE_META_TYPE = "akgentic.catalog.models.namespace_meta.NamespaceMeta"


def _seed_team(catalog: Catalog, namespace: str, user_id: str = "anonymous") -> None:
    catalog.create(
        Entry(
            id="team",
            kind="team",
            namespace=namespace,
            user_id=user_id,
            model_type=_TEAM_TYPE,
            payload=team_payload(),
        )
    )


def _put_raw_meta(repo: FakeEntryRepository, namespace: str, payload: dict[str, Any]) -> None:
    """Stash a raw meta entry directly in the repo (bypass NamespaceMeta validation)."""
    repo.put(
        Entry(
            id="_meta",
            kind="meta",
            namespace=namespace,
            user_id="anonymous",
            model_type=_NAMESPACE_META_TYPE,
            description="",
            payload=payload,
        )
    )


class TestPublicFlagCacheLazyPopulation:
    """``_is_namespace_public`` reads the meta entry once per namespace, then caches."""

    def test_first_lookup_queries_repository(self) -> None:
        inner = FakeEntryRepository()
        counting = CountingEntryRepository(inner)
        catalog = Catalog(counting)  # type: ignore[arg-type]
        _seed_team(catalog, "ns")
        catalog.create(make_meta_entry("ns", public=True))
        counting.reset()
        # First lookup hits the repository.
        assert catalog._is_namespace_public("ns") is True
        meta_lookups = sum(
            1 for name, args, _ in counting.calls if name == "get" and args == ("ns", "_meta")
        )
        assert meta_lookups == 1

    def test_second_lookup_uses_cache_no_repository_round_trip(self) -> None:
        inner = FakeEntryRepository()
        counting = CountingEntryRepository(inner)
        catalog = Catalog(counting)  # type: ignore[arg-type]
        _seed_team(catalog, "ns")
        catalog.create(make_meta_entry("ns", public=True))
        # Prime the cache.
        catalog._is_namespace_public("ns")
        counting.reset()
        # Second lookup must NOT hit the repository.
        assert catalog._is_namespace_public("ns") is True
        meta_lookups = sum(
            1 for name, args, _ in counting.calls if name == "get" and args == ("ns", "_meta")
        )
        assert meta_lookups == 0

    def test_returns_false_when_no_meta_entry(self) -> None:
        repo = FakeEntryRepository()
        catalog = Catalog(repo)
        _seed_team(catalog, "ns-no-meta")
        # No meta entry → False (private by default).
        assert catalog._is_namespace_public("ns-no-meta") is False

    def test_returns_false_when_meta_payload_lacks_public_key(self) -> None:
        repo = FakeEntryRepository()
        catalog = Catalog(repo)
        _seed_team(catalog, "ns")
        # Meta exists but payload has no `public` key (legacy shape).
        _put_raw_meta(repo, "ns", {"name": "ns", "description": "", "properties": {}})
        assert catalog._is_namespace_public("ns") is False

    def test_returns_false_when_meta_payload_public_is_truthy_but_not_strict_true(self) -> None:
        """Strict-bool — only typed ``True`` flips the flag."""
        for non_true_value in ("true", "True", 1, [1], {"x": 1}):
            repo = FakeEntryRepository()
            catalog = Catalog(repo)
            _seed_team(catalog, "ns")
            _put_raw_meta(
                repo,
                "ns",
                {
                    "name": "ns",
                    "description": "",
                    "properties": {},
                    "public": non_true_value,
                },
            )
            assert catalog._is_namespace_public("ns") is False, (
                f"public={non_true_value!r} must NOT be coerced to True"
            )

    def test_returns_true_when_meta_payload_public_is_strict_true(self) -> None:
        repo = FakeEntryRepository()
        catalog = Catalog(repo)
        _seed_team(catalog, "ns")
        catalog.create(make_meta_entry("ns", public=True))
        assert catalog._is_namespace_public("ns") is True


class TestUnreadableMetaGrantsNothing:
    """A meta entry the catalog cannot parse leaves every flag at its safe default."""

    def test_a_meta_payload_that_does_not_validate_grants_neither_flag(self) -> None:
        """No name means no usable metadata — and no access either.

        Reading the raw dict used to answer each flag independently, so a
        payload missing the one required field still handed out ``public`` and
        ``shareable``. Parsing it as a whole makes it all-or-nothing, and the
        nothing is what an unreadable declaration must get.
        """
        repo = FakeEntryRepository()
        catalog = Catalog(repo)
        _seed_team(catalog, "ns-broken")
        _put_raw_meta(repo, "ns-broken", {"public": True, "shareable": True})
        assert catalog._is_namespace_public("ns-broken") is False
        assert catalog._is_namespace_shareable("ns-broken") is False


class TestNamespaceWithNoMetaCachesItsMiss:
    """A namespace having no metadata is itself an answer worth remembering."""

    def test_a_namespace_with_no_meta_is_read_once_not_once_per_call(self) -> None:
        """A cache that treats ``None`` as a miss caches nothing at all.

        Namespaces without metadata are the common case, so re-reading them on
        every visibility check would be a silent per-call round trip — one no
        test on a namespace that *has* a meta entry could ever notice.
        """
        inner = FakeEntryRepository()
        counting = CountingEntryRepository(inner)
        catalog = Catalog(counting)  # type: ignore[arg-type]
        _seed_team(catalog, "ns-bare")
        counting.reset()
        assert catalog._is_namespace_public("ns-bare") is False
        assert catalog._is_namespace_public("ns-bare") is False
        meta_lookups = sum(
            1 for name, args, _ in counting.calls if name == "get" and args == ("ns-bare", "_meta")
        )
        assert meta_lookups == 1


class TestPublicFlagCacheInvalidation:
    """Meta-entry mutations drop the cached flag for the affected namespace."""

    def test_create_meta_entry_invalidates_cache(self) -> None:
        repo = FakeEntryRepository()
        catalog = Catalog(repo)
        _seed_team(catalog, "ns")
        # Cache the negative result (no meta yet).
        assert catalog._is_namespace_public("ns") is False
        # Now create the meta entry; cache MUST invalidate.
        catalog.create(make_meta_entry("ns", public=True))
        assert catalog._is_namespace_public("ns") is True

    def test_update_meta_entry_invalidates_cache(self) -> None:
        repo = FakeEntryRepository()
        catalog = Catalog(repo)
        _seed_team(catalog, "ns")
        catalog.create(make_meta_entry("ns", public=True))
        assert catalog._is_namespace_public("ns") is True
        # Flip to public=False and update; cache MUST invalidate.
        catalog.update(make_meta_entry("ns", public=False))
        assert catalog._is_namespace_public("ns") is False

    def test_delete_meta_entry_invalidates_cache(self) -> None:
        repo = FakeEntryRepository()
        catalog = Catalog(repo)
        _seed_team(catalog, "ns")
        catalog.create(make_meta_entry("ns", public=True))
        assert catalog._is_namespace_public("ns") is True
        catalog.delete("ns", "_meta")
        # After delete, the next lookup re-reads the (now-absent) meta.
        assert catalog._is_namespace_public("ns") is False

    def test_create_non_meta_entry_does_not_invalidate_cache(self) -> None:
        inner = FakeEntryRepository()
        counting = CountingEntryRepository(inner)
        catalog = Catalog(counting)  # type: ignore[arg-type]
        _seed_team(catalog, "ns")
        catalog.create(make_meta_entry("ns", public=True))
        # Prime the cache.
        catalog._is_namespace_public("ns")
        counting.reset()
        # Create a non-meta entry — cache must not invalidate.
        catalog.create(
            Entry(
                id="prompt-1",
                kind="prompt",
                namespace="ns",
                user_id="anonymous",
                model_type=_NAMESPACE_META_TYPE,
                payload={"name": "p", "description": "", "properties": {}},
            )
        )
        catalog._is_namespace_public("ns")
        meta_lookups = sum(
            1 for name, args, _ in counting.calls if name == "get" and args == ("ns", "_meta")
        )
        assert meta_lookups == 0

    def test_namespace_isolation_meta_write_in_one_ns_does_not_invalidate_other_ns(self) -> None:
        inner = FakeEntryRepository()
        counting = CountingEntryRepository(inner)
        catalog = Catalog(counting)  # type: ignore[arg-type]
        _seed_team(catalog, "ns-A")
        _seed_team(catalog, "ns-B")
        catalog.create(make_meta_entry("ns-A", public=True))
        catalog.create(make_meta_entry("ns-B", public=False))
        # Prime both.
        catalog._is_namespace_public("ns-A")
        catalog._is_namespace_public("ns-B")
        counting.reset()
        # Update ns-A's meta; ns-B's cache MUST stay intact.
        catalog.update(make_meta_entry("ns-A", public=False))
        catalog._is_namespace_public("ns-B")
        ns_b_lookups = sum(
            1 for name, args, _ in counting.calls if name == "get" and args == ("ns-B", "_meta")
        )
        assert ns_b_lookups == 0


class TestPublicFlagCachePerInstance:
    """The cache is per-Catalog-instance, not module-global."""

    def test_two_catalogs_have_independent_caches(self) -> None:
        repo = FakeEntryRepository()
        c1 = Catalog(repo)
        c2 = Catalog(repo)
        c1._meta_cache["ns"] = NamespaceMeta(name="ns", public=True)
        assert c2._meta_cache == {}


class TestUnifiedMetaCacheInvalidation:
    """One meta write clears what BOTH flags are read from, in a single call.

    Both flags now come off one cached read of the namespace's metadata, so
    the invalidation contract that used to span two cache slots must still
    hold end to end for both.
    """

    def test_meta_write_invalidates_both_caches(self) -> None:
        repo = FakeEntryRepository()
        catalog = Catalog(repo)
        _seed_team(catalog, "ns")
        catalog.create(make_meta_entry("ns", shareable=True, public=True))
        # Prime both caches.
        assert catalog._is_namespace_shareable("ns") is True
        assert catalog._is_namespace_public("ns") is True
        # Update meta to flip both flags off — both caches MUST invalidate.
        catalog.update(make_meta_entry("ns", shareable=False, public=False))
        assert catalog._is_namespace_shareable("ns") is False
        assert catalog._is_namespace_public("ns") is False

    def test_non_meta_write_invalidates_neither(self) -> None:
        inner = FakeEntryRepository()
        counting = CountingEntryRepository(inner)
        catalog = Catalog(counting)  # type: ignore[arg-type]
        _seed_team(catalog, "ns")
        catalog.create(make_meta_entry("ns", shareable=True, public=True))
        # Prime both caches.
        catalog._is_namespace_shareable("ns")
        catalog._is_namespace_public("ns")
        counting.reset()
        # Write a non-meta entry — neither cache slot should invalidate.
        catalog.create(
            Entry(
                id="prompt-1",
                kind="prompt",
                namespace="ns",
                user_id="anonymous",
                model_type=_NAMESPACE_META_TYPE,
                payload={"name": "p", "description": "", "properties": {}},
            )
        )
        catalog._is_namespace_shareable("ns")
        catalog._is_namespace_public("ns")
        meta_lookups = sum(
            1 for name, args, _ in counting.calls if name == "get" and args == ("ns", "_meta")
        )
        assert meta_lookups == 0
