"""Namespace ``_meta`` flags are read live, so every replica sees the same answer.

Two ``Catalog`` instances over one repository stand in for two server
replicas. A flag written through one must be what the other reads next.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from akgentic.catalog.catalog import Catalog
from akgentic.catalog.models.entry import Entry
from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.namespace_meta import NamespaceMeta
from akgentic.catalog.models.queries import EntryQuery

from ..conftest import team_payload
from .conftest import (
    CountingEntryRepository,
    FakeEntryRepository,
    make_entry,
    make_meta_entry,
    register_akgentic_test_module,
)

_TEAM_TYPE = "akgentic.team.models.TeamCard"
_NAMESPACE_META_TYPE = "akgentic.catalog.models.namespace_meta.NamespaceMeta"


class _Leaf(BaseModel):
    provider: str = "openai"


class _Holder(BaseModel):
    model_cfg: _Leaf | None = None


@pytest.fixture
def model_paths(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    module_name = register_akgentic_test_module(
        monkeypatch, "tests_fixture_namespace_flags", Leaf=_Leaf, Holder=_Holder
    )
    return f"{module_name}.Leaf", f"{module_name}.Holder"


def _seed_team(catalog: Catalog, namespace: str) -> None:
    catalog.create(
        Entry(
            id="team",
            kind="team",
            namespace=namespace,
            user_id="anonymous",
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


def _agent(holder: str, id: str) -> Entry:
    return Entry(
        id=id,
        kind="agent",
        namespace="tenant-A",
        user_id="anonymous",
        model_type=holder,
        payload={"model_cfg": {"__ref__": "shared-prompt", "__namespace__": "global"}},
    )


class TestFlagWrittenOnOneReplicaIsReadOnAnother:
    def test_public(self) -> None:
        repo = FakeEntryRepository()
        a, b = Catalog(repo), Catalog(repo)
        repo.put(make_meta_entry("ns", public=True, shareable=False, user_id="alice"))
        repo.put(make_entry(namespace="ns", user_id="alice"))
        with Catalog.as_caller("bob"):
            assert len(b.list(EntryQuery(namespace="ns"))) == 2
            b.get("ns", "entry-1")

        a.put_namespace_meta("ns", NamespaceMeta(name="ns", public=False))

        with Catalog.as_caller("bob"):
            assert b.list(EntryQuery(namespace="ns")) == []
            with pytest.raises(EntryNotFoundError):
                b.get("ns", "entry-1")

    def test_shareable(self, model_paths: tuple[str, str]) -> None:
        leaf, holder = model_paths
        repo = FakeEntryRepository()
        a, b = Catalog(repo), Catalog(repo)
        _seed_team(b, "global")
        b.create(make_meta_entry("global", shareable=True, public=False))
        b.create(make_entry(id="shared-prompt", kind="prompt", namespace="global", model_type=leaf))
        _seed_team(b, "tenant-A")
        b.create(_agent(holder, "agent-1"))

        a.put_namespace_meta("global", NamespaceMeta(name="global", shareable=False))

        with pytest.raises(CatalogValidationError) as exc_info:
            b.create(_agent(holder, "agent-2"))
        assert "is not shareable" in exc_info.value.errors[0]


class TestListReadsEachNamespaceMetaOncePerCall:
    @staticmethod
    def _seeded() -> tuple[Catalog, CountingEntryRepository]:
        counting = CountingEntryRepository(FakeEntryRepository())
        for ns in ("ns-0", "ns-1", "ns-2"):
            for id in ("t-0", "t-1"):
                counting.inner.put(make_entry(id=id, namespace=ns, user_id="alice"))
        counting.inner.put(make_entry(namespace="ns-bob", user_id="bob"))
        return Catalog(counting), counting  # type: ignore[arg-type]

    @staticmethod
    def _meta_reads(counting: CountingEntryRepository) -> int:
        return sum(1 for name, args, _ in counting.calls if name == "get" and args[1] == "_meta")

    def test_one_read_per_distinct_non_owned_namespace_every_call(self) -> None:
        catalog, counting = self._seeded()
        with Catalog.as_caller("bob"):
            assert len(catalog.list(EntryQuery(kind="tool"))) == 1
            assert self._meta_reads(counting) == 3
            counting.reset()
            catalog.list(EntryQuery(kind="tool"))
            assert self._meta_reads(counting) == 3

    def test_community_tier_reads_no_meta(self) -> None:
        catalog, counting = self._seeded()
        assert len(catalog.list(EntryQuery(kind="tool"))) == 7
        assert self._meta_reads(counting) == 0


class TestPublicFlagStrictBool:
    def test_returns_false_when_no_meta_entry(self) -> None:
        repo = FakeEntryRepository()
        catalog = Catalog(repo)
        _seed_team(catalog, "ns-no-meta")
        assert catalog._is_namespace_public("ns-no-meta") is False

    def test_returns_false_when_meta_payload_lacks_public_key(self) -> None:
        repo = FakeEntryRepository()
        catalog = Catalog(repo)
        _seed_team(catalog, "ns")
        _put_raw_meta(repo, "ns", {"name": "ns", "description": "", "properties": {}})
        assert catalog._is_namespace_public("ns") is False

    def test_returns_false_when_meta_payload_public_is_truthy_but_not_strict_true(self) -> None:
        for non_true_value in ("true", "True", 1, [1], {"x": 1}):
            repo = FakeEntryRepository()
            catalog = Catalog(repo)
            _seed_team(catalog, "ns")
            _put_raw_meta(
                repo,
                "ns",
                {"name": "ns", "description": "", "properties": {}, "public": non_true_value},
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
    def test_a_meta_payload_that_does_not_validate_grants_neither_flag(self) -> None:
        """No name means no usable metadata — and no access either."""
        repo = FakeEntryRepository()
        catalog = Catalog(repo)
        _seed_team(catalog, "ns-broken")
        _put_raw_meta(repo, "ns-broken", {"public": True, "shareable": True})
        assert catalog._is_namespace_public("ns-broken") is False
        assert catalog._is_namespace_shareable("ns-broken") is False


class TestShareableFlagStrictBool:
    @pytest.mark.parametrize(
        "non_true_payload",
        [
            {"properties": {"shared": "true"}},
            {"shareable": "true"},
            {"shareable": 1},
            {"shareable": False},
            {},
        ],
        ids=["legacy-nested", "string-true", "int-1", "bool-false", "absent"],
    )
    def test_non_typed_true_values_are_not_shareable(
        self, non_true_payload: dict[str, object], model_paths: tuple[str, str]
    ) -> None:
        leaf, holder = model_paths
        repo = FakeEntryRepository()
        catalog = Catalog(repo)
        _seed_team(catalog, "global")
        _put_raw_meta(
            repo,
            "global",
            {"name": "global", "description": "", "properties": {}} | non_true_payload,
        )
        catalog.create(
            make_entry(id="shared-prompt", kind="prompt", namespace="global", model_type=leaf)
        )
        _seed_team(catalog, "tenant-A")
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.create(_agent(holder, "agent-1"))
        assert "is not shareable" in exc_info.value.errors[0]

    def test_typed_bool_true_at_root_is_shareable(self, model_paths: tuple[str, str]) -> None:
        leaf, holder = model_paths
        repo = FakeEntryRepository()
        catalog = Catalog(repo)
        _seed_team(catalog, "global")
        catalog.create(make_meta_entry("global", shareable=True))
        catalog.create(
            make_entry(id="shared-prompt", kind="prompt", namespace="global", model_type=leaf)
        )
        _seed_team(catalog, "tenant-A")
        catalog.create(_agent(holder, "agent-1"))
