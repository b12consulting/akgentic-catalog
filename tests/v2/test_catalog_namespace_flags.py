"""Namespace ``_meta`` flags are read live, so every replica sees the same answer.

A ``Catalog`` stands in for one server replica; a write straight to the shared
repository stands in for another. The catalog must read that write next.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from akgentic.catalog.catalog import Catalog
from akgentic.catalog.models.entry import Entry
from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.queries import EntryQuery

from .conftest import (
    _NAMESPACE_META_TYPE,
    CountingEntryRepository,
    FakeEntryRepository,
    _seed_team,
    make_entry,
    make_meta_entry,
    register_akgentic_test_module,
)


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


def _catalog_with_raw_meta(payload: dict[str, object]) -> Catalog:
    """A catalog whose ``ns/_meta`` holds ``payload`` as-is (no validation on write)."""
    repo = FakeEntryRepository()
    repo.put(
        Entry(
            id="_meta",
            kind="meta",
            namespace="ns",
            user_id="anonymous",
            model_type=_NAMESPACE_META_TYPE,
            payload=payload,
        )
    )
    return Catalog(repo)


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
        b = Catalog(repo)
        repo.put(make_meta_entry("ns", public=True, shareable=False, user_id="alice"))
        repo.put(make_entry(namespace="ns", user_id="alice"))
        with Catalog.as_caller("bob"):
            assert len(b.list(EntryQuery(namespace="ns"))) == 2
            b.get("ns", "entry-1")

        repo.put(make_meta_entry("ns", public=False, shareable=False, user_id="alice"))

        with Catalog.as_caller("bob"):
            assert b.list(EntryQuery(namespace="ns")) == []
            with pytest.raises(EntryNotFoundError):
                b.get("ns", "entry-1")

    def test_shareable(self, model_paths: tuple[str, str]) -> None:
        leaf, holder = model_paths
        repo = FakeEntryRepository()
        b = Catalog(repo)
        _seed_team(b, "global")
        b.create(make_meta_entry("global", shareable=True, public=False))
        b.create(make_entry(id="shared-prompt", kind="prompt", namespace="global", model_type=leaf))
        _seed_team(b, "tenant-A")
        b.create(_agent(holder, "agent-1"))

        repo.put(make_meta_entry("global", shareable=False, public=False))

        with pytest.raises(CatalogValidationError) as exc_info:
            b.create(_agent(holder, "agent-2"))
        assert "is not shareable" in exc_info.value.errors[0]


class TestListReadsEachNamespaceMetaOncePerCall:
    @staticmethod
    def _seed(counting: CountingEntryRepository) -> None:
        for ns in ("ns-0", "ns-1", "ns-2"):
            for id in ("t-0", "t-1"):
                counting.inner.put(make_entry(id=id, namespace=ns, user_id="alice"))
        counting.inner.put(make_entry(namespace="ns-bob", user_id="bob"))

    @staticmethod
    def _meta_reads(counting: CountingEntryRepository) -> int:
        return sum(1 for name, args, _ in counting.calls if name == "get" and args[1] == "_meta")

    def test_one_read_per_distinct_non_owned_namespace_every_call(
        self, counting_catalog: tuple[Catalog, CountingEntryRepository]
    ) -> None:
        catalog, counting = counting_catalog
        self._seed(counting)
        with Catalog.as_caller("bob"):
            assert len(catalog.list(EntryQuery(kind="tool"))) == 1
            assert self._meta_reads(counting) == 3
            counting.reset()
            catalog.list(EntryQuery(kind="tool"))
            assert self._meta_reads(counting) == 3

    def test_community_tier_reads_no_meta(
        self, counting_catalog: tuple[Catalog, CountingEntryRepository]
    ) -> None:
        catalog, counting = counting_catalog
        self._seed(counting)
        assert len(catalog.list(EntryQuery(kind="tool"))) == 7
        assert self._meta_reads(counting) == 0


class TestOnlyAStrictTrueGrantsAFlag:
    @pytest.mark.parametrize("flag", ["public", "shareable"])
    @pytest.mark.parametrize("value", ["true", "True", 1, [1], {"x": 1}, False])
    def test_a_value_that_is_not_strict_true_grants_nothing(self, flag: str, value: object) -> None:
        catalog = _catalog_with_raw_meta({"name": "ns", flag: value})
        assert catalog._is_namespace_public("ns") is False
        assert catalog._is_namespace_shareable("ns") is False

    @pytest.mark.parametrize(
        "payload",
        [
            {"name": "ns"},
            {"name": "ns", "properties": {"shared": "true"}},
            {"public": True, "shareable": True},
        ],
        ids=["absent", "legacy-nested", "does-not-validate"],
    )
    def test_meta_without_a_typed_true_grants_nothing(self, payload: dict[str, object]) -> None:
        catalog = _catalog_with_raw_meta(payload)
        assert catalog._is_namespace_public("ns") is False
        assert catalog._is_namespace_shareable("ns") is False

    def test_no_meta_entry_grants_nothing(self) -> None:
        catalog = Catalog(FakeEntryRepository())
        assert catalog._is_namespace_public("ns") is False
        assert catalog._is_namespace_shareable("ns") is False

    def test_a_typed_true_grants_the_flag(self) -> None:
        catalog = _catalog_with_raw_meta({"name": "ns", "public": True, "shareable": True})
        assert catalog._is_namespace_public("ns") is True
        assert catalog._is_namespace_shareable("ns") is True
