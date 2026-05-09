"""Tests for Story 17.4 / AC10, AC11 — per-Catalog shared-flag cache + invalidation.

The resolver caches the per-namespace shared flag for the lifetime of a
``Catalog`` instance, with cache invalidation on meta-entry mutation
(create / update / delete).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from akgentic.catalog.catalog import Catalog
from akgentic.catalog.models.entry import Entry
from akgentic.catalog.models.errors import CatalogValidationError

from .conftest import (
    CountingEntryRepository,
    FakeEntryRepository,
    make_meta_entry,
    register_akgentic_test_module,
)

_TEAM_TYPE = "akgentic.team.models.TeamCard"


class _Leaf(BaseModel):
    provider: str = "openai"


class _Holder(BaseModel):
    model_cfg: _Leaf | None = None


def _team_payload() -> dict[str, Any]:
    return {
        "name": "team",
        "description": "",
        "entry_point": {
            "card": {
                "role": "entry",
                "description": "",
                "skills": [],
                "agent_class": "akgentic.core.agent.Akgent",
                "config": {"name": "entry", "role": "entry"},
            },
            "headcount": 1,
            "members": [],
        },
        "members": [],
        "agent_profiles": [],
    }


@pytest.fixture
def model_paths(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    module_name = register_akgentic_test_module(
        monkeypatch,
        "tests_fixture_17_4_cache",
        Leaf=_Leaf,
        Holder=_Holder,
    )
    return f"{module_name}.Leaf", f"{module_name}.Holder"


def _seed_team(catalog: Catalog, namespace: str) -> None:
    catalog.create(
        Entry(
            id="team",
            kind="team",
            namespace=namespace,
            user_id=None,
            model_type=_TEAM_TYPE,
            payload=_team_payload(),
        )
    )


class TestSharedFlagCacheHit:
    """AC10 — repeated cross-ns resolves issue exactly one ``repository.get`` per ns."""

    def test_repeated_resolve_does_not_reissue_meta_lookup(
        self, model_paths: tuple[str, str]
    ) -> None:
        leaf, holder = model_paths
        inner = FakeEntryRepository()
        counting = CountingEntryRepository(inner)
        catalog = Catalog(counting)  # type: ignore[arg-type]
        _seed_team(catalog, "global")
        catalog.create(make_meta_entry("global", shared=True))
        catalog.create(
            Entry(
                id="shared-prompt",
                kind="prompt",
                namespace="global",
                user_id=None,
                model_type=leaf,
                payload={"provider": "shared"},
            )
        )
        _seed_team(catalog, "tenant-A")
        catalog.create(
            Entry(
                id="agent-1",
                kind="agent",
                namespace="tenant-A",
                user_id=None,
                model_type=holder,
                payload={
                    "model_cfg": {
                        "__ref__": "shared-prompt",
                        "__namespace__": "global",
                    }
                },
            )
        )
        # Cold cache: first resolve hits the meta lookup once.
        catalog.resolve_by_id("tenant-A", "agent-1")
        counting.reset()
        # Second resolve: cache should serve, no meta lookup.
        catalog.resolve_by_id("tenant-A", "agent-1")
        meta_lookups = sum(
            1 for name, args, _ in counting.calls if name == "get" and args == ("global", "_meta")
        )
        assert meta_lookups == 0, f"expected 0 meta lookups, got {meta_lookups}: {counting.calls}"


class TestSharedFlagCacheInvalidation:
    """AC10 — flipping the meta entry invalidates the cache."""

    def test_flip_shared_true_to_false_makes_resolve_fail(
        self, model_paths: tuple[str, str]
    ) -> None:
        leaf, holder = model_paths
        repo = FakeEntryRepository()
        catalog = Catalog(repo)
        _seed_team(catalog, "global")
        catalog.create(make_meta_entry("global", shared=True))
        catalog.create(
            Entry(
                id="shared-prompt",
                kind="prompt",
                namespace="global",
                user_id=None,
                model_type=leaf,
                payload={"provider": "shared"},
            )
        )
        _seed_team(catalog, "tenant-A")
        catalog.create(
            Entry(
                id="agent-1",
                kind="agent",
                namespace="tenant-A",
                user_id=None,
                model_type=holder,
                payload={
                    "model_cfg": {
                        "__ref__": "shared-prompt",
                        "__namespace__": "global",
                    }
                },
            )
        )
        # Cold resolve succeeds.
        catalog.resolve_by_id("tenant-A", "agent-1")

        # Flip global meta to shared=false via update — this must invalidate
        # the cache so the next resolve fails.
        catalog.update(make_meta_entry("global", shared=False))
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.resolve_by_id("tenant-A", "agent-1")
        assert "is not shared" in exc_info.value.errors[0]

    def test_delete_meta_invalidates_cache(self, model_paths: tuple[str, str]) -> None:
        leaf, holder = model_paths
        repo = FakeEntryRepository()
        catalog = Catalog(repo)
        _seed_team(catalog, "global")
        catalog.create(make_meta_entry("global", shared=True))
        catalog.create(
            Entry(
                id="shared-prompt",
                kind="prompt",
                namespace="global",
                user_id=None,
                model_type=leaf,
                payload={"provider": "shared"},
            )
        )
        _seed_team(catalog, "tenant-A")
        catalog.create(
            Entry(
                id="agent-1",
                kind="agent",
                namespace="tenant-A",
                user_id=None,
                model_type=holder,
                payload={
                    "model_cfg": {
                        "__ref__": "shared-prompt",
                        "__namespace__": "global",
                    }
                },
            )
        )
        # Prime the cache.
        catalog.resolve_by_id("tenant-A", "agent-1")

        # Deleting the meta entry must invalidate the cache.
        catalog.delete("global", "_meta")
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.resolve_by_id("tenant-A", "agent-1")
        assert "is not shared" in exc_info.value.errors[0]

    def test_create_meta_invalidates_cache(self, model_paths: tuple[str, str]) -> None:
        leaf, holder = model_paths
        repo = FakeEntryRepository()
        catalog = Catalog(repo)
        # No meta on global initially.
        _seed_team(catalog, "global")
        catalog.create(
            Entry(
                id="shared-prompt",
                kind="prompt",
                namespace="global",
                user_id=None,
                model_type=leaf,
                payload={"provider": "shared"},
            )
        )
        _seed_team(catalog, "tenant-A")
        # Prime the negative cache by trying a resolve that fails.
        try:
            catalog.resolve(
                Entry(
                    id="probe",
                    kind="agent",
                    namespace="tenant-A",
                    user_id="alice",
                    model_type=holder,
                    payload={
                        "model_cfg": {
                            "__ref__": "shared-prompt",
                            "__namespace__": "global",
                        }
                    },
                )
            )
        except CatalogValidationError:
            pass
        # Now mark global shared by creating a meta entry — cache should
        # invalidate so the next resolve succeeds.
        catalog.create(make_meta_entry("global", shared=True))
        # Now we can create the agent without hitting the gate.
        catalog.create(
            Entry(
                id="agent-1",
                kind="agent",
                namespace="tenant-A",
                user_id=None,
                model_type=holder,
                payload={
                    "model_cfg": {
                        "__ref__": "shared-prompt",
                        "__namespace__": "global",
                    }
                },
            )
        )


class TestSharedFlagCachePerInstance:
    """AC11 — the cache is per-Catalog-instance, not module-global."""

    def test_two_catalogs_have_independent_caches(self) -> None:
        repo = FakeEntryRepository()
        c1 = Catalog(repo)
        c2 = Catalog(repo)
        # Independent dicts — mutating one must not affect the other.
        c1._shared_flag_cache["global"] = True
        assert c2._shared_flag_cache == {}


class TestSharedFlagExactStringSemantics:
    """AC1 — only the literal lowercase string "true" enables sharing."""

    @pytest.mark.parametrize(
        "shared_value",
        ["false", "True", "TRUE", "1", "", "yes"],
    )
    def test_non_true_values_are_not_shared(
        self, shared_value: str, model_paths: tuple[str, str]
    ) -> None:
        leaf, holder = model_paths
        repo = FakeEntryRepository()
        catalog = Catalog(repo)
        _seed_team(catalog, "global")
        catalog.create(make_meta_entry("global", shared=shared_value))
        catalog.create(
            Entry(
                id="shared-prompt",
                kind="prompt",
                namespace="global",
                user_id=None,
                model_type=leaf,
                payload={"provider": "shared"},
            )
        )
        _seed_team(catalog, "tenant-A")
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.create(
                Entry(
                    id="agent-1",
                    kind="agent",
                    namespace="tenant-A",
                    user_id=None,
                    model_type=holder,
                    payload={
                        "model_cfg": {
                            "__ref__": "shared-prompt",
                            "__namespace__": "global",
                        }
                    },
                )
            )
        assert "is not shared" in exc_info.value.errors[0]
