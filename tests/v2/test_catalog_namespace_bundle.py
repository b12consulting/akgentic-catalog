"""Tests for ``Catalog.export_namespace_yaml`` / ``import_namespace_yaml`` (Story 16.2).

Every round-trip test runs against both backends via the parametrised
``catalog_factory`` fixture. The dangling-ref and prepare-for-write-failure
no-op assertions use the single-backend ``counting_catalog`` fixture so the
test can verify ``put`` was never invoked.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from akgentic.catalog.catalog import Catalog
from akgentic.catalog.models.entry import Entry
from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.catalog.serialization import dump_namespace, load_namespace

from .conftest import (
    CatalogFactory,
    CountingEntryRepository,
    register_akgentic_test_module,
)

_TEAM_TYPE = "akgentic.team.models.TeamCard"


def _team_payload() -> dict[str, Any]:
    """Return a minimal valid ``TeamCard`` payload (copied from test_catalog_crud)."""
    return {
        "name": "team",
        "description": "",
        "entry_point": {
            "card": {
                "role": "entry",
                "description": "entry",
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


class _LeafPayloadModel(BaseModel):
    provider: str = "openai"
    temperature: float = 0.0


class _AgentPayloadModel(BaseModel):
    provider: str = "openai"
    temperature: float = 0.0
    model_cfg: _LeafPayloadModel | None = None


def _register_agent_models(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """Register stub agent + leaf payload models; return their FQCN paths."""
    module_name = register_akgentic_test_module(
        monkeypatch,
        "tests_fixture_16_2_bundle",
        _AgentPayloadModel=_AgentPayloadModel,
        _LeafPayloadModel=_LeafPayloadModel,
    )
    return f"{module_name}._AgentPayloadModel", f"{module_name}._LeafPayloadModel"


def _seed_team(
    catalog: Catalog,
    namespace: str,
    user_id: str | None = "alice",
) -> Entry:
    return catalog.create(
        Entry(
            id="team",
            kind="team",
            namespace=namespace,
            user_id=user_id,
            model_type=_TEAM_TYPE,
            payload=_team_payload(),
        )
    )


def _seed_agent(
    catalog: Catalog,
    namespace: str,
    id: str,
    user_id: str | None = "alice",
    payload: dict[str, Any] | None = None,
    model_type: str | None = None,
) -> Entry:
    return catalog.create(
        Entry(
            id=id,
            kind="agent",
            namespace=namespace,
            user_id=user_id,
            model_type=model_type or "akgentic.core.agent_card.AgentCard",
            payload=payload if payload is not None else _agent_payload(id),
        )
    )


def _agent_payload(id: str = "a") -> dict[str, Any]:
    return {
        "role": "r",
        "description": "",
        "skills": [],
        "agent_class": "akgentic.core.agent.Akgent",
        "config": {"name": id, "role": "r"},
        "routes_to": [],
        "metadata": {},
    }


# --- Export -----------------------------------------------------------------


class TestExportNamespaceYaml:
    """``Catalog.export_namespace_yaml``."""

    def test_export_empty_namespace_raises(self, catalog_factory: CatalogFactory) -> None:
        catalog, _ = catalog_factory()
        with pytest.raises(CatalogValidationError):
            catalog.export_namespace_yaml("nope")

    def test_export_round_trip_idempotent(self, catalog_factory: CatalogFactory) -> None:
        catalog, _ = catalog_factory()
        _seed_team(catalog, "ns-rt")
        _seed_agent(catalog, "ns-rt", "a")
        _seed_agent(catalog, "ns-rt", "b")
        yaml_text = catalog.export_namespace_yaml("ns-rt")
        parsed = load_namespace(yaml_text)
        # dump → load → dump yields the same yaml.
        again = dump_namespace(parsed)
        assert again == yaml_text


# --- Import -----------------------------------------------------------------


class TestImportNamespaceYaml:
    """``Catalog.import_namespace_yaml``."""

    def test_import_into_empty_namespace_creates_entries(
        self, catalog_factory: CatalogFactory
    ) -> None:
        catalog, repo = catalog_factory()
        _seed_team(catalog, "ns-src")
        _seed_agent(catalog, "ns-src", "a")
        yaml_text = catalog.export_namespace_yaml("ns-src")
        # Rewrite to a different destination namespace.
        new_text = yaml_text.replace("ns-src", "ns-dst")
        result = catalog.import_namespace_yaml(new_text)
        assert {e.id for e in result} == {"team", "a"}
        fetched = repo.list_by_namespace("ns-dst")
        assert {e.id for e in fetched} == {"team", "a"}

    def test_import_atomic_replace(self, catalog_factory: CatalogFactory) -> None:
        catalog, repo = catalog_factory()
        _seed_team(catalog, "ns-swap")
        _seed_agent(catalog, "ns-swap", "A")
        _seed_agent(catalog, "ns-swap", "B")
        _seed_agent(catalog, "ns-swap", "C")

        # New bundle: team + updated A + new D; B and C removed.
        bundle_entries = [
            Entry(
                id="team",
                kind="team",
                namespace="ns-swap",
                user_id="alice",
                model_type=_TEAM_TYPE,
                payload=_team_payload(),
            ),
            Entry(
                id="A",
                kind="agent",
                namespace="ns-swap",
                user_id="alice",
                model_type="akgentic.core.agent_card.AgentCard",
                payload=_agent_payload("A-updated"),
                description="updated A",
            ),
            Entry(
                id="D",
                kind="agent",
                namespace="ns-swap",
                user_id="alice",
                model_type="akgentic.core.agent_card.AgentCard",
                payload=_agent_payload("D"),
            ),
        ]
        yaml_text = dump_namespace(bundle_entries)
        catalog.import_namespace_yaml(yaml_text)
        fetched = {e.id: e for e in repo.list_by_namespace("ns-swap")}
        assert set(fetched.keys()) == {"team", "A", "D"}
        assert fetched["A"].description == "updated A"

    def test_import_rejects_bundle_with_no_team(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        catalog, _ = catalog_factory()
        agent_type, _ = _register_agent_models(monkeypatch)
        bundle = [
            Entry(
                id="a",
                kind="agent",
                namespace="ns-nt",
                user_id="alice",
                model_type=agent_type,
                payload={},
            )
        ]
        yaml_text = dump_namespace(bundle)
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.import_namespace_yaml(yaml_text)
        assert any("no team entry" in e for e in exc_info.value.errors)

    def test_import_rejects_multiple_team_entries(self, catalog_factory: CatalogFactory) -> None:
        catalog, _ = catalog_factory()
        bundle = [
            Entry(
                id="team1",
                kind="team",
                namespace="ns-mt",
                user_id="alice",
                model_type=_TEAM_TYPE,
                payload=_team_payload(),
            ),
            Entry(
                id="team2",
                kind="team",
                namespace="ns-mt",
                user_id="alice",
                model_type=_TEAM_TYPE,
                payload=_team_payload(),
            ),
        ]
        yaml_text = dump_namespace(bundle)
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.import_namespace_yaml(yaml_text)
        assert any("multiple team entries" in e for e in exc_info.value.errors)

    def test_validate_bundle_invariants_rejects_ownership_mismatch(
        self, catalog_factory: CatalogFactory
    ) -> None:
        """Exercise ``Catalog._validate_bundle_invariants`` with hand-crafted mismatch.

        The YAML bundle format assigns ``user_id`` from the document-level
        key to every entry, so a mismatch cannot surface through
        ``load_namespace``. This test invokes the invariant helper directly
        with a list whose entries disagree on ``user_id`` to prove the
        Catalog-level check still rejects the mismatch — matching the
        ``_check_ownership`` error shape referenced in AC22.
        """
        catalog, _ = catalog_factory()
        prepared = [
            Entry(
                id="team",
                kind="team",
                namespace="ns-own",
                user_id="alice",
                model_type=_TEAM_TYPE,
                payload=_team_payload(),
            ),
            Entry(
                id="rogue",
                kind="agent",
                namespace="ns-own",
                user_id="bob",
                model_type="akgentic.core.agent_card.AgentCard",
                payload=_agent_payload("rogue"),
            ),
        ]
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog._validate_bundle_invariants(prepared)
        assert any(
            "Ownership mismatch" in e and "entry 'rogue'" in e for e in exc_info.value.errors
        )

    def test_import_rejects_dangling_ref(
        self,
        counting_catalog: tuple[Catalog, CountingEntryRepository],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        catalog, counting = counting_catalog
        agent_type, leaf_type = _register_agent_models(monkeypatch)
        # Pre-seed a team + a leaf target that exists in the current
        # namespace state. ``prepare_for_write`` therefore succeeds (the ref
        # resolves against the repo), but the bundle check catches that
        # ``ghost`` is absent from the imported bundle's id set.
        _seed_team(catalog, "ns-dr")
        catalog.create(
            Entry(
                id="ghost",
                kind="model",
                namespace="ns-dr",
                user_id="alice",
                model_type=leaf_type,
                payload={"provider": "openai", "temperature": 0.0},
            )
        )
        # The bundle omits 'ghost' but an agent payload refers to it.
        bundle_team = Entry(
            id="team",
            kind="team",
            namespace="ns-dr",
            user_id="alice",
            model_type=_TEAM_TYPE,
            payload=_team_payload(),
        )
        bundle_agent = Entry(
            id="dangler",
            kind="agent",
            namespace="ns-dr",
            user_id="alice",
            model_type=agent_type,
            payload={
                "provider": "openai",
                "model_cfg": {"__ref__": "ghost", "__type__": leaf_type},
            },
        )
        yaml_text = dump_namespace([bundle_team, bundle_agent])
        counting.reset()
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.import_namespace_yaml(yaml_text)
        assert any("not found in bundle" in e for e in exc_info.value.errors)
        # Atomic-failure contract: no put during the failing call.
        assert counting.count("put") == 0
        assert counting.count("delete") == 0

    def test_import_rejects_prepare_for_write_failure(
        self,
        counting_catalog: tuple[Catalog, CountingEntryRepository],
    ) -> None:
        catalog, counting = counting_catalog
        _seed_team(catalog, "ns-pfw")
        # Build a bundle whose agent payload is structurally incompatible
        # with AgentCard (missing required ``role`` field) — prepare_for_write
        # will surface a CatalogValidationError during model validation.
        team_entry = Entry(
            id="team",
            kind="team",
            namespace="ns-pfw",
            user_id="alice",
            model_type=_TEAM_TYPE,
            payload=_team_payload(),
        )
        bad_agent = Entry(
            id="broken",
            kind="agent",
            namespace="ns-pfw",
            user_id="alice",
            model_type="akgentic.core.agent_card.AgentCard",
            payload={"not_a_real_field": "x"},  # missing required role/agent_class/config
        )
        yaml_text = dump_namespace([team_entry, bad_agent])
        counting.reset()
        with pytest.raises(CatalogValidationError):
            catalog.import_namespace_yaml(yaml_text)
        # Atomic-failure contract: no put / delete during the failing call.
        assert counting.count("put") == 0
        assert counting.count("delete") == 0

    def test_import_malformed_yaml_raises(self, catalog_factory: CatalogFactory) -> None:
        catalog, _ = catalog_factory()
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.import_namespace_yaml("{{{ not yaml }")
        assert any("Failed to parse bundle YAML" in e for e in exc_info.value.errors)

    def test_import_with_in_bundle_ref_succeeds(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        catalog, repo = catalog_factory()
        agent_type, leaf_type = _register_agent_models(monkeypatch)
        # Pre-seed the namespace with a team so create() is happy.
        bundle = [
            Entry(
                id="team",
                kind="team",
                namespace="ns-ref",
                user_id="alice",
                model_type=_TEAM_TYPE,
                payload=_team_payload(),
            ),
            Entry(
                id="leaf",
                kind="model",
                namespace="ns-ref",
                user_id="alice",
                model_type=leaf_type,
                payload={"provider": "openai", "temperature": 0.0},
            ),
            Entry(
                id="agent",
                kind="agent",
                namespace="ns-ref",
                user_id="alice",
                model_type=agent_type,
                payload={
                    "provider": "openai",
                    "model_cfg": {"__ref__": "leaf", "__type__": leaf_type},
                },
            ),
        ]
        yaml_text = dump_namespace(bundle)
        catalog.import_namespace_yaml(yaml_text)
        stored = {e.id: e for e in repo.list_by_namespace("ns-ref")}
        assert set(stored.keys()) == {"team", "leaf", "agent"}
        # Ref markers preserved in stored payload.
        assert stored["agent"].payload["model_cfg"] == {
            "__ref__": "leaf",
            "__type__": leaf_type,
        }
