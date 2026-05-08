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
from akgentic.catalog.serialization import (
    dump_namespace,
    dump_namespace_v2,
    load_namespace,
)

from .conftest import (
    CatalogFactory,
    CountingEntryRepository,
    make_meta_entry,
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
        """Story 17.5: export → re-export is byte-identical for the v2 wire shape."""
        catalog, _ = catalog_factory()
        _seed_team(catalog, "ns-rt")
        _seed_agent(catalog, "ns-rt", "a")
        _seed_agent(catalog, "ns-rt", "b")
        yaml_text = catalog.export_namespace_yaml("ns-rt")
        again = catalog.export_namespace_yaml("ns-rt")
        # Two consecutive exports against unchanged state ⇒ byte-identical YAML.
        assert again == yaml_text
        # And the v2 shape parses to entries (load_namespace handles both shapes).
        parsed = load_namespace(yaml_text)
        assert {e.id for e in parsed} == {"team", "a", "b"}


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


# --- Story 17.2 — bundle import meta singleton ----------------------------


_NAMESPACE_META_TYPE_BUNDLE = "akgentic.catalog.models.namespace_meta.NamespaceMeta"


def _meta_entry_for_bundle(
    namespace: str,
    user_id: str | None,
    entry_id: str = "_meta",
    name: str = "primary",
) -> Entry:
    return Entry(
        id=entry_id,
        kind="meta",
        namespace=namespace,
        user_id=user_id,
        model_type=_NAMESPACE_META_TYPE_BUNDLE,
        description=f"meta {entry_id}",
        payload={"name": name, "description": "", "properties": {}},
    )


class TestBundleImportMetaSingleton:
    """Story 17.2 AC4 — bundle import rejects two ``kind="meta"`` entries."""

    def test_bundle_with_two_meta_entries_rejected_no_writes(
        self,
        catalog_factory: CatalogFactory,
    ) -> None:
        catalog, repo = catalog_factory()
        # Pre-seed the namespace so we can verify state is unchanged on rollback.
        team = _seed_team(catalog, namespace="tenant-42", user_id="alice")
        before = sorted(e.id for e in repo.list_by_namespace("tenant-42"))

        bundle = [
            Entry(
                id="team",
                kind="team",
                namespace="tenant-42",
                user_id="alice",
                model_type=_TEAM_TYPE,
                payload=_team_payload(),
            ),
            _meta_entry_for_bundle("tenant-42", user_id="alice", entry_id="_meta", name="A"),
            _meta_entry_for_bundle(
                "tenant-42",
                user_id="alice",
                entry_id="meta-extra",
                name="B",
            ),
        ]
        yaml_text = dump_namespace(bundle)

        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.import_namespace_yaml(yaml_text)
        assert any("has multiple meta entries" in m for m in exc_info.value.errors)
        # Atomic-failure contract: the namespace state is byte-identical to
        # the pre-import state.
        after = sorted(e.id for e in repo.list_by_namespace("tenant-42"))
        assert after == before
        assert team.id == "team"

    def test_bundle_with_one_meta_entry_imports_cleanly(
        self,
        catalog_factory: CatalogFactory,
    ) -> None:
        catalog, repo = catalog_factory()
        bundle = [
            Entry(
                id="team",
                kind="team",
                namespace="tenant-42",
                user_id="alice",
                model_type=_TEAM_TYPE,
                payload=_team_payload(),
            ),
            _meta_entry_for_bundle("tenant-42", user_id="alice", name="primary"),
        ]
        yaml_text = dump_namespace(bundle)
        catalog.import_namespace_yaml(yaml_text)
        ids = {e.id for e in repo.list_by_namespace("tenant-42")}
        assert ids == {"team", "_meta"}


class TestImportBundleCrossNs:
    """Story 17.4 — cross-ns markers exempt from bundle dangling-ref rule + shared-flag gate."""

    def test_cross_ns_ref_with_shared_target_imports(
        self,
        catalog_factory: CatalogFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bundle agent payload references global.shared — target exists in a shared namespace."""
        agent_type, leaf_type = _register_agent_models(monkeypatch)
        catalog, repo = catalog_factory()
        # Seed the global target + meta with shared=true.
        _seed_team(catalog, "global", user_id=None)
        catalog.create(make_meta_entry("global", shared=True))
        catalog.create(
            Entry(
                id="shared-prompt",
                kind="prompt",
                namespace="global",
                user_id=None,
                model_type=leaf_type,
                payload={"provider": "openai"},
            )
        )
        bundle = [
            Entry(
                id="team",
                kind="team",
                namespace="tenant-A",
                user_id=None,
                model_type=_TEAM_TYPE,
                payload=_team_payload(),
            ),
            Entry(
                id="agent-1",
                kind="agent",
                namespace="tenant-A",
                user_id=None,
                model_type=agent_type,
                payload={
                    "model_cfg": {
                        "__ref__": "shared-prompt",
                        "__namespace__": "global",
                    }
                },
            ),
        ]
        yaml_text = dump_namespace(bundle)
        catalog.import_namespace_yaml(yaml_text)
        ids = {e.id for e in repo.list_by_namespace("tenant-A")}
        assert ids == {"team", "agent-1"}

    def test_cross_ns_ref_to_non_shared_namespace_rejected(
        self,
        catalog_factory: CatalogFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bundle with cross-ns ref to a non-shared namespace fails at prepare_for_write."""
        agent_type, _leaf = _register_agent_models(monkeypatch)
        catalog, _repo = catalog_factory()
        # global has no meta entry — namespace is not shared.
        bundle = [
            Entry(
                id="team",
                kind="team",
                namespace="tenant-A",
                user_id=None,
                model_type=_TEAM_TYPE,
                payload=_team_payload(),
            ),
            Entry(
                id="agent-1",
                kind="agent",
                namespace="tenant-A",
                user_id=None,
                model_type=agent_type,
                payload={"model_cfg": {"__ref__": "global.shared-prompt"}},
            ),
        ]
        yaml_text = dump_namespace(bundle)
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.import_namespace_yaml(yaml_text)
        msg = " | ".join(exc_info.value.errors)
        assert "is not shared" in msg

    def test_cross_ns_ref_target_missing_in_shared_namespace(
        self,
        catalog_factory: CatalogFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Shared namespace exists but target id missing ⇒ standard not-found."""
        agent_type, _leaf = _register_agent_models(monkeypatch)
        catalog, _repo = catalog_factory()
        # Mark global shared but seed no target id.
        _seed_team(catalog, "global", user_id=None)
        catalog.create(make_meta_entry("global", shared=True))
        bundle = [
            Entry(
                id="team",
                kind="team",
                namespace="tenant-A",
                user_id=None,
                model_type=_TEAM_TYPE,
                payload=_team_payload(),
            ),
            Entry(
                id="agent-1",
                kind="agent",
                namespace="tenant-A",
                user_id=None,
                model_type=agent_type,
                payload={"model_cfg": {"__ref__": "global.does-not-exist"}},
            ),
        ]
        yaml_text = dump_namespace(bundle)
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.import_namespace_yaml(yaml_text)
        msg = " | ".join(exc_info.value.errors)
        assert "not found in namespace" in msg
        assert "global" in msg


# --- Story 17.5 — Export external_refs ------------------------------------


class _OpenLeafModel(BaseModel):
    """Permissive leaf payload model — accepts arbitrary ref-style sub-payloads.

    Used by Story 17.5 export-side tests where the test seeds cross-ns
    targets whose payloads themselves carry refs (transitive reachability,
    cycle protection). The ``ConfigDict(extra='allow')`` accepts the
    populated/resolved sub-payload values that ``populate_refs`` produces
    at validation time, plus arbitrary leaf fields the tests stash.
    """

    model_config = {"extra": "allow"}
    provider: str = "openai"


class _OpenAgentModel(BaseModel):
    """Permissive agent payload model — accepts arbitrary ref-style sub-payloads.

    Mirrors :class:`_OpenLeafModel` for entries seeded as ``kind="agent"``.
    Story 17.5's tests stash cross-ns ref markers under arbitrary keys
    (``model_cfg``, ``list``, ``ref``, …) so the model must accept any
    field at validation time without rejecting unknown keys.
    """

    model_config = {"extra": "allow"}
    provider: str = "openai"


def _register_open_models(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """Register the permissive agent/leaf payload models; return (agent_type, leaf_type) FQCNs."""
    module_name = register_akgentic_test_module(
        monkeypatch,
        "tests_fixture_17_5_open",
        _OpenAgentModel=_OpenAgentModel,
        _OpenLeafModel=_OpenLeafModel,
    )
    return f"{module_name}._OpenAgentModel", f"{module_name}._OpenLeafModel"


def _seed_shared_namespace_with_targets(
    catalog: Catalog,
    namespace: str,
    user_id: str | None,
    target_ids: list[str],
    leaf_type: str | None = None,
) -> str:
    """Seed a shared namespace (team + meta with shared=true) and N model-kind targets.

    Returns the leaf-type FQCN used for the seeded targets so callers can
    pass it back when constructing referring agents. If ``leaf_type`` is
    ``None``, the test stub ``_LeafPayloadModel`` registered by
    :func:`_register_agent_models` is assumed to already be importable in
    the current process — callers MUST pass a registered FQCN explicitly.
    """
    assert leaf_type is not None, "callers must pass an already-registered leaf_type FQCN"
    _seed_team(catalog, namespace, user_id=user_id)
    catalog.create(make_meta_entry(namespace, shared=True, user_id=user_id))
    for tid in target_ids:
        catalog.create(
            Entry(
                id=tid,
                kind="model",
                namespace=namespace,
                user_id=user_id,
                model_type=leaf_type,
                payload={"provider": "openai", "temperature": 0.0},
            )
        )
    return leaf_type


def _seed_referrer_agent(
    catalog: Catalog,
    namespace: str,
    agent_id: str,
    payload: dict[str, Any],
    user_id: str | None,
    agent_type: str,
) -> Entry:
    """Seed a single agent in ``namespace`` whose payload carries cross-ns refs.

    ``agent_type`` is the FQCN registered for the permissive
    ``_OpenAgentModel`` via :func:`_register_open_models` so the test's
    payload accepts any field shape (refs, lists, nested overrides).
    """
    return catalog.create(
        Entry(
            id=agent_id,
            kind="agent",
            namespace=namespace,
            user_id=user_id,
            model_type=agent_type,
            payload=payload,
        )
    )


class TestExportExternalRefs:
    """Story 17.5 ACs 3, 5, 6, 7, 8, 9 — ``external_refs:`` content semantics.

    Tests directly exercise the catalog service's repository to seed
    cross-ns ref scenarios that ``populate_refs`` would otherwise reject —
    the export endpoint is a display projection, NOT a validator, so the
    tests focus on what export emits given persisted state, not on whether
    that state was reachable through the create pipeline.
    """

    def test_no_cross_ns_refs_yields_empty_external_refs(
        self, catalog_factory: CatalogFactory
    ) -> None:
        """A namespace with no cross-ns refs exports ``external_refs: []``."""
        import yaml

        catalog, _ = catalog_factory()
        _seed_team(catalog, "tenant-clean", user_id="alice")
        _seed_agent(catalog, "tenant-clean", "a", user_id="alice")
        text = catalog.export_namespace_yaml("tenant-clean")
        doc = yaml.safe_load(text)
        assert doc["external_refs"] == []

    def test_one_cross_ns_ref_yields_one_external(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One agent with one cross-ns ref ⇒ one ``external_refs:`` entry."""
        import yaml

        agent_type, leaf_type = _register_open_models(monkeypatch)
        catalog, _ = catalog_factory()
        _seed_shared_namespace_with_targets(
            catalog, "global", user_id=None, target_ids=["m1"], leaf_type=leaf_type
        )
        _seed_team(catalog, "tenant-A", user_id=None)
        _seed_referrer_agent(
            catalog,
            "tenant-A",
            "a",
            payload={"model_cfg": {"__ref__": "m1", "__namespace__": "global"}},
            user_id=None,
            agent_type=agent_type,
        )
        text = catalog.export_namespace_yaml("tenant-A")
        doc = yaml.safe_load(text)
        ext = doc["external_refs"]
        assert len(ext) == 1
        assert ext[0]["namespace"] == "global"
        assert ext[0]["id"] == "m1"
        assert ext[0]["kind"] == "model"

    def test_multiple_refs_to_same_target_dedup_to_one(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC3 dedup: multiple refs from different entries to ``(global, m1)`` collapse to one."""
        import yaml

        agent_type, leaf_type = _register_open_models(monkeypatch)
        catalog, _ = catalog_factory()
        _seed_shared_namespace_with_targets(
            catalog, "global", user_id=None, target_ids=["m1"], leaf_type=leaf_type
        )
        _seed_team(catalog, "tenant-D", user_id=None)
        _seed_referrer_agent(
            catalog,
            "tenant-D",
            "a",
            payload={"model_cfg": {"__ref__": "global.m1"}},
            user_id=None,
            agent_type=agent_type,
        )
        _seed_referrer_agent(
            catalog,
            "tenant-D",
            "b",
            payload={"model_cfg": {"__ref__": "global.m1"}},
            user_id=None,
            agent_type=agent_type,
        )
        text = catalog.export_namespace_yaml("tenant-D")
        doc = yaml.safe_load(text)
        assert len(doc["external_refs"]) == 1
        assert doc["external_refs"][0]["id"] == "m1"

    def test_external_refs_sorted_namespace_kind_id(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC4 sort key: ``external_refs:`` sorted by (namespace, kind, id) ascending."""
        import yaml

        agent_type, leaf_type = _register_open_models(monkeypatch)
        catalog, _ = catalog_factory()
        _seed_shared_namespace_with_targets(
            catalog, "alpha", user_id=None, target_ids=["m-a", "m-b"], leaf_type=leaf_type
        )
        _seed_shared_namespace_with_targets(
            catalog, "zulu", user_id=None, target_ids=["m-z"], leaf_type=leaf_type
        )
        _seed_team(catalog, "tenant-S", user_id=None)
        _seed_referrer_agent(
            catalog,
            "tenant-S",
            "agent",
            payload={
                "list": [
                    {"__ref__": "zulu.m-z"},
                    {"__ref__": "alpha.m-b"},
                    {"__ref__": "alpha.m-a"},
                ]
            },
            user_id=None,
            agent_type=agent_type,
        )
        text = catalog.export_namespace_yaml("tenant-S")
        doc = yaml.safe_load(text)
        keys = [(e["namespace"], e["kind"], e["id"]) for e in doc["external_refs"]]
        # alpha sorts before zulu; within alpha, "m-a" < "m-b".
        assert keys == [
            ("alpha", "model", "m-a"),
            ("alpha", "model", "m-b"),
            ("zulu", "model", "m-z"),
        ]

    def test_non_shared_target_silently_omitted(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC5: refs into a non-shared namespace are dropped from ``external_refs:``.

        Mixes one shared target and one non-shared target — only the shared
        one appears. The non-shared agent is staged via ``_repository.put``
        directly (bypassing ``populate_refs``) to simulate the data-drift
        scenario where a referrer was persisted before the target's
        namespace was flipped to non-shared.
        """
        import yaml

        agent_type, leaf_type = _register_open_models(monkeypatch)
        catalog, repo = catalog_factory()
        # Shared global with a target.
        _seed_shared_namespace_with_targets(
            catalog,
            "global",
            user_id=None,
            target_ids=["shared-target"],
            leaf_type=leaf_type,
        )
        # Non-shared "legacy-global" with a target — meta declares shared=false.
        _seed_team(catalog, "legacy-global", user_id=None)
        catalog.create(make_meta_entry("legacy-global", shared=False))
        catalog.create(
            Entry(
                id="legacy-target",
                kind="model",
                namespace="legacy-global",
                user_id=None,
                model_type=leaf_type,
                payload={"provider": "openai"},
            )
        )
        # Tenant team + agent with both refs (use repository.put so the
        # non-shared ref doesn't trip populate_refs).
        _seed_team(catalog, "tenant-NS", user_id=None)
        repo.put(
            Entry(
                id="a",
                kind="agent",
                namespace="tenant-NS",
                user_id=None,
                model_type=agent_type,
                payload={
                    "list": [
                        {"__ref__": "global.shared-target"},
                        {"__ref__": "legacy-global.legacy-target"},
                    ]
                },
            )
        )
        text = catalog.export_namespace_yaml("tenant-NS")
        doc = yaml.safe_load(text)
        ext_keys = {(e["namespace"], e["id"]) for e in doc["external_refs"]}
        # Only the shared one survives.
        assert ext_keys == {("global", "shared-target")}

    def test_missing_target_silently_omitted(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC6: a ref to a non-existent id in a shared namespace ⇒ silently omitted on export."""
        import yaml

        agent_type, leaf_type = _register_open_models(monkeypatch)
        catalog, repo = catalog_factory()
        _seed_shared_namespace_with_targets(
            catalog, "global", user_id=None, target_ids=["real"], leaf_type=leaf_type
        )
        _seed_team(catalog, "tenant-M", user_id=None)
        # Use repo.put to bypass populate_refs — simulates a delete-guard
        # bypass (target was deleted out from under the referrer).
        repo.put(
            Entry(
                id="agent-ghost-ref",
                kind="agent",
                namespace="tenant-M",
                user_id=None,
                model_type=agent_type,
                payload={"extra_ref": {"__ref__": "global.ghost"}},
            )
        )
        text = catalog.export_namespace_yaml("tenant-M")
        doc = yaml.safe_load(text)
        # Ghost is silently omitted — no entry for (global, ghost).
        assert doc["external_refs"] == []

    def test_same_namespace_ref_not_in_external_refs(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC7: same-namespace refs do not appear in ``external_refs:``."""
        import yaml

        agent_type, leaf_type = _register_open_models(monkeypatch)
        catalog, _ = catalog_factory()
        _seed_team(catalog, "tenant-LR", user_id="alice")
        # Pre-create a sibling target in the same namespace.
        catalog.create(
            Entry(
                id="sibling",
                kind="model",
                namespace="tenant-LR",
                user_id="alice",
                model_type=leaf_type,
                payload={"provider": "openai"},
            )
        )
        _seed_referrer_agent(
            catalog,
            "tenant-LR",
            "a",
            payload={"model_cfg": {"__ref__": "sibling"}},
            user_id="alice",
            agent_type=agent_type,
        )
        text = catalog.export_namespace_yaml("tenant-LR")
        doc = yaml.safe_load(text)
        assert doc["external_refs"] == []

    def test_explicit_namespace_matching_bundle_treated_as_same_namespace(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC7 second clause: ``__namespace__`` matching bundle's namespace ⇒ not external."""
        import yaml

        agent_type, leaf_type = _register_open_models(monkeypatch)
        catalog, repo = catalog_factory()
        _seed_team(catalog, "tenant-X", user_id="alice")
        catalog.create(
            Entry(
                id="sibling",
                kind="model",
                namespace="tenant-X",
                user_id="alice",
                model_type=leaf_type,
                payload={"provider": "openai"},
            )
        )
        # Use repo.put because cross-ns markers pointing at the same namespace
        # would normally be flagged by populate_refs.
        repo.put(
            Entry(
                id="agent",
                kind="agent",
                namespace="tenant-X",
                user_id="alice",
                model_type=agent_type,
                payload={"ref": {"__ref__": "sibling", "__namespace__": "tenant-X"}},
            )
        )
        text = catalog.export_namespace_yaml("tenant-X")
        doc = yaml.safe_load(text)
        assert doc["external_refs"] == []

    def test_transitive_reachability_two_hops(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC8: cross-ns target's payload carrying a cross-ns ref is followed transitively."""
        import yaml

        agent_type, leaf_type = _register_open_models(monkeypatch)
        catalog, repo = catalog_factory()
        _seed_shared_namespace_with_targets(
            catalog, "global", user_id=None, target_ids=[], leaf_type=leaf_type
        )
        _seed_shared_namespace_with_targets(
            catalog, "default", user_id=None, target_ids=[], leaf_type=leaf_type
        )
        # ``default.prompt-y`` is a leaf with no cross-ns refs.
        catalog.create(
            Entry(
                id="prompt-y",
                kind="model",
                namespace="default",
                user_id=None,
                model_type=leaf_type,
                payload={"provider": "openai"},
            )
        )
        # ``global.agent-x`` references ``default.prompt-y`` — staged via
        # repo.put so cross-ns ownership rules don't reject it.
        repo.put(
            Entry(
                id="agent-x",
                kind="agent",
                namespace="global",
                user_id=None,
                model_type=agent_type,
                payload={"model_cfg": {"__ref__": "default.prompt-y"}},
            )
        )
        # Tenant references ``global.agent-x``.
        _seed_team(catalog, "tenant-T", user_id=None)
        _seed_referrer_agent(
            catalog,
            "tenant-T",
            "agent",
            payload={"model_cfg": {"__ref__": "global.agent-x"}},
            user_id=None,
            agent_type=agent_type,
        )
        text = catalog.export_namespace_yaml("tenant-T")
        doc = yaml.safe_load(text)
        ext_keys = {(e["namespace"], e["id"]) for e in doc["external_refs"]}
        assert ext_keys == {("global", "agent-x"), ("default", "prompt-y")}

    def test_cycle_in_cross_ns_refs_terminates(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC8 cycle clause: ``tenant → global.x → default.y → global.x`` terminates."""
        import yaml

        agent_type, leaf_type = _register_open_models(monkeypatch)
        catalog, repo = catalog_factory()
        _seed_shared_namespace_with_targets(
            catalog, "global", user_id=None, target_ids=[], leaf_type=leaf_type
        )
        _seed_shared_namespace_with_targets(
            catalog, "default", user_id=None, target_ids=[], leaf_type=leaf_type
        )
        # default.y points back at global.x — a cycle through cross-ns refs.
        repo.put(
            Entry(
                id="y",
                kind="model",
                namespace="default",
                user_id=None,
                model_type=leaf_type,
                payload={"back_ref": {"__ref__": "global.x"}},
            )
        )
        # global.x references default.y.
        repo.put(
            Entry(
                id="x",
                kind="model",
                namespace="global",
                user_id=None,
                model_type=leaf_type,
                payload={"fwd_ref": {"__ref__": "default.y"}},
            )
        )
        _seed_team(catalog, "tenant-C", user_id=None)
        # Use repo.put so the cycle in the persisted state doesn't trip
        # populate_refs at create time — the export path is a display
        # projection that must tolerate cyclic state.
        repo.put(
            Entry(
                id="agent",
                kind="agent",
                namespace="tenant-C",
                user_id=None,
                model_type=agent_type,
                payload={"model_cfg": {"__ref__": "global.x"}},
            )
        )
        text = catalog.export_namespace_yaml("tenant-C")
        doc = yaml.safe_load(text)
        ext_keys = {(e["namespace"], e["id"]) for e in doc["external_refs"]}
        # Both targets appear exactly once — cycle terminated.
        assert ext_keys == {("global", "x"), ("default", "y")}
        # And the dedup invariant: each target appears exactly once.
        ids = [(e["namespace"], e["id"]) for e in doc["external_refs"]]
        assert len(ids) == len(set(ids))

    def test_same_ns_ref_inside_cross_ns_target_does_not_widen(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC9: a same-ns ref inside a cross-ns target does NOT widen ``external_refs:``."""
        import yaml

        agent_type, leaf_type = _register_open_models(monkeypatch)
        catalog, repo = catalog_factory()
        _seed_shared_namespace_with_targets(
            catalog, "global", user_id=None, target_ids=[], leaf_type=leaf_type
        )
        # Local same-ns ref inside global.
        catalog.create(
            Entry(
                id="prompt-z",
                kind="prompt",
                namespace="global",
                user_id=None,
                model_type=leaf_type,
                payload={"provider": "openai"},
            )
        )
        # global.agent-x's payload carries a same-ns ref to prompt-z.
        repo.put(
            Entry(
                id="agent-x",
                kind="agent",
                namespace="global",
                user_id=None,
                model_type=agent_type,
                payload={"model_cfg": {"__ref__": "prompt-z"}},
            )
        )
        # Tenant references global.agent-x.
        _seed_team(catalog, "tenant-W", user_id=None)
        _seed_referrer_agent(
            catalog,
            "tenant-W",
            "agent",
            payload={"model_cfg": {"__ref__": "global.agent-x"}},
            user_id=None,
            agent_type=agent_type,
        )
        text = catalog.export_namespace_yaml("tenant-W")
        doc = yaml.safe_load(text)
        ext_keys = {(e["namespace"], e["id"]) for e in doc["external_refs"]}
        # Only agent-x, NOT prompt-z. Per AC9 the local ref inside the
        # cross-ns target does NOT widen external_refs.
        assert ext_keys == {("global", "agent-x")}


# --- Story 17.5 — Import backward-compat ----------------------------------


class TestImportBackwardCompat:
    """Story 17.5 AC10, AC11 — both wire shapes accepted on import."""

    def test_import_v2_bundle_round_trip(self, catalog_factory: CatalogFactory) -> None:
        """A v2-shape bundle from ``export_namespace_yaml`` round-trips through import."""
        catalog, repo = catalog_factory()
        _seed_team(catalog, "ns-rt2", user_id="alice")
        _seed_agent(catalog, "ns-rt2", "a", user_id="alice")
        yaml_text = catalog.export_namespace_yaml("ns-rt2")
        # Rewrite to a fresh destination to avoid colliding with current state.
        new_text = yaml_text.replace("ns-rt2", "ns-rt2-dst")
        catalog.import_namespace_yaml(new_text)
        ids = {e.id for e in repo.list_by_namespace("ns-rt2-dst")}
        assert ids == {"team", "a"}

    def test_import_legacy_flat_list_bundle(self, catalog_factory: CatalogFactory) -> None:
        """A pre-17.5 legacy bundle (dict-keyed entries) imports cleanly."""
        catalog, repo = catalog_factory()
        legacy_text = (
            "namespace: ns-legacy-imp\n"
            "user_id: alice\n"
            "entries:\n"
            "  team:\n"
            "    kind: team\n"
            "    model_type: akgentic.team.models.TeamCard\n"
            "    parent_namespace: null\n"
            "    parent_id: null\n"
            "    description: ''\n"
            f"    payload: {_team_payload()!r}\n"
            "  a:\n"
            "    kind: agent\n"
            "    model_type: akgentic.core.agent_card.AgentCard\n"
            "    parent_namespace: null\n"
            "    parent_id: null\n"
            "    description: ''\n"
            f"    payload: {_agent_payload('a')!r}\n"
        )
        catalog.import_namespace_yaml(legacy_text)
        ids = {e.id for e in repo.list_by_namespace("ns-legacy-imp")}
        assert ids == {"team", "a"}

    def test_v2_external_refs_ignored_even_when_invalid(
        self, catalog_factory: CatalogFactory
    ) -> None:
        """AC10: ``external_refs:`` items would-fail-validation are still ignored on import."""
        catalog, repo = catalog_factory()
        _seed_team(catalog, "ns-imp-ext", user_id="alice")
        # Build a v2 bundle with a valid entries: list and a malformed
        # external_refs: item (missing required fields). Import must
        # ignore it entirely.
        entries = [
            Entry(
                id="team",
                kind="team",
                namespace="ns-imp-ext-dst",
                user_id="alice",
                model_type=_TEAM_TYPE,
                payload=_team_payload(),
            )
        ]
        # Build by serialising entries:, then surgically inject a malformed
        # external_refs: section into the YAML text.
        valid_yaml = dump_namespace_v2(entries, [])
        # Replace the empty external_refs: [] with a malformed list.
        broken = valid_yaml.replace(
            "external_refs: []",
            "external_refs:\n- {id: bad, namespace: foreign}\n",
        )
        # Import succeeds — external_refs is ignored.
        catalog.import_namespace_yaml(broken)
        ids = {e.id for e in repo.list_by_namespace("ns-imp-ext-dst")}
        assert ids == {"team"}

    def test_bare_list_root_rejected_with_existing_error(
        self, catalog_factory: CatalogFactory
    ) -> None:
        """AC11 second clause: a bare list at the YAML root is rejected with the existing error."""
        catalog, _ = catalog_factory()
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.import_namespace_yaml("- a\n- b\n")
        assert any("mapping" in m for m in exc_info.value.errors)


# --- Story 17.5 — Round-trip integrity ------------------------------------


class TestRoundTripBundle:
    """Story 17.5 AC12 — ``export → import → export`` is byte-identical on ``entries:``."""

    def test_export_import_export_byte_identical_entries(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Round-trip through ``dst-ns`` preserves the ``entries:`` block byte-for-byte."""
        agent_type, leaf_type = _register_open_models(monkeypatch)
        catalog, _ = catalog_factory()
        # Shared global with a target.
        _seed_shared_namespace_with_targets(
            catalog, "global", user_id=None, target_ids=["shared-m"], leaf_type=leaf_type
        )
        # Source namespace src-ns.
        _seed_team(catalog, "src-ns", user_id=None)
        _seed_referrer_agent(
            catalog,
            "src-ns",
            "agent",
            payload={"model_cfg": {"__ref__": "global.shared-m"}},
            user_id=None,
            agent_type=agent_type,
        )

        # 1) Export src-ns → bundle A.
        bundle_a = catalog.export_namespace_yaml("src-ns")

        # 2) Rewrite bundle A's namespace to dst-ns (each per-entry
        # ``namespace: src-ns`` line) and import — atomic replace into the
        # empty dst-ns.
        bundle_a_for_dst = bundle_a.replace("namespace: src-ns", "namespace: dst-ns")
        catalog.import_namespace_yaml(bundle_a_for_dst)

        # 3) Re-export dst-ns → bundle B.
        bundle_b = catalog.export_namespace_yaml("dst-ns")

        # The entries: block of bundle B (after the same namespace rewrite of
        # bundle A) must be byte-identical. Compare the entries: blocks
        # directly via slicing on the section markers.
        a_entries = _extract_section(bundle_a_for_dst, "entries:", "external_refs:")
        b_entries = _extract_section(bundle_b, "entries:", "external_refs:")
        assert a_entries == b_entries

    def test_external_refs_byte_identical_when_repo_unchanged(
        self, catalog_factory: CatalogFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two consecutive exports of the same namespace produce byte-identical YAML."""
        agent_type, leaf_type = _register_open_models(monkeypatch)
        catalog, _ = catalog_factory()
        _seed_shared_namespace_with_targets(
            catalog, "global", user_id=None, target_ids=["m1", "m2"], leaf_type=leaf_type
        )
        _seed_team(catalog, "tenant-X", user_id=None)
        _seed_referrer_agent(
            catalog,
            "tenant-X",
            "agent",
            payload={
                "list": [
                    {"__ref__": "global.m1"},
                    {"__ref__": "global.m2"},
                ],
            },
            user_id=None,
            agent_type=agent_type,
        )
        first = catalog.export_namespace_yaml("tenant-X")
        second = catalog.export_namespace_yaml("tenant-X")
        assert first == second


def _extract_section(yaml_text: str, start_key: str, end_key: str) -> str:
    """Slice the substring of ``yaml_text`` between two top-level keys.

    Used by round-trip tests to compare the ``entries:`` block byte-for-byte
    while ignoring the ``external_refs:`` block (which depends on global
    namespace state and may not be relevant to the assertion).
    """
    start = yaml_text.index(start_key)
    end = yaml_text.index(end_key, start)
    return yaml_text[start:end]
