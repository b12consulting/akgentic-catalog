"""Entry CRUD, listing, search and namespace deletion on the v2 ``/catalog`` router."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from akgentic.catalog.catalog import Catalog  # noqa: E402
from akgentic.catalog.models.entry import Entry  # noqa: E402

from ..conftest import team_payload  # noqa: E402
from .conftest import (  # noqa: E402
    _AGENT_TYPE,
    _TEAM_TYPE,
    _agent_payload,
    _seed_agent,
    _seed_team,
    register_akgentic_test_module,
)


class _RefLeaf(BaseModel):
    """Permissive leaf used for cross-ns ref payloads (Story 27.1)."""

    provider: str = "openai"


class _RefHolder(BaseModel):
    """Holder payload carrying a cross-ns ref (Story 27.1)."""

    model_cfg: _RefLeaf | None = None


@pytest.fixture
def ref_model_paths(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """Register permissive cross-ns ref models and return their dotted paths."""
    module_name = register_akgentic_test_module(
        monkeypatch,
        "tests_fixture_api_27_1_delete_namespace",
        Leaf=_RefLeaf,
        Holder=_RefHolder,
    )
    return f"{module_name}.Leaf", f"{module_name}.Holder"


# --- CRUD -------------------------------------------------------------------


class TestCreate:
    """POST /catalog/{kind} — AC7, AC23."""

    def test_create_team_returns_201(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        body = {
            "id": "team",
            "kind": "team",
            "namespace": "ns-create",
            "model_type": _TEAM_TYPE,
            "payload": team_payload(),
        }
        response = client.post("/catalog/team", json=body)
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "team"
        assert data["namespace"] == "ns-create"
        assert data["kind"] == "team"

    def test_create_kind_mismatch_400(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        body = {
            "id": "team",
            "kind": "team",
            "namespace": "ns-mismatch",
            "model_type": _TEAM_TYPE,
            "payload": team_payload(),
        }
        response = client.post("/catalog/agent", json=body)
        assert response.status_code == 400

    def test_create_duplicate_409(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-dup")
        body = {
            "id": "team",
            "kind": "team",
            "namespace": "ns-dup",
            "model_type": _TEAM_TYPE,
            "payload": team_payload(),
        }
        response = client.post("/catalog/team", json=body)
        assert response.status_code == 409

    def test_create_body_missing_model_type_422(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, _ = api_client
        body = {
            "id": "agent",
            "kind": "agent",
            "namespace": "ns-422",
            "payload": {},
        }
        response = client.post("/catalog/agent", json=body)
        assert response.status_code == 422

    def test_create_agent_without_team_409(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        body = {
            "id": "lone",
            "kind": "agent",
            "namespace": "ns-no-team",
            "model_type": _AGENT_TYPE,
            "payload": _agent_payload("lone"),
        }
        response = client.post("/catalog/agent", json=body)
        assert response.status_code == 409


class TestGet:
    """GET /catalog/{kind}/{id} — AC8."""

    def test_get_happy_path(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-g")
        response = client.get("/catalog/team/team", params={"namespace": "ns-g"})
        assert response.status_code == 200
        assert response.json()["id"] == "team"

    def test_get_missing_404(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        response = client.get("/catalog/team/team", params={"namespace": "nope"})
        assert response.status_code == 404

    def test_get_kind_mismatch_404(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-km")
        response = client.get("/catalog/agent/team", params={"namespace": "ns-km"})
        assert response.status_code == 404

    def test_get_without_namespace_422(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        response = client.get("/catalog/team/team")
        assert response.status_code == 422

    def test_get_native_value_entry_returns_standard_shape(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """Story 26.1 / AC 16 — ``GET /catalog/{ns}/{id}`` on a NativeValue
        entry returns the standard entry JSON shape (``model_type``,
        ``payload``) with no special-casing of the kind.
        """
        client, catalog = api_client
        _seed_team(catalog, "ns-native")
        catalog.create(
            Entry(
                id="id_native",
                kind="prompt",
                namespace="ns-native",
                user_id="anonymous",
                model_type="akgentic.catalog.NativeValue",
                payload={"value": "shared-prompt-body"},
            )
        )
        response = client.get("/catalog/prompt/id_native", params={"namespace": "ns-native"})
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == "id_native"
        assert body["model_type"] == "akgentic.catalog.NativeValue"
        assert body["payload"] == {"value": "shared-prompt-body"}


class TestUpdate:
    """PUT /catalog/{kind}/{id} — AC9."""

    def test_update_happy_path(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-u")
        body = {
            "id": "team",
            "kind": "team",
            "namespace": "ns-u",
            "model_type": _TEAM_TYPE,
            "description": "updated",
            "payload": team_payload(),
        }
        response = client.put("/catalog/team/team", params={"namespace": "ns-u"}, json=body)
        assert response.status_code == 200
        assert response.json()["description"] == "updated"
        stored = catalog.get("ns-u", "team")
        assert stored.description == "updated"

    def test_update_missing_404(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        body = {
            "id": "team",
            "kind": "team",
            "namespace": "ns-none",
            "model_type": _TEAM_TYPE,
            "payload": team_payload(),
        }
        response = client.put("/catalog/team/team", params={"namespace": "ns-none"}, json=body)
        assert response.status_code == 404

    def test_update_id_mismatch_400(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-um")
        _seed_agent(catalog, "ns-um", id="foo")
        body = {
            "id": "bar",
            "kind": "agent",
            "namespace": "ns-um",
            "model_type": _AGENT_TYPE,
            "payload": _agent_payload("bar"),
        }
        response = client.put("/catalog/agent/foo", params={"namespace": "ns-um"}, json=body)
        assert response.status_code == 400

    def test_update_without_namespace_422(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        response = client.put(
            "/catalog/team/team",
            json={
                "id": "team",
                "kind": "team",
                "namespace": "x",
                "model_type": _TEAM_TYPE,
                "payload": team_payload(),
            },
        )
        assert response.status_code == 422


class TestDelete:
    """DELETE /catalog/{kind}/{id} — AC10."""

    def test_delete_returns_204_and_gone(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-d")
        _seed_agent(catalog, "ns-d", id="a-1")
        response = client.delete("/catalog/agent/a-1", params={"namespace": "ns-d"})
        assert response.status_code == 204
        follow = client.get("/catalog/agent/a-1", params={"namespace": "ns-d"})
        assert follow.status_code == 404

    def test_delete_missing_404(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        response = client.delete("/catalog/agent/nope", params={"namespace": "ns-d-missing"})
        assert response.status_code == 404

    def test_delete_without_namespace_422(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        response = client.delete("/catalog/agent/foo")
        assert response.status_code == 422


class TestDeleteNamespace:
    """DELETE /catalog/namespace/{namespace} — Story 27.1 (ADR-028 §Decision 5)."""

    def test_delete_namespace_returns_204_and_gone(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-nuke")
        _seed_agent(catalog, "ns-nuke", id="a-1")
        response = client.delete("/catalog/namespace/ns-nuke")
        assert response.status_code == 204
        follow = client.get("/catalog/agent/a-1", params={"namespace": "ns-nuke"})
        assert follow.status_code == 404

    def test_delete_namespace_absent_404(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        response = client.delete("/catalog/namespace/ns-absent")
        assert response.status_code == 404

    def test_delete_namespace_external_ref_blocked_409(
        self,
        api_client: tuple[TestClient, Catalog],
        ref_model_paths: tuple[str, str],
    ) -> None:
        client, catalog = api_client
        leaf, holder = ref_model_paths
        # Shareable namespace 'global' with a prompt referenced cross-ns.
        _seed_team(catalog, "global")
        catalog.create(
            Entry(
                id="_meta",
                kind="meta",
                namespace="global",
                user_id="anonymous",
                model_type="akgentic.catalog.models.namespace_meta.NamespaceMeta",
                payload={
                    "name": "global",
                    "description": "",
                    "properties": {},
                    "shareable": True,
                    "public": False,
                },
            )
        )
        catalog.create(
            Entry(
                id="shared-prompt",
                kind="prompt",
                namespace="global",
                user_id="anonymous",
                model_type=leaf,
                payload={"provider": "shared"},
            )
        )
        _seed_team(catalog, "tenant-A")
        catalog.create(
            Entry(
                id="agent-ref",
                kind="agent",
                namespace="tenant-A",
                user_id="anonymous",
                model_type=holder,
                payload={
                    "model_cfg": {
                        "__ref__": "shared-prompt",
                        "__namespace__": "global",
                    }
                },
            )
        )
        response = client.delete("/catalog/namespace/global")
        assert response.status_code == 409


# --- Listing and search -----------------------------------------------------


class TestList:
    """GET /catalog/{kind} — AC11."""

    def test_list_filters_by_namespace(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-a", user_id="alice")
        _seed_team(catalog, "ns-b", user_id="alice")
        _seed_agent(catalog, "ns-a", id="a-a", user_id="alice")
        _seed_agent(catalog, "ns-b", id="a-b", user_id="alice")
        response = client.get("/catalog/agent", params={"namespace": "ns-a"})
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "a-a"

    def test_list_without_filter_returns_all_for_kind(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-la")
        _seed_team(catalog, "ns-lb")
        response = client.get("/catalog/team")
        assert response.status_code == 200
        assert len(response.json()) == 2


class TestSearch:
    """POST /catalog/{kind}/search — AC12, AC26."""

    def test_search_filters_by_user_id_and_namespace(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-s", user_id="alice")
        _seed_team(catalog, "ns-s2", user_id="bob")
        _seed_agent(catalog, "ns-s", id="a1", user_id="alice")
        _seed_agent(catalog, "ns-s2", id="a2", user_id="bob")
        body = {"namespace": "ns-s", "user_id": "alice"}
        response = client.post("/catalog/agent/search", json=body)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "a1"

    def test_search_kind_mismatch_400(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        body = {"kind": "tool"}
        response = client.post("/catalog/agent/search", json=body)
        assert response.status_code == 400

    def test_search_with_no_kind_uses_path(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-sk")
        _seed_agent(catalog, "ns-sk", id="only-agent")
        response = client.post("/catalog/agent/search", json={})
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["kind"] == "agent"
