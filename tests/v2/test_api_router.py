"""Integration tests for the v2 unified ``/catalog`` FastAPI router.

Covers every route in the router — happy path and every documented error
status code. See Story 16.1 ACs 21-31 for the mapping from tests to ACs.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from akgentic.catalog.catalog import Catalog  # noqa: E402
from akgentic.catalog.models.entry import Entry  # noqa: E402

_TEAM_TYPE = "akgentic.team.models.TeamCard"
_AGENT_TYPE = "akgentic.core.agent_card.AgentCard"


# --- helpers ----------------------------------------------------------------


def _team_payload() -> dict[str, Any]:
    """Return a minimal valid ``TeamCard`` payload."""
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


def _agent_payload(name: str = "a") -> dict[str, Any]:
    """Return a minimal valid ``AgentCard`` payload."""
    return {
        "role": "r",
        "description": "",
        "skills": [],
        "agent_class": "akgentic.core.agent.Akgent",
        "config": {"name": name, "role": "r"},
        "routes_to": [],
        "metadata": {},
    }


def _seed_team(catalog: Catalog, namespace: str, user_id: str | None = None) -> Entry:
    """Seed a team entry in ``namespace``."""
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
    id: str = "agent-a",
    user_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> Entry:
    """Seed a minimal agent entry sharing the team's ownership."""
    return catalog.create(
        Entry(
            id=id,
            kind="agent",
            namespace=namespace,
            user_id=user_id,
            model_type=_AGENT_TYPE,
            payload=payload if payload is not None else _agent_payload(id),
        )
    )


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
            "payload": _team_payload(),
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
            "payload": _team_payload(),
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
            "payload": _team_payload(),
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
            "payload": _team_payload(),
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
            "payload": _team_payload(),
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
                "payload": _team_payload(),
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


# --- Graph routes -----------------------------------------------------------


class TestClone:
    """POST /catalog/clone — AC13, AC27."""

    def test_clone_copies_tree(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "src-ns", user_id="alice")
        _seed_agent(catalog, "src-ns", id="agent-a", user_id="alice")
        response = client.post(
            "/catalog/clone",
            json={
                "src_namespace": "src-ns",
                "src_id": "team",
                "dst_namespace": "dst-ns",
                "dst_user_id": "alice",
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["namespace"] == "dst-ns"
        dst_entries = catalog.list_by_namespace("dst-ns")
        ids = {e.id for e in dst_entries}
        assert "team" in ids

    def test_clone_missing_source_404(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        response = client.post(
            "/catalog/clone",
            json={
                "src_namespace": "nope",
                "src_id": "team",
                "dst_namespace": "dst",
                "dst_user_id": None,
            },
        )
        assert response.status_code == 404


class TestResolveEntry:
    """GET /catalog/{kind}/{id}/resolve — AC14."""

    def test_resolve_happy_path(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-re")
        _seed_agent(catalog, "ns-re", id="agent-r")
        response = client.get("/catalog/agent/agent-r/resolve", params={"namespace": "ns-re"})
        assert response.status_code == 200
        body = response.json()
        assert body["config"]["name"] == "agent-r"

    def test_resolve_missing_404(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        response = client.get("/catalog/agent/nope/resolve", params={"namespace": "ns-none"})
        assert response.status_code == 404

    def test_resolve_kind_mismatch_404(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-rk")
        response = client.get("/catalog/agent/team/resolve", params={"namespace": "ns-rk"})
        assert response.status_code == 404


class TestResolveTeam:
    """GET /catalog/team/{namespace}/resolve — AC15, AC31."""

    def test_resolve_team_happy_path(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-rt")
        response = client.get("/catalog/team/ns-rt/resolve")
        assert response.status_code == 200
        body = response.json()
        # TeamCard has a ``name`` field and an ``entry_point`` field — see
        # akgentic.team.models.TeamCard for the pinned shape.
        assert body["name"] == "team"
        assert "entry_point" in body

    def test_resolve_team_missing_409(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        response = client.get("/catalog/team/ns-empty/resolve")
        assert response.status_code == 409


class TestReferences:
    """GET /catalog/{kind}/{id}/references — AC16, AC30."""

    def test_references_returns_referrers(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-1")
        # Seed a model entry that agent-a, agent-b will reference via __ref__.
        catalog._repository.put(
            Entry(
                id="id_gpt_41",
                kind="model",
                namespace="ns-1",
                model_type="akgentic.core.agent_card.AgentCard",
                payload=_agent_payload("gpt41"),
            )
        )
        # Put agents with a ref marker in metadata.
        agent_payload = _agent_payload("a")
        agent_payload["metadata"] = {"model": {"__ref__": "id_gpt_41"}}
        catalog._repository.put(
            Entry(
                id="agent-a",
                kind="agent",
                namespace="ns-1",
                model_type=_AGENT_TYPE,
                payload=agent_payload,
            )
        )
        agent_payload_b = _agent_payload("b")
        agent_payload_b["metadata"] = {"model": {"__ref__": "id_gpt_41"}}
        catalog._repository.put(
            Entry(
                id="agent-b",
                kind="agent",
                namespace="ns-1",
                model_type=_AGENT_TYPE,
                payload=agent_payload_b,
            )
        )
        response = client.get("/catalog/model/id_gpt_41/references", params={"namespace": "ns-1"})
        assert response.status_code == 200
        ids = {row["id"] for row in response.json()}
        assert ids == {"agent-a", "agent-b"}

    def test_references_missing_entry_404(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        response = client.get(
            "/catalog/model/id_gpt_41/references",
            params={"namespace": "ns-empty"},
        )
        assert response.status_code == 404


# --- Schema introspection ---------------------------------------------------


class TestSchema:
    """GET /catalog/schema — AC17, AC28."""

    def test_schema_allowlisted(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        response = client.get(
            "/catalog/schema",
            params={"model_type": "akgentic.core.agent_card.AgentCard"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body.get("type") == "object"
        assert "properties" in body

    def test_schema_disallowed_prefix_409(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        response = client.get("/catalog/schema", params={"model_type": "datetime.datetime"})
        assert response.status_code == 409

    def test_schema_missing_class_409(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        response = client.get(
            "/catalog/schema",
            params={"model_type": "akgentic.does.not.exist.Foo"},
        )
        assert response.status_code == 409


class TestCreateV2App:
    """AC4, AC5 — ``create_v2_app`` factory wires YAML + MongoDB backends."""

    def test_yaml_backend_default_path(self, tmp_path: Any) -> None:
        from akgentic.catalog.api.app import create_v2_app

        base = tmp_path / "custom-root"
        app = create_v2_app(backend="yaml", yaml_base_path=base)
        assert app.title == "Akgentic Catalog"
        assert base.exists()
        client = TestClient(app)
        response = client.get("/catalog/model_types")
        assert response.status_code == 200

    def test_unknown_backend_raises(self) -> None:
        from akgentic.catalog.api.app import create_v2_app

        with pytest.raises(ValueError, match="Unknown backend"):
            create_v2_app(backend="sqlite")  # type: ignore[arg-type]

    def test_mongodb_missing_config_raises(self) -> None:
        from akgentic.catalog.api.app import create_v2_app

        with pytest.raises(ValueError, match="mongo_config is required"):
            create_v2_app(backend="mongodb")

    def test_mongodb_backend_wires_collection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("mongomock")
        import mongomock

        from akgentic.catalog.api.app import create_v2_app
        from akgentic.catalog.repositories.mongo._config import MongoCatalogConfig

        config = MongoCatalogConfig(connection_string="mongodb://x", database="db_test")

        def _fake_client(self: MongoCatalogConfig) -> mongomock.MongoClient:
            return mongomock.MongoClient()

        monkeypatch.setattr(MongoCatalogConfig, "create_client", _fake_client)
        app = create_v2_app(backend="mongodb", mongo_config=config)
        assert app.title == "Akgentic Catalog"


class TestModelTypes:
    """GET /catalog/model_types — AC18, AC29."""

    def test_model_types_lists_imported_classes(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, _ = api_client
        # Ensure at least one known class is loaded in-process.
        import akgentic.core.agent_card  # noqa: F401

        response = client.get("/catalog/model_types")
        assert response.status_code == 200
        paths = response.json()
        assert isinstance(paths, list)
        assert all(isinstance(p, str) and p.startswith("akgentic.") for p in paths)
        assert "akgentic.core.agent_card.AgentCard" in paths


# --- Namespace bundle routes (Story 16.2) ----------------------------------


class TestNamespaceExport:
    """GET /catalog/namespace/{namespace}/export — AC23, AC25."""

    def test_export_happy_path(self, api_client: tuple[TestClient, Catalog]) -> None:
        import yaml

        client, catalog = api_client
        _seed_team(catalog, "ns-exp", user_id="alice")
        _seed_agent(catalog, "ns-exp", id="a-1", user_id="alice")
        response = client.get("/catalog/namespace/ns-exp/export")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/yaml")
        doc = yaml.safe_load(response.text)
        assert list(doc.keys()) == ["namespace", "user_id", "entries"]
        assert doc["namespace"] == "ns-exp"
        assert set(doc["entries"].keys()) == {"team", "a-1"}

    def test_export_empty_namespace_409(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        response = client.get("/catalog/namespace/nope/export")
        assert response.status_code == 409


class TestNamespaceImport:
    """POST /catalog/namespace/import — AC24, AC25, AC33."""

    def _build_bundle(self) -> str:
        import yaml as _yaml

        doc = {
            "namespace": "ns-imp",
            "user_id": "alice",
            "entries": {
                "team": {
                    "kind": "team",
                    "model_type": _TEAM_TYPE,
                    "parent_namespace": None,
                    "parent_id": None,
                    "description": "",
                    "payload": _team_payload(),
                },
                "a": {
                    "kind": "agent",
                    "model_type": _AGENT_TYPE,
                    "parent_namespace": None,
                    "parent_id": None,
                    "description": "",
                    "payload": _agent_payload("a"),
                },
            },
        }
        return _yaml.safe_dump(doc, sort_keys=False)

    def test_import_happy_path(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        yaml_text = self._build_bundle()
        response = client.post(
            "/catalog/namespace/import",
            content=yaml_text.encode("utf-8"),
            headers={"Content-Type": "application/yaml"},
        )
        assert response.status_code == 201
        data = response.json()
        assert isinstance(data, list)
        assert {e["id"] for e in data} == {"team", "a"}

    def test_import_malformed_yaml_422(self, api_client: tuple[TestClient, Catalog]) -> None:
        """Malformed YAML is a transport-level structural failure → HTTP 422.

        Mirrors the ``/namespace/validate`` contract — the router intercepts
        ``yaml.YAMLError`` before the catalog-service call so clients can
        distinguish syntactic YAML breakage from catalog-invariant (409)
        failures.
        """
        client, _ = api_client
        response = client.post(
            "/catalog/namespace/import",
            content=b"{{{ not yaml }",
        )
        assert response.status_code == 422
        assert "failed to parse bundle YAML" in response.json()["detail"]

    def test_import_missing_team_409(self, api_client: tuple[TestClient, Catalog]) -> None:
        import yaml as _yaml

        client, _ = api_client
        doc = {
            "namespace": "ns-noteam",
            "user_id": "alice",
            "entries": {
                "a": {
                    "kind": "agent",
                    "model_type": _AGENT_TYPE,
                    "parent_namespace": None,
                    "parent_id": None,
                    "description": "",
                    "payload": _agent_payload("a"),
                }
            },
        }
        response = client.post(
            "/catalog/namespace/import",
            content=_yaml.safe_dump(doc).encode("utf-8"),
        )
        assert response.status_code == 409
        assert any("no team entry" in e for e in response.json()["errors"])

    def test_import_dangling_ref_409(self, api_client: tuple[TestClient, Catalog]) -> None:
        import yaml as _yaml

        client, catalog = api_client
        # Pre-seed namespace with a ghost target so prepare_for_write passes,
        # leaving the dangling-ref-in-bundle check as the failure surface.
        _seed_team(catalog, "ns-dref", user_id="alice")
        catalog.create(
            Entry(
                id="ghost",
                kind="model",
                namespace="ns-dref",
                user_id="alice",
                model_type=_AGENT_TYPE,  # any allowlisted class with payload shape compat
                payload=_agent_payload("ghost"),
            )
        )
        doc = {
            "namespace": "ns-dref",
            "user_id": "alice",
            "entries": {
                "team": {
                    "kind": "team",
                    "model_type": _TEAM_TYPE,
                    "parent_namespace": None,
                    "parent_id": None,
                    "description": "",
                    "payload": _team_payload(),
                },
                "dangler": {
                    "kind": "agent",
                    "model_type": _AGENT_TYPE,
                    "parent_namespace": None,
                    "parent_id": None,
                    "description": "",
                    "payload": {
                        "role": "r",
                        "description": "",
                        "skills": [],
                        "agent_class": "akgentic.core.agent.Akgent",
                        "config": {"name": "dangler", "role": "r"},
                        "routes_to": [],
                        "metadata": {"ref": {"__ref__": "ghost", "__type__": _AGENT_TYPE}},
                    },
                },
            },
        }
        response = client.post(
            "/catalog/namespace/import",
            content=_yaml.safe_dump(doc).encode("utf-8"),
        )
        assert response.status_code == 409
        assert any("not found in bundle" in e for e in response.json()["errors"])

    def test_import_non_utf8_body_400(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        response = client.post(
            "/catalog/namespace/import",
            content=b"\xff\xfe\xfd",
        )
        assert response.status_code == 400
        assert "UTF-8" in response.json()["detail"]


class TestNamespaceBundleRoundTrip:
    """Atomic replace through HTTP (AC34)."""

    def test_round_trip_atomic_replace(self, api_client: tuple[TestClient, Catalog]) -> None:
        import yaml as _yaml

        client, catalog = api_client
        # Seed ns-a with {team, agent_a, tool_x}.
        _seed_team(catalog, "ns-a", user_id="alice")
        _seed_agent(catalog, "ns-a", id="agent_a", user_id="alice")
        catalog.create(
            Entry(
                id="tool_x",
                kind="tool",
                namespace="ns-a",
                user_id="alice",
                model_type=_AGENT_TYPE,
                payload=_agent_payload("tool_x"),
            )
        )

        # Bundle: team + agent_a_modified + tool_y (tool_x dropped).
        doc = {
            "namespace": "ns-a",
            "user_id": "alice",
            "entries": {
                "team": {
                    "kind": "team",
                    "model_type": _TEAM_TYPE,
                    "parent_namespace": None,
                    "parent_id": None,
                    "description": "",
                    "payload": _team_payload(),
                },
                "agent_a": {
                    "kind": "agent",
                    "model_type": _AGENT_TYPE,
                    "parent_namespace": None,
                    "parent_id": None,
                    "description": "updated",
                    "payload": _agent_payload("agent_a"),
                },
                "tool_y": {
                    "kind": "tool",
                    "model_type": _AGENT_TYPE,
                    "parent_namespace": None,
                    "parent_id": None,
                    "description": "",
                    "payload": _agent_payload("tool_y"),
                },
            },
        }
        response = client.post(
            "/catalog/namespace/import",
            content=_yaml.safe_dump(doc).encode("utf-8"),
        )
        assert response.status_code == 201

        # Verify per-entry state via HTTP GETs.
        r_agent = client.get("/catalog/agent/agent_a", params={"namespace": "ns-a"})
        assert r_agent.status_code == 200
        assert r_agent.json()["description"] == "updated"

        r_tool_y = client.get("/catalog/tool/tool_y", params={"namespace": "ns-a"})
        assert r_tool_y.status_code == 200

        r_tool_x = client.get("/catalog/tool/tool_x", params={"namespace": "ns-a"})
        assert r_tool_x.status_code == 404


# --- Namespace validation endpoints (Story 16.3) ---------------------------


def _validation_bundle(
    namespace: str = "ns-v",
    user_id: str | None = "alice",
    extra_entries: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Build a minimal valid bundle YAML, optionally appending extra entries."""
    import yaml as _yaml

    entries_map: dict[str, Any] = {
        "team": {
            "kind": "team",
            "model_type": _TEAM_TYPE,
            "parent_namespace": None,
            "parent_id": None,
            "description": "",
            "payload": _team_payload(),
        },
        "a": {
            "kind": "agent",
            "model_type": _AGENT_TYPE,
            "parent_namespace": None,
            "parent_id": None,
            "description": "",
            "payload": _agent_payload("a"),
        },
    }
    if extra_entries:
        entries_map.update(extra_entries)
    doc = {"namespace": namespace, "user_id": user_id, "entries": entries_map}
    return _yaml.safe_dump(doc, sort_keys=False)


class TestNamespaceValidateGet:
    """``GET /catalog/namespace/{namespace}/validate`` — AC36."""

    def test_get_happy_path(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-get-ok", user_id="alice")
        _seed_agent(catalog, "ns-get-ok", id="agent-a", user_id="alice")
        response = client.get("/catalog/namespace/ns-get-ok/validate")
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["namespace"] == "ns-get-ok"
        assert body["global_errors"] == []
        assert body["entry_issues"] == []

    def test_get_empty_namespace(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        response = client.get("/catalog/namespace/ns-empty/validate")
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        assert body["namespace"] == "ns-empty"  # AC18 patch
        assert body["global_errors"] == ["namespace has no entries"]

    def test_get_dangling_ref_corruption(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-get-dr", user_id="alice")
        # Bypass the service to seed a payload with a dangling ref.
        dangler_payload = _agent_payload("dangler")
        dangler_payload["metadata"] = {"ref": {"__ref__": "ghost"}}
        catalog._repository.put(
            Entry(
                id="dangler",
                kind="agent",
                namespace="ns-get-dr",
                user_id="alice",
                model_type=_AGENT_TYPE,
                payload=dangler_payload,
            )
        )
        response = client.get("/catalog/namespace/ns-get-dr/validate")
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        assert any("dangling ref" in m for m in body["global_errors"])


class TestNamespaceValidatePost:
    """``POST /catalog/namespace/validate`` — AC36, AC37."""

    def test_post_happy_path(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        yaml_text = _validation_bundle(namespace="ns-post-ok", user_id="alice")
        response = client.post(
            "/catalog/namespace/validate",
            content=yaml_text.encode("utf-8"),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["namespace"] == "ns-post-ok"

    def test_post_malformed_yaml_422(self, api_client: tuple[TestClient, Catalog]) -> None:
        """Malformed YAML → HTTP 422 per shard 07 transport-level contract."""
        client, _ = api_client
        response = client.post(
            "/catalog/namespace/validate",
            content=b"{{{ not yaml",
        )
        assert response.status_code == 422
        assert "failed to parse bundle YAML" in response.json()["detail"]

    def test_post_non_utf8_body_422(self, api_client: tuple[TestClient, Catalog]) -> None:
        """Non-UTF-8 request body → HTTP 422 (structural request-body failure)."""
        client, _ = api_client
        response = client.post(
            "/catalog/namespace/validate",
            content=b"\xff\xfe\xfd",
        )
        assert response.status_code == 422
        assert "not valid UTF-8" in response.json()["detail"]

    def test_post_allowlist_violation_returns_200_with_ok_false(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        import yaml as _yaml

        client, _ = api_client
        doc = {
            "namespace": "ns-post-allow",
            "user_id": "alice",
            "entries": {
                "team": {
                    "kind": "team",
                    "model_type": _TEAM_TYPE,
                    "parent_namespace": None,
                    "parent_id": None,
                    "description": "",
                    "payload": _team_payload(),
                },
                "bad": {
                    "kind": "model",
                    "model_type": "builtins.dict",
                    "parent_namespace": None,
                    "parent_id": None,
                    "description": "",
                    "payload": {},
                },
            },
        }
        yaml_text = _yaml.safe_dump(doc, sort_keys=False)
        response = client.post(
            "/catalog/namespace/validate",
            content=yaml_text.encode("utf-8"),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        assert any("outside allowlist" in m for m in body["global_errors"])

    def test_post_dangling_ref_returns_200_with_ok_false(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, _ = api_client
        dangler_payload = _agent_payload("dangler")
        dangler_payload["metadata"] = {"ref": {"__ref__": "ghost"}}
        yaml_text = _validation_bundle(
            namespace="ns-post-dangling",
            user_id="alice",
            extra_entries={
                "dangler": {
                    "kind": "agent",
                    "model_type": _AGENT_TYPE,
                    "parent_namespace": None,
                    "parent_id": None,
                    "description": "",
                    "payload": dangler_payload,
                }
            },
        )
        response = client.post(
            "/catalog/namespace/validate",
            content=yaml_text.encode("utf-8"),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        assert any("dangling ref" in m for m in body["global_errors"])

    def test_post_missing_team_returns_200_with_ok_false(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        import yaml as _yaml

        client, _ = api_client
        doc = {
            "namespace": "ns-post-noteam",
            "user_id": "alice",
            "entries": {
                "a": {
                    "kind": "agent",
                    "model_type": _AGENT_TYPE,
                    "parent_namespace": None,
                    "parent_id": None,
                    "description": "",
                    "payload": _agent_payload("a"),
                }
            },
        }
        response = client.post(
            "/catalog/namespace/validate",
            content=_yaml.safe_dump(doc, sort_keys=False).encode("utf-8"),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        assert any("no team entry" in m for m in body["global_errors"])

    def test_service_vs_http_divergence_on_malformed_yaml(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """AC24 — service returns report; HTTP returns 422.

        Service-level: ``Catalog.validate_namespace_yaml("{{{")`` returns a
        report with ``ok=False`` and the parse error in ``global_errors``
        (no exception). HTTP-level: ``POST`` with the same payload surfaces a
        422 at the transport boundary, per shard 07's "structural
        request-body errors (malformed YAML) still surface as 422" contract.
        """
        client, catalog = api_client
        report = catalog.validate_namespace_yaml("{{{")
        assert report.ok is False
        assert report.namespace is None
        assert any("Failed to parse bundle YAML" in m for m in report.global_errors)

        response = client.post(
            "/catalog/namespace/validate",
            content=b"{{{",
        )
        assert response.status_code == 422

    def test_post_json_round_trip(self, api_client: tuple[TestClient, Catalog]) -> None:
        """AC37 — the 200 response body deserialises into NamespaceValidationReport."""
        from akgentic.catalog.validation import NamespaceValidationReport

        client, _ = api_client
        yaml_text = _validation_bundle(namespace="ns-roundtrip", user_id="alice")
        response = client.post(
            "/catalog/namespace/validate",
            content=yaml_text.encode("utf-8"),
        )
        assert response.status_code == 200
        parsed = NamespaceValidationReport.model_validate_json(response.text)
        assert parsed.ok is True
        assert parsed.namespace == "ns-roundtrip"
        assert parsed.global_errors == []
        assert parsed.entry_issues == []


# --- Compliance-review regression tests (Epic 16 spec-compliance fix) -------


def test_router_namespace_validate_malformed_yaml_returns_422(
    api_client: tuple[TestClient, Catalog],
) -> None:
    """``POST /catalog/namespace/validate`` returns 422 on malformed YAML.

    Regression for Epic 16 spec-compliance review (BLOCKING V4): shard 07
    pins 422 — not 400 — as the transport-level status for structural
    request-body errors (malformed YAML) on validation endpoints.
    """
    client, _ = api_client
    response = client.post(
        "/catalog/namespace/validate",
        content=b"{{{ still : not : yaml",
    )
    assert response.status_code == 422
    assert "failed to parse bundle YAML" in response.json()["detail"]


def test_router_namespace_import_malformed_yaml_returns_422(
    api_client: tuple[TestClient, Catalog],
) -> None:
    """``POST /catalog/namespace/import`` returns 422 on malformed YAML.

    Regression for Epic 16 spec-compliance review: without the router-level
    ``yaml.YAMLError`` guard, malformed YAML reaches ``load_namespace`` and
    surfaces as ``CatalogValidationError`` → 409, which the client cannot
    distinguish from catalog-invariant failures. Mirrors the ``/validate``
    endpoint contract.
    """
    client, _ = api_client
    response = client.post(
        "/catalog/namespace/import",
        content=b"{{{ still : not : yaml",
    )
    assert response.status_code == 422
    assert "failed to parse bundle YAML" in response.json()["detail"]
