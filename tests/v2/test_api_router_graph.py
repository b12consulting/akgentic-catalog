"""Graph, resolve and schema-introspection routes on the v2 ``/catalog`` router."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from akgentic.catalog.allowlist import set_allowed_prefixes  # noqa: E402
from akgentic.catalog.catalog import Catalog  # noqa: E402
from akgentic.catalog.models.entry import Entry  # noqa: E402

from ..conftest import team_payload  # noqa: E402
from .conftest import (  # noqa: E402
    _AGENT_TYPE,
    _TEAM_TYPE,
    _agent_payload,
    _seed_agent,
    _seed_team,
    register_test_module,
)

_CUSTOMER_MODULE = "acme.core.models"


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
                "dst_user_id": "anonymous",
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


class TestResolveTeamProjection:
    """GET /catalog/team/{namespace}/resolve — Story 17.9 display projection.

    Asserts the wire-level behaviour of ``Catalog.load_team``'s ``name`` /
    ``description`` projection. The route's response schema is unchanged
    (the route already returns ``TeamCard.model_dump()``); only the
    values of ``name`` / ``description`` change when the team entry leaves
    them blank.
    """

    @staticmethod
    def _put_team_no_name(catalog: Catalog, namespace: str) -> None:
        """Direct repository ``put`` for a team entry whose ``name`` is blank.

        Bypasses ``Catalog.create``'s ``prepare_for_write`` so the stored
        payload is exactly what is passed here. The catalog bootstrap is
        unaffected — these tests do not add sub-entries.

        The blank ``name`` is an empty string rather than an omitted key or
        ``None``, because the pinned ``akgentic-team`` declares
        ``TeamCard.name: str`` as required and Pydantic rejects ``None``. The
        display-name projection under test fires on ``""`` just as it would
        on ``None``.
        """
        catalog._repository.put(
            Entry(
                id="team",
                kind="team",
                namespace=namespace,
                model_type=_TEAM_TYPE,
                payload=team_payload(name=""),
            )
        )

    @staticmethod
    def _put_meta(
        catalog: Catalog,
        namespace: str,
        *,
        payload_name: str | None,
        entry_description: str = "",
    ) -> None:
        """Direct repository ``put`` for a meta entry — bypasses validation."""
        payload: dict[str, Any] = {
            "description": "",
            "properties": {},
            "shareable": False,
        }
        if payload_name is not None:
            payload["name"] = payload_name
        catalog._repository.put(
            Entry(
                id="_meta",
                kind="meta",
                namespace=namespace,
                model_type="akgentic.catalog.models.namespace_meta.NamespaceMeta",
                description=entry_description,
                payload=payload,
            )
        )

    def test_response_name_projects_meta_name_when_team_name_is_none(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, catalog = api_client
        self._put_team_no_name(catalog, "ns-r-meta")
        self._put_meta(catalog, "ns-r-meta", payload_name="Friendly")
        response = client.get("/catalog/team/ns-r-meta/resolve")
        assert response.status_code == 200
        assert response.json()["name"] == "Friendly"

    def test_response_name_falls_back_to_namespace_when_no_meta(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, catalog = api_client
        self._put_team_no_name(catalog, "ns-r-fallback")
        response = client.get("/catalog/team/ns-r-fallback/resolve")
        assert response.status_code == 200
        assert response.json()["name"] == "ns-r-fallback"

    def test_response_name_preserves_explicit_team_name(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, catalog = api_client
        # Use the standard seed which sets payload["name"] = "team".
        _seed_team(catalog, "ns-r-explicit")
        # Add a meta entry with a different name to confirm no override.
        self._put_meta(catalog, "ns-r-explicit", payload_name="Friendly")
        response = client.get("/catalog/team/ns-r-explicit/resolve")
        assert response.status_code == 200
        assert response.json()["name"] == "team"


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

    def test_model_types_includes_a_configured_prefix(
        self, api_client: tuple[TestClient, Catalog], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Story 28.1 — a deployment-owned class shows up once its prefix is allowed."""
        client, _ = api_client

        class CaseIngestionConfig(BaseModel):
            case_id: str = ""

        CaseIngestionConfig.__module__ = _CUSTOMER_MODULE
        register_test_module(monkeypatch, _CUSTOMER_MODULE, CaseIngestionConfig=CaseIngestionConfig)
        set_allowed_prefixes([f"{_CUSTOMER_MODULE}."])

        response = client.get("/catalog/model_types")
        assert response.status_code == 200
        assert f"{_CUSTOMER_MODULE}.CaseIngestionConfig" in response.json()
