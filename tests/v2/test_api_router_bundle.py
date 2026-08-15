"""Namespace export / import bundle routes on the v2 ``/catalog`` router (Story 16.2)."""

from __future__ import annotations

import pytest

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
)


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
        # Story 18.2 — eight top-level keys in declaration order. The header
        # (name/description/properties/shareable/public) is auto-synthesised
        # from the team-payload fallback when no _meta entry exists in the
        # namespace.
        assert list(doc.keys()) == [
            "namespace",
            "user_id",
            "name",
            "description",
            "properties",
            "shareable",
            "public",
            "entries",
        ]
        assert doc["namespace"] == "ns-exp"
        assert doc["shareable"] is False
        assert doc["public"] is False
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
                    "description": "",
                    "payload": team_payload(),
                },
                "a": {
                    "kind": "agent",
                    "model_type": _AGENT_TYPE,
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
        assert any("has no team entry and no meta entry" in e for e in response.json()["errors"])

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
                    "description": "",
                    "payload": team_payload(),
                },
                "dangler": {
                    "kind": "agent",
                    "model_type": _AGENT_TYPE,
                    "description": "",
                    "payload": {
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
                    "description": "",
                    "payload": team_payload(),
                },
                "agent_a": {
                    "kind": "agent",
                    "model_type": _AGENT_TYPE,
                    "description": "updated",
                    "payload": _agent_payload("agent_a"),
                },
                "tool_y": {
                    "kind": "tool",
                    "model_type": _AGENT_TYPE,
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


# --- Compliance-review regression tests (Epic 16 spec-compliance fix) -------


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
