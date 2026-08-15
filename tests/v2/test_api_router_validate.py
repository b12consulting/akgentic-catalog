"""Namespace validation endpoints on the v2 ``/catalog`` router (Story 16.3)."""

from __future__ import annotations

from typing import Any

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


def _validation_bundle(
    namespace: str = "ns-v",
    user_id: str = "alice",
    extra_entries: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Build a minimal valid bundle YAML, optionally appending extra entries."""
    import yaml as _yaml

    entries_map: dict[str, Any] = {
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
                    "description": "",
                    "payload": team_payload(),
                },
                "bad": {
                    "kind": "model",
                    "model_type": "builtins.dict",
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
        assert any("has no team entry and no meta entry" in m for m in body["global_errors"])

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
