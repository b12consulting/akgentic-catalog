"""``/catalog/namespace/{ns}/meta`` GET and PUT routes on the v2 router (Story 17.2)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from akgentic.catalog.catalog import Catalog  # noqa: E402
from akgentic.catalog.models.namespace_meta import NamespaceMeta  # noqa: E402

from .conftest import _NAMESPACE_META_TYPE, _seed_meta_entry, _seed_team  # noqa: E402


class TestGetNamespaceMeta:
    """Story 17.2 AC7 — ``GET /catalog/namespace/{ns}/meta``."""

    def test_returns_meta_entry_when_present(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "tenant-42")
        _seed_meta_entry(catalog, "tenant-42", name="Tenant 42")
        response = client.get("/catalog/namespace/tenant-42/meta")
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == "_meta"
        assert body["kind"] == "meta"
        assert body["namespace"] == "tenant-42"
        assert body["payload"]["name"] == "Tenant 42"

    def test_returns_404_when_absent(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "tenant-42")
        response = client.get("/catalog/namespace/tenant-42/meta")
        assert response.status_code == 404
        assert "errors" in response.json()


class TestPutNamespaceMeta:
    """Story 17.2 AC8 — ``PUT /catalog/namespace/{ns}/meta`` upsert."""

    def test_create_branch_returns_201(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "tenant-42")
        body = {"name": "Tenant 42", "description": "primary", "properties": {"tier": "gold"}}
        response = client.put("/catalog/namespace/tenant-42/meta", json=body)
        assert response.status_code == 201
        entry = response.json()
        assert entry["kind"] == "meta"
        assert entry["id"] == "_meta"
        assert entry["namespace"] == "tenant-42"
        assert entry["payload"]["name"] == "Tenant 42"
        assert entry["payload"]["properties"] == {"tier": "gold"}
        assert entry["model_type"] == _NAMESPACE_META_TYPE

    def test_update_branch_returns_200(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "tenant-42")
        _seed_meta_entry(catalog, "tenant-42", name="Old Name")
        body = {"name": "New Name", "description": "updated", "properties": {}}
        response = client.put("/catalog/namespace/tenant-42/meta", json=body)
        assert response.status_code == 200
        entry = response.json()
        assert entry["payload"]["name"] == "New Name"
        assert entry["description"] == "updated"

    def test_body_kind_id_namespace_model_type_silently_ignored(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """The handler substitutes ``kind`` / ``id`` / ``namespace`` / ``model_type``
        regardless of body content. ``NamespaceMeta`` does not declare those
        fields, so a JSON payload that carries them is simply not parsed.
        """
        client, catalog = api_client
        _seed_team(catalog, "tenant-42")
        body = {
            "name": "Tenant 42",
            "description": "",
            "properties": {},
            # These should be ignored by the handler:
            "kind": "team",
            "id": "evil-id",
            "namespace": "evil-namespace",
            "model_type": "akgentic.evil",
        }
        response = client.put("/catalog/namespace/tenant-42/meta", json=body)
        assert response.status_code == 201
        entry = response.json()
        assert entry["kind"] == "meta"
        assert entry["id"] == "_meta"
        assert entry["namespace"] == "tenant-42"
        assert entry["model_type"] == _NAMESPACE_META_TYPE

    def test_malformed_body_missing_name_returns_422(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """Malformed body (missing required ``name``) surfaces as 422 from
        Pydantic ``ValidationError`` raised inside ``NamespaceMeta.model_validate``.
        """
        client, catalog = api_client
        _seed_team(catalog, "tenant-42")
        body = {"description": "missing name", "properties": {}}
        response = client.put("/catalog/namespace/tenant-42/meta", json=body)
        assert response.status_code == 422

    def test_no_team_creates_meta_with_anonymous_user_id(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """When no team exists, PUT meta creates the first anchor entry with
        user_id == "anonymous" (Story 17.10 — meta-only namespace bootstrap).
        """
        client, _ = api_client
        body = {"name": "ghost", "description": "", "properties": {}}
        response = client.put("/catalog/namespace/ghost-ns/meta", json=body)
        assert response.status_code == 201
        entry = response.json()
        assert entry["user_id"] == "anonymous"
        assert entry["kind"] == "meta"
        assert entry["namespace"] == "ghost-ns"

    def test_no_team_update_preserves_user_id(self, api_client: tuple[TestClient, Catalog]) -> None:
        """Subsequent PUT (update) preserves the _meta.user_id from the first write."""
        client, _ = api_client
        body = {"name": "ghost", "description": "", "properties": {}}
        response = client.put("/catalog/namespace/ghost-ns/meta", json=body)
        assert response.status_code == 201
        first_user_id = response.json()["user_id"]
        assert first_user_id == "anonymous"

        # Update the meta entry
        body_update = {"name": "ghost-updated", "description": "updated", "properties": {}}
        response2 = client.put("/catalog/namespace/ghost-ns/meta", json=body_update)
        assert response2.status_code == 200
        entry = response2.json()
        assert entry["user_id"] == first_user_id
        assert entry["payload"]["name"] == "ghost-updated"


class TestPutNamespaceMetaPublicField:
    """Story 18.2 AC6 — PUT/GET /catalog/namespace/{ns}/meta round-trips ``public``."""

    def test_put_public_true_round_trips(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "tenant-42")
        body = {"name": "Tenant 42", "description": "", "properties": {}, "public": True}
        response = client.put("/catalog/namespace/tenant-42/meta", json=body)
        assert response.status_code == 201
        entry = response.json()
        assert entry["payload"]["public"] is True
        # GET surfaces the same typed bool.
        get_resp = client.get("/catalog/namespace/tenant-42/meta")
        assert get_resp.status_code == 200
        assert get_resp.json()["payload"]["public"] is True

    def test_put_public_false_round_trips(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "tenant-42")
        body = {"name": "Tenant 42", "description": "", "properties": {}, "public": False}
        response = client.put("/catalog/namespace/tenant-42/meta", json=body)
        assert response.status_code == 201
        entry = response.json()
        assert entry["payload"]["public"] is False

    def test_put_public_invalid_type_yields_422(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        # Story 18.2 — the route's body parser runs ``model_validate``
        # (non-strict) so ``"yes"`` coerces to True (matching ``shareable``'s
        # established route-boundary behaviour — see story Open Question #6:
        # "match shareable exactly: lenient projection, strict upsert"). A
        # value Pydantic cannot coerce (e.g. a list / object) DOES surface
        # as 422, locking the validation envelope at the route boundary.
        client, catalog = api_client
        _seed_team(catalog, "tenant-42")
        body = {"name": "Tenant 42", "description": "", "properties": {}, "public": [1, 2, 3]}
        response = client.put("/catalog/namespace/tenant-42/meta", json=body)
        assert response.status_code == 422

    def test_put_public_string_coerces_at_route_strict_at_projection(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        # Story 18.2 — the route accepts a coercible string (``"true"``) and
        # the upsert path stores ``payload["public"] = True`` (Pydantic's
        # default-mode coercion). The strict-bool guarantee — only a real
        # bool ``True`` flips the picker / bundle-header projections —
        # lives at the read side (``NamespaceSummary._build_namespace_summary``,
        # ``BundleHeader._project_header``). This test pins both sides:
        #   1. The route accepts the body (no 422) — matches ``shareable``.
        #   2. ``NamespaceMeta.model_validate({...}, strict=True)`` on the
        #      same body raises ``ValidationError`` — strict mode is
        #      available at the model level for callers who opt in.
        client, catalog = api_client
        _seed_team(catalog, "tenant-42")
        body = {"name": "Tenant 42", "description": "", "properties": {}, "public": "true"}
        response = client.put("/catalog/namespace/tenant-42/meta", json=body)
        assert response.status_code == 201
        # Strict mode at the model level rejects the same body — the
        # contract is available, just not auto-applied at the route.
        from pydantic import ValidationError as _ValidationError

        with pytest.raises(_ValidationError):
            NamespaceMeta.model_validate(body, strict=True)

    def test_put_default_public_is_false(self, api_client: tuple[TestClient, Catalog]) -> None:
        # A body that omits ``public`` defaults to False on the upserted
        # entry — the field default flows through NamespaceMeta.
        client, catalog = api_client
        _seed_team(catalog, "tenant-42")
        body = {"name": "Tenant 42", "description": "", "properties": {}}
        response = client.put("/catalog/namespace/tenant-42/meta", json=body)
        assert response.status_code == 201
        entry = response.json()
        assert entry["payload"]["public"] is False
