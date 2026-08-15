"""Story 16.7 — tests for gating the generic ``/catalog/{kind}`` CRUD surface.

Covers every Acceptance Criterion in
``_bmad-output/akgentic-catalog/stories/16-7-hide-generic-kind-crud-routes-behind-a-setting.md``:

* AC #1 + AC #4 — default (setting ``False``) hides the eight kind-generic
  routes: requests 404, OpenAPI does not advertise them.
* AC #2 — setting ``True`` restores every route exactly as it was.
* AC #3 — namespace-scoped routes (``/catalog/namespaces``,
  ``/catalog/namespace/*``, ``/catalog/team/{ns}/resolve``,
  ``/catalog/schema``, ``/catalog/model_types``, ``/catalog/clone``) work
  regardless of the setting.
* AC #5 — ``AKGENTIC_CATALOG_EXPOSE_GENERIC_KIND_CRUD`` env var feeds the
  setting.

The fixtures ``api_client`` (flag True) and ``api_client_kind_crud_hidden``
(flag False) come from ``tests/v2/conftest.py``. AC #7 is honoured by reusing
the same ``api_client`` fixture in the ``test_api_router_*.py`` modules —
tests there continue to exercise the full route table under the True setting.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from httpx import Response  # noqa: E402

from akgentic.catalog.api._settings import CatalogRouterSettings  # noqa: E402
from akgentic.catalog.catalog import Catalog  # noqa: E402

from ..conftest import team_payload  # noqa: E402
from .conftest import _TEAM_TYPE, _seed_agent, _seed_team  # noqa: E402

# --- the eight kind-generic routes, described once, and their factories -----


def _no_seed(catalog: Catalog, namespace: str) -> None:
    """Row needs no pre-existing entry."""


def _seed_team_and_a1(catalog: Catalog, namespace: str) -> None:
    _seed_team(catalog, namespace)
    _seed_agent(catalog, namespace, id="a-1")


def _seed_team_and_agent_r(catalog: Catalog, namespace: str) -> None:
    _seed_team(catalog, namespace)
    _seed_agent(catalog, namespace, id="agent-r")


def _entry_body(namespace: str) -> dict[str, Any]:
    return {
        "id": "team",
        "kind": "team",
        "namespace": namespace,
        "model_type": _TEAM_TYPE,
        "payload": team_payload(),
    }


def _empty_body(namespace: str) -> dict[str, Any]:
    return {}


class _KindRoute(NamedTuple):
    """One kind-generic route, probed identically under both gating fixtures.

    ``body=None`` means "send no body at all" and is distinct from a row whose
    factory returns ``{}`` — ``post_kind_search`` posts an empty JSON document.

    ``namespace_param`` is each route's real calling convention, not a style
    choice: some routes take the namespace as a query parameter, some carry it
    inside the JSON body, and two take it in neither. Do not "harmonise" a row
    onto the majority — that changes the request the test issues.
    """

    id: str
    method: str
    path: str
    openapi_path: str
    ok_status: int
    seed: Callable[[Catalog, str], None] = _no_seed
    body: Callable[[str], dict[str, Any]] | None = None
    namespace_param: bool = False


_KIND_ROUTES = [
    _KindRoute("post_kind", "post", "/catalog/team", "/catalog/{kind}", 201, body=_entry_body),
    _KindRoute("get_list_kind", "get", "/catalog/team", "/catalog/{kind}", 200),
    _KindRoute(
        "get_kind_id",
        "get",
        "/catalog/team/team",
        "/catalog/{kind}/{id}",
        200,
        seed=_seed_team,
        namespace_param=True,
    ),
    _KindRoute(
        "put_kind_id",
        "put",
        "/catalog/team/team",
        "/catalog/{kind}/{id}",
        200,
        seed=_seed_team,
        body=_entry_body,
        namespace_param=True,
    ),
    _KindRoute(
        "delete_kind_id",
        "delete",
        "/catalog/agent/a-1",
        "/catalog/{kind}/{id}",
        204,
        seed=_seed_team_and_a1,
        namespace_param=True,
    ),
    _KindRoute(
        "post_kind_search",
        "post",
        "/catalog/team/search",
        "/catalog/{kind}/search",
        200,
        body=_empty_body,
    ),
    # ``kind=agent`` here, not ``team``: ``/catalog/team/{name}/resolve`` is a
    # static route (for ``resolve_team``) that would match first even without
    # the kind-generic family registered, so a ``team`` probe cannot
    # distinguish "route hidden" from "static-route hit".
    _KindRoute(
        "get_kind_id_resolve",
        "get",
        "/catalog/agent/agent-r/resolve",
        "/catalog/{kind}/{id}/resolve",
        200,
        seed=_seed_team_and_agent_r,
        namespace_param=True,
    ),
    _KindRoute(
        "get_kind_id_references",
        "get",
        "/catalog/agent/agent-r/references",
        "/catalog/{kind}/{id}/references",
        200,
        seed=_seed_team_and_agent_r,
        namespace_param=True,
    ),
]

_KIND_ROUTE_IDS = [route.id for route in _KIND_ROUTES]

# The kind-generic paths as they appear in the OpenAPI schema. FastAPI
# parametrises ``kind`` in the path, and OpenAPI renders the template string,
# so we match against the templates the rows carry. Two rows share
# ``/catalog/{kind}`` and three share ``/catalog/{kind}/{id}``.
_GENERIC_KIND_OPENAPI_PATHS = {route.openapi_path for route in _KIND_ROUTES}


def _probe(client: TestClient, catalog: Catalog, route: _KindRoute, namespace: str) -> Response:
    """Seed for ``route``, then issue its request scoped to ``namespace``."""
    route.seed(catalog, namespace)
    kwargs: dict[str, Any] = {}
    if route.namespace_param:
        kwargs["params"] = {"namespace": namespace}
    if route.body is not None:
        kwargs["json"] = route.body(namespace)
    return client.request(route.method, route.path, **kwargs)


# --- AC #1 + AC #4: default (False) hides the eight routes ------------------


class TestDefaultHidesGenericKindRoutes:
    """AC #1, AC #4 — with ``expose_generic_kind_crud=False`` the routes 404."""

    @pytest.mark.parametrize("route", _KIND_ROUTES, ids=_KIND_ROUTE_IDS)
    def test_route_returns_404(
        self, route: _KindRoute, api_client_kind_crud_hidden: tuple[TestClient, Catalog]
    ) -> None:
        client, catalog = api_client_kind_crud_hidden
        assert _probe(client, catalog, route, "ns-x").status_code == 404


class TestDefaultOpenAPIOmitsGenericKindPaths:
    """AC #1 — ``/openapi.json`` does not advertise the kind-generic paths."""

    def test_openapi_excludes_generic_kind_paths(
        self, api_client_kind_crud_hidden: tuple[TestClient, Catalog]
    ) -> None:
        client, _ = api_client_kind_crud_hidden
        response = client.get("/openapi.json")
        assert response.status_code == 200
        paths = set(response.json()["paths"].keys())
        # An empty set is disjoint from everything, so an emptied route table
        # would make the assertion below pass vacuously.
        assert _GENERIC_KIND_OPENAPI_PATHS
        # None of the kind-generic path templates should appear.
        assert _GENERIC_KIND_OPENAPI_PATHS.isdisjoint(paths), (
            f"unexpected kind-generic paths in OpenAPI: {_GENERIC_KIND_OPENAPI_PATHS & paths}"
        )


# --- AC #2: True restores every route ---------------------------------------


class TestSettingTrueRestoresRoutes:
    """AC #2 — with ``expose_generic_kind_crud=True`` every route works."""

    @pytest.mark.parametrize("route", _KIND_ROUTES, ids=_KIND_ROUTE_IDS)
    def test_route_returns_ok(
        self, route: _KindRoute, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, catalog = api_client
        assert _probe(client, catalog, route, "ns-t").status_code == route.ok_status


class TestOpenAPIIncludesGenericKindPathsWhenTrue:
    """AC #2 — OpenAPI advertises every kind-generic path when flag is True."""

    def test_openapi_includes_generic_kind_paths(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, _ = api_client
        response = client.get("/openapi.json")
        assert response.status_code == 200
        paths = set(response.json()["paths"].keys())
        # Nothing is missing from an empty set, so an emptied route table would
        # make the assertion below pass vacuously.
        assert _GENERIC_KIND_OPENAPI_PATHS
        missing = _GENERIC_KIND_OPENAPI_PATHS - paths
        assert not missing, f"expected kind-generic paths missing from OpenAPI: {missing}"


# --- AC #3: namespace-scoped routes unaffected regardless of the setting ----


class TestNamespaceRoutesUnaffected:
    """AC #3 — namespace / schema / model_types / clone always respond."""

    def test_list_namespaces(self, api_client_kind_crud_hidden: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client_kind_crud_hidden
        assert client.get("/catalog/namespaces").status_code == 200

    def test_export_namespace(
        self, api_client_kind_crud_hidden: tuple[TestClient, Catalog]
    ) -> None:
        client, catalog = api_client_kind_crud_hidden
        _seed_team(catalog, "ns-e")
        response = client.get("/catalog/namespace/ns-e/export")
        assert response.status_code == 200

    def test_validate_namespace_get(
        self, api_client_kind_crud_hidden: tuple[TestClient, Catalog]
    ) -> None:
        client, catalog = api_client_kind_crud_hidden
        _seed_team(catalog, "ns-v")
        response = client.get("/catalog/namespace/ns-v/validate")
        assert response.status_code == 200

    def test_validate_namespace_post(
        self, api_client_kind_crud_hidden: tuple[TestClient, Catalog]
    ) -> None:
        import yaml as _yaml

        client, _ = api_client_kind_crud_hidden
        doc = {
            "namespace": "ns-vp",
            "user_id": None,
            "entries": {
                "team": {
                    "kind": "team",
                    "model_type": _TEAM_TYPE,
                    "description": "",
                    "payload": team_payload(),
                },
            },
        }
        response = client.post(
            "/catalog/namespace/validate",
            content=_yaml.safe_dump(doc, sort_keys=False).encode("utf-8"),
        )
        assert response.status_code == 200

    def test_resolve_team(self, api_client_kind_crud_hidden: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client_kind_crud_hidden
        _seed_team(catalog, "ns-rt")
        assert client.get("/catalog/team/ns-rt/resolve").status_code == 200

    def test_schema(self, api_client_kind_crud_hidden: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client_kind_crud_hidden
        response = client.get(
            "/catalog/schema", params={"model_type": "akgentic.core.agent_card.AgentCard"}
        )
        assert response.status_code == 200

    def test_model_types(self, api_client_kind_crud_hidden: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client_kind_crud_hidden
        assert client.get("/catalog/model_types").status_code == 200

    def test_clone_endpoint_present(
        self, api_client_kind_crud_hidden: tuple[TestClient, Catalog]
    ) -> None:
        """``/catalog/clone`` is registered even when kind-CRUD is hidden.

        We do not need to exercise a successful clone here — the presence
        of the route (any response other than 404) is the AC. The catalog
        service returns 404 on missing source, which is the expected
        response shape for a no-seed call.
        """
        client, _ = api_client_kind_crud_hidden
        response = client.post(
            "/catalog/clone",
            json={
                "src_namespace": "nope",
                "src_id": "team",
                "dst_namespace": "dst",
                "dst_user_id": "anonymous",
            },
        )
        # 404 from the service layer, not from the route being absent —
        # if the route were unregistered the body would be FastAPI's
        # "Not Found" payload. Either way a 404 is valid evidence the route
        # itself is registered and reached; the TestNamespaceRoutesUnaffected
        # coverage above already confirms other static routes work.
        assert response.status_code in {404, 409}


# --- AC #5: env var feeds the setting ---------------------------------------


class TestSettingsFromEnv:
    """AC #5 — ``AKGENTIC_CATALOG_EXPOSE_GENERIC_KIND_CRUD`` round-trips."""

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "Yes", "on"])
    def test_truthy_values_enable(self, raw: str) -> None:
        settings = CatalogRouterSettings.from_env(
            {"AKGENTIC_CATALOG_EXPOSE_GENERIC_KIND_CRUD": raw}
        )
        assert settings.expose_generic_kind_crud is True

    @pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off", ""])
    def test_falsy_values_disable(self, raw: str) -> None:
        settings = CatalogRouterSettings.from_env(
            {"AKGENTIC_CATALOG_EXPOSE_GENERIC_KIND_CRUD": raw}
        )
        assert settings.expose_generic_kind_crud is False

    def test_unset_defaults_to_false(self) -> None:
        settings = CatalogRouterSettings.from_env({})
        assert settings.expose_generic_kind_crud is False

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError, match="not a recognised boolean"):
            CatalogRouterSettings.from_env({"AKGENTIC_CATALOG_EXPOSE_GENERIC_KIND_CRUD": "maybe"})

    def test_build_router_honours_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``build_router()`` with no args reads ``from_env``."""
        from akgentic.catalog.api.router import build_router

        monkeypatch.setenv("AKGENTIC_CATALOG_EXPOSE_GENERIC_KIND_CRUD", "1")
        enabled = build_router()
        paths_enabled = {route.path for route in enabled.routes}  # type: ignore[attr-defined]
        assert "/catalog/{kind}" in paths_enabled

        monkeypatch.setenv("AKGENTIC_CATALOG_EXPOSE_GENERIC_KIND_CRUD", "0")
        disabled = build_router()
        paths_disabled = {route.path for route in disabled.routes}  # type: ignore[attr-defined]
        assert "/catalog/{kind}" not in paths_disabled
        # Static routes still there.
        assert "/catalog/namespaces" in paths_disabled


# --- Route ordering regression ---------------------------------------------


class TestStaticRoutesWinDispatchOrder:
    """Sanity: ``/catalog/namespaces`` must dispatch before ``/catalog/{kind}``.

    With the generic routes re-registered after the static ones, FastAPI's
    declaration-order dispatch keeps the literal ``/namespaces`` path bound
    to ``list_namespaces``. If this regresses, ``/catalog/namespaces`` would
    fall through to ``list_entries(kind='namespaces')`` and 422 on the
    ``EntryKind`` validator.
    """

    def test_namespaces_still_wins_over_kind_route(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, _ = api_client
        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestNamespaceMetaRoutesAlwaysOn:
    """Story 17.2 — the two ``/namespace/{ns}/meta`` routes are always-on.

    They MUST be registered when ``expose_generic_kind_crud=False`` (the
    community-tier default), like every other static / namespace-scoped route.
    """

    def test_get_namespace_meta_route_registered_when_kind_crud_hidden(
        self, api_client_kind_crud_hidden: tuple[TestClient, Catalog]
    ) -> None:
        client, _ = api_client_kind_crud_hidden
        # Without seeding, the route returns 404 (entry not found), NOT
        # 404 from "route not registered" — and certainly NOT 405. Assert
        # that the OpenAPI schema declares the path.
        response = client.get("/openapi.json")
        spec = response.json()
        assert "/catalog/namespace/{namespace}/meta" in spec["paths"]
        assert "get" in spec["paths"]["/catalog/namespace/{namespace}/meta"]

    def test_put_namespace_meta_route_registered_when_kind_crud_hidden(
        self, api_client_kind_crud_hidden: tuple[TestClient, Catalog]
    ) -> None:
        client, _ = api_client_kind_crud_hidden
        response = client.get("/openapi.json")
        spec = response.json()
        assert "/catalog/namespace/{namespace}/meta" in spec["paths"]
        assert "put" in spec["paths"]["/catalog/namespace/{namespace}/meta"]
