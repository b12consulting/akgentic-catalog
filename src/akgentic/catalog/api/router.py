"""Unified v2 ``/catalog`` FastAPI router — CRUD, search, graph, schema.

This is the v2 HTTP surface for the unified ``Catalog`` service. It exposes:

* Per-entry CRUD (``POST /{kind}``, ``GET /{kind}/{id}``, ``PUT /{kind}/{id}``,
  ``DELETE /{kind}/{id}``) — every per-entry route requires ``namespace`` as a
  mandatory query parameter. **These routes are only registered when the
  ``expose_generic_kind_crud`` setting is True** (Story 16.7 — see
  :mod:`akgentic.catalog.api._settings`).
* Listing and search (``GET /{kind}``, ``POST /{kind}/search``) — also gated
  by ``expose_generic_kind_crud``.
* Graph ops (``POST /clone``, ``GET /{kind}/{id}/resolve``,
  ``GET /team/{namespace}/resolve``, ``GET /{kind}/{id}/references``). Of
  these, ``/clone`` and ``/team/{namespace}/resolve`` are always registered;
  ``/{kind}/{id}/resolve`` and ``/{kind}/{id}/references`` are gated by
  ``expose_generic_kind_crud`` because they consume the kind-generic surface.
* Schema introspection (``GET /schema``, ``GET /model_types``) and namespace
  ops (``/namespaces``, ``/namespace/*``) are always registered.

Business logic lives entirely in :class:`akgentic.catalog.catalog.Catalog`; the
router is a thin adapter. ``CatalogValidationError`` and ``EntryNotFoundError``
propagate unchanged to the app-wide exception handlers registered by
``api/_errors.py``. The one exception is URL/body coherence checks — those raise
``HTTPException(400)`` directly inside the handler.

Injection follows the v1 pattern: a module-level ``_catalog`` is initialised via
``set_catalog(catalog)`` and accessed through ``_get_catalog()``. See
architecture shard 07 for the full route table.

**Route declaration order matters** — FastAPI routes are dispatched in the
order they are registered. The static-path routes (``/clone``, ``/schema``,
``/model_types``, ``/team/{namespace}/resolve``, ``/namespaces``,
``/namespace/*``) are always declared first so literal paths take precedence
over the ``EntryKind``-typed ``/{kind}`` path segment. When
``expose_generic_kind_crud`` is True, the kind-generic routes are appended
after the static ones — preserving dispatch priority.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import yaml
from fastapi import APIRouter, Body, HTTPException, Query, Request, Response
from pydantic import BaseModel

from akgentic.catalog.api._settings import CatalogRouterSettings
from akgentic.catalog.models.entry import Entry, EntryKind
from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.queries import CloneRequest, EntryQuery
from akgentic.catalog.resolver import enumerate_allowlisted_model_types, load_model_type
from akgentic.catalog.validation import NamespaceValidationReport

if TYPE_CHECKING:
    from akgentic.catalog.catalog import Catalog

__all__ = ["NamespaceSummary", "build_router", "router", "set_catalog"]

# Content types accepted on the YAML branch of ``_parse_body_as``. Both
# variants are in the wild; ``application/x-yaml`` is the historical
# unregistered spelling and ``application/yaml`` the RFC9512-registered one.
_YAML_CONTENT_TYPES: frozenset[str] = frozenset({"application/yaml", "application/x-yaml"})


class NamespaceSummary(BaseModel):
    """Flat DTO for ``GET /catalog/namespaces`` — one row per namespace.

    Every catalog namespace contains exactly one ``kind="team"`` entry
    (enforced by the catalog service), so "list namespaces" is equivalent to
    "list team entries, project to this DTO". The shape intentionally omits
    ``user_id``, ``parent_namespace``, and other entry-model fields to keep
    the payload minimal and to avoid leaking tenancy design to the picker.
    """

    namespace: str
    name: str
    description: str


logger = logging.getLogger(__name__)

_catalog: Catalog | None = None


def set_catalog(catalog: Catalog) -> None:
    """Inject the unified v2 ``Catalog`` service instance.

    Args:
        catalog: The ``Catalog`` service bound to a concrete repository.
    """
    global _catalog  # noqa: PLW0603
    _catalog = catalog


def _get_catalog() -> Catalog:
    """Return the injected catalog or raise if not configured."""
    if _catalog is None:
        raise RuntimeError("Catalog not configured — call set_catalog() first")
    return _catalog


def _ensure_kind(entry_kind: EntryKind, path_kind: EntryKind) -> None:
    """Raise 400 if ``entry_kind`` does not match ``path_kind``."""
    if entry_kind != path_kind:
        raise HTTPException(status_code=400, detail="kind mismatch between path and body")


async def _parse_body_as[T: BaseModel](request: Request, model_type: type[T]) -> T:
    """Parse a raw request body as ``model_type`` from JSON or YAML.

    Dispatches on the request's ``Content-Type`` header (normalized: split on
    ``;``, strip, lower-case). ``application/json`` (or missing header) uses
    ``json.loads``; ``application/yaml`` and ``application/x-yaml`` use
    ``yaml.safe_load``. Any other non-empty content type is rejected with a
    415. Malformed JSON/YAML payloads surface as 422 with a descriptive
    detail string.

    An empty body on either JSON or YAML path is treated as ``{}`` so that
    downstream Pydantic validation produces the familiar ``field required``
    422 contract — matching the v1 ``_parse_yaml_or_json`` shape ported from
    ``akgentic-infra`` (see ``akgentic-infra`` ADR-023 §D4).

    Pydantic ``ValidationError`` raised by ``model_type.model_validate`` is
    NOT wrapped here — FastAPI's registered exception handlers surface it as
    the default 422 envelope. Wrapping would double-encode the error and
    drift from the typed-body contract used elsewhere in the router.

    Args:
        request: The inbound Starlette ``Request`` (body already buffered).
        model_type: The Pydantic ``BaseModel`` subclass to validate into.

    Returns:
        A validated ``model_type`` instance.

    Raises:
        HTTPException: 415 for unknown content types; 422 for malformed
            JSON or YAML payloads.
        pydantic.ValidationError: Propagates unwrapped for FastAPI's default
            422 handler.
    """
    raw = await request.body()
    raw_ct = request.headers.get("content-type", "application/json")
    ct = raw_ct.split(";")[0].strip().lower()
    payload: Any
    if ct == "application/json" or ct == "":
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"invalid JSON body: {exc}") from exc
    elif ct in _YAML_CONTENT_TYPES:
        try:
            payload = yaml.safe_load(raw) if raw else {}
        except yaml.YAMLError as exc:
            raise HTTPException(status_code=422, detail=f"invalid YAML body: {exc}") from exc
    else:
        raise HTTPException(
            status_code=415,
            detail=(
                f"unsupported Content-Type: {ct!r}; expected application/json or application/yaml"
            ),
        )
    return model_type.model_validate(payload)


def _multi_format_body_openapi(model_name: str) -> dict[str, Any]:
    """Return an ``openapi_extra`` dict advertising JSON + YAML request bodies.

    The schema entry points at the already-registered Pydantic component
    (``#/components/schemas/{model_name}``) so no additional schema objects
    are emitted — FastAPI collects the schema once when the model is used
    as a ``response_model`` or body argument elsewhere in the app.

    Args:
        model_name: The Pydantic class name FastAPI emits in
            ``components.schemas`` (e.g. ``"Entry"``).

    Returns:
        A dict merged into the operation's OpenAPI object by FastAPI.
    """
    schema_ref = {"$ref": f"#/components/schemas/{model_name}"}
    content_entry = {"schema": schema_ref}
    return {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": content_entry,
                "application/yaml": content_entry,
                "application/x-yaml": content_entry,
            },
        }
    }


# --- Handler implementations ------------------------------------------------


async def list_namespaces() -> list[NamespaceSummary]:
    """List every catalog namespace with its team name and description.

    Equivalent to listing ``kind="team"`` entries and projecting each to
    ``NamespaceSummary``. The list is sorted alphabetically by ``namespace``.
    No ``user_id`` filter is applied — callers (or tier-specific middleware)
    are responsible for tenancy filtering. Namespaces that somehow lack a
    team entry are skipped silently (defensive guard for a state the catalog
    invariants should prevent).
    """
    logger.debug("GET /catalog/namespaces")
    teams = _get_catalog().list(EntryQuery(kind="team"))
    summaries = [
        NamespaceSummary(
            namespace=t.namespace,
            name=str(t.payload.get("name", "")),
            description=t.description,
        )
        for t in teams
    ]
    return sorted(summaries, key=lambda s: s.namespace)


async def clone_entry(request: Request) -> Entry:
    """Deep-copy an entry tree into ``dst_namespace`` — AC13.

    Accepts a ``CloneRequest`` body in either JSON or YAML (see Epic 21).
    """
    req = await _parse_body_as(request, CloneRequest)
    logger.debug(
        "POST /catalog/clone (%s,%s) -> (%s,%s)",
        req.src_namespace,
        req.src_id,
        req.dst_namespace,
        req.dst_user_id,
    )
    return _get_catalog().clone(req.src_namespace, req.src_id, req.dst_namespace, req.dst_user_id)


async def get_schema(model_type: str = Query(...)) -> dict[str, Any]:
    """Return the JSON Schema for ``model_type`` — AC17."""
    logger.debug("GET /catalog/schema?model_type=%s", model_type)
    try:
        cls = load_model_type(model_type)
    except (ImportError, AttributeError) as exc:
        raise CatalogValidationError(
            [f"model_type '{model_type}' could not be imported: {exc}"]
        ) from exc
    return cls.model_json_schema()


async def list_model_types() -> list[str]:
    """Return allowlisted Pydantic model classes loaded in the current process."""
    logger.debug("GET /catalog/model_types")
    return enumerate_allowlisted_model_types()


async def resolve_team(namespace: str) -> dict[str, Any]:
    """Resolve the team entry in ``namespace`` into a dumped ``TeamCard`` — AC15."""
    logger.debug("GET /catalog/team/%s/resolve", namespace)
    team_card = _get_catalog().load_team(namespace)
    return team_card.model_dump(mode="json")


async def export_namespace(namespace: str) -> Response:
    """Export ``namespace`` as a single ``application/yaml`` bundle document."""
    logger.debug("GET /catalog/namespace/%s/export", namespace)
    yaml_text = _get_catalog().export_namespace_yaml(namespace)
    return Response(content=yaml_text, media_type="application/yaml")


async def import_namespace(
    body: bytes = Body(..., media_type="application/yaml"),
) -> list[Entry]:
    """Import a bundle YAML document as an atomic namespace replacement."""
    logger.debug("POST /catalog/namespace/import")
    try:
        yaml_text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400, detail=f"bundle body is not valid UTF-8: {exc}"
        ) from exc
    try:
        yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=422, detail=f"failed to parse bundle YAML: {exc}") from exc
    return _get_catalog().import_namespace_yaml(yaml_text)


async def validate_namespace_get(namespace: str) -> NamespaceValidationReport:
    """Validate the persisted state of ``namespace`` — AC22."""
    logger.debug("GET /catalog/namespace/%s/validate", namespace)
    return _get_catalog().validate_namespace(namespace)


async def validate_namespace_post(
    body: bytes = Body(..., media_type="application/yaml"),
) -> NamespaceValidationReport:
    """Dry-run validate a proposed bundle YAML — AC23."""
    logger.debug("POST /catalog/namespace/validate")
    try:
        yaml_text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=422, detail=f"bundle body is not valid UTF-8: {exc}"
        ) from exc
    try:
        yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=422, detail=f"failed to parse bundle YAML: {exc}") from exc
    return _get_catalog().validate_namespace_yaml(yaml_text)


async def resolve_entry(
    kind: EntryKind,
    id: str,
    namespace: str = Query(...),
) -> dict[str, Any]:
    """Resolve an entry into its dumped runtime model — AC14."""
    logger.debug("GET /catalog/%s/%s/resolve?namespace=%s", kind, id, namespace)
    catalog = _get_catalog()
    entry = catalog.get(namespace, id)
    if entry.kind != kind:
        raise EntryNotFoundError(f"Entry ({namespace}, {id}, kind={kind}) not found")
    model = catalog.resolve_by_id(namespace, id)
    return model.model_dump(mode="json")


async def list_references(
    kind: EntryKind,
    id: str,
    namespace: str = Query(...),
) -> list[Entry]:
    """Return entries in ``namespace`` referencing ``(kind, id)`` — AC16."""
    logger.debug("GET /catalog/%s/%s/references?namespace=%s", kind, id, namespace)
    catalog = _get_catalog()
    entry = catalog.get(namespace, id)
    if entry.kind != kind:
        raise EntryNotFoundError(f"Entry ({namespace}, {id}, kind={kind}) not found")
    return catalog.find_references(namespace, id)


async def search_entries(kind: EntryKind, request: Request) -> list[Entry]:
    """Search entries of ``kind`` via an ``EntryQuery`` body — AC12.

    Accepts the ``EntryQuery`` body in either JSON or YAML (see Epic 21).
    """
    query = await _parse_body_as(request, EntryQuery)
    logger.debug("POST /catalog/%s/search", kind)
    if query.kind is not None and query.kind != kind:
        raise HTTPException(status_code=400, detail="kind mismatch between path and body")
    effective = query if query.kind is not None else query.model_copy(update={"kind": kind})
    return _get_catalog().list(effective)


async def create_entry(kind: EntryKind, request: Request) -> Entry:
    """Create a new entry — AC7.

    Accepts the ``Entry`` body in either JSON or YAML (see Epic 21).
    """
    entry = await _parse_body_as(request, Entry)
    logger.debug("POST /catalog/%s — creating (%s, %s)", kind, entry.namespace, entry.id)
    _ensure_kind(entry.kind, kind)
    return _get_catalog().create(entry)


async def list_entries(
    kind: EntryKind,
    namespace: str | None = None,
    user_id: str | None = None,
    user_id_set: bool | None = None,
    parent_namespace: str | None = None,
    parent_id: str | None = None,
) -> list[Entry]:
    """List entries of ``kind`` with optional filters — AC11."""
    logger.debug("GET /catalog/%s (list)", kind)
    query = EntryQuery(
        kind=kind,
        namespace=namespace,
        user_id=user_id,
        user_id_set=user_id_set,
        parent_namespace=parent_namespace,
        parent_id=parent_id,
    )
    return _get_catalog().list(query)


async def get_entry(
    kind: EntryKind,
    id: str,
    namespace: str = Query(...),
) -> Entry:
    """Get an entry by ``(namespace, id)``; kind mismatch → 404. AC8."""
    logger.debug("GET /catalog/%s/%s?namespace=%s", kind, id, namespace)
    entry = _get_catalog().get(namespace, id)
    if entry.kind != kind:
        raise EntryNotFoundError(f"Entry ({namespace}, {id}, kind={kind}) not found")
    return entry


async def update_entry(
    kind: EntryKind,
    id: str,
    request: Request,
    namespace: str = Query(...),
) -> Entry:
    """Update an entry; URL is authoritative over body. AC9.

    Accepts the ``Entry`` body in either JSON or YAML (see Epic 21).
    """
    entry = await _parse_body_as(request, Entry)
    logger.debug("PUT /catalog/%s/%s?namespace=%s", kind, id, namespace)
    _ensure_kind(entry.kind, kind)
    if entry.namespace != namespace or entry.id != id:
        raise HTTPException(status_code=400, detail="namespace/id mismatch between URL and body")
    return _get_catalog().update(entry)


async def delete_entry(
    kind: EntryKind,
    id: str,
    namespace: str = Query(...),
) -> Response:
    """Delete an entry; missing or kind-mismatched entry raises 404. AC10."""
    logger.debug("DELETE /catalog/%s/%s?namespace=%s", kind, id, namespace)
    catalog = _get_catalog()
    existing = catalog.get(namespace, id)
    if existing.kind != kind:
        raise EntryNotFoundError(f"Entry ({namespace}, {id}, kind={kind}) not found")
    catalog.delete(namespace, id)
    return Response(status_code=204)


# --- Route registration -----------------------------------------------------


def _register_static_routes(target: APIRouter) -> None:
    """Register the namespace-scoped and schema routes that always ship.

    These routes are declared **first** so the literal paths take precedence
    over the dynamic ``/{kind}`` segment when the kind-generic family is
    also registered (Story 16.7 — FastAPI dispatches in declaration order).
    """
    target.get("/namespaces", response_model=list[NamespaceSummary])(list_namespaces)
    target.post(
        "/clone",
        response_model=Entry,
        status_code=201,
        openapi_extra=_multi_format_body_openapi("CloneRequest"),
    )(clone_entry)
    target.get("/schema")(get_schema)
    target.get("/model_types", response_model=list[str])(list_model_types)
    target.get("/team/{namespace}/resolve")(resolve_team)
    target.get("/namespace/{namespace}/export")(export_namespace)
    target.post("/namespace/import", response_model=list[Entry], status_code=201)(import_namespace)
    target.get("/namespace/{namespace}/validate", response_model=NamespaceValidationReport)(
        validate_namespace_get
    )
    target.post("/namespace/validate", response_model=NamespaceValidationReport)(
        validate_namespace_post
    )


def _register_generic_kind_routes(target: APIRouter) -> None:
    """Register the eight generic ``/catalog/{kind}`` CRUD routes.

    Registered only when ``CatalogRouterSettings.expose_generic_kind_crud``
    is True (Story 16.7). The order matters — graph/search routes on
    ``/{kind}/{id}/...`` and ``/{kind}/search`` must appear before the bare
    ``/{kind}/{id}`` routes, and all of them must appear **after** the
    static routes registered by :func:`_register_static_routes` so literal
    paths win dispatch order.
    """
    # Graph routes on /{kind}/{id}/... (must precede /{kind}/{id}).
    target.get("/{kind}/{id}/resolve")(resolve_entry)
    target.get("/{kind}/{id}/references", response_model=list[Entry])(list_references)
    # Search (must precede /{kind}/{id}).
    target.post(
        "/{kind}/search",
        response_model=list[Entry],
        openapi_extra=_multi_format_body_openapi("EntryQuery"),
    )(search_entries)
    # CRUD routes.
    target.post(
        "/{kind}",
        response_model=Entry,
        status_code=201,
        openapi_extra=_multi_format_body_openapi("Entry"),
    )(create_entry)
    target.get("/{kind}", response_model=list[Entry])(list_entries)
    target.get("/{kind}/{id}", response_model=Entry)(get_entry)
    target.put(
        "/{kind}/{id}",
        response_model=Entry,
        openapi_extra=_multi_format_body_openapi("Entry"),
    )(update_entry)
    target.delete("/{kind}/{id}", status_code=204)(delete_entry)


def build_router(settings: CatalogRouterSettings | None = None) -> APIRouter:
    """Construct a fresh ``/catalog`` ``APIRouter`` gated by ``settings``.

    Use this factory in tests (or in any deployment that wants explicit
    control over the flag) instead of the module-level :data:`router`.
    When ``settings.expose_generic_kind_crud`` is True, the eight generic
    kind routes are registered after the static routes; otherwise only the
    static routes are registered and requests to ``/catalog/{kind}*`` return
    404.

    Args:
        settings: Router configuration. Defaults to
            :meth:`CatalogRouterSettings.from_env` — i.e. whatever the
            ``AKGENTIC_CATALOG_EXPOSE_GENERIC_KIND_CRUD`` environment
            variable says, or the safe default (``False``) if unset.

    Returns:
        A configured :class:`fastapi.APIRouter` ready for
        ``app.include_router()``.
    """
    effective = settings if settings is not None else CatalogRouterSettings.from_env()
    new_router = APIRouter(prefix="/catalog", tags=["catalog"])
    _register_static_routes(new_router)
    if effective.expose_generic_kind_crud:
        _register_generic_kind_routes(new_router)
    return new_router


# Module-level router — materialised at import time from the ambient
# environment. Downstream callers that `from akgentic.catalog.api.router
# import router` pick up whichever routes match
# ``AKGENTIC_CATALOG_EXPOSE_GENERIC_KIND_CRUD`` at process start. Tests that
# need to flip the flag per-test should use :func:`build_router` directly.
router = build_router()
