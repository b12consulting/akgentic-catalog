"""Unified v2 ``/catalog`` FastAPI router — CRUD, search, graph, schema.

This is the v2 HTTP surface for the unified ``Catalog`` service. It exposes:

* Per-entry CRUD (``POST /{kind}``, ``GET /{kind}/{id}``, ``PUT /{kind}/{id}``,
  ``DELETE /{kind}/{id}``) — every per-entry route requires ``namespace`` as a
  mandatory query parameter.
* Listing and search (``GET /{kind}``, ``POST /{kind}/search``).
* Graph ops (``POST /clone``, ``GET /{kind}/{id}/resolve``,
  ``GET /team/{namespace}/resolve``, ``GET /{kind}/{id}/references``).
* Schema introspection (``GET /schema``, ``GET /model_types``).

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
``/model_types``, ``/team/{namespace}/resolve``) are declared before the
dynamic ``/{kind}`` family so literal paths take precedence over the
``EntryKind``-typed path segment.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

from akgentic.catalog.models.entry import Entry, EntryKind
from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.queries import CloneRequest, EntryQuery
from akgentic.catalog.resolver import load_model_type

if TYPE_CHECKING:
    from akgentic.catalog.catalog import Catalog

__all__ = ["router", "set_catalog"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/catalog", tags=["catalog"])

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


# --- Static-path routes (must precede /{kind} routes) -----------------------


@router.post("/clone", response_model=Entry, status_code=201)
async def clone_entry(req: CloneRequest) -> Entry:
    """Deep-copy an entry tree into ``dst_namespace`` — AC13."""
    logger.debug(
        "POST /catalog/clone (%s,%s) -> (%s,%s)",
        req.src_namespace,
        req.src_id,
        req.dst_namespace,
        req.dst_user_id,
    )
    return _get_catalog().clone(
        req.src_namespace, req.src_id, req.dst_namespace, req.dst_user_id
    )


@router.get("/schema")
async def get_schema(model_type: str = Query(...)) -> dict[str, Any]:
    """Return the JSON Schema for ``model_type`` — AC17.

    ``load_model_type`` covers allowlist, ``BaseModel``, and reserved-key
    checks. Dynamic-import failures (missing module, missing attribute) are
    surfaced here as ``CatalogValidationError`` so the app-wide 409 handler
    takes over, matching the shard 07 error mapping.
    """
    logger.debug("GET /catalog/schema?model_type=%s", model_type)
    try:
        cls = load_model_type(model_type)
    except (ImportError, AttributeError) as exc:
        raise CatalogValidationError(
            [f"model_type '{model_type}' could not be imported: {exc}"]
        ) from exc
    return cls.model_json_schema()


@router.get("/model_types", response_model=list[str])
async def list_model_types() -> list[str]:
    """Return allowlisted Pydantic model classes loaded in the current process.

    Reflection-based enumeration — the result is "known-imported, not
    known-existing" (see architecture shard 07). AC18.
    """
    logger.debug("GET /catalog/model_types")
    return _enumerate_allowlisted_model_types()


@router.get("/team/{namespace}/resolve")
async def resolve_team(namespace: str) -> dict[str, Any]:
    """Resolve the team entry in ``namespace`` into a dumped ``TeamCard`` — AC15."""
    logger.debug("GET /catalog/team/%s/resolve", namespace)
    team_card = _get_catalog().load_team(namespace)
    return team_card.model_dump(mode="json")


# --- Graph routes on /{kind}/{id}/... (must precede /{kind}/{id}) -----------


@router.get("/{kind}/{id}/resolve")
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


@router.get("/{kind}/{id}/references", response_model=list[Entry])
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


# --- Search (must precede /{kind}/{id}) -------------------------------------


@router.post("/{kind}/search", response_model=list[Entry])
async def search_entries(kind: EntryKind, query: EntryQuery) -> list[Entry]:
    """Search entries of ``kind`` via an ``EntryQuery`` body — AC12."""
    logger.debug("POST /catalog/%s/search", kind)
    if query.kind is not None and query.kind != kind:
        raise HTTPException(status_code=400, detail="kind mismatch between path and body")
    effective = query if query.kind is not None else query.model_copy(update={"kind": kind})
    return _get_catalog().list(effective)


# --- CRUD routes ------------------------------------------------------------


@router.post("/{kind}", response_model=Entry, status_code=201)
async def create_entry(kind: EntryKind, entry: Entry) -> Entry:
    """Create a new entry — AC7."""
    logger.debug("POST /catalog/%s — creating (%s, %s)", kind, entry.namespace, entry.id)
    _ensure_kind(entry.kind, kind)
    return _get_catalog().create(entry)


@router.get("/{kind}", response_model=list[Entry])
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


@router.get("/{kind}/{id}", response_model=Entry)
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


@router.put("/{kind}/{id}", response_model=Entry)
async def update_entry(
    kind: EntryKind,
    id: str,
    entry: Entry,
    namespace: str = Query(...),
) -> Entry:
    """Update an entry; URL is authoritative over body. AC9."""
    logger.debug("PUT /catalog/%s/%s?namespace=%s", kind, id, namespace)
    _ensure_kind(entry.kind, kind)
    if entry.namespace != namespace or entry.id != id:
        raise HTTPException(status_code=400, detail="namespace/id mismatch between URL and body")
    return _get_catalog().update(entry)


@router.delete("/{kind}/{id}", status_code=204)
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


# --- Helpers ----------------------------------------------------------------


def _enumerate_allowlisted_model_types() -> list[str]:
    """Enumerate allowlisted ``BaseModel`` subclasses loaded under ``akgentic.*``.

    Walks a snapshot of ``sys.modules`` to avoid mutation-during-iteration
    issues. Any per-module introspection error is swallowed — optional
    dependencies may be absent, partially imported, or raise on attribute
    access. ``load_model_type`` is used as the authoritative allowlist +
    BaseModel + reserved-key gate.
    """
    results: set[str] = set()
    modules_snapshot = list(sys.modules.items())
    for module_name, module in modules_snapshot:
        if not module_name.startswith("akgentic.") or module is None:
            continue
        _collect_allowlisted(module, results)
    return sorted(results)


def _collect_allowlisted(module: Any, results: set[str]) -> None:
    """Add every allowlisted ``BaseModel`` from ``module`` into ``results``."""
    try:
        items = list(vars(module).items())
    except Exception:  # noqa: BLE001 — defensive; partially imported modules
        return
    for _name, value in items:
        if not isinstance(value, type) or not issubclass(value, BaseModel):
            continue
        path = f"{value.__module__}.{value.__name__}"
        if not path.startswith("akgentic.") or path in results:
            continue
        try:
            load_model_type(path)
        except Exception:  # noqa: BLE001 — swallow reserved-key or import errors
            continue
        results.add(path)
