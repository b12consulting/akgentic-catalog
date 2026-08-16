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
from typing import TYPE_CHECKING, Any, Final, get_args

import yaml
from fastapi import APIRouter, Body, HTTPException, Query, Request, Response
from pydantic import BaseModel

from akgentic.catalog.api._settings import CatalogRouterSettings
from akgentic.catalog.models.entry import Entry, EntryKind
from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.namespace_meta import NamespaceMeta
from akgentic.catalog.models.queries import CloneRequest, EntryQuery
from akgentic.catalog.resolver import enumerate_allowlisted_model_types, load_model_type
from akgentic.catalog.validation import NamespaceValidationReport

if TYPE_CHECKING:
    from akgentic.catalog.catalog import Catalog

__all__ = ["NamespaceKindCount", "NamespaceSummary", "build_router", "router", "set_catalog"]

# Content types accepted on the YAML branch of ``_parse_body_as``. Both
# variants are in the wild; ``application/x-yaml`` is the historical
# unregistered spelling and ``application/yaml`` the RFC9512-registered one.
_YAML_CONTENT_TYPES: frozenset[str] = frozenset({"application/yaml", "application/x-yaml"})

# The six entry kinds, derived from the ``EntryKind`` alias rather than
# hand-written, so a seventh kind reaches ``NamespaceSummary.counts``
# without a second edit here.
_ENTRY_KINDS: Final[tuple[EntryKind, ...]] = get_args(EntryKind)


class NamespaceKindCount(BaseModel):
    """One kind's tally on a namespace row (ADR-021 §D1).

    A model wrapping a single integer, deliberately — ``counts`` maps to
    this rather than to a bare ``int``. A second tally (how many entries
    of that kind reach the namespace from *another* namespace) is wanted
    later and is out of scope here; the nested object makes that an
    additive field instead of a breaking reshape of a response consumers
    already parse. Only ``total`` ships: a placeholder that is always
    zero would be worse than an absent one, because a consumer cannot
    tell a stub from a real zero.
    """

    total: int


class NamespaceSummary(BaseModel):
    """Flat DTO for ``GET /catalog/namespaces`` — one row per namespace.

    Every namespace surfaced here has at least one ``kind="team"`` OR
    ``kind="meta"`` entry; the picker now sees both classes via the
    union-discovery handler (Story 17.7). The shape intentionally omits
    ``user_id`` and other entry-model fields to keep the payload minimal
    and to avoid leaking tenancy design to the picker.

    Eight pinned fields in declaration order — the original six, then the
    two appended by ADR-021 §D1. Nothing above ``owner`` was renamed,
    reordered or re-derived when the two were added.

    * ``namespace`` — the namespace identifier.
    * ``name`` — the display name (meta-then-team fallback).
    * ``description`` — the human-readable description.
    * ``team`` — ``True`` iff a ``kind="team"`` entry exists in the
      namespace; ``False`` for team-less library namespaces (e.g. a
      platform-defined ``global`` carrying shared models / tools / prompts
      without a team).
    * ``shareable`` — ``True`` iff a ``kind="meta"`` entry exists AND its
      ``payload["shareable"] is True`` (typed bool at the root, strict-bool
      comparison — ADR-008 §D2 as updated 2026-05-08 rev 2).
    * ``public`` — ``True`` iff a ``kind="meta"`` entry exists AND its
      ``payload["public"] is True`` (typed bool at the root, strict-bool
      comparison — ADR-009 §D2). Visibility filtering at the catalog
      list/get/search/clone boundary lands in Story 18.4; the picker
      surfaces the flag so frontends can render a "public library" badge.
    * ``owner`` — the ``user_id`` of the namespace's ``kind="team"``
      entry, falling back to its ``kind="meta"`` entry, ``None`` when
      neither carries one. The same ownership anchor ``akgentic-infra``'s
      owner-or-admin gate already resolves — but **reported, never
      enforced** (ADR-021 §D3): this field authorizes nothing, filters
      nothing, and changes no row's visibility.
    * ``counts`` — per-kind entry tallies for the namespace. **All six
      ``EntryKind`` keys are always present**, zero-valued where the
      namespace holds none of that kind, so a consumer never has to
      distinguish an absent key from a zero. Each value is a
      :class:`NamespaceKindCount` rather than a bare ``int`` — see that
      class for why. The tallies come from the visibility-filtered
      ``Catalog.list``, so they agree with what the same caller could
      list through ``GET /catalog/{kind}``.
    """

    namespace: str
    name: str
    description: str
    team: bool
    shareable: bool
    public: bool
    owner: str | None
    counts: dict[str, NamespaceKindCount]


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
    """List every namespace surfaced by team OR meta entries.

    Issues exactly SIX repository round-trips — one `catalog.list` per
    `EntryKind` — independent of the namespace count (ADR-021 §D2). The
    `team` and `meta` slices drive discovery exactly as they did when
    they were the only two queries; the other four exist solely to feed
    `counts`.

    **Discovery is unchanged and must stay that way.** A namespace is
    surfaced iff it has a `team` OR a `meta` entry — the union below is
    taken over those two slices only, never over the whole grouped
    result. A namespace holding only tools is still absent from this
    listing; widening the row set to match the widened query would put
    library-only namespaces into the team-creation picker.

    Counting goes through `Catalog.list`, which applies the visibility
    filter, and never through `Catalog.list_by_namespace`, which does
    not: a caller's tallies must match what that caller could actually
    list. No `user_id` filter is applied on top — callers (or
    tier-specific middleware) are responsible for tenancy filtering.

    The result list contains one row per namespace in the union, sorted
    alphabetically ascending by `namespace`. Each row is built via
    `_build_namespace_summary` from the (optional) team and (optional)
    meta entries already in memory — no extra `catalog.get` round-trip
    per row.

    Projection rules:

    * `name` — `meta.payload["name"]` if a meta entry exists and its
      name is a non-empty string; else the namespace identifier itself
      (`team.namespace` / `meta.namespace`, identical by definition for
      a row in the union — the dict key from the union-discovery loop
      below). The team entry's `payload["name"]` is intentionally NOT
      consulted — operators wanting a different display string create a
      meta entry.
    * `description` — `meta.description` if a meta entry exists; else
      `team.description` if a team exists; else `""`.
    * `team` — `True` iff a team entry was found by the team query.
    * `shareable` — `True` iff a meta entry was found AND
      `meta.payload.get("shareable") is True` (typed-bool, strict
      comparison).
    * `public` — `True` iff a meta entry was found AND
      `meta.payload.get("public") is True` (typed-bool, strict
      comparison — ADR-009 §D2).
    * `owner` — `_derive_owner` on the same two entries.
    * `counts` — the namespace's slice of `_count_by_namespace`.
    """
    logger.debug("GET /catalog/namespaces")
    catalog = _get_catalog()
    by_kind: dict[EntryKind, list[Entry]] = {
        kind: catalog.list(EntryQuery(kind=kind)) for kind in _ENTRY_KINDS
    }
    teams_by_ns: dict[str, Entry] = {e.namespace: e for e in by_kind["team"]}
    metas_by_ns: dict[str, Entry] = {e.namespace: e for e in by_kind["meta"]}
    counts_by_ns = _count_by_namespace(by_kind)
    # Discovery: team OR meta only — NOT the union of all six slices.
    namespaces = sorted(set(teams_by_ns) | set(metas_by_ns))
    return [
        _build_namespace_summary(
            ns,
            teams_by_ns.get(ns),
            metas_by_ns.get(ns),
            owner=_derive_owner(teams_by_ns.get(ns), metas_by_ns.get(ns)),
            counts=counts_by_ns.get(ns, _zero_counts()),
        )
        for ns in namespaces
    ]


def _zero_counts() -> dict[str, NamespaceKindCount]:
    """Return the all-zero per-kind tally map, one key per ``EntryKind``.

    The default a row falls back to when it is in the discovery union but
    absent from the grouped tallies — unreachable in practice (a row is
    in the union only because a team or meta entry put it there, and that
    entry is itself counted), kept as the map's zero value so no caller
    can ever receive a row with a missing key.
    """
    return {kind: NamespaceKindCount(total=0) for kind in _ENTRY_KINDS}


def _count_by_namespace(
    by_kind: dict[EntryKind, list[Entry]],
) -> dict[str, dict[str, NamespaceKindCount]]:
    """Tally the grouped entries per namespace, per kind, in one pass.

    Performs NO repository I/O — ``by_kind`` is the result of the six
    queries :func:`list_namespaces` has already issued. Every namespace
    present in the result carries **all six** ``EntryKind`` keys,
    zero-valued where it holds none of that kind.

    Args:
        by_kind: Entries grouped by kind, as returned by the six
            ``catalog.list`` calls.

    Returns:
        A map from namespace to its complete six-key tally map. Only
        namespaces that hold at least one entry appear — callers project
        the rest through :func:`_zero_counts`.
    """
    tallies: dict[str, dict[str, int]] = {}
    for kind, entries in by_kind.items():
        for entry in entries:
            if entry.namespace not in tallies:
                tallies[entry.namespace] = {k: 0 for k in _ENTRY_KINDS}
            tallies[entry.namespace][kind] += 1
    return {
        namespace: {k: NamespaceKindCount(total=n) for k, n in per_kind.items()}
        for namespace, per_kind in tallies.items()
    }


def _derive_owner(team: Entry | None, meta: Entry | None) -> str | None:
    """Return the namespace's owner: the team entry's ``user_id``, else the meta's, else ``None``.

    Performs NO repository I/O — both entries come from the caller.
    ``Entry.user_id`` is a non-empty string by construction, so the
    emptiness guards below are defensive only; they cost nothing and keep
    the chain honest if that ever loosens.

    Reported, never enforced (ADR-021 §D3) — this value gates nothing.
    """
    if team is not None and team.user_id:
        return team.user_id
    if meta is not None and meta.user_id:
        return meta.user_id
    return None


def _build_namespace_summary(
    namespace: str,
    team: Entry | None,
    meta: Entry | None,
    *,
    owner: str | None,
    counts: dict[str, NamespaceKindCount],
) -> NamespaceSummary:
    """Project optional team/meta entries to a ``NamespaceSummary`` row.

    Performs NO repository I/O — both entries, the owner and the tally
    map are passed in by :func:`list_namespaces` from the queries it has
    already issued. That property is load-bearing: it is what makes this
    helper unit-testable with no catalog in hand, and what keeps the
    handler's round-trip count constant in the namespace count.

    The helper applies the five projection rules:

    * ``name`` — two-rung chain: ``meta.payload["name"]`` if a meta entry
      is present AND its ``name`` is a non-empty string; else the
      ``namespace`` argument verbatim (the dict key from the
      union-discovery loop in :func:`list_namespaces` — identical to
      ``team.namespace`` / ``meta.namespace`` for any row in the union).
      The team entry's ``payload["name"]`` is intentionally NOT consulted
      by the picker — operators who want a different display string
      create a meta entry. The team payload's ``name`` is preserved on
      disk byte-identically; it is no longer authoritative for the
      picker.
    * ``description`` — meta if present, else team, else empty string
      (meta-then-team fallback, unchanged).
    * ``team`` — ``True`` iff ``team is not None``.
    * ``shareable`` — ``True`` iff a meta entry exists AND its
      ``payload.get("shareable") is True`` (strict-bool, no truthy-string
      coercion).
    * ``public`` — ``True`` iff a meta entry exists AND its
      ``payload.get("public") is True`` (strict-bool, no truthy-string
      coercion — ADR-009 §D2).

    ``owner`` and ``counts`` are set verbatim from the keyword arguments
    — the helper derives neither. They are keyword-only so no existing
    positional call site can silently shift.

    Args:
        namespace: The namespace identifier (the union-discovery dict key).
        team: The namespace's ``kind="team"`` entry, if any.
        meta: The namespace's ``kind="meta"`` entry, if any.
        owner: The derived owner (see :func:`_derive_owner`).
        counts: The namespace's complete six-key tally map.
    """
    description = ""
    if team is not None:
        description = team.description

    name = namespace
    shareable = False
    public = False
    if meta is not None:
        meta_payload = meta.payload if isinstance(meta.payload, dict) else {}
        meta_name = meta_payload.get("name")
        if isinstance(meta_name, str) and meta_name != "":
            name = meta_name
        description = meta.description
        shareable = meta_payload.get("shareable") is True
        public = meta_payload.get("public") is True

    return NamespaceSummary(
        namespace=namespace,
        name=name,
        description=description,
        team=team is not None,
        shareable=shareable,
        public=public,
        owner=owner,
        counts=counts,
    )


async def get_namespace_meta(namespace: str) -> Entry:
    """Return the `(namespace, "_meta")` entry; 404 when absent.

    Delegates to `Catalog.get(namespace, "_meta")`. `EntryNotFoundError`
    propagates unchanged (mapped to HTTP 404 by `api/_errors.py`). A stored
    entry whose `kind` is not `"meta"` (defensive — the catalog
    invariants should prevent this) is treated as absent and the handler
    re-raises `EntryNotFoundError` so the response shape stays consistent.
    """
    logger.debug("GET /catalog/namespace/%s/meta", namespace)
    entry = _get_catalog().get(namespace, "_meta")
    if entry.kind != "meta":
        raise EntryNotFoundError(f"Entry ({namespace}, _meta, kind=meta) not found")
    return entry


async def put_namespace_meta(namespace: str, request: Request) -> Response:
    """Upsert the `(namespace, "_meta")` entry from a `NamespaceMeta` body.

    Accepts the body in JSON or YAML via `_parse_body_as`. The body is a
    `NamespaceMeta`, which declares none of the entry's identity fields
    (`id` / `kind` / `namespace` / `model_type`), so a payload that carries
    them by accident is simply not parsed.

    Ownership derivation, the create/update dispatch and the `_meta` entry's
    shape all live in `Catalog.put_namespace_meta` — the single writer, shared
    with the bundle-import path. The handler owns only the HTTP envelope:
    201 when the entry was created, 200 when it was updated.
    """
    meta = await _parse_body_as(request, NamespaceMeta)
    logger.debug("PUT /catalog/namespace/%s/meta", namespace)
    entry, created = _get_catalog().put_namespace_meta(namespace, meta)
    return Response(
        content=entry.model_dump_json(),
        media_type="application/json",
        status_code=201 if created else 200,
    )


async def clone_entry(request: Request) -> Entry:
    """Deep-copy an entry tree into `dst_namespace`; the body may be JSON or YAML."""
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
    """Return the JSON Schema for `model_type`."""
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
    """Resolve the team entry in `namespace` into a dumped `TeamCard`."""
    logger.debug("GET /catalog/team/%s/resolve", namespace)
    team_card = _get_catalog().load_team(namespace)
    return team_card.model_dump(mode="json")


async def export_namespace(namespace: str) -> Response:
    """Export `namespace` as a single `application/yaml` bundle document."""
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
    """Validate the persisted state of `namespace`."""
    logger.debug("GET /catalog/namespace/%s/validate", namespace)
    return _get_catalog().validate_namespace(namespace)


async def validate_namespace_post(
    body: bytes = Body(..., media_type="application/yaml"),
) -> NamespaceValidationReport:
    """Dry-run validate a proposed bundle YAML."""
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
    """Resolve an entry into its dumped runtime model."""
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
    """Return entries in `namespace` referencing `(kind, id)`."""
    logger.debug("GET /catalog/%s/%s/references?namespace=%s", kind, id, namespace)
    catalog = _get_catalog()
    entry = catalog.get(namespace, id)
    if entry.kind != kind:
        raise EntryNotFoundError(f"Entry ({namespace}, {id}, kind={kind}) not found")
    return catalog.find_references(namespace, id)


async def search_entries(kind: EntryKind, request: Request) -> list[Entry]:
    """Search entries of `kind` via an `EntryQuery` body; JSON or YAML."""
    query = await _parse_body_as(request, EntryQuery)
    logger.debug("POST /catalog/%s/search", kind)
    if query.kind is not None and query.kind != kind:
        raise HTTPException(status_code=400, detail="kind mismatch between path and body")
    effective = query if query.kind is not None else query.model_copy(update={"kind": kind})
    return _get_catalog().list(effective)


async def create_entry(kind: EntryKind, request: Request) -> Entry:
    """Create a new entry; the `Entry` body may be JSON or YAML."""
    entry = await _parse_body_as(request, Entry)
    logger.debug("POST /catalog/%s — creating (%s, %s)", kind, entry.namespace, entry.id)
    _ensure_kind(entry.kind, kind)
    return _get_catalog().create(entry)


async def list_entries(
    kind: EntryKind,
    namespace: str | None = None,
    user_id: str | None = None,
    user_id_set: bool | None = None,
) -> list[Entry]:
    """List entries of `kind` with optional filters."""
    logger.debug("GET /catalog/%s (list)", kind)
    query = EntryQuery(
        kind=kind,
        namespace=namespace,
        user_id=user_id,
        user_id_set=user_id_set,
    )
    return _get_catalog().list(query)


async def get_entry(
    kind: EntryKind,
    id: str,
    namespace: str = Query(...),
) -> Entry:
    """Get an entry by `(namespace, id)`; kind mismatch → 404."""
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
    """Update an entry; URL is authoritative over body. Body may be JSON or YAML."""
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
    """Delete an entry; missing or kind-mismatched entry raises 404."""
    logger.debug("DELETE /catalog/%s/%s?namespace=%s", kind, id, namespace)
    catalog = _get_catalog()
    existing = catalog.get(namespace, id)
    if existing.kind != kind:
        raise EntryNotFoundError(f"Entry ({namespace}, {id}, kind={kind}) not found")
    catalog.delete(namespace, id)
    return Response(status_code=204)


async def delete_namespace(namespace: str) -> Response:
    """Delete an entire namespace (all entries + `_meta`); 204 on success.

    Implements ADR-028 §Decision 5. Delegates to
    `Catalog.delete_namespace` with no added try/except so the per-entry
    delete error contract is preserved: `EntryNotFoundError` (empty/absent
    namespace) propagates → HTTP 404, and `CatalogValidationError` (an
    external inbound reference blocks the delete) propagates → HTTP 409, both
    via the app-wide handlers in `api/_errors.py`. The route carries NO
    authorization dependency — owner-or-admin gating is enforced upstream by
    the infra route gate (a separate epic).
    """
    logger.debug("DELETE /catalog/namespace/%s", namespace)
    _get_catalog().delete_namespace(namespace)
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
    target.get("/namespace/{namespace}/meta", response_model=Entry)(get_namespace_meta)
    target.put(
        "/namespace/{namespace}/meta",
        response_model=Entry,
        openapi_extra=_multi_format_body_openapi("NamespaceMeta"),
    )(put_namespace_meta)
    target.delete("/namespace/{namespace}", status_code=204)(delete_namespace)


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
