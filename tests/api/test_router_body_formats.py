"""Round-trip and unit tests for Epic 21 — multi-format body handling.

Covers:

* Story 21.1 ACs — every per-kind CRUD handler (``create_entry``,
  ``update_entry``, ``search_entries``, ``clone_entry``) accepts JSON and
  YAML request bodies with identical semantics, and ``openapi.json``
  advertises all three content types (``application/json``,
  ``application/yaml``, ``application/x-yaml``) on the four endpoints.
* Unit tests for ``_parse_body_as`` covering every content-type branch,
  empty-body handling, malformed JSON/YAML error responses, and case /
  parameter-stripping normalization.

All integration tests use the ``api_client`` fixture from
``tests/api/conftest.py`` — a real ``TestClient`` around a YAML-backed
``Catalog``. No ``MagicMock``; no shortcut past the handler signature.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import pydantic
import pytest
import yaml

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.requests import Request  # noqa: E402

from akgentic.catalog.api.router import _parse_body_as  # noqa: E402
from akgentic.catalog.catalog import Catalog  # noqa: E402
from akgentic.catalog.models.entry import Entry  # noqa: E402
from akgentic.catalog.models.queries import CloneRequest, EntryQuery  # noqa: E402

from ..conftest import team_payload  # noqa: E402

_TEAM_TYPE = "akgentic.team.models.TeamCard"
_PROMPT_TYPE = "akgentic.llm.models.PromptTemplate"


# --- helpers ----------------------------------------------------------------


def _prompt_payload() -> dict[str, Any]:
    """Return a minimal dict payload for a ``prompt`` entry (free-form)."""
    return {"text": "hello"}


def _team_entry(
    namespace: str = "ns-body",
    entry_id: str = "team",
    user_id: str = "anonymous",
) -> Entry:
    """Build an ``Entry`` of kind ``team`` with a valid payload."""
    return Entry(
        id=entry_id,
        kind="team",
        namespace=namespace,
        user_id=user_id,
        model_type=_TEAM_TYPE,
        payload=team_payload(),
    )


def _prompt_entry(
    namespace: str = "ns-body",
    entry_id: str = "prompt-1",
    user_id: str = "anonymous",
) -> Entry:
    """Build an ``Entry`` of kind ``prompt`` with a valid payload."""
    return Entry(
        id=entry_id,
        kind="prompt",
        namespace=namespace,
        user_id=user_id,
        model_type="akgentic.core.agent_card.AgentCard",  # allowlisted placeholder
        payload={
            # AgentCard-shaped payload so model_type validates at resolve time;
            # structural validity at the Entry layer is all we need here.
            "description": "",
            "skills": [],
            "agent_class": "akgentic.core.agent.Akgent",
            "config": {"name": entry_id, "role": "r"},
            "routes_to": [],
            "metadata": {},
        },
    )


def _json_body(model: pydantic.BaseModel) -> str:
    """Serialize ``model`` as a JSON request body."""
    return model.model_dump_json()


def _yaml_body(model: pydantic.BaseModel) -> str:
    """Serialize ``model`` as a YAML request body."""
    return yaml.safe_dump(model.model_dump(mode="json"), sort_keys=False)


class _BodyFormat(NamedTuple):
    """One request-body encoding: its id, its header and its serializer.

    ``name`` doubles as the per-case namespace suffix, so each case keeps its
    own namespace and the round-trip assertions read back what they sent.
    """

    name: str
    content_type: str
    serialize: Callable[[pydantic.BaseModel], str]


_BODY_FORMATS = [
    _BodyFormat("json", "application/json", _json_body),
    _BodyFormat("yaml", "application/yaml", _yaml_body),
]

_BODY_FORMAT_IDS = [fmt.name for fmt in _BODY_FORMATS]


# --- Task 7: round-trip handler tests --------------------------------------


class TestCreateEntryBodyFormats:
    """POST /catalog/{kind} under JSON and YAML bodies."""

    @pytest.mark.parametrize("fmt", _BODY_FORMATS, ids=_BODY_FORMAT_IDS)
    def test_body(self, fmt: _BodyFormat, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        namespace = f"ns-create-{fmt.name}"
        entry = _team_entry(namespace=namespace)
        response = client.post(
            "/catalog/team",
            content=fmt.serialize(entry).encode("utf-8"),
            headers={"Content-Type": fmt.content_type},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["id"] == "team"
        assert body["namespace"] == namespace
        assert body["kind"] == "team"

    def test_yaml_body_prompt(self, api_client: tuple[TestClient, Catalog]) -> None:
        """Second kind covered — AC requires at least two kinds exercised."""
        client, catalog = api_client
        # Seed a team first so the prompt has a namespace to live in.
        catalog.create(_team_entry(namespace="ns-create-yaml-prompt"))
        entry = _prompt_entry(namespace="ns-create-yaml-prompt", entry_id="p1")
        response = client.post(
            "/catalog/prompt",
            content=_yaml_body(entry).encode("utf-8"),
            headers={"Content-Type": "application/yaml"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["id"] == "p1"
        assert body["kind"] == "prompt"


class TestUpdateEntryBodyFormats:
    """PUT /catalog/{kind}/{id} under JSON and YAML bodies."""

    @pytest.mark.parametrize("fmt", _BODY_FORMATS, ids=_BODY_FORMAT_IDS)
    def test_body(self, fmt: _BodyFormat, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        namespace = f"ns-update-{fmt.name}"
        catalog.create(_team_entry(namespace=namespace))
        updated = _team_entry(namespace=namespace)
        updated = updated.model_copy(update={"description": f"updated-{fmt.name}"})
        response = client.put(
            f"/catalog/team/team?namespace={namespace}",
            content=fmt.serialize(updated).encode("utf-8"),
            headers={"Content-Type": fmt.content_type},
        )
        assert response.status_code == 200
        assert response.json()["description"] == f"updated-{fmt.name}"

    def test_url_mismatch_400_under_yaml(self, api_client: tuple[TestClient, Catalog]) -> None:
        """URL authoritative check fires regardless of body format."""
        client, catalog = api_client
        right = "ns-update-mismatch-right"
        catalog.create(_team_entry(namespace=right))
        # Body namespace deliberately differs from URL namespace.
        body_entry = _team_entry(namespace="ns-update-mismatch-wrong")
        response = client.put(
            f"/catalog/team/team?namespace={right}",
            content=_yaml_body(body_entry).encode("utf-8"),
            headers={"Content-Type": "application/yaml"},
        )
        assert response.status_code == 400


class TestSearchEntriesBodyFormats:
    """POST /catalog/{kind}/search under JSON and YAML bodies."""

    @pytest.mark.parametrize("fmt", _BODY_FORMATS, ids=_BODY_FORMAT_IDS)
    def test_body(self, fmt: _BodyFormat, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        namespace = f"ns-search-{fmt.name}"
        catalog.create(_team_entry(namespace=namespace))
        query = EntryQuery(namespace=namespace)
        response = client.post(
            "/catalog/team/search",
            content=fmt.serialize(query).encode("utf-8"),
            headers={"Content-Type": fmt.content_type},
        )
        assert response.status_code == 200
        rows = response.json()
        assert len(rows) == 1
        assert rows[0]["namespace"] == namespace

    def test_kind_mismatch_400_under_yaml(self, api_client: tuple[TestClient, Catalog]) -> None:
        """Cross-kind mismatch check fires under YAML too."""
        client, _ = api_client
        query = EntryQuery(kind="agent")
        response = client.post(
            "/catalog/team/search",
            content=_yaml_body(query).encode("utf-8"),
            headers={"Content-Type": "application/yaml"},
        )
        assert response.status_code == 400


class TestCloneEntryBodyFormats:
    """POST /catalog/clone under JSON and YAML bodies."""

    @pytest.mark.parametrize("fmt", _BODY_FORMATS, ids=_BODY_FORMAT_IDS)
    def test_body(self, fmt: _BodyFormat, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        src_ns = f"ns-clone-src-{fmt.name}"
        dst_ns = f"ns-clone-dst-{fmt.name}"
        catalog.create(_team_entry(namespace=src_ns))
        req = CloneRequest(src_namespace=src_ns, src_id="team", dst_namespace=dst_ns)
        response = client.post(
            "/catalog/clone",
            content=fmt.serialize(req).encode("utf-8"),
            headers={"Content-Type": fmt.content_type},
        )
        assert response.status_code == 201
        assert response.json()["namespace"] == dst_ns


# --- OpenAPI advertisement --------------------------------------------------


class TestOpenAPIAdvertisement:
    """``openapi.json`` advertises all three content types on the four endpoints."""

    _EXPECTED = {"application/json", "application/yaml", "application/x-yaml"}

    def _schema(self, api_client: tuple[TestClient, Catalog]) -> dict[str, Any]:
        client, _ = api_client
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema: dict[str, Any] = response.json()
        return schema

    def _assert_three_formats(self, schema: dict[str, Any], path: str, method: str) -> None:
        op = schema["paths"][path][method]
        content = op["requestBody"]["content"]
        assert self._EXPECTED.issubset(content.keys()), (
            f"{method.upper()} {path} missing content types: {self._EXPECTED - content.keys()}"
        )

    def test_openapi_declares_three_content_types_on_four_endpoints(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        schema = self._schema(api_client)
        self._assert_three_formats(schema, "/catalog/{kind}", "post")
        self._assert_three_formats(schema, "/catalog/{kind}/{id}", "put")
        self._assert_three_formats(schema, "/catalog/{kind}/search", "post")
        self._assert_three_formats(schema, "/catalog/clone", "post")


# --- Task 8: _parse_body_as unit tests -------------------------------------


def _build_request(body: bytes, content_type: str | None) -> Request:
    """Build a minimal Starlette ``Request`` around raw bytes.

    Uses an in-memory ``receive`` coroutine to feed the body in a single
    ``http.request`` message. Headers are encoded to bytes per the ASGI
    contract. A missing ``content_type`` means no header is attached (tests
    the "default to application/json" fallback).
    """
    headers: list[tuple[bytes, bytes]] = []
    if content_type is not None:
        headers.append((b"content-type", content_type.encode("latin-1")))
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": headers,
    }
    sent = {"done": False}

    async def receive() -> dict[str, Any]:
        if sent["done"]:
            return {"type": "http.disconnect"}
        sent["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)  # type: ignore[arg-type]


def _valid_entry_json_bytes() -> bytes:
    """Canonical JSON bytes for a minimal valid ``Entry``."""
    return _json_body(_team_entry(namespace="ns-unit")).encode("utf-8")


def _valid_entry_yaml_bytes() -> bytes:
    """Canonical YAML bytes for a minimal valid ``Entry``."""
    return _yaml_body(_team_entry(namespace="ns-unit")).encode("utf-8")


class TestParseBodyAs:
    """Unit tests for every branch of ``_parse_body_as``."""

    async def test_json_happy_path(self) -> None:
        request = _build_request(_valid_entry_json_bytes(), "application/json")
        entry = await _parse_body_as(request, Entry)
        assert isinstance(entry, Entry)
        assert entry.kind == "team"

    async def test_yaml_happy_path_application_yaml(self) -> None:
        request = _build_request(_valid_entry_yaml_bytes(), "application/yaml")
        entry = await _parse_body_as(request, Entry)
        assert isinstance(entry, Entry)
        assert entry.kind == "team"

    async def test_yaml_happy_path_application_x_yaml(self) -> None:
        request = _build_request(_valid_entry_yaml_bytes(), "application/x-yaml")
        entry = await _parse_body_as(request, Entry)
        assert isinstance(entry, Entry)
        assert entry.kind == "team"

    async def test_empty_body_json_propagates_validation_error(self) -> None:
        """Empty JSON body -> ``model_validate({})`` -> Pydantic ValidationError."""
        request = _build_request(b"", "application/json")
        with pytest.raises(pydantic.ValidationError):
            await _parse_body_as(request, Entry)

    async def test_empty_body_yaml_propagates_validation_error(self) -> None:
        """Empty YAML body -> ``model_validate({})`` -> Pydantic ValidationError."""
        request = _build_request(b"", "application/yaml")
        with pytest.raises(pydantic.ValidationError):
            await _parse_body_as(request, Entry)

    async def test_unknown_content_type_raises_415(self) -> None:
        request = _build_request(b"<xml/>", "application/xml")
        with pytest.raises(HTTPException) as exc_info:
            await _parse_body_as(request, Entry)
        assert exc_info.value.status_code == 415
        assert "unsupported Content-Type" in str(exc_info.value.detail)
        assert "application/xml" in str(exc_info.value.detail)

    async def test_malformed_yaml_raises_422(self) -> None:
        request = _build_request(b":: not yaml ::", "application/yaml")
        with pytest.raises(HTTPException) as exc_info:
            await _parse_body_as(request, Entry)
        assert exc_info.value.status_code == 422
        assert "invalid YAML body" in str(exc_info.value.detail)

    async def test_malformed_json_raises_422(self) -> None:
        request = _build_request(b"{not json", "application/json")
        with pytest.raises(HTTPException) as exc_info:
            await _parse_body_as(request, Entry)
        assert exc_info.value.status_code == 422
        assert "invalid JSON body" in str(exc_info.value.detail)

    async def test_content_type_parameter_stripping(self) -> None:
        """``application/yaml; charset=utf-8`` is treated as YAML."""
        request = _build_request(_valid_entry_yaml_bytes(), "application/yaml; charset=utf-8")
        entry = await _parse_body_as(request, Entry)
        assert isinstance(entry, Entry)
        assert entry.kind == "team"

    async def test_case_insensitive_content_type(self) -> None:
        """``Application/YAML`` normalizes to ``application/yaml``."""
        request = _build_request(_valid_entry_yaml_bytes(), "Application/YAML")
        entry = await _parse_body_as(request, Entry)
        assert isinstance(entry, Entry)
        assert entry.kind == "team"

    async def test_missing_content_type_defaults_to_json(self) -> None:
        """No ``Content-Type`` header -> JSON branch (Starlette default)."""
        request = _build_request(_valid_entry_json_bytes(), content_type=None)
        entry = await _parse_body_as(request, Entry)
        assert isinstance(entry, Entry)
        assert entry.kind == "team"
