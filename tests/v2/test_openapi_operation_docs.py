"""Guard the text the catalog API publishes as its operation documentation.

No route handler in ``api/router.py`` passes ``description=`` to its
registration, so FastAPI publishes ``inspect.cleandoc(handler.__doc__)``
verbatim as the operation ``description`` in ``/openapi.json`` and in Swagger
UI. That makes every route docstring a user-facing artefact read by external API
consumers, and it is the one place where sprint metadata a consumer cannot look
up (``ACnn``, ``Epic N``, ``Story N.M``) and RST ``literal`` markup — which the
Markdown renderer at the far end shows as literal backticks — actually reach a
reader.

These tests walk **every** operation the spec contains rather than a hardcoded
route table, so a route added later is covered the day it is written. They read
the generated spec and assert a property of it; they never assert on source text
or on a docstring's presence.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from akgentic.catalog.catalog import Catalog  # noqa: E402

# A path item also carries non-operation keys (``parameters``, ``summary``,
# ``$ref``); only these eight are operations.
_HTTP_METHODS: frozenset[str] = frozenset(
    {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
)

# Sprint metadata that must never reach a published description. Deliberately a
# pattern rather than a list of identifiers: the point is to catch strings
# nobody has written yet.
_PLANNING_REFERENCE = re.compile(r"AC\d|Epic \d|Story \d")

# The router registers 12 static + 8 kind-gated operations; the ``api_client``
# fixture exposes both families. A future route must not turn this red merely by
# existing, hence a floor rather than an exact count.
_MINIMUM_OPERATIONS = 20


def _operations(client: TestClient) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(label, operation)`` for every operation in ``/openapi.json``.

    ``label`` is ``"METHOD /path"`` so a failure names the offender directly.
    """
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec: dict[str, Any] = response.json()
    return [
        (f"{method.upper()} {path}", operation)
        for path, path_item in spec["paths"].items()
        for method, operation in path_item.items()
        if method in _HTTP_METHODS
    ]


class TestPublishedOperationDocs:
    """``/openapi.json`` describes the endpoints, and nothing else."""

    def test_no_planning_reference_in_any_operation(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, _ = api_client
        offenders = [
            f"{label}: {field}={match.group(0)!r}"
            for label, operation in _operations(client)
            for field in ("summary", "description")
            if (match := _PLANNING_REFERENCE.search(operation.get(field) or ""))
        ]
        assert not offenders, "planning references published in the OpenAPI spec:\n" + "\n".join(
            offenders
        )

    def test_no_rst_literal_markup_in_any_description(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, _ = api_client
        offenders = [
            label
            for label, operation in _operations(client)
            if "``" in (operation.get("description") or "")
        ]
        assert not offenders, (
            "RST double-backtick markup published where Markdown is rendered:\n"
            + "\n".join(offenders)
        )

    def test_every_operation_is_documented(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, _ = api_client
        operations = _operations(client)
        undocumented = [
            label for label, operation in operations if not operation.get("description")
        ]
        assert not undocumented, "operations published without a description:\n" + "\n".join(
            undocumented
        )
        assert len(operations) >= _MINIMUM_OPERATIONS
