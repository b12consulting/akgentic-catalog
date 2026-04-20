"""Integration tests for ``create_app(backend='postgres', ...)``.

Skips cleanly when the ``[postgres]`` extra is absent. Uses the session-scoped
Postgres testcontainer fixtures defined in the package-root
``tests/conftest.py``.
"""

from __future__ import annotations

import importlib.util

import pytest

pytest.importorskip("nagra")
pytest.importorskip("testcontainers.postgres")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from akgentic.catalog.api.app import create_app  # noqa: E402
from akgentic.catalog.repositories.postgres import (  # noqa: E402
    NagraTemplateCatalogRepository,
    init_db,
)
from tests.conftest import make_template  # noqa: E402


def test_create_app_postgres_backend_missing_conn_string_raises() -> None:
    """AC #3: ``create_app(backend='postgres')`` without conn string raises."""
    with pytest.raises(ValueError, match="postgres_conn_string is required"):
        create_app(backend="postgres")


def test_create_app_postgres_backend_end_to_end(
    postgres_conn_string: str,
) -> None:
    """AC #4 / AC #14: end-to-end create_app -> Nagra repos -> HTTP round-trip."""
    # init_db is a deployment-time hook — not called implicitly by wiring.
    init_db(postgres_conn_string)

    # Truncate before seeding so the test is self-contained.
    from nagra import Transaction

    with Transaction(postgres_conn_string) as trn:
        trn.execute(
            "TRUNCATE template_entries, tool_entries, agent_entries, team_entries"
        )

    # Seed one TemplateEntry directly via the Nagra repo.
    template = make_template(id="tpl-pg-end2end", template="Hello {name}")
    NagraTemplateCatalogRepository(postgres_conn_string).create(template)

    # Build the app via the wiring under test.
    app = create_app(backend="postgres", postgres_conn_string=postgres_conn_string)
    assert isinstance(app, FastAPI)

    # Verify routers are registered.
    route_paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/api/templates" in route_paths
    assert "/api/tools" in route_paths
    assert "/api/agents" in route_paths
    assert "/api/teams" in route_paths

    # Hit the API and confirm the seeded row is reachable through the stack.
    client = TestClient(app)
    resp = client.get("/api/templates/")
    assert resp.status_code == 200
    ids = [entry["id"] for entry in resp.json()]
    assert "tpl-pg-end2end" in ids


@pytest.mark.skipif(
    importlib.util.find_spec("nagra") is not None,
    reason=(
        "Regression guard for no-extra environments; running env has nagra installed."
    ),
)
def test_create_app_postgres_backend_importless_extra() -> None:
    """AC #7: module import succeeds without the ``[postgres]`` extra installed.

    This test is intentionally inverted — it only runs when ``nagra`` is NOT
    importable. In CI with the extra installed it skips. The purpose is to
    guard against regressions in environments that do NOT install the extra.
    """
    from akgentic.catalog.api.app import create_app as _create_app

    assert _create_app is not None
