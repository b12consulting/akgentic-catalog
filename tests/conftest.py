"""Shared test configuration and payload factories for akgentic-catalog.

Defines session-scoped infrastructure fixtures used across the
``tests/api/``, ``tests/cli/``, ``tests/scripts/``, and
``tests/repositories/`` sub-suites, plus the FastAPI ``TestClient``
fixtures and the mongomock ``entries_collection`` shared by every
sub-suite. In particular, the Postgres DSN
fixture lives here so every sub-suite shares ONE live database per
pytest session (start-up is expensive; repeated TRUNCATE is cheap).

It also holds shared payload factories — plain functions, not fixtures,
per the convention recorded in ``tests/v2/conftest.py``: a stateless
construction is a helper, not a fixture. Sub-suites reach them by
package-relative import (``from ..conftest import team_payload``).

Two DSN sources are supported, in priority order:

1. The ``DB_CONN_STRING_PERSISTENCE`` environment variable — CI runs use
   a service container via this channel (the GitHub Actions ``services:``
   block exposes Postgres on ``localhost:5432``). Tests skip the
   testcontainers import path entirely when the env var is set.
2. A local ``testcontainers.postgres.PostgresContainer`` — default for
   developer machines where Docker is running but no external Postgres
   is configured.

Skip-clean discipline: the fixture skips cleanly when the
``[postgres]`` extra is absent (``nagra`` / ``psycopg`` unimportable) OR
when neither channel is viable (no env var AND Docker unavailable).

Implements ADR-011 §"Wiring surface" — the shared fixture lives at the
package-level conftest so every sub-suite can consume it. Navigation-only
reference.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    import pymongo.collection

    from akgentic.catalog.catalog import Catalog


def team_payload(
    *,
    name: str = "team",
    card_description: str = "",
    **overrides: Any,
) -> dict[str, Any]:
    """Return a minimal valid ``TeamCard`` payload.

    ``AgentCard.role`` is a derived property (reads ``config.role``) rather
    than a declared field, so inlining ``"role": ...`` at the card level
    would be stripped by Pydantic on validation and break a strict
    round-trip equality check. Use ``config.role`` exclusively.

    ``card_description`` sets ``entry_point.card.description`` — a nested
    key, unreachable through ``**overrides``, which lands on the top-level
    dict only (applied after the literal is built, so existing keys keep
    their position and unknown keys append).

    The literal is built inside the body on every call, never copied from a
    module-level constant: callers routinely mutate the result — assigning
    to ``["name"]``, and in one case ``pop``-ing the key outright — and a
    shared object would leak those edits into every later test in the
    process. See ``test_team_payload_returns_independent_objects``.
    """
    payload: dict[str, Any] = {
        "name": name,
        "description": "",
        "entry_point": {
            "card": {
                "description": card_description,
                "skills": [],
                "agent_class": "akgentic.core.agent.Akgent",
                "config": {"name": "entry", "role": "entry"},
            },
            "headcount": 1,
            "members": [],
        },
        "members": [],
        "agent_profiles": [],
    }
    payload.update(overrides)
    return payload


def _build_api_client(tmp_path: Path, *, expose_generic_kind_crud: bool) -> tuple[Any, Catalog]:
    """Build a ``(TestClient, Catalog)`` pair wired to a YAML-backed v2 router.

    ``expose_generic_kind_crud`` gates the generic ``/catalog/{kind}`` CRUD
    family (Story 16.7) and is the only thing distinguishing the two fixtures
    below. It is keyword-only — a bare positional boolean at the call sites
    says nothing about which surface is being built.

    Callers must leave those fixtures function-scoped: ``set_catalog`` writes
    the module-level ``_catalog`` in ``api/router.py``, so a wider scope leaks
    one test's catalog into the next.

    ``fastapi`` is guarded via ``importorskip`` inside the body, and every
    ``akgentic.catalog`` import is local, so this conftest — loaded for every
    run in the package — stays importable when the ``api`` extra is absent.
    """
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from akgentic.catalog.api._errors import add_exception_handlers
    from akgentic.catalog.api._settings import CatalogRouterSettings
    from akgentic.catalog.api.router import build_router, set_catalog
    from akgentic.catalog.catalog import Catalog
    from akgentic.catalog.repositories.yaml import YamlEntryRepository

    repo = YamlEntryRepository(tmp_path)
    catalog = Catalog(repo)

    app = FastAPI(title="Akgentic Catalog")
    app.include_router(
        build_router(CatalogRouterSettings(expose_generic_kind_crud=expose_generic_kind_crud))
    )
    set_catalog(catalog)
    add_exception_handlers(app)

    return TestClient(app), catalog


@pytest.fixture
def api_client(tmp_path: Path) -> tuple[Any, Catalog]:
    """Yield a ``(TestClient, Catalog)`` pair opting **in** to kind-generic CRUD.

    See :func:`_build_api_client` for the wiring and the function-scope constraint.
    """
    return _build_api_client(tmp_path, expose_generic_kind_crud=True)


@pytest.fixture
def api_client_kind_crud_hidden(tmp_path: Path) -> tuple[Any, Catalog]:
    """Same as ``api_client`` but with ``expose_generic_kind_crud=False``.

    See :func:`_build_api_client` for the wiring and the function-scope constraint.
    """
    return _build_api_client(tmp_path, expose_generic_kind_crud=False)


@pytest.fixture
def entries_collection() -> pymongo.collection.Collection:  # type: ignore[type-arg]
    """Provide a fresh mongomock-backed ``catalog_entries`` collection per test.

    Builds an in-memory ``mongomock.MongoClient`` on demand so tests that do
    not touch Mongo pay no import cost. Each test gets an isolated collection
    — no cross-test state. ``pymongo`` is an optional dep per the package
    ``pyproject.toml``; ``mongomock`` ships under the ``dev`` extra.
    """
    import mongomock

    client = mongomock.MongoClient()
    return client["test_catalog"]["catalog_entries"]


@pytest.fixture(scope="session")
def postgres_dsn() -> Iterator[str]:
    """Yield a Postgres DSN backed by one live database per session.

    Skips cleanly when:

    * ``nagra`` / ``psycopg`` are not installed (the ``[postgres]`` extra
      is absent).
    * No ``DB_CONN_STRING_PERSISTENCE`` env var is set AND
      ``testcontainers.postgres`` is unavailable or Docker itself is not
      running.

    Strips the ``+psycopg2`` / ``+psycopg`` driver suffix from the DSN
    so Nagra (which accepts only plain libpq DSNs) works uniformly
    across both source channels.

    Applies the schema once per session via ``init_db`` — every sub-suite
    consuming this fixture finds an empty-but-initialised
    ``catalog_entries`` table. Per-test isolation is the consumer's
    concern (see :func:`postgres_clean_dsn` below).
    """
    pytest.importorskip("nagra")
    pytest.importorskip("psycopg")

    env_dsn = os.environ.get("DB_CONN_STRING_PERSISTENCE")
    if env_dsn:
        dsn = _normalise_dsn(env_dsn)
        _apply_schema(dsn)
        yield dsn
        return

    pg_module = pytest.importorskip("testcontainers.postgres")
    try:
        container = pg_module.PostgresContainer("postgres:16-alpine")
        container.start()
    except Exception as exc:  # pragma: no cover — infra-dependent branch
        pytest.skip(f"Docker unavailable for testcontainers.postgres: {exc}")

    try:
        dsn = _normalise_dsn(container.get_connection_url())
        _apply_schema(dsn)
        yield dsn
    finally:
        container.stop()


def _normalise_dsn(dsn: str) -> str:
    """Strip SQLAlchemy-style driver tokens — Nagra wants a bare libpq DSN."""
    dsn = dsn.replace("postgresql+psycopg2://", "postgresql://")
    dsn = dsn.replace("postgresql+psycopg://", "postgresql://")
    return dsn


def _apply_schema(dsn: str) -> None:
    """Run ``init_db`` against ``dsn`` — idempotent, safe to re-run."""
    from akgentic.catalog.repositories.postgres import PostgresCatalogConfig, init_db

    init_db(PostgresCatalogConfig(connection_string=dsn))


@pytest.fixture
def postgres_clean_dsn(postgres_dsn: str) -> str:
    """Truncate ``catalog_entries`` before each test; return the shared DSN.

    Uses a direct psycopg connection (outside Nagra) so the truncate is a
    focused maintenance op rather than a transaction with broader intent.
    """
    import psycopg

    with psycopg.connect(postgres_dsn) as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE catalog_entries")
    return postgres_dsn
