"""Fixtures for Postgres (Nagra) repository tests.

Uses ``testcontainers[postgres]`` to spin up a single session-scoped
``postgres:16-alpine`` container. If ``nagra`` or ``testcontainers.postgres``
is not importable, the whole module (and therefore every Postgres test) is
skipped via ``pytest.importorskip`` — mirrors the Mongo test layout.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("nagra")
pytest.importorskip("testcontainers.postgres")

from testcontainers.postgres import PostgresContainer  # noqa: E402

from akgentic.catalog.repositories.postgres import init_db  # noqa: E402

if TYPE_CHECKING:
    pass


def _to_nagra_conn_string(sqlalchemy_url: str) -> str:
    """Convert the testcontainers SQLAlchemy-style URL to a libpq URL.

    ``testcontainers`` emits URLs like
    ``postgresql+psycopg2://user:pw@host:port/db``. Nagra's ``Transaction``
    wraps a psycopg / libpq connection, which accepts the standard
    ``postgresql://`` scheme without the driver suffix. Strip the driver so
    the URL is portable regardless of Nagra's current psycopg binding.
    """
    if "+" in sqlalchemy_url.split("://", 1)[0]:
        scheme, rest = sqlalchemy_url.split("://", 1)
        scheme = scheme.split("+", 1)[0]
        return f"{scheme}://{rest}"
    return sqlalchemy_url


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    """Start a single postgres:16-alpine container for the test session."""
    with PostgresContainer("postgres:16-alpine") as container:
        yield container


@pytest.fixture(scope="session")
def postgres_conn_string(postgres_container: PostgresContainer) -> str:
    """Nagra-compatible connection string derived from the session container."""
    raw_url = postgres_container.get_connection_url()
    return _to_nagra_conn_string(raw_url)


@pytest.fixture(scope="session")
def postgres_initialized(postgres_conn_string: str) -> str:
    """Run ``init_db`` exactly once against the session container."""
    init_db(postgres_conn_string)
    return postgres_conn_string


@pytest.fixture
def postgres_clean_tables(postgres_initialized: str) -> Iterator[str]:
    """Truncate the four catalog tables between tests."""
    from nagra import Transaction

    yield postgres_initialized
    with Transaction(postgres_initialized) as trn:
        trn.execute(
            "TRUNCATE template_entries, tool_entries, agent_entries, team_entries"
        )
