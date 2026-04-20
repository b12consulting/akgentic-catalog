"""Nagra-backed tool repository (stub — CRUD arrives in story 15.2)."""

from __future__ import annotations

import builtins
import logging
from typing import TYPE_CHECKING

from akgentic.catalog.models.tool import ToolEntry
from akgentic.catalog.repositories.base import ToolCatalogRepository

if TYPE_CHECKING:
    from akgentic.catalog.models.queries import ToolQuery

logger = logging.getLogger(__name__)

_list = builtins.list  # repository's list() method shadows the built-in


class NagraToolCatalogRepository(ToolCatalogRepository):
    """Nagra/Postgres-backed tool catalog repository.

    Scaffolded in story 15.1 — CRUD methods raise ``NotImplementedError`` and
    land in story 15.2. The constructor calls :func:`_ensure_schema_loaded`
    so instantiation is always safe once ``nagra`` is available.
    """

    def __init__(self, conn_string: str) -> None:
        """Initialise with a Nagra connection string and load the shared schema.

        Args:
            conn_string: Nagra-compatible Postgres connection string.
        """
        from akgentic.catalog.repositories.postgres import _ensure_schema_loaded

        _ensure_schema_loaded()
        self._conn_string = conn_string
        logger.info("NagraToolCatalogRepository initialised")

    def create(self, tool_entry: ToolEntry) -> str:
        """Scaffolded — real CRUD arrives in story 15.2."""
        raise NotImplementedError("Implemented in story 15.2/15.3")

    def get(self, id: str) -> ToolEntry | None:
        """Scaffolded — real CRUD arrives in story 15.2."""
        raise NotImplementedError("Implemented in story 15.2/15.3")

    def list(self) -> _list[ToolEntry]:
        """Scaffolded — real CRUD arrives in story 15.2."""
        raise NotImplementedError("Implemented in story 15.2/15.3")

    def search(self, query: ToolQuery) -> _list[ToolEntry]:
        """Scaffolded — real CRUD arrives in story 15.2."""
        raise NotImplementedError("Implemented in story 15.2/15.3")

    def update(self, id: str, tool_entry: ToolEntry) -> None:
        """Scaffolded — real CRUD arrives in story 15.2."""
        raise NotImplementedError("Implemented in story 15.2/15.3")

    def delete(self, id: str) -> None:
        """Scaffolded — real CRUD arrives in story 15.2."""
        raise NotImplementedError("Implemented in story 15.2/15.3")
