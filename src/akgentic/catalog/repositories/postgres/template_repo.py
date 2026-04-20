"""Nagra-backed template repository (stub — CRUD arrives in story 15.2)."""

from __future__ import annotations

import builtins
import logging
from typing import TYPE_CHECKING

from akgentic.catalog.models.template import TemplateEntry
from akgentic.catalog.repositories.base import TemplateCatalogRepository

if TYPE_CHECKING:
    from akgentic.catalog.models.queries import TemplateQuery

logger = logging.getLogger(__name__)

_list = builtins.list  # repository's list() method shadows the built-in


class NagraTemplateCatalogRepository(TemplateCatalogRepository):
    """Nagra/Postgres-backed template catalog repository.

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
        logger.info("NagraTemplateCatalogRepository initialised")

    def create(self, template_entry: TemplateEntry) -> str:
        """Scaffolded — real CRUD arrives in story 15.2."""
        raise NotImplementedError("Implemented in story 15.2/15.3")

    def get(self, id: str) -> TemplateEntry | None:
        """Scaffolded — real CRUD arrives in story 15.2."""
        raise NotImplementedError("Implemented in story 15.2/15.3")

    def list(self) -> _list[TemplateEntry]:
        """Scaffolded — real CRUD arrives in story 15.2."""
        raise NotImplementedError("Implemented in story 15.2/15.3")

    def search(self, query: TemplateQuery) -> _list[TemplateEntry]:
        """Scaffolded — real CRUD arrives in story 15.2."""
        raise NotImplementedError("Implemented in story 15.2/15.3")

    def update(self, id: str, template_entry: TemplateEntry) -> None:
        """Scaffolded — real CRUD arrives in story 15.2."""
        raise NotImplementedError("Implemented in story 15.2/15.3")

    def delete(self, id: str) -> None:
        """Scaffolded — real CRUD arrives in story 15.2."""
        raise NotImplementedError("Implemented in story 15.2/15.3")
