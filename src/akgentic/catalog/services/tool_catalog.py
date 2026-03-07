"""ToolCatalog service — domain logic over tool repository."""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.tool import ToolEntry
from akgentic.catalog.repositories.base import ToolCatalogRepository

if TYPE_CHECKING:
    from akgentic.catalog.models.queries import ToolQuery
    from akgentic.catalog.services.agent_catalog import AgentCatalog

__all__ = ["ToolCatalog"]

_list = builtins.list


class ToolCatalog:
    """Service layer for tool catalog entries."""

    def __init__(self, repository: ToolCatalogRepository) -> None:
        self.repository = repository
        self._agent_catalog: AgentCatalog | None = None

    @property
    def agent_catalog(self) -> AgentCatalog | None:
        """Optional downstream agent catalog for delete protection."""
        return self._agent_catalog

    @agent_catalog.setter
    def agent_catalog(self, value: AgentCatalog | None) -> None:
        self._agent_catalog = value

    def validate_create(self, entry: ToolEntry) -> _list[str]:
        """Check for duplicate id. Returns list of error strings."""
        errors: _list[str] = []
        if self.repository.get(entry.id) is not None:
            errors.append(f"Tool id '{entry.id}' already exists")
        return errors

    def create(self, entry: ToolEntry) -> str:
        """Persist a new tool entry. Raises CatalogValidationError on duplicate id."""
        errors = self.validate_create(entry)
        if errors:
            raise CatalogValidationError(errors)
        return self.repository.create(entry)

    def get(self, id: str) -> ToolEntry | None:
        """Retrieve a tool entry by id."""
        return self.repository.get(id)

    def list(self) -> _list[ToolEntry]:
        """List all tool entries."""
        return self.repository.list()

    def search(self, query: ToolQuery) -> _list[ToolEntry]:
        """Search tool entries by query."""
        return self.repository.search(query)

    def update(self, id: str, entry: ToolEntry) -> None:
        """Update an existing tool entry. Raises EntryNotFoundError if missing."""
        if self.repository.get(id) is None:
            raise EntryNotFoundError(f"Tool id '{id}' not found")
        if entry.id != id:
            raise CatalogValidationError(
                [f"Entry id '{entry.id}' does not match update target '{id}'"]
            )
        self.repository.update(id, entry)

    def validate_delete(self, id: str) -> _list[str]:
        """Check existence and downstream references before delete."""
        errors: _list[str] = []
        if self.repository.get(id) is None:
            errors.append(f"Tool id '{id}' not found")
            return errors
        if self._agent_catalog is not None:
            for agent in self._agent_catalog.repository.list():
                if id in agent.tool_ids:
                    errors.append(
                        f"Agent '{agent.id}' references tool '{id}'"
                        f" in tool_ids — cannot delete"
                    )
        return errors

    def delete(self, id: str) -> None:
        """Delete a tool entry. Raises errors if not found or referenced downstream."""
        errors = self.validate_delete(id)
        if errors:
            if "not found" in errors[0]:
                raise EntryNotFoundError(errors[0])
            raise CatalogValidationError(errors)
        self.repository.delete(id)
