"""TemplateCatalog service — domain logic over template repository."""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

from akgentic.agent.config import AgentConfig
from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.template import TemplateEntry
from akgentic.catalog.refs import _is_catalog_ref, _resolve_ref
from akgentic.catalog.repositories.base import TemplateCatalogRepository
from akgentic.catalog.services._protocols import _AgentCatalogProtocol

if TYPE_CHECKING:
    from akgentic.catalog.models.queries import TemplateQuery

__all__ = ["TemplateCatalog"]

_list = builtins.list


class TemplateCatalog:
    """Service layer for template catalog entries."""

    def __init__(self, repository: TemplateCatalogRepository) -> None:
        self.repository = repository
        self._agent_catalog: _AgentCatalogProtocol | None = None

    @property
    def agent_catalog(self) -> _AgentCatalogProtocol | None:
        """Optional downstream agent catalog for delete protection."""
        return self._agent_catalog

    @agent_catalog.setter
    def agent_catalog(self, value: _AgentCatalogProtocol | None) -> None:
        self._agent_catalog = value

    def validate_create(self, entry: TemplateEntry) -> _list[str]:
        """Check for duplicate id. Returns list of error strings."""
        errors: _list[str] = []
        if self.repository.get(entry.id) is not None:
            errors.append(f"Template id '{entry.id}' already exists")
        return errors

    def create(self, entry: TemplateEntry) -> str:
        """Persist a new template entry. Raises CatalogValidationError on duplicate id."""
        errors = self.validate_create(entry)
        if errors:
            raise CatalogValidationError(errors)
        return self.repository.create(entry)

    def get(self, id: str) -> TemplateEntry | None:
        """Retrieve a template entry by id."""
        return self.repository.get(id)

    def list(self) -> _list[TemplateEntry]:
        """List all template entries."""
        return self.repository.list()

    def search(self, query: TemplateQuery) -> _list[TemplateEntry]:
        """Search template entries by query."""
        return self.repository.search(query)

    def update(self, id: str, entry: TemplateEntry) -> None:
        """Update an existing template entry. Raises EntryNotFoundError if missing."""
        if self.repository.get(id) is None:
            raise EntryNotFoundError(f"Template id '{id}' not found")
        if entry.id != id:
            raise CatalogValidationError(
                [f"Entry id '{entry.id}' does not match update target '{id}'"]
            )
        self.repository.update(id, entry)

    def validate_delete(self, id: str) -> _list[str]:
        """Check existence and downstream references before delete."""
        errors: _list[str] = []
        if self.repository.get(id) is None:
            errors.append(f"Template id '{id}' not found")
            return errors
        if self._agent_catalog is not None:
            for agent in self._agent_catalog.repository.list():
                config = agent.card.config
                if isinstance(config, AgentConfig):
                    if _is_catalog_ref(config.prompt.template):
                        ref_id = _resolve_ref(config.prompt.template)
                        if ref_id == id:
                            errors.append(
                                f"Agent '{agent.id}' references template '@{id}'"
                                f" — cannot delete"
                            )
        return errors

    def delete(self, id: str) -> None:
        """Delete a template entry. Raises errors if not found or referenced downstream."""
        errors = self.validate_delete(id)
        if errors:
            if "not found" in errors[0]:
                raise EntryNotFoundError(errors[0])
            raise CatalogValidationError(errors)
        self.repository.delete(id)
