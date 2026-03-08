"""TeamCatalog service — domain logic over team repository with cross-validation."""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.team import TeamMemberSpec, TeamSpec, agent_in_members
from akgentic.catalog.repositories.base import TeamCatalogRepository
from akgentic.catalog.services.agent_catalog import AgentCatalog
from akgentic.core.utils.deserializer import import_class

if TYPE_CHECKING:
    from akgentic.catalog.models.queries import TeamQuery

__all__ = ["TeamCatalog"]

_list = builtins.list  # Alias: the service's list() method shadows the built-in


class TeamCatalog:
    """Service layer for team catalog entries with cross-validation."""

    def __init__(
        self,
        repository: TeamCatalogRepository,
        agent_catalog: AgentCatalog,
    ) -> None:
        """Initialize with repository and agent catalog for cross-validation.

        Args:
            repository: Storage backend for team entries.
            agent_catalog: For validating member and profile agent references.
        """
        self.repository = repository
        self._agent_catalog = agent_catalog

    @staticmethod
    def _collect_agent_ids(members: _list[TeamMemberSpec]) -> _list[str]:
        """Recursively collect all agent_ids from the members tree."""
        ids: _list[str] = []
        for m in members:
            ids.append(m.agent_id)
            if m.members:
                ids.extend(TeamCatalog._collect_agent_ids(m.members))
        return ids

    def _validate_entry(
        self,
        entry: TeamSpec,
        *,
        exclude_id: str | None = None,
    ) -> _list[str]:
        """Cross-validate a team entry. Shared by create and update.

        Args:
            entry: The team spec to validate.
            exclude_id: If set, skip the duplicate-id check for this id (used by update).
        """
        errors: _list[str] = []

        # AC5: Duplicate ID rejection (skipped when updating own entry)
        if exclude_id != entry.id and self.repository.get(entry.id) is not None:
            errors.append(f"Team id '{entry.id}' already exists")

        # AC2: entry_point must be an agent_id in the members tree
        if not agent_in_members(entry.entry_point, entry.members):
            errors.append(
                f"Entry point '{entry.entry_point}' not found in members tree"
            )

        # AC1/AC15: Every agent_id in members tree must exist in AgentCatalog
        for agent_id in TeamCatalog._collect_agent_ids(entry.members):
            if self._agent_catalog.get(agent_id) is None:
                errors.append(
                    f"Agent '{agent_id}' not found in AgentCatalog"
                )

        # AC3: Every agent_id in profiles must exist in AgentCatalog
        for agent_id in entry.profiles:
            if self._agent_catalog.get(agent_id) is None:
                errors.append(
                    f"Profile agent '{agent_id}' not found in AgentCatalog"
                )

        # AC4: Every message_type must be resolvable
        for mt in entry.message_types:
            try:
                import_class(mt)
            except (ImportError, AttributeError, ValueError) as e:
                errors.append(f"Cannot resolve message_type '{mt}': {e}")

        return errors

    def validate_create(self, entry: TeamSpec) -> _list[str]:
        """Validate a team entry for creation. Returns list of error strings."""
        return self._validate_entry(entry)

    def create(self, entry: TeamSpec) -> str:
        """Persist a new team entry. Raises CatalogValidationError on validation failure."""
        errors = self.validate_create(entry)
        if errors:
            raise CatalogValidationError(errors)
        return self.repository.create(entry)

    def get(self, id: str) -> TeamSpec | None:
        """Retrieve a team entry by id."""
        return self.repository.get(id)

    def list(self) -> _list[TeamSpec]:
        """List all team entries."""
        return self.repository.list()

    def search(self, query: TeamQuery) -> _list[TeamSpec]:
        """Search team entries by query."""
        return self.repository.search(query)

    def update(self, id: str, entry: TeamSpec) -> None:
        """Update an existing team entry. Raises EntryNotFoundError if missing."""
        if self.repository.get(id) is None:
            raise EntryNotFoundError(f"Team id '{id}' not found")
        if entry.id != id:
            raise CatalogValidationError(
                [f"Entry id '{entry.id}' does not match update target '{id}'"]
            )
        errors = self._validate_entry(entry, exclude_id=id)
        if errors:
            raise CatalogValidationError(errors)
        self.repository.update(id, entry)

    def validate_delete(self, id: str) -> _list[str]:
        """Check existence before delete. No downstream refs for teams (v1: D9)."""
        errors: _list[str] = []
        if self.repository.get(id) is None:
            errors.append(f"Team id '{id}' not found")
        return errors

    def delete(self, id: str) -> None:
        """Delete a team entry. Raises EntryNotFoundError if not found."""
        errors = self.validate_delete(id)
        if errors:
            raise EntryNotFoundError(errors[0])
        self.repository.delete(id)
