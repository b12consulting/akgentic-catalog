"""Abstract base classes for catalog repositories."""

import builtins
from abc import ABC, abstractmethod

from akgentic.catalog.models.agent import AgentEntry
from akgentic.catalog.models.queries import AgentQuery, TeamQuery, TemplateQuery, ToolQuery
from akgentic.catalog.models.team import TeamSpec
from akgentic.catalog.models.template import TemplateEntry
from akgentic.catalog.models.tool import ToolEntry

__all__ = [
    "AgentCatalogRepository",
    "TeamCatalogRepository",
    "TemplateCatalogRepository",
    "ToolCatalogRepository",
]

_list = builtins.list  # Alias: the repository's list() method shadows the built-in


class TemplateCatalogRepository(ABC):
    """Abstract repository for template catalog entries."""

    @abstractmethod
    def create(self, template_entry: TemplateEntry) -> str:
        """Persist a new template entry.

        Args:
            template_entry: The template entry to create.

        Returns:
            The id of the created entry.

        Raises:
            CatalogValidationError: If an entry with the same id already exists.
        """

    @abstractmethod
    def get(self, id: str) -> TemplateEntry | None:
        """Retrieve a template entry by id.

        Args:
            id: The template entry id.

        Returns:
            The template entry, or None if not found.
        """

    @abstractmethod
    def list(self) -> _list[TemplateEntry]:
        """Return all template entries."""

    @abstractmethod
    def search(self, query: TemplateQuery) -> _list[TemplateEntry]:
        """Filter template entries matching query criteria.

        Args:
            query: Query with optional filter fields (AND semantics).

        Returns:
            Matching template entries.
        """

    @abstractmethod
    def update(self, id: str, template_entry: TemplateEntry) -> None:
        """Update an existing template entry.

        Args:
            id: The id of the entry to update.
            template_entry: The new entry data.

        Raises:
            EntryNotFoundError: If no entry with the given id exists.
        """

    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete a template entry by id.

        Args:
            id: The id of the entry to delete.

        Raises:
            EntryNotFoundError: If no entry with the given id exists.
        """


class ToolCatalogRepository(ABC):
    """Abstract repository for tool catalog entries."""

    @abstractmethod
    def create(self, tool_entry: ToolEntry) -> str:
        """Persist a new tool entry.

        Args:
            tool_entry: The tool entry to create.

        Returns:
            The id of the created entry.

        Raises:
            CatalogValidationError: If an entry with the same id already exists.
        """

    @abstractmethod
    def get(self, id: str) -> ToolEntry | None:
        """Retrieve a tool entry by id.

        Args:
            id: The tool entry id.

        Returns:
            The tool entry, or None if not found.
        """

    @abstractmethod
    def list(self) -> _list[ToolEntry]:
        """Return all tool entries."""

    @abstractmethod
    def search(self, query: ToolQuery) -> _list[ToolEntry]:
        """Filter tool entries matching query criteria.

        Args:
            query: Query with optional filter fields (AND semantics).

        Returns:
            Matching tool entries.
        """

    @abstractmethod
    def update(self, id: str, tool_entry: ToolEntry) -> None:
        """Update an existing tool entry.

        Args:
            id: The id of the entry to update.
            tool_entry: The new entry data.

        Raises:
            EntryNotFoundError: If no entry with the given id exists.
        """

    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete a tool entry by id.

        Args:
            id: The id of the entry to delete.

        Raises:
            EntryNotFoundError: If no entry with the given id exists.
        """


class AgentCatalogRepository(ABC):
    """Abstract repository for agent catalog entries."""

    @abstractmethod
    def create(self, agent_entry: AgentEntry) -> str:
        """Persist a new agent entry.

        Args:
            agent_entry: The agent entry to create.

        Returns:
            The id of the created entry.

        Raises:
            CatalogValidationError: If an entry with the same id already exists.
        """

    @abstractmethod
    def get(self, id: str) -> AgentEntry | None:
        """Retrieve an agent entry by id.

        Args:
            id: The agent entry id.

        Returns:
            The agent entry, or None if not found.
        """

    @abstractmethod
    def list(self) -> _list[AgentEntry]:
        """Return all agent entries."""

    @abstractmethod
    def search(self, query: AgentQuery) -> _list[AgentEntry]:
        """Filter agent entries matching query criteria.

        Args:
            query: Query with optional filter fields (AND semantics).

        Returns:
            Matching agent entries.
        """

    @abstractmethod
    def update(self, id: str, agent_entry: AgentEntry) -> None:
        """Update an existing agent entry.

        Args:
            id: The id of the entry to update.
            agent_entry: The new entry data.

        Raises:
            EntryNotFoundError: If no entry with the given id exists.
        """

    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete an agent entry by id.

        Args:
            id: The id of the entry to delete.

        Raises:
            EntryNotFoundError: If no entry with the given id exists.
        """


class TeamCatalogRepository(ABC):
    """Abstract repository for team catalog entries."""

    @abstractmethod
    def create(self, team_spec: TeamSpec) -> str:
        """Persist a new team spec.

        Args:
            team_spec: The team spec to create.

        Returns:
            The id of the created entry.

        Raises:
            CatalogValidationError: If an entry with the same id already exists.
        """

    @abstractmethod
    def get(self, id: str) -> TeamSpec | None:
        """Retrieve a team spec by id.

        Args:
            id: The team spec id.

        Returns:
            The team spec, or None if not found.
        """

    @abstractmethod
    def list(self) -> _list[TeamSpec]:
        """Return all team specs."""

    @abstractmethod
    def search(self, query: TeamQuery) -> _list[TeamSpec]:
        """Filter team specs matching query criteria.

        Args:
            query: Query with optional filter fields (AND semantics).

        Returns:
            Matching team specs.
        """

    @abstractmethod
    def update(self, id: str, team_spec: TeamSpec) -> None:
        """Update an existing team spec.

        Args:
            id: The id of the entry to update.
            team_spec: The new entry data.

        Raises:
            EntryNotFoundError: If no entry with the given id exists.
        """

    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete a team spec by id.

        Args:
            id: The id of the entry to delete.

        Raises:
            EntryNotFoundError: If no entry with the given id exists.
        """
