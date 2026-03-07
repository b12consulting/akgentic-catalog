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

_list = builtins.list


class TemplateCatalogRepository(ABC):
    """Abstract repository for template catalog entries."""

    @abstractmethod
    def create(self, template_entry: TemplateEntry) -> str: ...

    @abstractmethod
    def get(self, id: str) -> TemplateEntry | None: ...

    @abstractmethod
    def list(self) -> _list[TemplateEntry]: ...

    @abstractmethod
    def search(self, query: TemplateQuery) -> _list[TemplateEntry]: ...

    @abstractmethod
    def update(self, id: str, template_entry: TemplateEntry) -> None: ...

    @abstractmethod
    def delete(self, id: str) -> None: ...


class ToolCatalogRepository(ABC):
    """Abstract repository for tool catalog entries."""

    @abstractmethod
    def create(self, tool_entry: ToolEntry) -> str: ...

    @abstractmethod
    def get(self, id: str) -> ToolEntry | None: ...

    @abstractmethod
    def list(self) -> _list[ToolEntry]: ...

    @abstractmethod
    def search(self, query: ToolQuery) -> _list[ToolEntry]: ...

    @abstractmethod
    def update(self, id: str, tool_entry: ToolEntry) -> None: ...

    @abstractmethod
    def delete(self, id: str) -> None: ...


class AgentCatalogRepository(ABC):
    """Abstract repository for agent catalog entries."""

    @abstractmethod
    def create(self, agent_entry: AgentEntry) -> str: ...

    @abstractmethod
    def get(self, id: str) -> AgentEntry | None: ...

    @abstractmethod
    def list(self) -> _list[AgentEntry]: ...

    @abstractmethod
    def search(self, query: AgentQuery) -> _list[AgentEntry]: ...

    @abstractmethod
    def update(self, id: str, agent_entry: AgentEntry) -> None: ...

    @abstractmethod
    def delete(self, id: str) -> None: ...


class TeamCatalogRepository(ABC):
    """Abstract repository for team catalog entries."""

    @abstractmethod
    def create(self, team_spec: TeamSpec) -> str: ...

    @abstractmethod
    def get(self, id: str) -> TeamSpec | None: ...

    @abstractmethod
    def list(self) -> _list[TeamSpec]: ...

    @abstractmethod
    def search(self, query: TeamQuery) -> _list[TeamSpec]: ...

    @abstractmethod
    def update(self, id: str, team_spec: TeamSpec) -> None: ...

    @abstractmethod
    def delete(self, id: str) -> None: ...
