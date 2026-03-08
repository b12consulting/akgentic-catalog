"""YAML-backed repository for agent catalog entries."""

import builtins
from pathlib import Path

from akgentic.catalog.models.agent import AgentEntry
from akgentic.catalog.models.queries import AgentQuery
from akgentic.catalog.repositories.base import AgentCatalogRepository
from akgentic.catalog.repositories.yaml._base import YamlRepositoryBase

_list = builtins.list  # Alias: the repository's list() method shadows the built-in


class YamlAgentCatalogRepository(AgentCatalogRepository, YamlRepositoryBase[AgentEntry]):
    """YAML directory-backed agent catalog repository."""

    _entry_type = AgentEntry

    def __init__(self, catalog_dir: Path) -> None:
        """Initialize with the directory containing agent YAML files.

        Args:
            catalog_dir: Path to the directory of agent catalog YAML files.
        """
        YamlRepositoryBase.__init__(self, catalog_dir)

    def create(self, agent_entry: AgentEntry) -> str:
        return YamlRepositoryBase.create(self, agent_entry)

    def get(self, id: str) -> AgentEntry | None:
        return YamlRepositoryBase.get(self, id)

    def list(self) -> _list[AgentEntry]:
        return YamlRepositoryBase.list(self)

    def update(self, id: str, agent_entry: AgentEntry) -> None:
        YamlRepositoryBase.update(self, id, agent_entry)

    def delete(self, id: str) -> None:
        YamlRepositoryBase.delete(self, id)

    def search(self, query: AgentQuery) -> _list[AgentEntry]:
        """Filter agents: AND all non-None fields."""
        results: _list[AgentEntry] = []
        for entry in self._ensure_loaded():
            if query.id is not None and entry.id != query.id:
                continue
            if query.role is not None and entry.card.role != query.role:
                continue
            if query.skills is not None and not (
                set(query.skills) & set(entry.card.skills)
            ):
                continue
            if (
                query.description is not None
                and query.description.lower() not in entry.card.description.lower()
            ):
                continue
            results.append(entry)
        return results
