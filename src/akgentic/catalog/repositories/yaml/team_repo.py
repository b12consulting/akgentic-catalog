"""YAML-backed repository for team catalog entries."""

import builtins
from pathlib import Path

from akgentic.catalog.models.queries import TeamQuery
from akgentic.catalog.models.team import TeamMemberSpec, TeamSpec
from akgentic.catalog.repositories.base import TeamCatalogRepository
from akgentic.catalog.repositories.yaml._base import YamlRepositoryBase

_list = builtins.list  # Alias: the repository's list() method shadows the built-in


def _agent_in_members(agent_id: str, members: _list[TeamMemberSpec]) -> bool:
    """Recursively check if agent_id appears anywhere in the members tree."""
    for member in members:
        if member.agent_id == agent_id:
            return True
        if member.members and _agent_in_members(agent_id, member.members):
            return True
    return False


class YamlTeamCatalogRepository(TeamCatalogRepository, YamlRepositoryBase[TeamSpec]):
    """YAML directory-backed team catalog repository."""

    _entry_type = TeamSpec

    def __init__(self, catalog_dir: Path) -> None:
        """Initialize with the directory containing team YAML files.

        Args:
            catalog_dir: Path to the directory of team catalog YAML files.
        """
        YamlRepositoryBase.__init__(self, catalog_dir)

    def create(self, team_spec: TeamSpec) -> str:
        return YamlRepositoryBase.create(self, team_spec)

    def get(self, id: str) -> TeamSpec | None:
        return YamlRepositoryBase.get(self, id)

    def list(self) -> _list[TeamSpec]:
        return YamlRepositoryBase.list(self)

    def update(self, id: str, team_spec: TeamSpec) -> None:
        YamlRepositoryBase.update(self, id, team_spec)

    def delete(self, id: str) -> None:
        YamlRepositoryBase.delete(self, id)

    def search(self, query: TeamQuery) -> _list[TeamSpec]:
        """Filter teams: AND all non-None fields."""
        results: _list[TeamSpec] = []
        for entry in self._ensure_loaded():
            if query.id is not None and entry.id != query.id:
                continue
            if query.name is not None and query.name.lower() not in entry.name.lower():
                continue
            if (
                query.description is not None
                and query.description.lower() not in entry.description.lower()
            ):
                continue
            if query.agent_id is not None and not _agent_in_members(
                query.agent_id, entry.members
            ):
                continue
            results.append(entry)
        return results
