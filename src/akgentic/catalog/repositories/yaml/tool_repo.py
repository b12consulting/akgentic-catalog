"""YAML-backed repository for tool catalog entries."""

import builtins
from pathlib import Path

from akgentic.catalog.models.queries import ToolQuery
from akgentic.catalog.models.tool import ToolEntry
from akgentic.catalog.repositories.base import ToolCatalogRepository
from akgentic.catalog.repositories.yaml._base import YamlRepositoryBase

_list = builtins.list


class YamlToolCatalogRepository(ToolCatalogRepository, YamlRepositoryBase[ToolEntry]):
    """YAML directory-backed tool catalog repository."""

    _entry_type = ToolEntry

    def __init__(self, catalog_dir: Path) -> None:
        YamlRepositoryBase.__init__(self, catalog_dir)

    def create(self, tool_entry: ToolEntry) -> str:
        return YamlRepositoryBase.create(self, tool_entry)

    def get(self, id: str) -> ToolEntry | None:
        return YamlRepositoryBase.get(self, id)

    def list(self) -> _list[ToolEntry]:
        return YamlRepositoryBase.list(self)

    def update(self, id: str, tool_entry: ToolEntry) -> None:
        YamlRepositoryBase.update(self, id, tool_entry)

    def delete(self, id: str) -> None:
        YamlRepositoryBase.delete(self, id)

    def search(self, query: ToolQuery) -> _list[ToolEntry]:
        """Filter tools: AND all non-None fields."""
        results: _list[ToolEntry] = []
        for entry in self._ensure_loaded():
            if query.id is not None and entry.id != query.id:
                continue
            if query.tool_class is not None and entry.tool_class != query.tool_class:
                continue
            if query.name is not None and query.name.lower() not in entry.tool.name.lower():
                continue
            if (
                query.description is not None
                and query.description.lower() not in entry.tool.description.lower()
            ):
                continue
            results.append(entry)
        return results
