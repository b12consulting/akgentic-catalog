"""YAML-backed repository for template catalog entries."""

import builtins
from pathlib import Path

from akgentic.catalog.models.queries import TemplateQuery
from akgentic.catalog.models.template import TemplateEntry
from akgentic.catalog.repositories.base import TemplateCatalogRepository
from akgentic.catalog.repositories.yaml._base import YamlRepositoryBase

_list = builtins.list  # Alias: the repository's list() method shadows the built-in


class YamlTemplateCatalogRepository(TemplateCatalogRepository, YamlRepositoryBase[TemplateEntry]):
    """YAML directory-backed template catalog repository."""

    _entry_type = TemplateEntry

    def __init__(self, catalog_dir: Path) -> None:
        """Initialize with the directory containing template YAML files.

        Args:
            catalog_dir: Path to the directory of template catalog YAML files.
        """
        YamlRepositoryBase.__init__(self, catalog_dir)

    def create(self, template_entry: TemplateEntry) -> str:
        return YamlRepositoryBase.create(self, template_entry)

    def get(self, id: str) -> TemplateEntry | None:
        return YamlRepositoryBase.get(self, id)

    def list(self) -> _list[TemplateEntry]:
        return YamlRepositoryBase.list(self)

    def update(self, id: str, template_entry: TemplateEntry) -> None:
        YamlRepositoryBase.update(self, id, template_entry)

    def delete(self, id: str) -> None:
        YamlRepositoryBase.delete(self, id)

    def search(self, query: TemplateQuery) -> _list[TemplateEntry]:
        """Filter templates: AND all non-None fields."""
        results: _list[TemplateEntry] = []
        for entry in self._ensure_loaded():
            if query.id is not None and entry.id != query.id:
                continue
            if query.placeholder is not None and query.placeholder not in entry.placeholders:
                continue
            results.append(entry)
        return results
