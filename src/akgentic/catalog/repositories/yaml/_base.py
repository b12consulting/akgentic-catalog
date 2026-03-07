"""Shared base for YAML-backed catalog repositories."""

import builtins
from pathlib import Path

import yaml
from pydantic import BaseModel

from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError

_list = builtins.list


class YamlRepositoryBase[T: BaseModel]:
    """Generic base providing YAML file loading, caching, duplicate detection, and CRUD.

    Subclasses must set ``_entry_type`` to the concrete Pydantic model class.
    """

    _entry_type: type[T]

    def __init__(self, catalog_dir: Path) -> None:
        self._catalog_dir = catalog_dir
        self._entries: _list[T] | None = None

    # ------------------------------------------------------------------
    # Loading & caching
    # ------------------------------------------------------------------

    def _load_all(self) -> _list[T]:
        """Scan catalog_dir for *.yaml files, validate entries, detect duplicates."""
        entries: _list[T] = []
        seen_ids: dict[str, Path] = {}

        if not self._catalog_dir.exists():
            return entries

        for yaml_path in sorted(self._catalog_dir.glob("*.yaml")):
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if raw is None:
                continue
            if not isinstance(raw, builtins.list):
                raw = [raw]
            for item in raw:
                entry = self._entry_type.model_validate(item)
                entry_id: str = getattr(entry, "id")
                if entry_id in seen_ids:
                    raise CatalogValidationError(
                        [
                            f"Duplicate id '{entry_id}' found in "
                            f"'{seen_ids[entry_id]}' and '{yaml_path}'"
                        ]
                    )
                seen_ids[entry_id] = yaml_path
                entries.append(entry)

        return entries

    def _ensure_loaded(self) -> _list[T]:
        """Return cached entries, loading from disk on first access."""
        if self._entries is None:
            self._entries = self._load_all()
        return self._entries

    def reload(self) -> None:
        """Force a re-scan from disk on next access."""
        self._entries = None

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, id: str) -> T | None:
        """Return entry by id, or None if not found."""
        for entry in self._ensure_loaded():
            if getattr(entry, "id") == id:
                return entry
        return None

    def list(self) -> _list[T]:
        """Return all cached entries."""
        return _list(self._ensure_loaded())

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def create(self, entry: T) -> str:
        """Persist a new entry to a YAML file and invalidate cache."""
        entry_id: str = getattr(entry, "id")
        file_path = self._catalog_dir / f"{entry_id}.yaml"
        self._catalog_dir.mkdir(parents=True, exist_ok=True)

        data = [entry.model_dump()]
        if file_path.exists():
            existing = yaml.safe_load(file_path.read_text(encoding="utf-8"))
            if existing is None:
                existing = []
            if not isinstance(existing, builtins.list):
                existing = [existing]
            existing.append(entry.model_dump())
            data = existing

        file_path.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        self._entries = None
        return entry_id

    def update(self, id: str, entry: T) -> None:
        """Find file containing id, update entry in place, invalidate cache."""
        for yaml_path in sorted(self._catalog_dir.glob("*.yaml")):
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if raw is None:
                continue
            if not isinstance(raw, builtins.list):
                raw = [raw]
            for i, item in enumerate(raw):
                if item.get("id") == id:
                    raw[i] = entry.model_dump()
                    yaml_path.write_text(
                        yaml.dump(
                            raw,
                            default_flow_style=False,
                            sort_keys=False,
                            allow_unicode=True,
                        ),
                        encoding="utf-8",
                    )
                    self._entries = None
                    return
        raise EntryNotFoundError(f"Entry with id '{id}' not found")

    def delete(self, id: str) -> None:
        """Find file containing id, remove entry, delete file if empty, invalidate cache."""
        for yaml_path in sorted(self._catalog_dir.glob("*.yaml")):
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if raw is None:
                continue
            if not isinstance(raw, builtins.list):
                raw = [raw]
            for i, item in enumerate(raw):
                if item.get("id") == id:
                    raw.pop(i)
                    if raw:
                        yaml_path.write_text(
                            yaml.dump(
                                raw,
                                default_flow_style=False,
                                sort_keys=False,
                                allow_unicode=True,
                            ),
                            encoding="utf-8",
                        )
                    else:
                        yaml_path.unlink()
                    self._entries = None
                    return
        raise EntryNotFoundError(f"Entry with id '{id}' not found")
