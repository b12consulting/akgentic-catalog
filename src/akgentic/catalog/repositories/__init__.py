"""Public API surface for catalog v2 repositories.

Re-exports the structural ``EntryRepository`` protocol and the concrete
YAML-backed implementation. The MongoDB-backed implementation is conditionally
re-exported when ``pymongo`` is installed.
"""

from __future__ import annotations

from akgentic.catalog.repositories.base import EntryRepository
from akgentic.catalog.repositories.yaml_entry_repo import YamlEntryRepository

__all__ = [
    "EntryRepository",
    "YamlEntryRepository",
]

try:
    from akgentic.catalog.repositories.mongo._config import MongoCatalogConfig
    from akgentic.catalog.repositories.mongo_entry_repo import MongoEntryRepository

    __all__ += [
        "MongoCatalogConfig",
        "MongoEntryRepository",
    ]
except ImportError:
    pass
