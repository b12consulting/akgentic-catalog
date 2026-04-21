"""MongoDB backend connection configuration for the v2 catalog.

Re-exports :class:`MongoCatalogConfig` — the single v2-alive configuration
model used by :mod:`akgentic.catalog.repositories.mongo_entry_repo` and the
v2 API/CLI wiring. Importing this package requires ``pymongo`` to be
installed.
"""

from __future__ import annotations

try:
    import pymongo  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only when pymongo missing
    raise ImportError(
        "pymongo is required to use the MongoDB catalog backend. "
        "Install with: pip install akgentic-catalog[mongo]"
    ) from exc

from akgentic.catalog.repositories.mongo._config import MongoCatalogConfig

__all__ = ["MongoCatalogConfig"]
