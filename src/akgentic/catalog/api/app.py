"""FastAPI application factory for the Akgentic v2 catalog API.

Provides ``create_v2_app()`` which assembles a FastAPI application serving
the unified ``/catalog`` router over a YAML- or MongoDB-backed
:class:`EntryRepository`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from fastapi import FastAPI

from akgentic.catalog.api._errors import add_exception_handlers
from akgentic.catalog.api.router import router as v2_router
from akgentic.catalog.api.router import set_catalog as set_v2_catalog
from akgentic.catalog.catalog import Catalog

if TYPE_CHECKING:
    from pathlib import Path

    from akgentic.catalog.repositories.base import EntryRepository
    from akgentic.catalog.repositories.mongo import MongoCatalogConfig

__all__ = ["create_v2_app"]

logger = logging.getLogger(__name__)


def create_v2_app(
    *,
    backend: Literal["yaml", "mongodb"] = "yaml",
    yaml_base_path: Path | None = None,
    mongo_config: MongoCatalogConfig | None = None,
) -> FastAPI:
    """Create a v2 FastAPI app serving the unified ``/catalog`` router.

    Args:
        backend: ``"yaml"`` for filesystem-backed storage or ``"mongodb"`` for
            MongoDB-backed storage.
        yaml_base_path: Root directory for YAML entries. Defaults to
            ``Path("./catalog")`` when ``backend="yaml"`` and this argument is
            ``None``. Created if absent.
        mongo_config: MongoDB connection + naming configuration. Required when
            ``backend="mongodb"``.

    Returns:
        A configured ``FastAPI`` app with the v2 router mounted and catalog
        exception handlers registered.

    Raises:
        ValueError: If the backend identifier is unknown or required arguments
            are missing.
    """
    repo = _build_v2_repository(
        backend=backend, yaml_base_path=yaml_base_path, mongo_config=mongo_config
    )
    catalog = Catalog(repository=repo)
    set_v2_catalog(catalog)

    app = FastAPI(title="Akgentic Catalog")
    app.include_router(v2_router)
    add_exception_handlers(app)

    logger.info("Created v2 Akgentic Catalog API with %s backend", backend)
    return app


def _build_v2_repository(
    *,
    backend: Literal["yaml", "mongodb"],
    yaml_base_path: Path | None,
    mongo_config: MongoCatalogConfig | None,
) -> EntryRepository:
    """Construct the concrete v2 ``EntryRepository`` for ``create_v2_app``."""
    if backend == "yaml":
        from pathlib import Path as _Path

        from akgentic.catalog.repositories.yaml import YamlEntryRepository

        base = yaml_base_path if yaml_base_path is not None else _Path("./catalog")
        base.mkdir(parents=True, exist_ok=True)
        return YamlEntryRepository(base)
    if backend == "mongodb":
        if mongo_config is None:
            msg = "mongo_config is required when backend='mongodb'"
            raise ValueError(msg)
        from akgentic.catalog.repositories.mongo import MongoEntryRepository

        client = mongo_config.create_client()
        collection = mongo_config.get_collection(client, mongo_config.catalog_entries_collection)
        return MongoEntryRepository(collection)
    msg = f"Unknown backend: {backend!r}. Must be 'yaml' or 'mongodb'."
    raise ValueError(msg)
