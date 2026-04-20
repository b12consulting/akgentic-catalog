"""FastAPI application factory for the Akgentic catalog API.

Provides ``create_app()`` which assembles a fully wired FastAPI application
with configurable storage backend (YAML files or MongoDB).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from fastapi import FastAPI

from akgentic.catalog.api._errors import add_exception_handlers
from akgentic.catalog.api.agent_router import router as agent_router
from akgentic.catalog.api.agent_router import set_catalog as set_agent_catalog
from akgentic.catalog.api.router import router as v2_router
from akgentic.catalog.api.router import set_catalog as set_v2_catalog
from akgentic.catalog.api.team_router import router as team_router
from akgentic.catalog.api.team_router import set_catalog as set_team_catalog
from akgentic.catalog.api.template_router import router as template_router
from akgentic.catalog.api.template_router import set_catalog as set_template_catalog
from akgentic.catalog.api.tool_router import router as tool_router
from akgentic.catalog.api.tool_router import set_catalog as set_tool_catalog
from akgentic.catalog.catalog import Catalog
from akgentic.catalog.services.agent_catalog import AgentCatalog
from akgentic.catalog.services.team_catalog import TeamCatalog
from akgentic.catalog.services.template_catalog import TemplateCatalog
from akgentic.catalog.services.tool_catalog import ToolCatalog

if TYPE_CHECKING:
    from pathlib import Path

    from akgentic.catalog.repositories.base import EntryRepository
    from akgentic.catalog.repositories.mongo._config import MongoCatalogConfig

__all__ = ["create_app", "create_v2_app"]

_V2_ENTRIES_COLLECTION = "catalog_entries"

logger = logging.getLogger(__name__)


def create_app(
    *,
    backend: Literal["yaml", "mongodb"] = "yaml",
    yaml_base_path: Path | None = None,
    mongo_config: MongoCatalogConfig | None = None,
) -> FastAPI:
    """Create a fully wired FastAPI application for the catalog API.

    Args:
        backend: Storage backend to use — ``"yaml"`` for file-based or
            ``"mongodb"`` for MongoDB.
        yaml_base_path: Root directory for YAML catalog files. Required when
            ``backend="yaml"``. Subdirectories are created if absent.
        mongo_config: MongoDB connection configuration. Required when
            ``backend="mongodb"``.

    Returns:
        A configured FastAPI application with all routers and exception
        handlers registered.

    Raises:
        ValueError: If required configuration for the chosen backend is missing.
    """
    if backend == "yaml":
        if yaml_base_path is None:
            msg = "yaml_base_path is required when backend='yaml'"
            raise ValueError(msg)
        template_catalog, tool_catalog, agent_catalog, team_catalog = _wire_yaml_backend(
            yaml_base_path
        )
    elif backend == "mongodb":
        if mongo_config is None:
            msg = "mongo_config is required when backend='mongodb'"
            raise ValueError(msg)
        template_catalog, tool_catalog, agent_catalog, team_catalog = _wire_mongodb_backend(
            mongo_config
        )
    else:
        msg = f"Unknown backend: {backend!r}. Must be 'yaml' or 'mongodb'."
        raise ValueError(msg)

    # Wire downstream back-references for delete protection
    template_catalog.agent_catalog = agent_catalog
    tool_catalog.agent_catalog = agent_catalog
    agent_catalog.team_catalog = team_catalog

    # Inject catalogs into router modules
    set_template_catalog(template_catalog)
    set_tool_catalog(tool_catalog)
    set_agent_catalog(agent_catalog)
    set_team_catalog(team_catalog)

    # Assemble FastAPI app
    app = FastAPI(title="Akgentic Org API")
    app.include_router(template_router)
    app.include_router(tool_router)
    app.include_router(agent_router)
    app.include_router(team_router)
    add_exception_handlers(app)

    logger.info("Created Akgentic Org API with %s backend", backend)
    return app


def _wire_yaml_backend(
    base_path: Path,
) -> tuple[TemplateCatalog, ToolCatalog, AgentCatalog, TeamCatalog]:
    """Create catalog services wired with YAML repositories.

    Args:
        base_path: Root directory containing ``templates/``, ``tools/``,
            ``agents/``, and ``teams/`` subdirectories.

    Returns:
        Tuple of (TemplateCatalog, ToolCatalog, AgentCatalog, TeamCatalog).
    """
    from akgentic.catalog.repositories.yaml.agent_repo import YamlAgentCatalogRepository
    from akgentic.catalog.repositories.yaml.team_repo import YamlTeamCatalogRepository
    from akgentic.catalog.repositories.yaml.template_repo import YamlTemplateCatalogRepository
    from akgentic.catalog.repositories.yaml.tool_repo import YamlToolCatalogRepository

    # Create subdirectories if absent
    for name in ("templates", "tools", "agents", "teams"):
        (base_path / name).mkdir(parents=True, exist_ok=True)

    # Create repositories
    template_repo = YamlTemplateCatalogRepository(base_path / "templates")
    tool_repo = YamlToolCatalogRepository(base_path / "tools")
    agent_repo = YamlAgentCatalogRepository(base_path / "agents")
    team_repo = YamlTeamCatalogRepository(base_path / "teams")

    # Create services in dependency order
    template_catalog = TemplateCatalog(template_repo)
    tool_catalog = ToolCatalog(tool_repo)
    agent_catalog = AgentCatalog(agent_repo, template_catalog, tool_catalog)
    team_catalog = TeamCatalog(team_repo, agent_catalog)

    return template_catalog, tool_catalog, agent_catalog, team_catalog


def _wire_mongodb_backend(
    config: MongoCatalogConfig,
) -> tuple[TemplateCatalog, ToolCatalog, AgentCatalog, TeamCatalog]:
    """Create catalog services wired with MongoDB repositories.

    Args:
        config: MongoDB connection and collection configuration.

    Returns:
        Tuple of (TemplateCatalog, ToolCatalog, AgentCatalog, TeamCatalog).
    """
    from akgentic.catalog.repositories.mongo.agent_repo import MongoAgentCatalogRepository
    from akgentic.catalog.repositories.mongo.team_repo import MongoTeamCatalogRepository
    from akgentic.catalog.repositories.mongo.template_repo import MongoTemplateCatalogRepository
    from akgentic.catalog.repositories.mongo.tool_repo import MongoToolCatalogRepository

    client = config.create_client()

    template_repo = MongoTemplateCatalogRepository(
        config.get_collection(client, config.template_entries_collection)
    )
    tool_repo = MongoToolCatalogRepository(
        config.get_collection(client, config.tool_entries_collection)
    )
    agent_repo = MongoAgentCatalogRepository(
        config.get_collection(client, config.agent_entries_collection)
    )
    team_repo = MongoTeamCatalogRepository(
        config.get_collection(client, config.team_entries_collection)
    )

    # Create services in dependency order
    template_catalog = TemplateCatalog(template_repo)
    tool_catalog = ToolCatalog(tool_repo)
    agent_catalog = AgentCatalog(agent_repo, template_catalog, tool_catalog)
    team_catalog = TeamCatalog(team_repo, agent_catalog)

    return template_catalog, tool_catalog, agent_catalog, team_catalog


def create_v2_app(
    *,
    backend: Literal["yaml", "mongodb"] = "yaml",
    yaml_base_path: Path | None = None,
    mongo_config: MongoCatalogConfig | None = None,
) -> FastAPI:
    """Create a v2 FastAPI app serving the unified ``/catalog`` router.

    The v2 factory lives alongside the v1 ``create_app`` intentionally — v1
    removal is Epic 19. Callers migrate to ``create_v2_app`` at their own pace.

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

        from akgentic.catalog.repositories.yaml_entry_repo import YamlEntryRepository

        base = yaml_base_path if yaml_base_path is not None else _Path("./catalog")
        base.mkdir(parents=True, exist_ok=True)
        return YamlEntryRepository(base)
    if backend == "mongodb":
        if mongo_config is None:
            msg = "mongo_config is required when backend='mongodb'"
            raise ValueError(msg)
        from akgentic.catalog.repositories.mongo_entry_repo import MongoEntryRepository

        client = mongo_config.create_client()
        collection = mongo_config.get_collection(client, _V2_ENTRIES_COLLECTION)
        return MongoEntryRepository(collection)
    msg = f"Unknown backend: {backend!r}. Must be 'yaml' or 'mongodb'."
    raise ValueError(msg)
