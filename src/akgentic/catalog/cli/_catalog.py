"""Catalog service wiring for the CLI.

Provides ``build_catalogs()`` to create all four catalog services from a
YAML base directory, replicating the wiring pattern from ``api/app.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from akgentic.catalog.services.agent_catalog import AgentCatalog
    from akgentic.catalog.services.team_catalog import TeamCatalog
    from akgentic.catalog.services.template_catalog import TemplateCatalog
    from akgentic.catalog.services.tool_catalog import ToolCatalog

__all__ = ["build_catalogs"]

logger = logging.getLogger(__name__)


def build_catalogs(
    catalog_dir: Path,
) -> tuple[TemplateCatalog, ToolCatalog, AgentCatalog, TeamCatalog]:
    """Wire all four catalog services from a YAML base directory.

    Creates subdirectories (``templates/``, ``tools/``, ``agents/``,
    ``teams/``) if they do not exist, then wires YAML repositories and
    catalog services in dependency order with back-references for delete
    protection.

    Args:
        catalog_dir: Root directory containing catalog YAML files.

    Returns:
        Tuple of (TemplateCatalog, ToolCatalog, AgentCatalog, TeamCatalog).
    """
    from akgentic.catalog.repositories.yaml.agent_repo import YamlAgentCatalogRepository
    from akgentic.catalog.repositories.yaml.team_repo import YamlTeamCatalogRepository
    from akgentic.catalog.repositories.yaml.template_repo import YamlTemplateCatalogRepository
    from akgentic.catalog.repositories.yaml.tool_repo import YamlToolCatalogRepository
    from akgentic.catalog.services.agent_catalog import AgentCatalog
    from akgentic.catalog.services.team_catalog import TeamCatalog
    from akgentic.catalog.services.template_catalog import TemplateCatalog
    from akgentic.catalog.services.tool_catalog import ToolCatalog

    # Create subdirectories if absent
    for name in ("templates", "tools", "agents", "teams"):
        (catalog_dir / name).mkdir(parents=True, exist_ok=True)

    # Create repositories
    template_repo = YamlTemplateCatalogRepository(catalog_dir / "templates")
    tool_repo = YamlToolCatalogRepository(catalog_dir / "tools")
    agent_repo = YamlAgentCatalogRepository(catalog_dir / "agents")
    team_repo = YamlTeamCatalogRepository(catalog_dir / "teams")

    # Create services in dependency order
    template_catalog = TemplateCatalog(template_repo)
    tool_catalog = ToolCatalog(tool_repo)
    agent_catalog = AgentCatalog(agent_repo, template_catalog, tool_catalog)
    team_catalog = TeamCatalog(team_repo, agent_catalog)

    # Wire downstream back-references for delete protection
    template_catalog.agent_catalog = agent_catalog
    tool_catalog.agent_catalog = agent_catalog
    agent_catalog.team_catalog = team_catalog

    logger.info("Wired catalog services from %s", catalog_dir)
    return template_catalog, tool_catalog, agent_catalog, team_catalog
