"""Public API surface for akgentic-catalog.

Re-exports entry models (TemplateEntry, ToolEntry, AgentEntry, TeamSpec),
query models, error types, abstract and YAML repository interfaces,
catalog services, and the resolve_env_vars utility. MongoDB backend exports
are conditionally available when pymongo is installed.
"""

from __future__ import annotations

from akgentic.catalog.env import resolve_env_vars
from akgentic.catalog.models.agent import AgentEntry
from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.queries import AgentQuery, TeamQuery, TemplateQuery, ToolQuery
from akgentic.catalog.models.team import TeamMemberSpec, TeamSpec
from akgentic.catalog.models.template import TemplateEntry
from akgentic.catalog.models.tool import ToolEntry
from akgentic.catalog.repositories.base import (
    AgentCatalogRepository,
    TeamCatalogRepository,
    TemplateCatalogRepository,
    ToolCatalogRepository,
)
from akgentic.catalog.repositories.yaml.agent_repo import YamlAgentCatalogRepository
from akgentic.catalog.repositories.yaml.team_repo import YamlTeamCatalogRepository
from akgentic.catalog.repositories.yaml.template_repo import YamlTemplateCatalogRepository
from akgentic.catalog.repositories.yaml.tool_repo import YamlToolCatalogRepository
from akgentic.catalog.services.agent_catalog import AgentCatalog
from akgentic.catalog.services.team_catalog import TeamCatalog
from akgentic.catalog.services.template_catalog import TemplateCatalog
from akgentic.catalog.services.tool_catalog import ToolCatalog

__all__ = [
    "AgentCatalog",
    "AgentCatalogRepository",
    "AgentEntry",
    "AgentQuery",
    "CatalogValidationError",
    "EntryNotFoundError",
    "TeamCatalog",
    "TeamCatalogRepository",
    "TeamMemberSpec",
    "TeamQuery",
    "TeamSpec",
    "TemplateCatalog",
    "TemplateCatalogRepository",
    "TemplateEntry",
    "TemplateQuery",
    "ToolCatalog",
    "ToolCatalogRepository",
    "ToolEntry",
    "ToolQuery",
    "YamlAgentCatalogRepository",
    "YamlTeamCatalogRepository",
    "YamlTemplateCatalogRepository",
    "YamlToolCatalogRepository",
    "resolve_env_vars",
]

try:
    from akgentic.catalog.repositories.mongo import (
        MongoAgentCatalogRepository,
        MongoCatalogConfig,
        MongoTeamCatalogRepository,
        MongoTemplateCatalogRepository,
        MongoToolCatalogRepository,
        from_document,
        to_document,
    )

    __all__ += [
        "MongoAgentCatalogRepository",
        "MongoCatalogConfig",
        "MongoTeamCatalogRepository",
        "MongoTemplateCatalogRepository",
        "MongoToolCatalogRepository",
        "from_document",
        "to_document",
    ]
except ImportError:
    pass
