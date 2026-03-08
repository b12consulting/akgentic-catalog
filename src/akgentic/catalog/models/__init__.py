"""Public API surface for catalog data models.

Re-exports entry models (TemplateEntry, ToolEntry, AgentEntry, TeamSpec,
TeamMemberSpec), query filter models (TemplateQuery, ToolQuery, AgentQuery,
TeamQuery), and error types (CatalogValidationError, EntryNotFoundError).
"""

from __future__ import annotations

from akgentic.catalog.models.agent import AgentEntry
from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.queries import AgentQuery, TeamQuery, TemplateQuery, ToolQuery
from akgentic.catalog.models.team import TeamMemberSpec, TeamSpec
from akgentic.catalog.models.template import TemplateEntry
from akgentic.catalog.models.tool import ToolEntry

__all__ = [
    "AgentEntry",
    "AgentQuery",
    "CatalogValidationError",
    "EntryNotFoundError",
    "TeamMemberSpec",
    "TeamQuery",
    "TeamSpec",
    "TemplateEntry",
    "TemplateQuery",
    "ToolEntry",
    "ToolQuery",
]
