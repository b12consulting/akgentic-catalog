"""AgentCatalog service — domain logic over agent repository with cross-validation."""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

from akgentic.agent.config import AgentConfig
from akgentic.catalog.models.agent import AgentEntry
from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.team import TeamMemberSpec
from akgentic.catalog.refs import _is_catalog_ref, _resolve_ref
from akgentic.catalog.repositories.base import AgentCatalogRepository
from akgentic.catalog.services.template_catalog import TemplateCatalog
from akgentic.catalog.services.tool_catalog import ToolCatalog

if TYPE_CHECKING:
    from akgentic.catalog.models.queries import AgentQuery

__all__ = ["AgentCatalog"]

_list = builtins.list


class _TeamCatalogProtocol:
    """Minimal duck-type for team catalog (avoids circular import)."""


class AgentCatalog:
    """Service layer for agent catalog entries with cross-validation."""

    def __init__(
        self,
        repository: AgentCatalogRepository,
        template_catalog: TemplateCatalog,
        tool_catalog: ToolCatalog,
    ) -> None:
        self.repository = repository
        self._template_catalog = template_catalog
        self._tool_catalog = tool_catalog
        self._team_catalog: _TeamCatalogProtocol | None = None

    @property
    def team_catalog(self) -> _TeamCatalogProtocol | None:
        """Optional downstream team catalog for delete protection."""
        return self._team_catalog

    @team_catalog.setter
    def team_catalog(self, value: _TeamCatalogProtocol | None) -> None:
        self._team_catalog = value

    def validate_create(
        self,
        entry: AgentEntry,
        pending_names: set[str] | None = None,
    ) -> _list[str]:
        """Validate an agent entry for creation. Returns list of error strings."""
        errors: _list[str] = []

        # AC8: Duplicate ID rejection
        if self.repository.get(entry.id) is not None:
            errors.append(f"Agent id '{entry.id}' already exists")

        # AC1: Tool reference validation
        for tool_id in entry.tool_ids:
            if self._tool_catalog.get(tool_id) is None:
                errors.append(f"Tool '{tool_id}' not found in ToolCatalog")

        # AC2: Template @-reference validation
        config = entry.card.config
        if isinstance(config, AgentConfig):
            prompt = config.prompt
            if _is_catalog_ref(prompt.template):
                template_id = _resolve_ref(prompt.template)
                tpl_entry = self._template_catalog.get(template_id)
                if tpl_entry is None:
                    errors.append(
                        f"Template '@{template_id}' not found in TemplateCatalog"
                    )
                else:
                    expected = set(tpl_entry.placeholders)
                    actual = set(prompt.params.keys())
                    missing = expected - actual
                    extra = actual - expected
                    if missing:
                        errors.append(
                            f"Template '@{template_id}' missing params: "
                            f"{', '.join(sorted(missing))}"
                        )
                    if extra:
                        errors.append(
                            f"Template '@{template_id}' extra params: "
                            f"{', '.join(sorted(extra))}"
                        )

        # AC3/AC4: Route target validation
        valid_names = pending_names or set()
        existing_names = {
            a.card.config.name for a in self.repository.list()
        }
        for route_name in entry.card.routes_to:
            if route_name not in existing_names and route_name not in valid_names:
                errors.append(
                    f"Route target '{route_name}' not found in AgentCatalog"
                )

        return errors

    def create(
        self,
        entry: AgentEntry,
        pending_names: set[str] | None = None,
    ) -> str:
        """Persist a new agent entry. Raises CatalogValidationError on validation failure."""
        errors = self.validate_create(entry, pending_names)
        if errors:
            raise CatalogValidationError(errors)
        return self.repository.create(entry)

    def get(self, id: str) -> AgentEntry | None:
        """Retrieve an agent entry by id."""
        return self.repository.get(id)

    def list(self) -> _list[AgentEntry]:
        """List all agent entries."""
        return self.repository.list()

    def search(self, query: AgentQuery) -> _list[AgentEntry]:
        """Search agent entries by query."""
        return self.repository.search(query)

    def update(self, id: str, entry: AgentEntry) -> None:
        """Update an existing agent entry. Raises EntryNotFoundError if missing."""
        if self.repository.get(id) is None:
            raise EntryNotFoundError(f"Agent id '{id}' not found")
        if entry.id != id:
            raise CatalogValidationError(
                [f"Entry id '{entry.id}' does not match update target '{id}'"]
            )
        errors = self.validate_create(entry)
        # Remove the "already exists" error since we're updating an existing entry
        errors = [e for e in errors if "already exists" not in e]
        if errors:
            raise CatalogValidationError(errors)
        self.repository.update(id, entry)

    def validate_delete(self, id: str) -> _list[str]:
        """Check existence, routing deps, and team refs before delete."""
        errors: _list[str] = []
        agent = self.repository.get(id)
        if agent is None:
            errors.append(f"Agent id '{id}' not found")
            return errors

        agent_name = agent.card.config.name

        # Check routing dependencies: no other agent routes to this agent's name
        for other in self.repository.list():
            if other.id == id:
                continue
            if agent_name in other.card.routes_to:
                errors.append(
                    f"Agent '{other.id}' routes to '{agent_name}' — cannot delete"
                )

        # Check downstream TeamCatalog references (only when wired)
        if self._team_catalog is not None:
            repo = getattr(self._team_catalog, "repository", None)
            if repo is not None:
                for team in repo.list():
                    if AgentCatalog._agent_in_members(id, team.members):
                        errors.append(
                            f"Team '{team.id}' references agent '{id}'"
                            f" in members — cannot delete"
                        )
                    if id in team.profiles:
                        errors.append(
                            f"Team '{team.id}' references agent '{id}'"
                            f" in profiles — cannot delete"
                        )

        return errors

    def delete(self, id: str) -> None:
        """Delete an agent entry. Raises errors if not found or referenced."""
        errors = self.validate_delete(id)
        if errors:
            if "not found" in errors[0]:
                raise EntryNotFoundError(errors[0])
            raise CatalogValidationError(errors)
        self.repository.delete(id)

    @staticmethod
    def _agent_in_members(agent_id: str, members: _list[TeamMemberSpec]) -> bool:
        """Recursively check if agent_id appears in a members tree."""
        for m in members:
            if m.agent_id == agent_id:
                return True
            if m.members and AgentCatalog._agent_in_members(agent_id, m.members):
                return True
        return False
