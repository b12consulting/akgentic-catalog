"""Team catalog entry models with hierarchical member trees."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from akgentic.catalog.models._types import NonEmptyStr
from akgentic.catalog.models.agent import AgentEntry
from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.core.utils.deserializer import import_class

__all__ = [
    "TeamMemberSpec",
    "TeamEntry",
    "agent_in_members",
]


class _AgentCatalogProtocol(Protocol):
    """Protocol for agent catalog lookup (avoids circular import)."""

    def get(self, agent_id: str) -> Any:  # noqa: ANN401
        """Return AgentEntry or None."""
        ...


class TeamMemberSpec(BaseModel):
    """A member within a team hierarchy, supporting recursive nesting."""

    agent_id: NonEmptyStr = Field(description="References an AgentEntry.id in the AgentCatalog")
    headcount: int = Field(default=1, ge=1, description="Number of agent instances to create")
    members: list[TeamMemberSpec] = Field(
        default=[], description="Nested sub-members under this agent in the hierarchy"
    )


def agent_in_members(agent_id: str, members: list[TeamMemberSpec]) -> bool:
    """Recursively check if agent_id appears in a members tree.

    Args:
        agent_id: The agent id to search for.
        members: The member tree to search.

    Returns:
        True if agent_id is found anywhere in the tree.
    """
    for m in members:
        if m.agent_id == agent_id:
            return True
        if m.members and agent_in_members(agent_id, m.members):
            return True
    return False


class TeamEntry(BaseModel):
    """A team composition with entry point, message types, and member hierarchy."""

    id: NonEmptyStr = Field(description="Unique catalog identifier for this team")
    name: NonEmptyStr = Field(description="Human-readable team name")
    entry_point: NonEmptyStr = Field(
        description="AgentEntry.id that serves as the team front door for external messages"
    )
    message_types: list[NonEmptyStr] = Field(
        min_length=1, description="Fully qualified class paths for accepted message types"
    )
    members: list[TeamMemberSpec] = Field(
        min_length=1, description="Team composition tree — agents instantiated at startup"
    )
    profiles: list[str] = Field(
        default=[], description="AgentEntry.ids available for runtime hiring, not instantiated"
    )
    description: str = Field(
        default="", description="Optional team description for catalog browsing"
    )

    def resolve_entry_point(self, agent_catalog: _AgentCatalogProtocol) -> AgentEntry:
        """Resolve entry_point id to the full AgentEntry from the catalog.

        Args:
            agent_catalog: Catalog service providing agent lookups.

        Returns:
            The AgentEntry for this team's entry point.

        Raises:
            CatalogValidationError: If the entry point agent is not found.
        """
        entry = agent_catalog.get(self.entry_point)
        if entry is None:
            raise CatalogValidationError([f"Entry point '{self.entry_point}' not found in catalog"])
        return entry  # type: ignore[no-any-return]

    def resolve_message_types(self) -> list[type]:
        """Resolve message_type class paths to actual Python classes.

        Collects ALL errors before raising, so users see every problem at once.

        Returns:
            List of resolved Python classes.

        Raises:
            CatalogValidationError: If any message_type cannot be resolved.
        """
        errors: list[str] = []
        resolved: list[type] = []
        for mt in self.message_types:
            try:
                cls = import_class(mt)
            except (ImportError, AttributeError, ValueError) as e:
                errors.append(f"Cannot resolve message_type '{mt}': {e}")
                continue
            if not isinstance(cls, type):
                errors.append(
                    f"Cannot resolve message_type '{mt}': "
                    f"resolved to {type(cls).__name__}, not a class"
                )
                continue
            resolved.append(cls)
        if errors:
            raise CatalogValidationError(errors)
        return resolved
