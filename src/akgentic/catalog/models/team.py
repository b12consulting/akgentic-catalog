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
    "TeamSpec",
]


class _AgentCatalogProtocol(Protocol):
    """Protocol for agent catalog lookup (duck-typed, Epic 3 not yet available)."""

    def get(self, agent_id: str) -> Any:  # noqa: ANN401
        """Return AgentEntry or None."""
        ...


class TeamMemberSpec(BaseModel):
    """A member within a team hierarchy, supporting recursive nesting."""

    agent_id: NonEmptyStr
    headcount: int = Field(default=1, ge=1)
    members: list[TeamMemberSpec] = []


class TeamSpec(BaseModel):
    """A team composition with entry point, message types, and member hierarchy."""

    id: NonEmptyStr
    name: NonEmptyStr
    entry_point: NonEmptyStr
    message_types: list[NonEmptyStr] = Field(min_length=1)
    members: list[TeamMemberSpec] = Field(min_length=1)
    profiles: list[str] = []
    description: str = ""

    def resolve_entry_point(self, agent_catalog: _AgentCatalogProtocol) -> AgentEntry:
        """Resolve entry_point id to the full AgentEntry from the catalog."""
        entry = agent_catalog.get(self.entry_point)
        if entry is None:
            raise CatalogValidationError(
                [f"Entry point '{self.entry_point}' not found in catalog"]
            )
        return entry  # type: ignore[no-any-return]

    def resolve_message_types(self) -> list[type]:
        """Resolve message_type class paths to actual Python classes.

        Collects ALL errors before raising, so users see every problem at once.
        """
        errors: list[str] = []
        resolved: list[type] = []
        for mt in self.message_types:
            try:
                resolved.append(import_class(mt))
            except (ImportError, AttributeError) as e:
                errors.append(f"Cannot resolve message_type '{mt}': {e}")
        if errors:
            raise CatalogValidationError(errors)
        return resolved
