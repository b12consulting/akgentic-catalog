"""Team catalog entry models with hierarchical member trees."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from akgentic.catalog.models._types import NonEmptyStr
from akgentic.catalog.models.agent import (
    AgentEntry,
    _TemplateCatalogProtocol,
    _ToolCatalogProtocol,
)
from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.core.utils.deserializer import import_class
from akgentic.team.models import TeamCard, TeamCardMember

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
        description=(
            "AgentEntry.id of the HumanProxy that sends the first message"
            " to the team (ADR-003 convention)"
        )
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

    @staticmethod
    def _resolve_members(
        specs: list[TeamMemberSpec],
        agent_catalog: _AgentCatalogProtocol,
        errors: list[str],
        tool_catalog: _ToolCatalogProtocol | None = None,
        template_catalog: _TemplateCatalogProtocol | None = None,
    ) -> list[tuple[str, TeamCardMember]]:
        """Recursively resolve TeamMemberSpec list to (agent_id, TeamCardMember) pairs.

        Args:
            specs: Member specifications to resolve.
            agent_catalog: Catalog service providing agent lookups.
            errors: Accumulator for error messages (mutated in place).
            tool_catalog: Optional tool catalog for resolving tool_ids to ToolCards.
            template_catalog: Optional template catalog for resolving @-refs.

        Returns:
            List of (agent_id, TeamCardMember) tuples for successfully resolved members.
        """
        resolved: list[tuple[str, TeamCardMember]] = []
        for spec in specs:
            entry: AgentEntry | None = agent_catalog.get(spec.agent_id)
            if entry is None:
                errors.append(f"Agent '{spec.agent_id}' not found in catalog")
                # Still recurse children to collect all errors
                TeamEntry._resolve_members(
                    spec.members, agent_catalog, errors, tool_catalog, template_catalog,
                )
                continue
            children = TeamEntry._resolve_members(
                spec.members, agent_catalog, errors, tool_catalog, template_catalog,
            )
            # Use fully resolved card (tools + templates) when catalogs are available
            if tool_catalog is not None and template_catalog is not None:
                try:
                    card = entry.to_agent_card(tool_catalog, template_catalog)
                except CatalogValidationError as e:
                    errors.extend(e.errors)
                    card = entry.card
            else:
                card = entry.card
            member = TeamCardMember(
                card=card,
                headcount=spec.headcount,
                members=[m for _, m in children],
            )
            resolved.append((spec.agent_id, member))
        return resolved

    def to_team_card(
        self,
        agent_catalog: _AgentCatalogProtocol,
        tool_catalog: _ToolCatalogProtocol | None = None,
        template_catalog: _TemplateCatalogProtocol | None = None,
    ) -> TeamCard:
        """Resolve all string IDs into a runtime-ready TeamCard.

        Converts the catalog-level TeamEntry (string IDs, FQCN message types)
        into a runtime-ready TeamCard (resolved AgentCards, Python classes).

        When ``tool_catalog`` and ``template_catalog`` are provided, each
        member's ``AgentCard`` is fully resolved via ``AgentEntry.to_agent_card``
        (tools populated, prompt templates expanded).  Without them, the raw
        ``entry.card`` is used (tools may be empty, templates unresolved).

        Args:
            agent_catalog: Catalog service providing agent lookups.
            tool_catalog: Optional tool catalog for resolving tool_ids to ToolCards.
            template_catalog: Optional template catalog for resolving @-refs.

        Returns:
            A TeamCard with fully resolved members, entry_point, and message_types.

        Raises:
            CatalogValidationError: If any agents are missing or message types
                cannot be resolved. Collects ALL errors before raising.
        """
        errors: list[str] = []

        # 1. Resolve members tree
        resolved_pairs = self._resolve_members(
            self.members, agent_catalog, errors, tool_catalog, template_catalog,
        )

        # 2. Resolve message types (collect errors, don't fail fast)
        resolved_message_types: list[type] = []
        try:
            resolved_message_types = self.resolve_message_types()
        except CatalogValidationError as e:
            errors.extend(e.errors)

        # 3. Raise all collected errors
        if errors:
            raise CatalogValidationError(errors)

        # 4. Find entry_point in resolved top-level members
        entry_point_member: TeamCardMember | None = None
        remaining_members: list[TeamCardMember] = []
        for agent_id, member in resolved_pairs:
            if agent_id == self.entry_point and entry_point_member is None:
                entry_point_member = member
            else:
                remaining_members.append(member)

        if entry_point_member is None:
            raise CatalogValidationError(
                [f"Entry point '{self.entry_point}' not found in resolved top-level members"]
            )

        return TeamCard(
            name=self.name,
            description=self.description,
            entry_point=entry_point_member,
            members=remaining_members,
            message_types=resolved_message_types,
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
