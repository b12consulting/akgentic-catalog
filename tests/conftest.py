"""Shared test factory functions and in-memory mock repositories.

Factory functions are plain functions (not fixtures) — they are pure constructors
with no state. Fixtures are only for stateful objects that benefit from pytest
lifecycle management.
"""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

from akgentic.catalog.models.agent import AgentEntry
from akgentic.catalog.models.team import TeamEntry, TeamMemberSpec
from akgentic.catalog.models.template import TemplateEntry
from akgentic.catalog.models.tool import ToolEntry
from akgentic.catalog.repositories.base import ToolCatalogRepository

if TYPE_CHECKING:
    from akgentic.catalog.models.queries import ToolQuery

_list = builtins.list  # Avoids shadowing by Pydantic 'list' fields


# --- Factory Functions ---


def make_template(
    id: str = "sys-prompt",
    template: str = "You are {role}. {instructions}",
) -> TemplateEntry:
    """Create a TemplateEntry for testing."""
    return TemplateEntry(id=id, template=template)


def make_tool(
    id: str = "search-1",
    tool_class: str = "akgentic.tool.search.SearchTool",
    name: str = "search",
    description: str = "Search the web",
) -> ToolEntry:
    """Create a ToolEntry for testing with configurable tool metadata."""
    return ToolEntry(
        id=id,
        tool_class=tool_class,
        tool={"name": name, "description": description},
    )


def make_agent(
    id: str = "agent-1",
    name: str = "test-agent",
    tool_ids: _list[str] | None = None,
    template_ref: str | None = None,
    params: dict[str, str] | None = None,
    routes_to: _list[str] | None = None,
) -> AgentEntry:
    """Create an AgentEntry for testing with optional tool/template/route config."""
    prompt: dict[str, str | dict[str, str]] = {}
    if template_ref is not None:
        prompt["template"] = template_ref
        if params is not None:
            prompt["params"] = params
    config: dict[str, str | dict[str, str | dict[str, str]]] = {"name": name}
    if prompt:
        config["prompt"] = prompt
    return AgentEntry(
        id=id,
        tool_ids=tool_ids or [],
        card={
            "role": "engineer",
            "description": "test agent",
            "skills": ["coding"],
            "agent_class": "akgentic.agent.BaseAgent",
            "config": config,
            "routes_to": routes_to or [],
        },
    )


def make_team(
    id: str = "team-1",
    name: str = "Test Team",
    entry_point: str = "agent-1",
    members: _list[TeamMemberSpec] | None = None,
    agent_profiles: _list[str] | None = None,
    message_types: _list[str] | None = None,
) -> TeamEntry:
    """Create a TeamEntry for testing with optional members and config."""
    default_members = members or [TeamMemberSpec(agent_id="agent-1")]
    return TeamEntry(
        id=id,
        name=name,
        entry_point=entry_point,
        message_types=message_types or ["akgentic.core.messages.UserMessage"],
        members=default_members,
        agent_profiles=agent_profiles or [],
    )


# --- In-Memory Mock Repositories (migrated from helpers.py) ---


class InMemoryToolCatalogRepository(ToolCatalogRepository):
    """Simple in-memory repository for testing service logic."""

    def __init__(self) -> None:
        self._entries: dict[str, ToolEntry] = {}

    def create(self, tool_entry: ToolEntry) -> str:
        self._entries[tool_entry.id] = tool_entry
        return tool_entry.id

    def get(self, id: str) -> ToolEntry | None:
        return self._entries.get(id)

    def list(self) -> _list[ToolEntry]:
        return _list(self._entries.values())

    def search(self, query: ToolQuery) -> _list[ToolEntry]:
        results = self.list()
        if query.id is not None:
            results = [e for e in results if e.id == query.id]
        if query.tool_class is not None:
            results = [e for e in results if e.tool_class == query.tool_class]
        return results

    def update(self, id: str, tool_entry: ToolEntry) -> None:
        self._entries[id] = tool_entry

    def delete(self, id: str) -> None:
        del self._entries[id]


class MockAgentCatalogRepository:
    """Mock repository that returns a fixed list of AgentEntry objects."""

    def __init__(self, entries: _list[AgentEntry]) -> None:
        self._entries = entries

    def list(self) -> _list[AgentEntry]:
        return self._entries


class MockAgentCatalog:
    """Mock agent catalog with .repository.list() for delete protection."""

    def __init__(self, entries: _list[AgentEntry]) -> None:
        self._repository = MockAgentCatalogRepository(entries)

    @property
    def repository(self) -> MockAgentCatalogRepository:
        return self._repository
