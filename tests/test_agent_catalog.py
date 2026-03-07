"""Tests for AgentCatalog service with cross-validation."""

import builtins

import pytest

from akgentic.catalog.models.agent import AgentEntry
from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.queries import AgentQuery, TemplateQuery, ToolQuery
from akgentic.catalog.models.team import TeamMemberSpec, TeamSpec
from akgentic.catalog.models.template import TemplateEntry
from akgentic.catalog.models.tool import ToolEntry
from akgentic.catalog.repositories.base import (
    AgentCatalogRepository,
    TemplateCatalogRepository,
    ToolCatalogRepository,
)
from akgentic.catalog.services.agent_catalog import AgentCatalog
from akgentic.catalog.services.template_catalog import TemplateCatalog
from akgentic.catalog.services.tool_catalog import ToolCatalog

_list = builtins.list


# --- In-memory repositories for testing ---


class InMemoryTemplateCatalogRepository(TemplateCatalogRepository):
    def __init__(self) -> None:
        self._entries: dict[str, TemplateEntry] = {}

    def create(self, template_entry: TemplateEntry) -> str:
        self._entries[template_entry.id] = template_entry
        return template_entry.id

    def get(self, id: str) -> TemplateEntry | None:
        return self._entries.get(id)

    def list(self) -> _list[TemplateEntry]:
        return _list(self._entries.values())

    def search(self, query: TemplateQuery) -> _list[TemplateEntry]:
        return self.list()

    def update(self, id: str, template_entry: TemplateEntry) -> None:
        self._entries[id] = template_entry

    def delete(self, id: str) -> None:
        del self._entries[id]


class InMemoryToolCatalogRepository(ToolCatalogRepository):
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
        return self.list()

    def update(self, id: str, tool_entry: ToolEntry) -> None:
        self._entries[id] = tool_entry

    def delete(self, id: str) -> None:
        del self._entries[id]


class InMemoryAgentCatalogRepository(AgentCatalogRepository):
    def __init__(self) -> None:
        self._entries: dict[str, AgentEntry] = {}

    def create(self, agent_entry: AgentEntry) -> str:
        self._entries[agent_entry.id] = agent_entry
        return agent_entry.id

    def get(self, id: str) -> AgentEntry | None:
        return self._entries.get(id)

    def list(self) -> _list[AgentEntry]:
        return _list(self._entries.values())

    def search(self, query: AgentQuery) -> _list[AgentEntry]:
        return self.list()

    def update(self, id: str, agent_entry: AgentEntry) -> None:
        self._entries[id] = agent_entry

    def delete(self, id: str) -> None:
        del self._entries[id]


class MockTeamCatalog:
    """Mock team catalog satisfying _TeamCatalogProtocol (has list() method)."""

    def __init__(self, teams: _list[TeamSpec]) -> None:
        self._teams = _list(teams)

    def list(self) -> _list[TeamSpec]:
        return self._teams


# --- Helper factories ---


def _make_template(
    id: str = "sys-prompt",
    template: str = "You are {role}. {instructions}",
) -> TemplateEntry:
    return TemplateEntry(id=id, template=template)


def _make_tool(id: str = "search-1") -> ToolEntry:
    return ToolEntry(
        id=id,
        tool_class="akgentic.tool.search.search.SearchTool",
        tool={"name": "search", "description": "Search the web"},
    )


def _make_agent(
    id: str = "agent-1",
    name: str = "test-agent",
    tool_ids: _list[str] | None = None,
    template_ref: str | None = None,
    params: dict[str, str] | None = None,
    routes_to: _list[str] | None = None,
) -> AgentEntry:
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


def _make_team(
    id: str = "team-1",
    members: _list[dict[str, str | int | _list[dict[str, str]]]] | None = None,
    profiles: _list[str] | None = None,
) -> TeamSpec:
    default_members = members or [{"agent_id": "agent-1"}]
    return TeamSpec(
        id=id,
        name="Test Team",
        entry_point="agent-1",
        message_types=["akgentic.core.messages.UserMessage"],
        members=default_members,
        profiles=profiles or [],
    )


# --- Fixtures ---


@pytest.fixture
def template_repo() -> InMemoryTemplateCatalogRepository:
    return InMemoryTemplateCatalogRepository()


@pytest.fixture
def tool_repo() -> InMemoryToolCatalogRepository:
    return InMemoryToolCatalogRepository()


@pytest.fixture
def agent_repo() -> InMemoryAgentCatalogRepository:
    return InMemoryAgentCatalogRepository()


@pytest.fixture
def template_catalog(
    template_repo: InMemoryTemplateCatalogRepository,
) -> TemplateCatalog:
    return TemplateCatalog(repository=template_repo)


@pytest.fixture
def tool_catalog(tool_repo: InMemoryToolCatalogRepository) -> ToolCatalog:
    return ToolCatalog(repository=tool_repo)


@pytest.fixture
def catalog(
    agent_repo: InMemoryAgentCatalogRepository,
    template_catalog: TemplateCatalog,
    tool_catalog: ToolCatalog,
) -> AgentCatalog:
    return AgentCatalog(
        repository=agent_repo,
        template_catalog=template_catalog,
        tool_catalog=tool_catalog,
    )


# --- Tests ---


class TestValidateCreateDuplicateId:
    """AC8: Duplicate ID rejection."""

    def test_returns_error_for_duplicate_id(self, catalog: AgentCatalog) -> None:
        entry = _make_agent("agent-1")
        catalog.create(entry)
        errors = catalog.validate_create(_make_agent("agent-1", name="other"))
        assert len(errors) == 1
        assert "already exists" in errors[0]

    def test_returns_empty_for_unique_id(self, catalog: AgentCatalog) -> None:
        errors = catalog.validate_create(_make_agent("agent-1"))
        assert errors == []


class TestValidateCreateToolRefs:
    """AC1: Tool reference validation."""

    def test_valid_tool_ids(
        self, catalog: AgentCatalog, tool_catalog: ToolCatalog
    ) -> None:
        tool_catalog.create(_make_tool("search-1"))
        entry = _make_agent("agent-1", tool_ids=["search-1"])
        errors = catalog.validate_create(entry)
        assert errors == []

    def test_missing_tool_id(self, catalog: AgentCatalog) -> None:
        entry = _make_agent("agent-1", tool_ids=["nonexistent-tool"])
        errors = catalog.validate_create(entry)
        assert len(errors) == 1
        assert "Tool 'nonexistent-tool' not found" in errors[0]

    def test_multiple_missing_tool_ids(self, catalog: AgentCatalog) -> None:
        entry = _make_agent("agent-1", tool_ids=["missing-1", "missing-2"])
        errors = catalog.validate_create(entry)
        assert len(errors) == 2
        assert "Tool 'missing-1' not found" in errors[0]
        assert "Tool 'missing-2' not found" in errors[1]


class TestValidateCreateTemplateRefs:
    """AC2: Template @-reference validation."""

    def test_valid_template_ref(
        self, catalog: AgentCatalog, template_catalog: TemplateCatalog
    ) -> None:
        template_catalog.create(_make_template("sys-prompt"))
        entry = _make_agent(
            "agent-1",
            template_ref="@sys-prompt",
            params={"role": "engineer", "instructions": "code well"},
        )
        errors = catalog.validate_create(entry)
        assert errors == []

    def test_missing_template_ref(self, catalog: AgentCatalog) -> None:
        entry = _make_agent("agent-1", template_ref="@nonexistent")
        errors = catalog.validate_create(entry)
        assert len(errors) == 1
        assert "Template '@nonexistent' not found" in errors[0]

    def test_missing_params(
        self, catalog: AgentCatalog, template_catalog: TemplateCatalog
    ) -> None:
        template_catalog.create(_make_template("sys-prompt"))
        entry = _make_agent("agent-1", template_ref="@sys-prompt", params={})
        errors = catalog.validate_create(entry)
        assert any("missing params" in e for e in errors)

    def test_extra_params(
        self, catalog: AgentCatalog, template_catalog: TemplateCatalog
    ) -> None:
        template_catalog.create(_make_template("sys-prompt"))
        entry = _make_agent(
            "agent-1",
            template_ref="@sys-prompt",
            params={"role": "x", "instructions": "y", "extra": "z"},
        )
        errors = catalog.validate_create(entry)
        assert any("extra params" in e for e in errors)

    def test_non_catalog_ref_template_skips_validation(
        self, catalog: AgentCatalog
    ) -> None:
        entry = _make_agent("agent-1", template_ref="You are a helpful assistant")
        errors = catalog.validate_create(entry)
        assert errors == []


class TestValidateCreateRouteTargets:
    """AC3/AC4: Route target validation with pending_names."""

    def test_valid_route_target(self, catalog: AgentCatalog) -> None:
        catalog.create(_make_agent("target-1", name="router-target"))
        entry = _make_agent("agent-1", routes_to=["router-target"])
        errors = catalog.validate_create(entry)
        assert errors == []

    def test_missing_route_target(self, catalog: AgentCatalog) -> None:
        entry = _make_agent("agent-1", routes_to=["nonexistent"])
        errors = catalog.validate_create(entry)
        assert len(errors) == 1
        assert "Route target 'nonexistent' not found" in errors[0]

    def test_pending_names_bypass(self, catalog: AgentCatalog) -> None:
        entry = _make_agent("agent-1", routes_to=["future-agent"])
        errors = catalog.validate_create(entry, pending_names={"future-agent"})
        assert errors == []

    def test_pending_names_partial(self, catalog: AgentCatalog) -> None:
        entry = _make_agent("agent-1", routes_to=["future-agent", "missing"])
        errors = catalog.validate_create(entry, pending_names={"future-agent"})
        assert len(errors) == 1
        assert "Route target 'missing' not found" in errors[0]


class TestValidateCreateMultipleErrors:
    """AC5: Multiple error collection in single call."""

    def test_multiple_errors_collected(self, catalog: AgentCatalog) -> None:
        catalog.create(_make_agent("agent-1"))
        entry = _make_agent(
            "agent-1",
            tool_ids=["missing-tool"],
            template_ref="@missing-template",
            routes_to=["missing-target"],
        )
        errors = catalog.validate_create(entry)
        assert len(errors) >= 3
        assert any("already exists" in e for e in errors)
        assert any("Tool 'missing-tool'" in e for e in errors)
        assert any("Template '@missing-template'" in e for e in errors)


class TestCreate:
    """AC5/AC8: create delegates to validate_create and raises."""

    def test_create_success(self, catalog: AgentCatalog) -> None:
        entry = _make_agent("agent-1")
        result = catalog.create(entry)
        assert result == "agent-1"
        assert catalog.get("agent-1") is not None

    def test_create_raises_on_validation_errors(
        self, catalog: AgentCatalog
    ) -> None:
        entry = _make_agent("agent-1", tool_ids=["nonexistent"])
        with pytest.raises(CatalogValidationError, match="not found"):
            catalog.create(entry)

    def test_create_with_pending_names(self, catalog: AgentCatalog) -> None:
        entry = _make_agent("agent-1", routes_to=["future-agent"])
        result = catalog.create(entry, pending_names={"future-agent"})
        assert result == "agent-1"


class TestCrudDelegation:
    """AC9: get, list, search delegate directly to repository."""

    def test_get_existing(self, catalog: AgentCatalog) -> None:
        catalog.create(_make_agent("agent-1"))
        result = catalog.get("agent-1")
        assert result is not None
        assert result.id == "agent-1"

    def test_get_nonexistent(self, catalog: AgentCatalog) -> None:
        assert catalog.get("nonexistent") is None

    def test_list_delegates(self, catalog: AgentCatalog) -> None:
        catalog.create(_make_agent("agent-1"))
        catalog.create(_make_agent("agent-2", name="agent-two"))
        result = catalog.list()
        assert len(result) == 2

    def test_search_delegates(self, catalog: AgentCatalog) -> None:
        catalog.create(_make_agent("agent-1"))
        query = AgentQuery(id="agent-1")
        result = catalog.search(query)
        assert len(result) >= 1


class TestUpdate:
    """AC6: update with existence check and cross-validation."""

    def test_update_existing_success(self, catalog: AgentCatalog) -> None:
        catalog.create(_make_agent("agent-1"))
        updated = _make_agent("agent-1", name="updated-name")
        catalog.update("agent-1", updated)
        result = catalog.get("agent-1")
        assert result is not None

    def test_update_nonexistent_raises(self, catalog: AgentCatalog) -> None:
        entry = _make_agent("agent-1")
        with pytest.raises(EntryNotFoundError, match="not found"):
            catalog.update("agent-1", entry)

    def test_update_id_mismatch_raises(self, catalog: AgentCatalog) -> None:
        catalog.create(_make_agent("agent-1"))
        mismatched = _make_agent("agent-2", name="agent-two")
        with pytest.raises(CatalogValidationError, match="does not match"):
            catalog.update("agent-1", mismatched)

    def test_update_cross_validates(self, catalog: AgentCatalog) -> None:
        catalog.create(_make_agent("agent-1"))
        updated = _make_agent("agent-1", tool_ids=["nonexistent"])
        with pytest.raises(CatalogValidationError, match="Tool"):
            catalog.update("agent-1", updated)


class TestValidateDelete:
    """AC7: Delete validation with routing and team checks."""

    def test_not_found_error(self, catalog: AgentCatalog) -> None:
        errors = catalog.validate_delete("nonexistent")
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_no_dependencies(self, catalog: AgentCatalog) -> None:
        catalog.create(_make_agent("agent-1"))
        errors = catalog.validate_delete("agent-1")
        assert errors == []

    def test_routing_dependency(self, catalog: AgentCatalog) -> None:
        catalog.create(_make_agent("target", name="target-name"))
        catalog.create(_make_agent("router", name="router-name", routes_to=["target-name"]))
        errors = catalog.validate_delete("target")
        assert len(errors) == 1
        assert "routes to" in errors[0]
        assert "cannot delete" in errors[0]

    def test_team_member_reference(self, catalog: AgentCatalog) -> None:
        catalog.create(_make_agent("agent-1"))
        team = _make_team("team-1", members=[{"agent_id": "agent-1"}])
        catalog.team_catalog = MockTeamCatalog([team])
        errors = catalog.validate_delete("agent-1")
        assert len(errors) == 1
        assert "members" in errors[0]
        assert "cannot delete" in errors[0]

    def test_team_profiles_reference(self, catalog: AgentCatalog) -> None:
        catalog.create(_make_agent("agent-1"))
        team = _make_team(
            "team-1",
            members=[{"agent_id": "other-agent"}],
            profiles=["agent-1"],
        )
        catalog.team_catalog = MockTeamCatalog([team])
        errors = catalog.validate_delete("agent-1")
        assert len(errors) == 1
        assert "profiles" in errors[0]
        assert "cannot delete" in errors[0]

    def test_no_team_catalog_wired(self, catalog: AgentCatalog) -> None:
        catalog.create(_make_agent("agent-1"))
        assert catalog.team_catalog is None
        errors = catalog.validate_delete("agent-1")
        assert errors == []

    def test_nested_team_member_reference(self, catalog: AgentCatalog) -> None:
        catalog.create(_make_agent("nested-agent"))
        team = _make_team(
            "team-1",
            members=[{
                "agent_id": "outer-agent",
                "members": [{"agent_id": "nested-agent"}],
            }],
        )
        catalog.team_catalog = MockTeamCatalog([team])
        errors = catalog.validate_delete("nested-agent")
        assert len(errors) == 1
        assert "members" in errors[0]


class TestDelete:
    """AC7: delete calls validate_delete and raises appropriately."""

    def test_delete_success(self, catalog: AgentCatalog) -> None:
        catalog.create(_make_agent("agent-1"))
        catalog.delete("agent-1")
        assert catalog.get("agent-1") is None

    def test_delete_not_found_raises(self, catalog: AgentCatalog) -> None:
        with pytest.raises(EntryNotFoundError, match="not found"):
            catalog.delete("nonexistent")

    def test_delete_routing_dependency_raises(
        self, catalog: AgentCatalog
    ) -> None:
        catalog.create(_make_agent("target", name="target-name"))
        catalog.create(_make_agent("router", name="router-name", routes_to=["target-name"]))
        with pytest.raises(CatalogValidationError, match="cannot delete"):
            catalog.delete("target")

    def test_delete_team_reference_raises(self, catalog: AgentCatalog) -> None:
        catalog.create(_make_agent("agent-1"))
        team = _make_team("team-1", members=[{"agent_id": "agent-1"}])
        catalog.team_catalog = MockTeamCatalog([team])
        with pytest.raises(CatalogValidationError, match="cannot delete"):
            catalog.delete("agent-1")


class TestTeamCatalogProperty:
    """AC7: team_catalog getter/setter."""

    def test_defaults_to_none(self, catalog: AgentCatalog) -> None:
        assert catalog.team_catalog is None

    def test_can_be_set(self, catalog: AgentCatalog) -> None:
        mock_tc = MockTeamCatalog([])
        catalog.team_catalog = mock_tc
        assert catalog.team_catalog is mock_tc

    def test_can_be_unset(self, catalog: AgentCatalog) -> None:
        mock_tc = MockTeamCatalog([])
        catalog.team_catalog = mock_tc
        catalog.team_catalog = None
        assert catalog.team_catalog is None


class TestAgentInMembers:
    """Static helper for recursive member check."""

    def test_direct_member(self) -> None:
        members = [TeamMemberSpec(agent_id="agent-1")]
        assert AgentCatalog._agent_in_members("agent-1", members) is True

    def test_nested_member(self) -> None:
        members = [
            TeamMemberSpec(
                agent_id="outer",
                members=[TeamMemberSpec(agent_id="inner")],
            )
        ]
        assert AgentCatalog._agent_in_members("inner", members) is True

    def test_not_found(self) -> None:
        members = [TeamMemberSpec(agent_id="agent-1")]
        assert AgentCatalog._agent_in_members("nonexistent", members) is False

    def test_empty_members(self) -> None:
        assert AgentCatalog._agent_in_members("agent-1", []) is False
