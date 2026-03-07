"""Tests for ToolCatalog service."""

import builtins

import pytest

from akgentic.catalog.models.agent import AgentEntry
from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.queries import ToolQuery
from akgentic.catalog.models.tool import ToolEntry
from akgentic.catalog.repositories.base import ToolCatalogRepository
from akgentic.catalog.services.tool_catalog import ToolCatalog

_list = builtins.list


# --- In-memory repository for testing ---


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


# --- Mock agent catalog for delete protection ---


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


# --- Helpers ---


def _make_tool(id: str = "search-1") -> ToolEntry:
    """Create a minimal ToolEntry using SearchTool (a real ToolCard subclass)."""
    return ToolEntry(
        id=id,
        tool_class="akgentic.tool.search.search.SearchTool",
        tool={"name": "search", "description": "Search the web"},
    )


def _make_agent_with_tool(agent_id: str, tool_ids: _list[str]) -> AgentEntry:
    """Create an AgentEntry with given tool_ids."""
    return AgentEntry(
        id=agent_id,
        tool_ids=tool_ids,
        card={
            "role": "engineer",
            "description": "test",
            "skills": ["coding"],
            "agent_class": "akgentic.agent.BaseAgent",
            "config": {"name": "test-agent"},
        },
    )


# --- Tests ---


@pytest.fixture
def repo() -> InMemoryToolCatalogRepository:
    return InMemoryToolCatalogRepository()


@pytest.fixture
def catalog(repo: InMemoryToolCatalogRepository) -> ToolCatalog:
    return ToolCatalog(repository=repo)


class TestCreate:
    """AC #2: create with unique id persists, duplicate raises CatalogValidationError."""

    def test_create_unique_id_persists(self, catalog: ToolCatalog) -> None:
        entry = _make_tool("search-1")
        result = catalog.create(entry)
        assert result == "search-1"
        assert catalog.get("search-1") is not None

    def test_create_duplicate_id_raises(self, catalog: ToolCatalog) -> None:
        entry = _make_tool("search-1")
        catalog.create(entry)
        with pytest.raises(CatalogValidationError, match="already exists"):
            catalog.create(entry)


class TestGet:
    """AC #3: get delegates correctly."""

    def test_get_existing(self, catalog: ToolCatalog) -> None:
        entry = _make_tool("search-1")
        catalog.create(entry)
        result = catalog.get("search-1")
        assert result is not None
        assert result.id == "search-1"

    def test_get_nonexistent_returns_none(self, catalog: ToolCatalog) -> None:
        assert catalog.get("nonexistent") is None


class TestList:
    """AC #3: list delegates correctly."""

    def test_list_delegates(self, catalog: ToolCatalog) -> None:
        catalog.create(_make_tool("search-1"))
        catalog.create(_make_tool("search-2"))
        result = catalog.list()
        assert len(result) == 2


class TestSearch:
    """AC #3: search delegates correctly."""

    def test_search_delegates(self, catalog: ToolCatalog) -> None:
        catalog.create(_make_tool("search-1"))
        catalog.create(_make_tool("search-2"))
        query = ToolQuery(id="search-1")
        result = catalog.search(query)
        assert len(result) == 1
        assert result[0].id == "search-1"


class TestUpdate:
    """AC #3: update with existing id delegates, nonexistent raises."""

    def test_update_existing_delegates(self, catalog: ToolCatalog) -> None:
        catalog.create(_make_tool("search-1"))
        updated = _make_tool("search-1")
        catalog.update("search-1", updated)
        result = catalog.get("search-1")
        assert result is not None

    def test_update_nonexistent_raises(self, catalog: ToolCatalog) -> None:
        entry = _make_tool("search-1")
        with pytest.raises(EntryNotFoundError, match="not found"):
            catalog.update("search-1", entry)


class TestDelete:
    """AC #3: delete with existing id delegates, nonexistent raises."""

    def test_delete_existing_no_downstream(self, catalog: ToolCatalog) -> None:
        catalog.create(_make_tool("search-1"))
        catalog.delete("search-1")
        assert catalog.get("search-1") is None

    def test_delete_nonexistent_raises(self, catalog: ToolCatalog) -> None:
        with pytest.raises(EntryNotFoundError, match="not found"):
            catalog.delete("nonexistent")


class TestAgentCatalogAttribute:
    """AC #3: agent_catalog defaults to None."""

    def test_defaults_to_none(self, catalog: ToolCatalog) -> None:
        assert catalog.agent_catalog is None

    def test_can_be_set(self, catalog: ToolCatalog) -> None:
        mock_ac = MockAgentCatalog([])
        catalog.agent_catalog = mock_ac
        assert catalog.agent_catalog is mock_ac


class TestDeleteProtection:
    """Delete protection when agent_catalog is set."""

    def test_delete_blocked_when_agent_references_tool(
        self, catalog: ToolCatalog
    ) -> None:
        catalog.create(_make_tool("search-1"))
        agent = _make_agent_with_tool("agent-1", ["search-1"])
        catalog.agent_catalog = MockAgentCatalog([agent])
        with pytest.raises(CatalogValidationError, match="cannot delete"):
            catalog.delete("search-1")

    def test_delete_allowed_when_agent_catalog_is_none(
        self, catalog: ToolCatalog
    ) -> None:
        catalog.create(_make_tool("search-1"))
        catalog.delete("search-1")
        assert catalog.get("search-1") is None

    def test_delete_allowed_when_no_agent_references_tool(
        self, catalog: ToolCatalog
    ) -> None:
        catalog.create(_make_tool("search-1"))
        agent = _make_agent_with_tool("agent-1", ["other-tool"])
        catalog.agent_catalog = MockAgentCatalog([agent])
        catalog.delete("search-1")
        assert catalog.get("search-1") is None

    def test_delete_blocked_multiple_agents_reference_tool(
        self, catalog: ToolCatalog
    ) -> None:
        catalog.create(_make_tool("search-1"))
        agent1 = _make_agent_with_tool("agent-1", ["search-1"])
        agent2 = _make_agent_with_tool("agent-2", ["search-1"])
        catalog.agent_catalog = MockAgentCatalog([agent1, agent2])
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.delete("search-1")
        assert len(exc_info.value.errors) == 2


class TestValidateCreate:
    """validate_create returns list[str], not raises."""

    def test_returns_empty_for_unique(self, catalog: ToolCatalog) -> None:
        entry = _make_tool("search-1")
        errors = catalog.validate_create(entry)
        assert errors == []

    def test_returns_errors_for_duplicate(self, catalog: ToolCatalog) -> None:
        entry = _make_tool("search-1")
        catalog.create(entry)
        errors = catalog.validate_create(entry)
        assert len(errors) == 1
        assert "already exists" in errors[0]


class TestValidateDelete:
    """validate_delete returns list[str] for downstream checks."""

    def test_returns_empty_when_no_refs(self, catalog: ToolCatalog) -> None:
        catalog.create(_make_tool("search-1"))
        errors = catalog.validate_delete("search-1")
        assert errors == []

    def test_returns_errors_when_referenced(self, catalog: ToolCatalog) -> None:
        catalog.create(_make_tool("search-1"))
        agent = _make_agent_with_tool("agent-1", ["search-1"])
        catalog.agent_catalog = MockAgentCatalog([agent])
        errors = catalog.validate_delete("search-1")
        assert len(errors) == 1
        assert "cannot delete" in errors[0]

    def test_returns_not_found_error(self, catalog: ToolCatalog) -> None:
        errors = catalog.validate_delete("nonexistent")
        assert len(errors) == 1
        assert "not found" in errors[0]
