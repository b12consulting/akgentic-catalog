"""Tests for TemplateCatalog service."""

import builtins

import pytest

from akgentic.catalog.models.agent import AgentEntry
from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.queries import TemplateQuery
from akgentic.catalog.models.template import TemplateEntry
from akgentic.catalog.repositories.base import TemplateCatalogRepository
from akgentic.catalog.services.template_catalog import TemplateCatalog

_list = builtins.list


# --- In-memory repository for testing ---


class InMemoryTemplateCatalogRepository(TemplateCatalogRepository):
    """Simple in-memory repository for testing service logic."""

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
        results = self.list()
        if query.id is not None:
            results = [e for e in results if e.id == query.id]
        if query.placeholder is not None:
            results = [e for e in results if query.placeholder in e.placeholders]
        return results

    def update(self, id: str, template_entry: TemplateEntry) -> None:
        self._entries[id] = template_entry

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


# --- Helper ---


def _make_template(id: str = "tpl-1", template: str = "Hello {name}") -> TemplateEntry:
    return TemplateEntry(id=id, template=template)


def _make_agent_with_template_ref(
    agent_id: str, template_ref: str
) -> AgentEntry:
    """Create an AgentEntry whose config.prompt.template is a catalog @-ref."""
    return AgentEntry(
        id=agent_id,
        card={
            "role": "engineer",
            "description": "test",
            "skills": ["coding"],
            "agent_class": "akgentic.agent.BaseAgent",
            "config": {
                "name": "test-agent",
                "prompt": {"template": template_ref},
            },
        },
    )


# --- Tests ---


@pytest.fixture
def repo() -> InMemoryTemplateCatalogRepository:
    return InMemoryTemplateCatalogRepository()


@pytest.fixture
def catalog(repo: InMemoryTemplateCatalogRepository) -> TemplateCatalog:
    return TemplateCatalog(repository=repo)


class TestCreate:
    """AC #1: create with unique id persists, duplicate raises CatalogValidationError."""

    def test_create_unique_id_persists(self, catalog: TemplateCatalog) -> None:
        entry = _make_template("tpl-1")
        result = catalog.create(entry)
        assert result == "tpl-1"
        assert catalog.get("tpl-1") is not None

    def test_create_duplicate_id_raises(self, catalog: TemplateCatalog) -> None:
        entry = _make_template("tpl-1")
        catalog.create(entry)
        with pytest.raises(CatalogValidationError, match="already exists"):
            catalog.create(entry)


class TestGet:
    """AC #3: get delegates to repository."""

    def test_get_existing(self, catalog: TemplateCatalog) -> None:
        entry = _make_template("tpl-1")
        catalog.create(entry)
        result = catalog.get("tpl-1")
        assert result is not None
        assert result.id == "tpl-1"

    def test_get_nonexistent_returns_none(self, catalog: TemplateCatalog) -> None:
        assert catalog.get("nonexistent") is None


class TestList:
    """AC #3: list delegates to repository."""

    def test_list_delegates(self, catalog: TemplateCatalog) -> None:
        catalog.create(_make_template("tpl-1"))
        catalog.create(_make_template("tpl-2", template="Bye {name}"))
        result = catalog.list()
        assert len(result) == 2


class TestSearch:
    """AC #3: search delegates to repository."""

    def test_search_delegates(self, catalog: TemplateCatalog) -> None:
        catalog.create(_make_template("tpl-1"))
        catalog.create(_make_template("tpl-2", template="Bye {name}"))
        query = TemplateQuery(id="tpl-1")
        result = catalog.search(query)
        assert len(result) == 1
        assert result[0].id == "tpl-1"


class TestUpdate:
    """AC #3: update validates id exists, then delegates."""

    def test_update_existing_delegates(self, catalog: TemplateCatalog) -> None:
        catalog.create(_make_template("tpl-1"))
        updated = _make_template("tpl-1", template="Updated {name}")
        catalog.update("tpl-1", updated)
        result = catalog.get("tpl-1")
        assert result is not None
        assert result.template == "Updated {name}"

    def test_update_nonexistent_raises(self, catalog: TemplateCatalog) -> None:
        entry = _make_template("tpl-1")
        with pytest.raises(EntryNotFoundError, match="not found"):
            catalog.update("tpl-1", entry)


class TestDelete:
    """AC #3: delete validates id exists and checks downstream refs."""

    def test_delete_existing_no_downstream(self, catalog: TemplateCatalog) -> None:
        catalog.create(_make_template("tpl-1"))
        catalog.delete("tpl-1")
        assert catalog.get("tpl-1") is None

    def test_delete_nonexistent_raises(self, catalog: TemplateCatalog) -> None:
        with pytest.raises(EntryNotFoundError, match="not found"):
            catalog.delete("nonexistent")


class TestAgentCatalogAttribute:
    """AC #3: agent_catalog defaults to None."""

    def test_defaults_to_none(self, catalog: TemplateCatalog) -> None:
        assert catalog.agent_catalog is None

    def test_can_be_set(self, catalog: TemplateCatalog) -> None:
        mock_ac = MockAgentCatalog([])
        catalog.agent_catalog = mock_ac
        assert catalog.agent_catalog is mock_ac


class TestDeleteProtection:
    """Delete protection when agent_catalog is set."""

    def test_delete_blocked_when_agent_references_template(
        self, catalog: TemplateCatalog
    ) -> None:
        catalog.create(_make_template("sys-prompt"))
        agent = _make_agent_with_template_ref("agent-1", "@sys-prompt")
        catalog.agent_catalog = MockAgentCatalog([agent])
        with pytest.raises(CatalogValidationError, match="cannot delete"):
            catalog.delete("sys-prompt")

    def test_delete_allowed_when_agent_catalog_is_none(
        self, catalog: TemplateCatalog
    ) -> None:
        catalog.create(_make_template("sys-prompt"))
        catalog.delete("sys-prompt")
        assert catalog.get("sys-prompt") is None

    def test_delete_allowed_when_no_agent_references_template(
        self, catalog: TemplateCatalog
    ) -> None:
        catalog.create(_make_template("sys-prompt"))
        agent = _make_agent_with_template_ref("agent-1", "@other-template")
        catalog.agent_catalog = MockAgentCatalog([agent])
        catalog.delete("sys-prompt")
        assert catalog.get("sys-prompt") is None

    def test_delete_blocked_multiple_agents_reference_template(
        self, catalog: TemplateCatalog
    ) -> None:
        catalog.create(_make_template("sys-prompt"))
        agent1 = _make_agent_with_template_ref("agent-1", "@sys-prompt")
        agent2 = _make_agent_with_template_ref("agent-2", "@sys-prompt")
        catalog.agent_catalog = MockAgentCatalog([agent1, agent2])
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.delete("sys-prompt")
        assert len(exc_info.value.errors) == 2


class TestValidateCreate:
    """validate_create returns list[str], not raises."""

    def test_returns_empty_list_for_unique(self, catalog: TemplateCatalog) -> None:
        entry = _make_template("tpl-1")
        errors = catalog.validate_create(entry)
        assert errors == []

    def test_returns_errors_for_duplicate(self, catalog: TemplateCatalog) -> None:
        entry = _make_template("tpl-1")
        catalog.create(entry)
        errors = catalog.validate_create(entry)
        assert len(errors) == 1
        assert "already exists" in errors[0]


class TestValidateDelete:
    """validate_delete returns list[str] for downstream checks."""

    def test_returns_empty_when_no_refs(self, catalog: TemplateCatalog) -> None:
        catalog.create(_make_template("tpl-1"))
        errors = catalog.validate_delete("tpl-1")
        assert errors == []

    def test_returns_errors_when_referenced(self, catalog: TemplateCatalog) -> None:
        catalog.create(_make_template("sys-prompt"))
        agent = _make_agent_with_template_ref("agent-1", "@sys-prompt")
        catalog.agent_catalog = MockAgentCatalog([agent])
        errors = catalog.validate_delete("sys-prompt")
        assert len(errors) == 1
        assert "cannot delete" in errors[0]

    def test_returns_not_found_error(self, catalog: TemplateCatalog) -> None:
        errors = catalog.validate_delete("nonexistent")
        assert len(errors) == 1
        assert "not found" in errors[0]
