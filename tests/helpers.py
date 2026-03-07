"""Shared test fixtures for catalog service tests."""

import builtins

from akgentic.catalog.models.agent import AgentEntry

_list = builtins.list


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
