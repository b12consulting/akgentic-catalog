"""Tests for environment variable substitution utility."""

import pytest

from akgentic.catalog.env import resolve_env_vars
from akgentic.catalog.models.tool import ToolEntry
from akgentic.catalog.services.tool_catalog import ToolCatalog
from tests.test_tool_catalog import InMemoryToolCatalogRepository


class TestResolveEnvVars:
    def test_replaces_single_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_VAR", "hello")
        assert resolve_env_vars("${MY_VAR}") == "hello"

    def test_replaces_multiple_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "8080")
        assert resolve_env_vars("${HOST}:${PORT}") == "localhost:8080"

    def test_preserves_text_around_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NAME", "world")
        assert resolve_env_vars("hello ${NAME}!") == "hello world!"

    def test_no_vars_returns_unchanged(self) -> None:
        assert resolve_env_vars("plain text") == "plain text"

    def test_empty_string_returns_empty(self) -> None:
        assert resolve_env_vars("") == ""

    def test_raises_on_missing_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NONEXISTENT_VAR_12345", raising=False)
        with pytest.raises(OSError, match="NONEXISTENT_VAR_12345"):
            resolve_env_vars("${NONEXISTENT_VAR_12345}")

    def test_underscore_in_var_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_VAR_2", "value")
        assert resolve_env_vars("${MY_VAR_2}") == "value"

    def test_ignores_invalid_var_syntax(self) -> None:
        # $VAR without braces should not be substituted
        assert resolve_env_vars("$VAR") == "$VAR"

    def test_ignores_empty_braces(self) -> None:
        # ${} should not match the pattern (requires valid identifier)
        assert resolve_env_vars("${}") == "${}"

    def test_var_starting_with_digit_not_matched(self) -> None:
        # ${1VAR} is not a valid identifier per the regex
        assert resolve_env_vars("${1VAR}") == "${1VAR}"


class TestCatalogStoresPlaceholdersAsIs:
    """AC5: Catalog stores ${VAR} placeholders as-is, not resolved at persistence time."""

    def test_tool_entry_with_env_var_placeholder_persists_unchanged(self) -> None:
        """Create a ToolEntry with ${VAR} in a string field, persist via repo, verify unchanged."""
        entry = ToolEntry(
            id="figma-tool",
            tool_class="akgentic.tool.search.search.SearchTool",
            tool={"name": "Figma", "description": "Figma API with key ${FIGMA_API_KEY}"},
        )

        repo = InMemoryToolCatalogRepository()
        catalog = ToolCatalog(repository=repo)
        catalog.create(entry)

        retrieved = catalog.get("figma-tool")
        assert retrieved is not None
        assert "${FIGMA_API_KEY}" in retrieved.tool.description
