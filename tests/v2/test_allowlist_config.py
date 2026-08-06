"""Tests for the deployment-configurable ``model_type`` allowlist.

Covers the shared source of truth (``akgentic.catalog._allowlist``) and both
layers that read it: the storage-side ``Entry`` annotation check and the
runtime ``load_model_type`` resolver gate. Configuration comes from the
``AKGENTIC_CATALOG_ALLOWED_PREFIXES`` environment variable; ``akgentic.`` is
always allowed regardless of configuration.
"""

from __future__ import annotations

import sys
import types

import pytest
from pydantic import BaseModel, ValidationError

from akgentic.catalog._allowlist import (
    ALLOWLIST_ENV_VAR,
    ALWAYS_ALLOWED_PREFIX,
    allowed_prefixes,
)
from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.catalog.resolver import load_model_type

from .conftest import make_entry


class _AppModel(BaseModel):
    """Stand-in for an application-owned config class outside ``akgentic.*``."""

    value: str = "x"


def _register_module(
    monkeypatch: pytest.MonkeyPatch, module_name: str, **attributes: object
) -> str:
    """Register a throwaway module under an arbitrary (non-akgentic) name."""
    module = types.ModuleType(module_name)
    for name, value in attributes.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module_name


class TestAllowedPrefixes:
    """The shared ``allowed_prefixes`` source of truth."""

    def test_default_is_akgentic_only(self) -> None:
        assert allowed_prefixes({}) == (ALWAYS_ALLOWED_PREFIX,)

    def test_env_adds_prefix_with_akgentic_first(self) -> None:
        result = allowed_prefixes({ALLOWLIST_ENV_VAR: "myapp."})
        assert result == ("akgentic.", "myapp.")

    def test_multiple_prefixes_parsed_and_trimmed(self) -> None:
        result = allowed_prefixes({ALLOWLIST_ENV_VAR: " myapp. , other.pkg. "})
        assert result == ("akgentic.", "myapp.", "other.pkg.")

    def test_akgentic_forced_even_if_omitted(self) -> None:
        result = allowed_prefixes({ALLOWLIST_ENV_VAR: "myapp."})
        assert result[0] == ALWAYS_ALLOWED_PREFIX

    def test_duplicate_and_redundant_akgentic_deduped(self) -> None:
        result = allowed_prefixes({ALLOWLIST_ENV_VAR: "akgentic.,myapp.,myapp."})
        assert result == ("akgentic.", "myapp.")

    def test_empty_and_whitespace_entries_ignored(self) -> None:
        result = allowed_prefixes({ALLOWLIST_ENV_VAR: ",  ,myapp.,"})
        assert result == ("akgentic.", "myapp.")


class TestEntryAllowlistConfigurable:
    """The storage-side ``Entry.model_type`` check honours the env var."""

    def test_app_prefix_rejected_by_default(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            make_entry(model_type="myapp.tools.MyToolCard")
        assert "outside allowlist" in str(exc_info.value)

    def test_app_prefix_accepted_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ALLOWLIST_ENV_VAR, "myapp.")
        entry = make_entry(model_type="myapp.tools.MyToolCard")
        assert entry.model_type == "myapp.tools.MyToolCard"

    def test_akgentic_still_accepted_when_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ALLOWLIST_ENV_VAR, "myapp.")
        entry = make_entry(model_type="akgentic.llm.ModelConfig")
        assert entry.model_type == "akgentic.llm.ModelConfig"


class TestLoadModelTypeAllowlistConfigurable:
    """The runtime ``load_model_type`` gate honours the same env var."""

    def test_app_class_rejected_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module_name = _register_module(
            monkeypatch, "myapp_fixture_default", AppModel=_AppModel
        )
        with pytest.raises(CatalogValidationError) as exc_info:
            load_model_type(f"{module_name}.AppModel")
        assert "outside allowlist" in exc_info.value.errors[0]

    def test_app_class_loaded_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module_name = _register_module(
            monkeypatch, "myapp_fixture_ok", AppModel=_AppModel
        )
        monkeypatch.setenv(ALLOWLIST_ENV_VAR, f"{module_name}.")
        assert load_model_type(f"{module_name}.AppModel") is _AppModel
