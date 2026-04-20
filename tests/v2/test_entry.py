"""Tests for the v2 ``Entry`` model, ``EntryKind``, and ``AllowlistedPath``."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from akgentic.catalog.models.entry import (
    AllowlistedPath,
    Entry,
    EntryKind,
    NonEmptyStr,
)

from .conftest import make_entry

# Reference the re-exports so the imports are not flagged as unused; also doubles
# as a smoke-level AC1 check that the names resolve to usable objects.
assert NonEmptyStr is not None
assert AllowlistedPath is not None
assert EntryKind is not None  # type: ignore[truthy-function]


class TestEntryImport:
    """AC1 — public importability of the v2 entry-model surface."""

    def test_entry_is_base_model_subclass(self) -> None:
        assert issubclass(Entry, BaseModel)

    def test_entry_kind_alias_has_expected_literal_values(self) -> None:
        # ``Literal`` aliases expose their members via typing.get_args
        from typing import get_args

        assert set(get_args(EntryKind)) == {"team", "agent", "tool", "model", "prompt"}


class TestEntryFields:
    """AC7 — Entry.model_fields exposes exactly the specified fields."""

    def test_field_set_is_exact(self) -> None:
        expected_fields = {
            "id",
            "kind",
            "namespace",
            "user_id",
            "parent_namespace",
            "parent_id",
            "model_type",
            "description",
            "payload",
        }
        assert set(Entry.model_fields.keys()) == expected_fields

    @pytest.mark.parametrize(
        "field_name",
        ["id", "kind", "namespace", "model_type", "payload"],
    )
    def test_required_fields(self, field_name: str) -> None:
        assert Entry.model_fields[field_name].is_required()

    @pytest.mark.parametrize(
        ("field_name", "expected_default"),
        [
            ("user_id", None),
            ("parent_namespace", None),
            ("parent_id", None),
            ("description", ""),
        ],
    )
    def test_optional_field_defaults(self, field_name: str, expected_default: object) -> None:
        field = Entry.model_fields[field_name]
        assert not field.is_required()
        assert field.default == expected_default

    def test_every_field_has_description(self) -> None:
        missing = [name for name, f in Entry.model_fields.items() if not f.description]
        assert not missing, f"fields without description: {missing}"


class TestEntryParentPairValidator:
    """AC8 — lineage pair consistency validator."""

    def test_both_none_is_valid(self) -> None:
        entry = make_entry(parent_namespace=None, parent_id=None)
        assert entry.parent_namespace is None
        assert entry.parent_id is None

    def test_parent_id_without_namespace_is_valid(self) -> None:
        # Same-namespace duplicate case
        entry = make_entry(parent_namespace=None, parent_id="parent-1")
        assert entry.parent_id == "parent-1"

    def test_both_set_is_valid(self) -> None:
        entry = make_entry(parent_namespace="ns-parent", parent_id="parent-1")
        assert entry.parent_namespace == "ns-parent"
        assert entry.parent_id == "parent-1"

    def test_namespace_without_id_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            make_entry(parent_namespace="ns-parent", parent_id=None)
        assert "parent_namespace set but parent_id is None" in str(exc_info.value)


class TestAllowlistedPath:
    """AC9 — ``AllowlistedPath`` prefix enforcement."""

    @pytest.mark.parametrize(
        "path",
        [
            "akgentic.llm.ModelConfig",
            "akgentic.core.agent_card.AgentCard",
            "akgentic.tool.search.SearchTool",
        ],
    )
    def test_accepts_akgentic_prefixed_paths(self, path: str) -> None:
        entry = make_entry(model_type=path)
        assert entry.model_type == path

    @pytest.mark.parametrize(
        "path",
        [
            "os.system",
            "builtins.eval",
            "",
            "akgentic",  # missing trailing dot — prefix check is strict
            "akgenticfake.Impersonator",
        ],
    )
    def test_rejects_non_allowlisted_paths(self, path: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            make_entry(model_type=path)
        assert "outside allowlist" in str(exc_info.value)


class TestNamespaceRequired:
    """The ``namespace`` field rejects None and empty string via ``NonEmptyStr``."""

    def test_none_namespace_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Entry(
                id="entry-1",
                kind="tool",
                namespace=None,  # type: ignore[arg-type]
                model_type="akgentic.core.agent_card.AgentCard",
                payload={},
            )

    def test_empty_namespace_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Entry(
                id="entry-1",
                kind="tool",
                namespace="",
                model_type="akgentic.core.agent_card.AgentCard",
                payload={},
            )
