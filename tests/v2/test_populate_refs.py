"""Tests for ``akgentic.catalog.resolver.populate_refs`` — AC3 through AC12."""

from __future__ import annotations

from typing import Any

import pytest

from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.catalog.resolver import populate_refs

from .conftest import FakeEntryRepository, make_entry


class TestLeafPassthrough:
    """AC4 — non-dict / non-list leaves pass through unchanged."""

    @pytest.mark.parametrize(
        "leaf",
        ["text", 42, 3.14, True, False, None],
    )
    def test_scalar_leaves_identity(self, leaf: Any) -> None:
        repo = FakeEntryRepository()
        result = populate_refs(leaf, repo, "ns-1")
        assert result == leaf
        assert type(result) is type(leaf)

    def test_empty_dict(self) -> None:
        repo = FakeEntryRepository()
        result = populate_refs({}, repo, "ns-1")
        assert result == {}

    def test_empty_list(self) -> None:
        repo = FakeEntryRepository()
        result = populate_refs([], repo, "ns-1")
        assert result == []


class TestDictRecursion:
    """AC5 — plain dicts recurse into values and build a fresh container."""

    def test_plain_dict_unchanged_values(self) -> None:
        repo = FakeEntryRepository()
        node = {"provider": "openai", "model": "gpt-4.1"}
        result = populate_refs(node, repo, "ns-1")
        assert result == node
        # Recursion builds a fresh dict — input is not mutated.
        node["provider"] = "mutated"
        assert result["provider"] == "openai"

    def test_nested_dicts_recurse(self) -> None:
        repo = FakeEntryRepository()
        node = {"outer": {"inner": {"deep": 1}}}
        result = populate_refs(node, repo, "ns-1")
        assert result == {"outer": {"inner": {"deep": 1}}}
        # Distinct containers at each level.
        assert result is not node
        assert result["outer"] is not node["outer"]


class TestListRecursion:
    """AC6 — lists recurse element-wise and build a fresh container."""

    def test_mixed_list(self) -> None:
        repo = FakeEntryRepository()
        repo.put(make_entry(id="x", namespace="ns-1", payload={"k": 1}))
        node = [{"__ref__": "x"}, "literal", 42]
        result = populate_refs(node, repo, "ns-1")
        assert result == [{"k": 1}, "literal", 42]

    def test_nested_lists(self) -> None:
        repo = FakeEntryRepository()
        node: Any = [[1, 2], [3, [4, 5]]]
        result = populate_refs(node, repo, "ns-1")
        assert result == [[1, 2], [3, [4, 5]]]
        assert result is not node


class TestRefReplacement:
    """AC7 — ref dict replaced by target payload (recursive)."""

    def test_flat_ref_replaced(self) -> None:
        repo = FakeEntryRepository()
        repo.put(
            make_entry(
                id="target-id",
                namespace="ns-1",
                payload={"provider": "openai", "model": "gpt-4.1"},
            )
        )
        result = populate_refs({"__ref__": "target-id"}, repo, "ns-1")
        assert result == {"provider": "openai", "model": "gpt-4.1"}

    def test_nested_ref_in_target_resolves_recursively(self) -> None:
        repo = FakeEntryRepository()
        repo.put(make_entry(id="leaf", namespace="ns-1", payload={"value": 7}))
        repo.put(
            make_entry(
                id="middle",
                namespace="ns-1",
                payload={"child": {"__ref__": "leaf"}},
            )
        )
        result = populate_refs({"__ref__": "middle"}, repo, "ns-1")
        assert result == {"child": {"value": 7}}


class TestTypeMismatch:
    """AC8 — ``__type__`` mismatch raises ``CatalogValidationError``."""

    def test_type_mismatch_raises(self) -> None:
        repo = FakeEntryRepository()
        repo.put(
            make_entry(
                id="target-id",
                namespace="ns-1",
                model_type="akgentic.llm.OtherConfig",
                payload={},
            )
        )
        marker = {"__ref__": "target-id", "__type__": "akgentic.llm.ModelConfig"}
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs(marker, repo, "ns-1")
        assert len(exc_info.value.errors) == 1
        msg = exc_info.value.errors[0]
        assert "expected akgentic.llm.ModelConfig" in msg
        assert "got akgentic.llm.OtherConfig" in msg
        assert "target-id" in msg

    def test_type_match_succeeds(self) -> None:
        repo = FakeEntryRepository()
        repo.put(
            make_entry(
                id="target-id",
                namespace="ns-1",
                model_type="akgentic.llm.ModelConfig",
                payload={"provider": "openai"},
            )
        )
        marker = {"__ref__": "target-id", "__type__": "akgentic.llm.ModelConfig"}
        result = populate_refs(marker, repo, "ns-1")
        assert result == {"provider": "openai"}

    def test_absent_type_skips_check(self) -> None:
        """Match-all case: no ``__type__`` in marker → no type check runs."""
        repo = FakeEntryRepository()
        repo.put(
            make_entry(
                id="target-id",
                namespace="ns-1",
                model_type="akgentic.llm.AnyConfig",
                payload={"k": 1},
            )
        )
        result = populate_refs({"__ref__": "target-id"}, repo, "ns-1")
        assert result == {"k": 1}


class TestMissingTarget:
    """AC9 — missing target raises ``CatalogValidationError``."""

    def test_missing_target_raises(self) -> None:
        repo = FakeEntryRepository()
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs({"__ref__": "missing-id"}, repo, "ns-1")
        assert len(exc_info.value.errors) == 1
        msg = exc_info.value.errors[0]
        assert "not found" in msg
        assert "missing-id" in msg
        assert "ns-1" in msg


class TestCycleDetection:
    """AC10 — ref cycles raise ``CatalogValidationError``."""

    def test_three_hop_cycle_raises(self) -> None:
        repo = FakeEntryRepository()
        repo.put(make_entry(id="A", namespace="ns-1", payload={"x": {"__ref__": "B"}}))
        repo.put(make_entry(id="B", namespace="ns-1", payload={"y": {"__ref__": "C"}}))
        repo.put(make_entry(id="C", namespace="ns-1", payload={"z": {"__ref__": "A"}}))
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs({"__ref__": "A"}, repo, "ns-1")
        msg = exc_info.value.errors[0].lower()
        assert "cycle" in msg
        assert "a" in msg  # id of the entry closing the cycle
        assert "ns-1" in msg

    def test_self_cycle_raises(self) -> None:
        repo = FakeEntryRepository()
        repo.put(make_entry(id="A", namespace="ns-1", payload={"self": {"__ref__": "A"}}))
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs({"__ref__": "A"}, repo, "ns-1")
        msg = exc_info.value.errors[0].lower()
        assert "cycle" in msg
        assert "a" in msg
        assert "ns-1" in msg

    def test_one_hop_ref_does_not_raise(self) -> None:
        repo = FakeEntryRepository()
        repo.put(make_entry(id="A", namespace="ns-1", payload={"v": 1}))
        result = populate_refs({"__ref__": "A"}, repo, "ns-1")
        assert result == {"v": 1}


class TestVisitingNotShared:
    """AC11 — ``_visiting`` is per-ref-chain, not global across siblings."""

    def test_same_target_on_two_siblings_is_not_a_cycle(self) -> None:
        repo = FakeEntryRepository()
        repo.put(make_entry(id="shared", namespace="ns-1", payload={"provider": "openai"}))
        node = {"left": {"__ref__": "shared"}, "right": {"__ref__": "shared"}}
        result = populate_refs(node, repo, "ns-1")
        assert result == {
            "left": {"provider": "openai"},
            "right": {"provider": "openai"},
        }

    def test_caller_visiting_set_not_mutated(self) -> None:
        """A fresh set passed by a caller is never mutated by populate_refs."""
        repo = FakeEntryRepository()
        repo.put(make_entry(id="A", namespace="ns-1", payload={"k": 1}))
        caller_set: set[tuple[str, str]] = set()
        populate_refs({"__ref__": "A"}, repo, "ns-1", caller_set)
        assert caller_set == set()


class TestNamespaceForwarding:
    """AC12 — namespace is forwarded unchanged; never derived from target."""

    def test_namespace_forwarded_through_nested_refs(self) -> None:
        """The recursion must keep calling repo.get with the original namespace."""
        repo = FakeEntryRepository()
        repo.put(make_entry(id="A", namespace="ns-alpha", payload={"v": {"__ref__": "B"}}))
        repo.put(make_entry(id="B", namespace="ns-alpha", payload={"leaf": 1}))
        result = populate_refs({"__ref__": "A"}, repo, "ns-alpha")
        assert result == {"v": {"leaf": 1}}

    def test_target_in_other_namespace_invisible(self) -> None:
        """Cross-namespace resolution is structurally impossible."""
        repo = FakeEntryRepository()
        repo.put(make_entry(id="A", namespace="ns-beta", payload={"v": 1}))
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs({"__ref__": "A"}, repo, "ns-alpha")
        assert "not found" in exc_info.value.errors[0]
        assert "ns-alpha" in exc_info.value.errors[0]
