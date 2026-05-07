"""Unit tests for ``akgentic.catalog.serialization`` (Story 16.2)."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from akgentic.catalog.models.entry import Entry
from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.catalog.serialization import (
    _KIND_HEADERS,
    dump_namespace,
    load_namespace,
)

_TEAM_TYPE = "akgentic.team.models.TeamCard"
_AGENT_TYPE = "akgentic.core.agent_card.AgentCard"
_PROMPT_TYPE = "akgentic.llm.prompts.PromptTemplate"
_TOOL_TYPE = "akgentic.tool.tool_card.ToolCard"
_MODEL_TYPE = "akgentic.llm.model_config.ModelConfig"


def _team(namespace: str = "ns-1", user_id: str | None = "alice") -> Entry:
    return Entry(
        id="team",
        kind="team",
        namespace=namespace,
        user_id=user_id,
        model_type=_TEAM_TYPE,
        payload={"name": "team"},
    )


def _agent(
    id: str,
    namespace: str = "ns-1",
    user_id: str | None = "alice",
    payload: dict[str, Any] | None = None,
) -> Entry:
    return Entry(
        id=id,
        kind="agent",
        namespace=namespace,
        user_id=user_id,
        model_type=_AGENT_TYPE,
        payload=payload if payload is not None else {"role": id},
    )


def _prompt(
    id: str,
    namespace: str = "ns-1",
    user_id: str | None = "alice",
) -> Entry:
    return Entry(
        id=id,
        kind="prompt",
        namespace=namespace,
        user_id=user_id,
        model_type=_PROMPT_TYPE,
        payload={"template": id},
    )


def _tool(
    id: str,
    namespace: str = "ns-1",
    user_id: str | None = "alice",
) -> Entry:
    return Entry(
        id=id,
        kind="tool",
        namespace=namespace,
        user_id=user_id,
        model_type=_TOOL_TYPE,
        payload={"name": id},
    )


def _model(
    id: str,
    namespace: str = "ns-1",
    user_id: str | None = "alice",
) -> Entry:
    return Entry(
        id=id,
        kind="model",
        namespace=namespace,
        user_id=user_id,
        model_type=_MODEL_TYPE,
        payload={"provider": "openai"},
    )


# --- dump_namespace ---------------------------------------------------------


class TestDumpNamespace:
    def test_round_trip(self) -> None:
        entries = [_team(), _agent("b"), _agent("a"), _agent("c")]
        text = dump_namespace(entries)
        parsed = load_namespace(text)
        # dump reorders (team first, non-team sorted); round-trip input-order is
        # the dumped order. Compare after reordering the input to match.
        expected_order = [_team(), _agent("a"), _agent("b"), _agent("c")]
        assert [e.model_dump() for e in parsed] == [e.model_dump() for e in expected_order]

    def test_root_keys_and_order(self) -> None:
        text = dump_namespace([_team(), _agent("a")])
        doc = yaml.safe_load(text)
        assert list(doc.keys()) == ["namespace", "user_id", "entries"]
        assert doc["namespace"] == "ns-1"
        assert doc["user_id"] == "alice"

    def test_enterprise_user_id_is_null(self) -> None:
        text = dump_namespace([_team(user_id=None), _agent("a", user_id=None)])
        doc = yaml.safe_load(text)
        assert doc["user_id"] is None
        # YAML null, not the literal string "null".
        assert "user_id: null" in text

    def test_entry_keys_and_order(self) -> None:
        text = dump_namespace([_team(), _agent("a")])
        doc = yaml.safe_load(text)
        agent_map = doc["entries"]["a"]
        assert list(agent_map.keys()) == [
            "kind",
            "model_type",
            "parent_namespace",
            "parent_id",
            "description",
            "payload",
        ]
        # id / namespace / user_id must NOT be duplicated inside the per-entry map.
        assert "id" not in agent_map
        assert "namespace" not in agent_map
        assert "user_id" not in agent_map

    def test_team_first_then_sorted(self) -> None:
        entries = [_agent("c"), _team(), _agent("a"), _agent("b")]
        text = dump_namespace(entries)
        doc = yaml.safe_load(text)
        assert list(doc["entries"].keys()) == ["team", "a", "b", "c"]

    def test_emit_order_groups_by_kind_then_id(self) -> None:
        """Entries emit in kind order (team, agent, prompt, tool, model) then id."""
        prompt_a = Entry(
            id="prompt_a",
            kind="prompt",
            namespace="ns-1",
            user_id="alice",
            model_type="akgentic.llm.prompts.PromptTemplate",
            payload={},
        )
        tool_a = Entry(
            id="tool_a",
            kind="tool",
            namespace="ns-1",
            user_id="alice",
            model_type="akgentic.tool.tool_card.ToolCard",
            payload={},
        )
        model_b = Entry(
            id="model_b",
            kind="model",
            namespace="ns-1",
            user_id="alice",
            model_type="akgentic.llm.model_config.ModelConfig",
            payload={},
        )
        model_a = Entry(
            id="model_a",
            kind="model",
            namespace="ns-1",
            user_id="alice",
            model_type="akgentic.llm.model_config.ModelConfig",
            payload={},
        )
        # Input order scrambled on purpose; two models verify intra-kind id sub-sort.
        entries = [model_b, tool_a, _agent("zulu"), prompt_a, model_a, _agent("alpha"), _team()]
        text = dump_namespace(entries)
        doc = yaml.safe_load(text)
        assert list(doc["entries"].keys()) == [
            "team",
            "alpha",
            "zulu",
            "prompt_a",
            "tool_a",
            "model_a",
            "model_b",
        ]

    def test_rejects_empty_list(self) -> None:
        with pytest.raises(CatalogValidationError) as exc_info:
            dump_namespace([])
        assert exc_info.value.errors == [
            "bundle must declare at least one entry, including a `kind=team` entry"
        ]

    def test_rejects_mismatched_user_id(self) -> None:
        entries = [
            _team(user_id="alice"),
            _agent("a", user_id="bob"),
            _agent("b", user_id="carol"),
        ]
        with pytest.raises(CatalogValidationError) as exc_info:
            dump_namespace(entries)
        errors = exc_info.value.errors
        assert len(errors) == 2
        assert "entry 'a'" in errors[0]
        assert "entry 'b'" in errors[1]

    def test_rejects_mismatched_namespace(self) -> None:
        entries = [
            _team(namespace="ns-1"),
            _agent("a", namespace="ns-2"),
        ]
        with pytest.raises(CatalogValidationError) as exc_info:
            dump_namespace(entries)
        errors = exc_info.value.errors
        assert any("entry 'a'" in e and "namespace" in e for e in errors)

    def test_preserves_ref_markers(self) -> None:
        ref_payload = {
            "prompt": {"__ref__": "p1", "__type__": "akgentic.llm.prompts.PromptTemplate"}
        }
        entries = [_team(), _agent("a", payload=ref_payload)]
        text = dump_namespace(entries)
        parsed = load_namespace(text)
        round_tripped = next(e for e in parsed if e.id == "a")
        assert round_tripped.payload == ref_payload


# --- load_namespace ---------------------------------------------------------


class TestLoadNamespace:
    def test_rejects_malformed_yaml(self) -> None:
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace("{{{ not yaml }")
        assert "Failed to parse bundle YAML" in exc_info.value.errors[0]

    def test_rejects_missing_root_keys(self) -> None:
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace("foo: bar\n")
        errors = exc_info.value.errors
        assert any("namespace" in e for e in errors)
        assert any("user_id" in e for e in errors)
        assert any("entries" in e for e in errors)

    def test_rejects_empty_entries(self) -> None:
        text = "namespace: ns-1\nuser_id: alice\nentries: {}\n"
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(text)
        assert exc_info.value.errors == [
            "bundle must declare at least one entry, including a `kind=team` entry"
        ]

    def test_rejects_entries_as_list(self) -> None:
        text = "namespace: ns-1\nuser_id: alice\nentries: []\n"
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(text)
        assert any("entries" in e and "mapping" in e for e in exc_info.value.errors)

    def test_rejects_namespace_empty(self) -> None:
        text = "namespace: ''\nuser_id: alice\nentries:\n  team: {kind: team}\n"
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(text)
        assert any("namespace" in e and "non-empty" in e for e in exc_info.value.errors)

    def test_rejects_user_id_wrong_type(self) -> None:
        text = (
            "namespace: ns-1\n"
            "user_id: 42\n"
            "entries:\n  team: {kind: team, model_type: akgentic.team.models.TeamCard}\n"
        )
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(text)
        assert any("user_id" in e and "string or null" in e for e in exc_info.value.errors)

    def test_rejects_root_not_mapping(self) -> None:
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace("- a\n- b\n")
        assert any("mapping" in e for e in exc_info.value.errors)

    def test_wraps_per_entry_validation(self) -> None:
        # Missing model_type — Pydantic ValidationError surfaces as CatalogValidationError.
        text = "namespace: ns-1\nuser_id: alice\nentries:\n  a:\n    kind: agent\n    payload: {}\n"
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(text)
        assert any("entry 'a' is invalid" in e for e in exc_info.value.errors)

    def test_rejects_entry_map_not_mapping(self) -> None:
        text = "namespace: ns-1\nuser_id: alice\nentries:\n  a: 42\n"
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(text)
        assert any("entry 'a' is invalid" in e for e in exc_info.value.errors)

    def test_preserves_dict_iteration_order(self) -> None:
        text = (
            "namespace: ns-1\n"
            "user_id: alice\n"
            "entries:\n"
            "  team:\n"
            "    kind: team\n"
            "    model_type: akgentic.team.models.TeamCard\n"
            "    payload: {}\n"
            "  zulu:\n"
            "    kind: agent\n"
            "    model_type: akgentic.core.agent_card.AgentCard\n"
            "    payload: {}\n"
            "  alpha:\n"
            "    kind: agent\n"
            "    model_type: akgentic.core.agent_card.AgentCard\n"
            "    payload: {}\n"
        )
        parsed = load_namespace(text)
        assert [e.id for e in parsed] == ["team", "zulu", "alpha"]


# --- section headers and spacing (Story 16.8) --------------------------------


class TestSectionHeadersAndSpacing:
    def test_emits_header_per_non_empty_kind(self) -> None:
        """Bundle with one entry of each kind produces all five section headers."""
        entries = [
            _team(),
            _agent("a1"),
            _prompt("p1"),
            _tool("t1"),
            _model("m1"),
        ]
        text = dump_namespace(entries)
        assert _KIND_HEADERS["team"] in text
        assert _KIND_HEADERS["agent"] in text
        assert _KIND_HEADERS["prompt"] in text
        assert _KIND_HEADERS["tool"] in text
        assert _KIND_HEADERS["model"] in text

    def test_omits_header_for_absent_kind(self) -> None:
        """Bundle with only a team entry produces only the Teams header."""
        text = dump_namespace([_team()])
        assert _KIND_HEADERS["team"] in text
        assert _KIND_HEADERS["agent"] not in text
        assert _KIND_HEADERS["prompt"] not in text
        assert _KIND_HEADERS["tool"] not in text
        assert _KIND_HEADERS["model"] not in text

    def test_header_order_matches_kind_emit_order(self) -> None:
        """Teams header appears before Agents, Agents before Prompts, etc."""
        entries = [
            _team(),
            _agent("a1"),
            _prompt("p1"),
            _tool("t1"),
            _model("m1"),
        ]
        text = dump_namespace(entries)
        teams_pos = text.index(_KIND_HEADERS["team"])
        agents_pos = text.index(_KIND_HEADERS["agent"])
        prompts_pos = text.index(_KIND_HEADERS["prompt"])
        tools_pos = text.index(_KIND_HEADERS["tool"])
        models_pos = text.index(_KIND_HEADERS["model"])
        assert teams_pos < agents_pos < prompts_pos < tools_pos < models_pos

    def test_same_kind_entries_separated_by_blank_line(self) -> None:
        """Three agents produce exactly two blank-line separators within the agent section."""
        entries = [_agent("a1"), _agent("a2"), _agent("a3"), _team()]
        text = dump_namespace(entries)
        agents_header = _KIND_HEADERS["agent"]
        agent_section_start = text.index(agents_header)
        agent_section = text[agent_section_start:]
        double_newlines = agent_section.count("\n\n")
        assert double_newlines == 2

    def test_no_blank_line_immediately_after_header(self) -> None:
        """The first entry of a kind immediately follows the header — no blank line in between."""
        entries = [_team(), _agent("a1"), _agent("a2")]
        text = dump_namespace(entries)
        for kind in ("team", "agent"):
            header = _KIND_HEADERS[kind]
            header_pos = text.index(header)
            after_header = text[header_pos + len(header):]
            # Exactly one newline ends the header line; no empty line before the first entry key.
            assert after_header.startswith("\n  "), (
                f"Expected header for {kind!r} to be followed immediately by an entry key line, "
                f"got: {after_header[:40]!r}"
            )

    def test_no_blank_line_at_end_of_final_section(self) -> None:
        """The output ends with exactly one trailing newline, no double newline at EOF."""
        entries = [_team(), _agent("a1")]
        text = dump_namespace(entries)
        assert text.endswith("\n")
        assert not text.endswith("\n\n")

    def test_round_trip_through_safe_load_preserves_values(self) -> None:
        """load_namespace(dump_namespace(entries)) returns structurally equal list.

        Covers: (a) single-team-only, (b) one of each kind, (c) multi-entry per kind
        (3 agents, 3 prompts, 3 tools, 1 model), (d) enterprise bundle with user_id=None.
        """
        entries = [
            _team(),
            _agent("agent_a"),
            _agent("agent_b"),
            _agent("agent_c"),
            _prompt("prompt_x"),
            _prompt("prompt_y"),
            _prompt("prompt_z"),
            _tool("tool_p"),
            _tool("tool_q"),
            _tool("tool_r"),
            _model("model_1"),
        ]
        text = dump_namespace(entries)
        recovered = load_namespace(text)
        sort_key = lambda e: (e.kind, e.id)  # noqa: E731
        assert [e.model_dump() for e in sorted(recovered, key=sort_key)] == [
            e.model_dump() for e in sorted(entries, key=sort_key)
        ]
        parsed_doc = yaml.safe_load(text)
        assert set(parsed_doc["entries"].keys()) == {e.id for e in entries}

    def test_round_trip_enterprise_bundle(self) -> None:
        """Enterprise bundle (user_id=None) round-trips correctly through dump/load."""
        entries = [
            _team(user_id=None),
            _agent("a1", user_id=None),
            _prompt("p1", user_id=None),
        ]
        text = dump_namespace(entries)
        recovered = load_namespace(text)
        sort_key = lambda e: (e.kind, e.id)  # noqa: E731
        assert [e.model_dump() for e in sorted(recovered, key=sort_key)] == [
            e.model_dump() for e in sorted(entries, key=sort_key)
        ]
        doc = yaml.safe_load(text)
        assert doc["user_id"] is None
