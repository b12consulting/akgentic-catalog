"""Unit tests for ``akgentic.catalog.serialization`` (Story 16.2)."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from akgentic.catalog.models.entry import Entry
from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.catalog.serialization import (
    _KIND_HEADERS,
    _iter_cross_ns_targets,
    dump_namespace,
    dump_namespace_v2,
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

    def test_rejects_empty_v2_entries_list(self) -> None:
        """Story 17.5: ``entries: []`` selects the v2 shape, then errors as empty bundle."""
        text = "namespace: ns-1\nuser_id: alice\nentries: []\n"
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(text)
        assert exc_info.value.errors == [
            "bundle must declare at least one entry, including a `kind=team` entry"
        ]

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
            after_header = text[header_pos + len(header) :]
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


# --- Story 17.2 — meta entry emit order and round-trip ---------------------


_META_TYPE = "akgentic.catalog.models.namespace_meta.NamespaceMeta"


def _meta(
    namespace: str = "ns-1",
    user_id: str | None = "alice",
    entry_id: str = "_meta",
) -> Entry:
    return Entry(
        id=entry_id,
        kind="meta",
        namespace=namespace,
        user_id=user_id,
        model_type=_META_TYPE,
        description="namespace meta",
        payload={"name": "Tenant", "description": "primary tenant", "properties": {}},
    )


class TestMetaEmitOrderAndRoundTrip:
    """Story 17.2 AC9 / AC10 — meta is emitted between team and agent."""

    def test_meta_section_header_present_between_team_and_agent(self) -> None:
        """When a bundle carries team + meta + agent, the Meta header sits between them."""
        entries = [_team(), _meta(), _agent("a1")]
        text = dump_namespace(entries)
        team_header = _KIND_HEADERS["team"]
        meta_header = _KIND_HEADERS["meta"]
        agent_header = _KIND_HEADERS["agent"]
        team_pos = text.index(team_header)
        meta_pos = text.index(meta_header)
        agent_pos = text.index(agent_header)
        assert team_pos < meta_pos < agent_pos

    def test_meta_entry_round_trips_through_dump_load(self) -> None:
        """``load_namespace(dump_namespace([team, meta, agent]))`` returns equal entries."""
        entries = [_team(), _meta(), _agent("a1")]
        text = dump_namespace(entries)
        recovered = load_namespace(text)
        sort_key = lambda e: (e.kind, e.id)  # noqa: E731
        assert [e.model_dump() for e in sorted(recovered, key=sort_key)] == [
            e.model_dump() for e in sorted(entries, key=sort_key)
        ]

    def test_meta_emit_order_index_one(self) -> None:
        """Bundle with team + meta + agent emits in (team, meta, agent) order."""
        text = dump_namespace([_agent("a1"), _meta(), _team()])
        doc = yaml.safe_load(text)
        # Outer keys are emitted in (kind emit order, id) — team @ 0, meta @ 1, agent @ 2.
        assert list(doc["entries"].keys()) == ["team", "_meta", "a1"]


# --- Story 17.5 — dump_namespace_v2 (entries: + external_refs:) ----------


class TestDumpNamespaceV2:
    """Story 17.5 AC1, AC2, AC4 — v2 wire shape with ``entries:`` + ``external_refs:``."""

    def test_root_has_entries_and_external_refs_in_order(self) -> None:
        """Document root has exactly two keys, ``entries`` first then ``external_refs``."""
        text = dump_namespace_v2([_team(), _agent("a")], [])
        doc = yaml.safe_load(text)
        assert list(doc.keys()) == ["entries", "external_refs"]
        assert isinstance(doc["entries"], list)
        assert isinstance(doc["external_refs"], list)

    def test_external_refs_always_present_when_empty(self) -> None:
        """``external_refs:`` is emitted as ``[]`` when no cross-ns targets exist."""
        text = dump_namespace_v2([_team()], [])
        doc = yaml.safe_load(text)
        assert doc["external_refs"] == []
        assert "external_refs:" in text

    def test_each_entry_carries_id_namespace_user_id(self) -> None:
        """v2 list items are self-contained — id/namespace/user_id sit on each item."""
        text = dump_namespace_v2(
            [_team(namespace="ns-1", user_id="alice"), _agent("a", namespace="ns-1")],
            [],
        )
        doc = yaml.safe_load(text)
        team_item = next(e for e in doc["entries"] if e["id"] == "team")
        assert team_item["namespace"] == "ns-1"
        assert team_item["user_id"] == "alice"
        assert team_item["kind"] == "team"

    def test_external_refs_sorted_by_namespace_kind_id(self) -> None:
        """``external_refs:`` items sort by ``(namespace, kind, id)`` ascending."""
        # Three foreign entries with mixed (namespace, kind, id) tuples.
        externals = [
            _prompt("z-prompt", namespace="alpha", user_id=None),
            _model("a-model", namespace="alpha", user_id=None),
            _agent("a-agent", namespace="zulu", user_id=None),
        ]
        text = dump_namespace_v2([_team()], externals)
        doc = yaml.safe_load(text)
        items = doc["external_refs"]
        keys = [(e["namespace"], e["kind"], e["id"]) for e in items]
        # alpha sorts before zulu; within alpha, model < prompt by string order.
        assert keys == [
            ("alpha", "model", "a-model"),
            ("alpha", "prompt", "z-prompt"),
            ("zulu", "agent", "a-agent"),
        ]

    def test_entries_block_has_section_headers(self) -> None:
        """The legacy section-header comments (Teams / Agents / …) survive in v2 ``entries:``."""
        text = dump_namespace_v2([_team(), _agent("a"), _prompt("p")], [])
        # All three section headers appear inside the entries: block.
        assert _KIND_HEADERS["team"] in text
        assert _KIND_HEADERS["agent"] in text
        assert _KIND_HEADERS["prompt"] in text
        # The headers appear BEFORE external_refs:.
        external_pos = text.index("external_refs:")
        assert text.index(_KIND_HEADERS["team"]) < external_pos
        assert text.index(_KIND_HEADERS["agent"]) < external_pos
        assert text.index(_KIND_HEADERS["prompt"]) < external_pos

    def test_external_refs_block_has_no_section_headers(self) -> None:
        """Section-header comments are reserved for ``entries:`` only."""
        external = _prompt("p1", namespace="global", user_id=None)
        text = dump_namespace_v2([_team()], [external])
        external_pos = text.index("external_refs:")
        external_block = text[external_pos:]
        # No kind-section headers under external_refs:.
        for header in _KIND_HEADERS.values():
            assert header not in external_block

    def test_rejects_empty_entries(self) -> None:
        """``entries: []`` raises with the legacy empty-bundle message."""
        with pytest.raises(CatalogValidationError) as exc_info:
            dump_namespace_v2([], [])
        assert exc_info.value.errors == [
            "bundle must declare at least one entry, including a `kind=team` entry"
        ]

    def test_rejects_mismatched_namespace(self) -> None:
        """Uniform-namespace invariant on ``entries`` is enforced (same as ``dump_namespace``)."""
        entries = [_team(namespace="ns-1"), _agent("a", namespace="ns-2")]
        with pytest.raises(CatalogValidationError) as exc_info:
            dump_namespace_v2(entries, [])
        assert any("namespace" in e for e in exc_info.value.errors)

    def test_two_consecutive_v2_dumps_byte_identical(self) -> None:
        """Repeat dumps with the same inputs produce byte-identical output."""
        entries = [_team(), _agent("a"), _agent("b")]
        externals = [_prompt("p1", namespace="global", user_id=None)]
        first = dump_namespace_v2(entries, externals)
        second = dump_namespace_v2(entries, externals)
        assert first == second


# --- Story 17.5 — load_namespace shape detection ---------------------------


class TestLoadNamespaceShapeDetection:
    """Story 17.5 AC10, AC11 — both shapes accepted on import; new ignores ``external_refs:``."""

    def test_v2_bundle_round_trip(self) -> None:
        """``load_namespace(dump_namespace_v2(entries, []))`` returns the same entries."""
        entries = [_team(), _agent("a"), _agent("b")]
        text = dump_namespace_v2(entries, [])
        recovered = load_namespace(text)
        sort_key = lambda e: (e.kind, e.id)  # noqa: E731
        assert [e.model_dump() for e in sorted(recovered, key=sort_key)] == [
            e.model_dump() for e in sorted(entries, key=sort_key)
        ]

    def test_v2_external_refs_silently_dropped_on_load(self) -> None:
        """Items inside ``external_refs:`` are NOT returned by ``load_namespace``."""
        entries = [_team(), _agent("a")]
        external = _prompt("foreign", namespace="global", user_id=None)
        text = dump_namespace_v2(entries, [external])
        recovered = load_namespace(text)
        # Only the namespace's own entries surface — the foreign prompt is ignored.
        assert {e.id for e in recovered} == {"team", "a"}
        # And every recovered entry's namespace is the bundle's own namespace.
        assert {e.namespace for e in recovered} == {"ns-1"}

    def test_legacy_bundle_round_trip_unchanged(self) -> None:
        """Pre-17.5 flat-list bundles still load — backward compat for on-disk fixtures."""
        # Construct the legacy YAML by hand to simulate an old fixture.
        legacy_text = (
            "namespace: ns-legacy\n"
            "user_id: alice\n"
            "entries:\n"
            "  team:\n"
            "    kind: team\n"
            "    model_type: akgentic.team.models.TeamCard\n"
            "    parent_namespace: null\n"
            "    parent_id: null\n"
            "    description: ''\n"
            "    payload: {name: team}\n"
            "  agent-a:\n"
            "    kind: agent\n"
            "    model_type: akgentic.core.agent_card.AgentCard\n"
            "    parent_namespace: null\n"
            "    parent_id: null\n"
            "    description: ''\n"
            "    payload: {role: r}\n"
        )
        recovered = load_namespace(legacy_text)
        assert {e.id for e in recovered} == {"team", "agent-a"}
        assert {e.namespace for e in recovered} == {"ns-legacy"}
        assert {e.user_id for e in recovered} == {"alice"}

    def test_external_refs_only_document_with_entries_list_loads_v2(self) -> None:
        """A v2 bundle with only ``entries: [...]`` (no namespace/user_id at root) parses."""
        entries = [_team(namespace="ns-vx"), _agent("a", namespace="ns-vx")]
        text = dump_namespace_v2(entries, [])
        # The v2 dump omits document-level namespace/user_id by design.
        doc = yaml.safe_load(text)
        assert "namespace" not in doc
        assert "user_id" not in doc
        # And it loads cleanly.
        recovered = load_namespace(text)
        assert {e.id for e in recovered} == {"team", "a"}

    def test_legacy_bundle_with_entries_dict_keeps_legacy_path(self) -> None:
        """Ambiguous-looking root: ``entries`` is a dict ⇒ legacy path is selected."""
        text = (
            "namespace: ns-l\n"
            "user_id: null\n"
            "entries:\n"
            "  team: {kind: team, model_type: akgentic.team.models.TeamCard, payload: {}}\n"
        )
        recovered = load_namespace(text)
        assert len(recovered) == 1
        assert recovered[0].id == "team"
        assert recovered[0].namespace == "ns-l"

    def test_root_not_mapping_rejected_for_both_shapes(self) -> None:
        """A bare list at the root is rejected with the existing error message."""
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace("- a\n- b\n")
        assert any("mapping" in e for e in exc_info.value.errors)

    def test_v2_per_item_validation_wraps_pydantic_error(self) -> None:
        """A v2 list item missing required fields raises ``entry '<id>' is invalid``."""
        text = (
            "entries:\n"
            "  - id: bad\n"
            "    namespace: ns-1\n"
            "    kind: agent\n"  # missing model_type
            "    payload: {}\n"
            "external_refs: []\n"
        )
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(text)
        assert any("entry 'bad' is invalid" in e for e in exc_info.value.errors)


# --- Story 17.5 — _iter_cross_ns_targets walker ----------------------------


class TestIterCrossNsTargets:
    """Story 17.5 AC13 — the cross-ns walker recognises both ref-marker shapes."""

    def test_canonical_marker_is_collected(self) -> None:
        """``{__ref__: x, __namespace__: ns}`` yields ``(ns, x)``."""
        payload = {"prompt": {"__ref__": "x", "__namespace__": "global"}}
        assert _iter_cross_ns_targets(payload) == [("global", "x")]

    def test_shorthand_marker_is_collected(self) -> None:
        """``{__ref__: ns.x}`` yields ``(ns, x)`` via first-dot split."""
        payload = {"prompt": {"__ref__": "global.x"}}
        assert _iter_cross_ns_targets(payload) == [("global", "x")]

    def test_same_namespace_marker_is_omitted(self) -> None:
        """``{__ref__: x}`` (no dot, no ``__namespace__``) yields nothing."""
        payload = {"prompt": {"__ref__": "x"}}
        assert _iter_cross_ns_targets(payload) == []

    def test_first_dot_split_for_shorthand_with_dotted_id(self) -> None:
        """Shorthand splits on the FIRST dot — ids may continue to contain dots."""
        payload = {"ref": {"__ref__": "ns.id.with.dots"}}
        assert _iter_cross_ns_targets(payload) == [("ns", "id.with.dots")]

    def test_walker_recurses_into_lists(self) -> None:
        """List items are visited."""
        payload = {
            "tools": [
                {"__ref__": "global.t1"},
                {"__ref__": "default.t2"},
            ]
        }
        result = _iter_cross_ns_targets(payload)
        assert ("global", "t1") in result
        assert ("default", "t2") in result

    def test_walker_recurses_into_sibling_overrides(self) -> None:
        """Sibling-override sub-payloads on a ref marker are walked for nested cross-ns refs."""
        # The marker itself targets (global, agent-x); its override carries
        # another cross-ns ref to (default, prompt-y) — both must be yielded.
        payload = {
            "agent": {
                "__ref__": "global.agent-x",
                "params": {"prompt": {"__ref__": "default.prompt-y"}},
            }
        }
        result = _iter_cross_ns_targets(payload)
        assert ("global", "agent-x") in result
        assert ("default", "prompt-y") in result

    def test_dedup_is_caller_responsibility(self) -> None:
        """The walker emits duplicates; dedup happens in ``_collect_external_refs``."""
        payload = {
            "a": {"__ref__": "global.x"},
            "b": {"__ref__": "global.x"},
        }
        result = _iter_cross_ns_targets(payload)
        assert result.count(("global", "x")) == 2

    def test_walker_is_parse_only_no_repository_access(self) -> None:
        """The walker accepts an arbitrary payload tree — no Pydantic, no repo lookups."""
        # An exotic but legal payload tree — primitives, nested dicts/lists.
        payload = {
            "level1": {
                "level2": [
                    1,
                    "string",
                    None,
                    {"__ref__": "global.deep"},
                ]
            }
        }
        # Does not raise.
        result = _iter_cross_ns_targets(payload)
        assert result == [("global", "deep")]
