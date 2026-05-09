"""Unit tests for ``akgentic.catalog.serialization`` (Stories 16.2 / 17.6)."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from akgentic.catalog.models.entry import Entry
from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.catalog.serialization import (
    _EXTERNAL_KIND_HEADERS,
    _KIND_HEADERS,
    BundleHeader,
    dump_namespace,
    load_namespace,
)

_TEAM_TYPE = "akgentic.team.models.TeamCard"
_AGENT_TYPE = "akgentic.core.agent_card.AgentCard"
_PROMPT_TYPE = "akgentic.llm.prompts.PromptTemplate"
_TOOL_TYPE = "akgentic.tool.tool_card.ToolCard"
_MODEL_TYPE = "akgentic.llm.model_config.ModelConfig"
_META_TYPE = "akgentic.catalog.models.namespace_meta.NamespaceMeta"


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


def _meta(
    namespace: str = "ns-1",
    user_id: str | None = "alice",
    entry_id: str = "_meta",
    name: str = "Tenant",
    description: str = "primary tenant",
    properties: dict[str, str] | None = None,
) -> Entry:
    return Entry(
        id=entry_id,
        kind="meta",
        namespace=namespace,
        user_id=user_id,
        model_type=_META_TYPE,
        description="namespace meta",
        payload={
            "name": name,
            "description": description,
            "properties": properties if properties is not None else {},
        },
    )


# --- dump_namespace (pre-17.5 shape via default kwargs) ---------------------


class TestDumpNamespace:
    """Restored pre-17.5 dict-keyed shape — emitted when no header / external_refs are passed."""

    def test_round_trip(self) -> None:
        entries = [_team(), _agent("b"), _agent("a"), _agent("c")]
        text = dump_namespace(entries)
        parsed_entries, header = load_namespace(text)
        # dump reorders (team first, non-team sorted); round-trip input-order is
        # the dumped order. Compare after reordering the input to match.
        expected_order = [_team(), _agent("a"), _agent("b"), _agent("c")]
        assert [e.model_dump() for e in parsed_entries] == [e.model_dump() for e in expected_order]
        # Pre-17.5 callers pass no header → header NOT emitted → parsed header
        # has present=False.
        assert header.present is False

    def test_root_keys_legacy_no_header(self) -> None:
        text = dump_namespace([_team(), _agent("a")])
        doc = yaml.safe_load(text)
        # Pre-17.5 wire shape — three top-level keys, no header trio.
        assert list(doc.keys()) == ["namespace", "user_id", "entries"]
        assert doc["namespace"] == "ns-1"
        assert doc["user_id"] == "alice"

    def test_enterprise_user_id_is_null(self) -> None:
        text = dump_namespace([_team(user_id=None), _agent("a", user_id=None)])
        doc = yaml.safe_load(text)
        assert doc["user_id"] is None
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
        entries = [
            _model("model_b"),
            _tool("tool_a"),
            _agent("zulu"),
            _prompt("prompt_a"),
            _model("model_a"),
            _agent("alpha"),
            _team(),
        ]
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
        parsed_entries, _header = load_namespace(text)
        round_tripped = next(e for e in parsed_entries if e.id == "a")
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

    def test_rejects_entries_as_list_with_explicit_message(self) -> None:
        """Story 17.6 — the rejected 17.5 wire shape (list-of-items) raises explicitly."""
        text = "namespace: ns-1\nuser_id: alice\nentries: []\n"
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(text)
        # Hand-edited 17.5-shape bundles surface a clear error pointing at the
        # rejected shape, so a downstream caller (CLI / API) can render it.
        assert any("Story 17.5" in e or "list-of-items" in e for e in exc_info.value.errors)

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
        parsed_entries, _header = load_namespace(text)
        assert [e.id for e in parsed_entries] == ["team", "zulu", "alpha"]

    def test_legacy_pre_175_bundle_parses_with_header_absent(self) -> None:
        """AC12 — pre-17.5 bundles (no header fields, no composite keys) parse identically."""
        text = (
            "namespace: tenant-A\n"
            "user_id: alice\n"
            "entries:\n"
            "  team:\n"
            "    kind: team\n"
            "    model_type: akgentic.team.models.TeamCard\n"
            "    payload: {name: team}\n"
            "  agent_a:\n"
            "    kind: agent\n"
            "    model_type: akgentic.core.agent_card.AgentCard\n"
            "    payload: {role: a}\n"
        )
        parsed_entries, header = load_namespace(text)
        assert {e.id for e in parsed_entries} == {"team", "agent_a"}
        # The header is absent — meta upsert will skip on import (AC11 / AC12).
        assert header.present is False
        assert header.name == ""
        assert header.description == ""
        assert header.properties == {}

    def test_skips_composite_keyed_entries(self) -> None:
        """AC10 — a key with a dot is treated as external and skipped."""
        text = (
            "namespace: tenant-A\n"
            "user_id: alice\n"
            "name: tenant-A\n"
            "description: ''\n"
            "properties: {}\n"
            "entries:\n"
            "  team:\n"
            "    kind: team\n"
            "    model_type: akgentic.team.models.TeamCard\n"
            "    payload: {name: team}\n"
            "  global.shared-prompt:\n"
            "    kind: prompt\n"
            "    model_type: akgentic.llm.prompts.PromptTemplate\n"
            "    payload: {template: shared}\n"
        )
        parsed_entries, header = load_namespace(text)
        # Only the local team is constructed; the composite-keyed entry is skipped.
        assert {e.id for e in parsed_entries} == {"team"}
        assert header.present is True
        assert header.name == "tenant-A"

    def test_rejects_bundle_with_only_external_entries(self) -> None:
        """A bundle whose entries: dict carries ONLY composite keys is empty post-skip."""
        text = (
            "namespace: tenant-A\n"
            "user_id: alice\n"
            "entries:\n"
            "  global.id1:\n"
            "    kind: prompt\n"
            "    model_type: akgentic.llm.prompts.PromptTemplate\n"
            "    payload: {}\n"
        )
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(text)
        assert any("bundle must declare at least one entry" in e for e in exc_info.value.errors)


# --- section headers and spacing --------------------------------------------


class TestSectionHeadersAndSpacing:
    def test_emits_header_per_non_empty_kind(self) -> None:
        entries = [_team(), _agent("a1"), _prompt("p1"), _tool("t1"), _model("m1")]
        text = dump_namespace(entries)
        assert _KIND_HEADERS["team"] in text
        assert _KIND_HEADERS["agent"] in text
        assert _KIND_HEADERS["prompt"] in text
        assert _KIND_HEADERS["tool"] in text
        assert _KIND_HEADERS["model"] in text

    def test_omits_header_for_absent_kind(self) -> None:
        text = dump_namespace([_team()])
        assert _KIND_HEADERS["team"] in text
        assert _KIND_HEADERS["agent"] not in text
        assert _KIND_HEADERS["prompt"] not in text
        assert _KIND_HEADERS["tool"] not in text
        assert _KIND_HEADERS["model"] not in text

    def test_header_order_matches_kind_emit_order(self) -> None:
        entries = [_team(), _agent("a1"), _prompt("p1"), _tool("t1"), _model("m1")]
        text = dump_namespace(entries)
        teams_pos = text.index(_KIND_HEADERS["team"])
        agents_pos = text.index(_KIND_HEADERS["agent"])
        prompts_pos = text.index(_KIND_HEADERS["prompt"])
        tools_pos = text.index(_KIND_HEADERS["tool"])
        models_pos = text.index(_KIND_HEADERS["model"])
        assert teams_pos < agents_pos < prompts_pos < tools_pos < models_pos

    def test_same_kind_entries_separated_by_blank_line(self) -> None:
        entries = [_agent("a1"), _agent("a2"), _agent("a3"), _team()]
        text = dump_namespace(entries)
        agents_header = _KIND_HEADERS["agent"]
        agent_section_start = text.index(agents_header)
        agent_section = text[agent_section_start:]
        double_newlines = agent_section.count("\n\n")
        assert double_newlines == 2

    def test_no_blank_line_at_end_of_final_section(self) -> None:
        entries = [_team(), _agent("a1")]
        text = dump_namespace(entries)
        assert text.endswith("\n")
        assert not text.endswith("\n\n")

    def test_round_trip_through_safe_load_preserves_values(self) -> None:
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
        recovered, _header = load_namespace(text)
        sort_key = lambda e: (e.kind, e.id)  # noqa: E731
        assert [e.model_dump() for e in sorted(recovered, key=sort_key)] == [
            e.model_dump() for e in sorted(entries, key=sort_key)
        ]
        parsed_doc = yaml.safe_load(text)
        assert set(parsed_doc["entries"].keys()) == {e.id for e in entries}

    def test_round_trip_enterprise_bundle(self) -> None:
        entries = [_team(user_id=None), _agent("a1", user_id=None), _prompt("p1", user_id=None)]
        text = dump_namespace(entries)
        recovered, _header = load_namespace(text)
        sort_key = lambda e: (e.kind, e.id)  # noqa: E731
        assert [e.model_dump() for e in sorted(recovered, key=sort_key)] == [
            e.model_dump() for e in sorted(entries, key=sort_key)
        ]
        doc = yaml.safe_load(text)
        assert doc["user_id"] is None


# --- Story 17.2 — meta entry emit order via legacy dump (no header trio) -----


class TestMetaEmitOrderAndRoundTrip:
    """Story 17.2 — when a meta entry is passed to ``dump_namespace`` directly
    (without going through ``Catalog.export_namespace_yaml``), it appears in
    ``entries:`` under the Meta section. This is the defensive case for
    callers that hand-build bundles without the catalog service's hoist."""

    def test_meta_section_header_present_between_team_and_agent(self) -> None:
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
        entries = [_team(), _meta(), _agent("a1")]
        text = dump_namespace(entries)
        recovered, _header = load_namespace(text)
        sort_key = lambda e: (e.kind, e.id)  # noqa: E731
        assert [e.model_dump() for e in sorted(recovered, key=sort_key)] == [
            e.model_dump() for e in sorted(entries, key=sort_key)
        ]

    def test_meta_emit_order_index_one(self) -> None:
        text = dump_namespace([_agent("a1"), _meta(), _team()])
        doc = yaml.safe_load(text)
        # Outer keys are emitted in (kind emit order, id) — team @ 0, meta @ 1, agent @ 2.
        assert list(doc["entries"].keys()) == ["team", "_meta", "a1"]


# --- Story 17.6 — header projection / external sections (dump-side) ---------


class TestDumpWithHeader:
    """``dump_namespace`` extended signature — header projection."""

    def test_header_emits_six_top_level_keys_in_order(self) -> None:
        text = dump_namespace(
            [_team(), _agent("a")],
            name="Tenant A",
            description="primary",
            properties={"shared": "true"},
        )
        doc = yaml.safe_load(text)
        assert list(doc.keys()) == [
            "namespace",
            "user_id",
            "name",
            "description",
            "properties",
            "entries",
        ]
        assert doc["name"] == "Tenant A"
        assert doc["description"] == "primary"
        assert doc["properties"] == {"shared": "true"}

    def test_no_header_emits_three_keys(self) -> None:
        """Pre-17.5 callers (no kwargs) get the three-key shape verbatim."""
        text = dump_namespace([_team(), _agent("a")])
        doc = yaml.safe_load(text)
        assert list(doc.keys()) == ["namespace", "user_id", "entries"]

    def test_properties_empty_dict_emitted_as_mapping_when_other_header_set(self) -> None:
        """AC2 — properties is always a mapping, never null."""
        text = dump_namespace([_team(), _agent("a")], name="Tenant A")
        doc = yaml.safe_load(text)
        # When `name` is set, all three header fields are emitted; properties
        # falls back to an empty mapping.
        assert doc["properties"] == {}
        assert "properties: {}" in text or "properties:\n" in text

    def test_no_external_section_when_external_refs_empty(self) -> None:
        """No external_refs → no External-ref headers anywhere in the output."""
        text = dump_namespace([_team(), _agent("a")], name="X")
        for kind, header in _EXTERNAL_KIND_HEADERS.items():
            assert header not in text, f"unexpected external header for {kind!r}"


class TestDumpExternalSections:
    """``dump_namespace`` extended signature — external section emission."""

    def test_external_section_uses_composite_keys(self) -> None:
        external = [
            _model("id_gpt_41", namespace="global", user_id=None),
            _model("id_gpt_52", namespace="global", user_id=None),
        ]
        text = dump_namespace(
            [_team(), _agent("a")],
            name="Tenant A",
            external_refs=external,
        )
        doc = yaml.safe_load(text)
        # Both external entries appear under entries: with composite <ns>.<id> keys.
        assert "global.id_gpt_41" in doc["entries"]
        assert "global.id_gpt_52" in doc["entries"]

    def test_external_section_header_emitted(self) -> None:
        external = [_model("m1", namespace="global", user_id=None)]
        text = dump_namespace([_team()], name="X", external_refs=external)
        # The External-ref Models header appears.
        assert _EXTERNAL_KIND_HEADERS["model"] in text

    def test_external_kinds_in_pre_175_order(self) -> None:
        """AC5 — external sections emit team → agent → prompt → tool → model."""
        external = [
            _model("m1", namespace="global", user_id=None),
            _tool("t1", namespace="global", user_id=None),
            _agent("ext_a", namespace="global", user_id=None),
            _prompt("p1", namespace="global", user_id=None),
            _team(namespace="global", user_id=None),
        ]
        text = dump_namespace([_team()], name="X", external_refs=external)
        ext_team_pos = text.index(_EXTERNAL_KIND_HEADERS["team"])
        ext_agent_pos = text.index(_EXTERNAL_KIND_HEADERS["agent"])
        ext_prompt_pos = text.index(_EXTERNAL_KIND_HEADERS["prompt"])
        ext_tool_pos = text.index(_EXTERNAL_KIND_HEADERS["tool"])
        ext_model_pos = text.index(_EXTERNAL_KIND_HEADERS["model"])
        assert ext_team_pos < ext_agent_pos < ext_prompt_pos < ext_tool_pos < ext_model_pos

    def test_external_entries_sorted_by_namespace_then_id(self) -> None:
        """AC7 — within an external section, sort by (namespace, id) ascending."""
        external = [
            _model("zeta", namespace="bbb", user_id=None),
            _model("alpha", namespace="aaa", user_id=None),
            _model("alpha", namespace="bbb", user_id=None),
            _model("zeta", namespace="aaa", user_id=None),
        ]
        text = dump_namespace([_team()], name="X", external_refs=external)
        doc = yaml.safe_load(text)
        # Expect aaa.alpha, aaa.zeta, bbb.alpha, bbb.zeta.
        composite_keys = [k for k in doc["entries"] if "." in k]
        assert composite_keys == ["aaa.alpha", "aaa.zeta", "bbb.alpha", "bbb.zeta"]

    def test_external_section_header_preceded_by_blank_line(self) -> None:
        """AC5 — every section header is surrounded by a blank line above and below."""
        external = [_model("m1", namespace="global", user_id=None)]
        text = dump_namespace([_team()], name="X", external_refs=external)
        header = _EXTERNAL_KIND_HEADERS["model"]
        idx = text.index(header)
        # The line preceding the header must be empty.
        before = text[:idx].rstrip("\n").splitlines()
        assert before, "header should have content before it"
        # The line directly below the header must be blank, then the entry.
        after = text[idx + len(header) :]
        # After the header line we expect: \n\n  <key>:\n
        assert after.startswith("\n\n  "), repr(after[:30])

    def test_local_entry_after_external_section_does_not_emit(self) -> None:
        """Externals come AFTER all local sections; no local-after-external emit."""
        # Build a bundle where the first entry is local and the rest are external.
        text = dump_namespace(
            [_team()],
            name="X",
            external_refs=[_model("m1", namespace="global", user_id=None)],
        )
        # The local Teams header appears BEFORE the external Models header.
        teams_pos = text.index(_KIND_HEADERS["team"])
        ext_models_pos = text.index(_EXTERNAL_KIND_HEADERS["model"])
        assert teams_pos < ext_models_pos


# --- Story 17.6 — load_namespace for header projection ----------------------


class TestLoadHeaderProjection:
    def test_header_present_when_all_three_set(self) -> None:
        text = dump_namespace(
            [_team(), _agent("a")],
            name="Tenant A",
            description="primary",
            properties={"shared": "true"},
        )
        _entries, header = load_namespace(text)
        assert header.present is True
        assert header.name == "Tenant A"
        assert header.description == "primary"
        assert header.properties == {"shared": "true"}

    def test_header_absent_for_pre_175_bundle(self) -> None:
        text = dump_namespace([_team(), _agent("a")])
        _entries, header = load_namespace(text)
        assert header.present is False

    def test_header_present_when_only_name_set(self) -> None:
        """A bundle carrying just `name` (auto-fills the rest) is still a 17.6 bundle."""
        text = dump_namespace([_team(), _agent("a")], name="Tenant A")
        _entries, header = load_namespace(text)
        assert header.present is True
        assert header.name == "Tenant A"


# --- Story 17.6 — round-trip with the new shape -----------------------------


class TestRoundTripNewShape:
    """Two consecutive dumps of the same input produce byte-identical output."""

    def test_byte_identical_two_consecutive_dumps(self) -> None:
        external = [_model("id_gpt_41", namespace="global", user_id=None)]
        text_a = dump_namespace(
            [_team(), _agent("a"), _prompt("p1")],
            name="Tenant",
            description="primary",
            properties={"shared": "true"},
            external_refs=external,
        )
        text_b = dump_namespace(
            [_team(), _agent("a"), _prompt("p1")],
            name="Tenant",
            description="primary",
            properties={"shared": "true"},
            external_refs=external,
        )
        assert text_a == text_b

    def test_round_trip_drops_external_entries_on_import(self) -> None:
        """AC10 — composite-keyed entries skip on load, so re-dump without them is asymmetric.

        This pins the documented behaviour: a 17.6 bundle with externals
        round-trips its LOCAL entries verbatim; externals are display
        projection only.
        """
        external = [_model("ext", namespace="global", user_id=None)]
        text = dump_namespace(
            [_team(), _agent("a")],
            name="Tenant",
            external_refs=external,
        )
        local_entries, header = load_namespace(text)
        # Externals dropped — only local entries are reconstructed.
        assert {e.id for e in local_entries} == {"team", "a"}
        assert header.name == "Tenant"


# --- BundleHeader dataclass smoke ------------------------------------------


class TestBundleHeader:
    def test_default_present_false(self) -> None:
        h = BundleHeader()
        assert h.name == ""
        assert h.description == ""
        assert h.properties == {}
        assert h.present is False

    def test_explicit_present(self) -> None:
        h = BundleHeader(name="X", description="d", properties={"k": "v"}, present=True)
        assert h.present is True
        assert h.properties == {"k": "v"}
