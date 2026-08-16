"""Unit tests for ``akgentic.catalog.serialization`` (Stories 16.2 / 17.6)."""

from __future__ import annotations

from typing import Any, get_origin

import pytest
import yaml

from akgentic.catalog.models.entry import Entry
from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.catalog.models.namespace_meta import NamespaceMeta
from akgentic.catalog.serialization import (
    _EXTERNAL_KIND_HEADERS,
    _KIND_HEADERS,
    BundleHeader,
    _project_header,
    dump_namespace,
    load_namespace,
)

_TEAM_TYPE = "akgentic.team.models.TeamCard"
_AGENT_TYPE = "akgentic.core.agent_card.AgentCard"
_PROMPT_TYPE = "akgentic.llm.prompts.PromptTemplate"
_TOOL_TYPE = "akgentic.tool.tool_card.ToolCard"
_MODEL_TYPE = "akgentic.llm.model_config.ModelConfig"
_META_TYPE = "akgentic.catalog.models.namespace_meta.NamespaceMeta"


def _team(namespace: str = "ns-1", user_id: str = "alice") -> Entry:
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
    user_id: str = "alice",
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
    user_id: str = "alice",
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
    user_id: str = "alice",
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
    user_id: str = "alice",
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
    user_id: str = "alice",
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

    def test_community_tier_user_id_is_anonymous(self) -> None:
        # Story 18.1 — community-tier exports emit ``user_id: anonymous``
        # (not ``null``). The catalog never writes ``user_id: null`` again.
        text = dump_namespace([_team(user_id="anonymous"), _agent("a", user_id="anonymous")])
        doc = yaml.safe_load(text)
        assert doc["user_id"] == "anonymous"
        assert "user_id: anonymous" in text
        assert "user_id: null" not in text

    def test_entry_keys_and_order(self) -> None:
        text = dump_namespace([_team(), _agent("a")])
        doc = yaml.safe_load(text)
        agent_map = doc["entries"]["a"]
        assert list(agent_map.keys()) == [
            "kind",
            "model_type",
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
        assert exc_info.value.errors == ["bundle must declare at least one entry"]

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

    def test_native_value_bundle_round_trip(self) -> None:
        """Story 26.1 / AC 14 — a bundle containing a NativeValue entry plus a
        composite entry that references it round-trips through
        ``dump_namespace`` / ``load_namespace``. On reload the reference still
        resolves to the unwrapped scalar — the bundle format treats NativeValue
        entries as normal entries with no special-casing.
        """
        native_type = "akgentic.catalog.NativeValue"
        native = Entry(
            id="id_native",
            kind="prompt",
            namespace="ns-1",
            user_id="alice",
            model_type=native_type,
            payload={"value": "shared-body"},
        )
        consumer = Entry(
            id="id_consumer",
            kind="prompt",
            namespace="ns-1",
            user_id="alice",
            model_type=_PROMPT_TYPE,
            payload={"template": {"__ref__": "id_native"}},
        )
        text = dump_namespace([_team(), native, consumer])
        parsed_entries, _header = load_namespace(text)
        # Byte-equal payloads after a dump/load cycle.
        reloaded_native = next(e for e in parsed_entries if e.id == "id_native")
        reloaded_consumer = next(e for e in parsed_entries if e.id == "id_consumer")
        assert reloaded_native.model_type == native_type
        assert reloaded_native.payload == {"value": "shared-body"}
        assert reloaded_consumer.payload == {"template": {"__ref__": "id_native"}}


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
        assert exc_info.value.errors == ["bundle must declare at least one entry"]

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

    def test_round_trip_anonymous_bundle(self) -> None:
        # Story 18.1 — community-tier exports carry ``user_id: anonymous`` at
        # the bundle root; round-trip preserves it.
        entries = [
            _team(user_id="anonymous"),
            _agent("a1", user_id="anonymous"),
            _prompt("p1", user_id="anonymous"),
        ]
        text = dump_namespace(entries)
        recovered, _header = load_namespace(text)
        sort_key = lambda e: (e.kind, e.id)  # noqa: E731
        assert [e.model_dump() for e in sorted(recovered, key=sort_key)] == [
            e.model_dump() for e in sorted(entries, key=sort_key)
        ]
        doc = yaml.safe_load(text)
        assert doc["user_id"] == "anonymous"


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

    def test_header_emits_eight_top_level_keys_in_order(self) -> None:
        # Story 18.2 — header now includes ``public`` between ``shareable``
        # and ``entries``. ``properties`` is fully free-form ``str -> str``
        # with NO catalog-reserved keys.
        text = dump_namespace(
            [_team(), _agent("a")],
            name="Tenant A",
            description="primary",
            properties={"owner_team": "platform"},
            shareable=True,
            public=True,
        )
        doc = yaml.safe_load(text)
        assert list(doc.keys()) == [
            "namespace",
            "user_id",
            "name",
            "description",
            "properties",
            "shareable",
            "public",
            "entries",
        ]
        assert doc["name"] == "Tenant A"
        assert doc["description"] == "primary"
        assert doc["properties"] == {"owner_team": "platform"}
        assert doc["shareable"] is True
        assert doc["public"] is True

    def test_default_shareable_and_public_false_emit_in_header(self) -> None:
        # Story 17.7 / 18.2 — when a header is forced (here by ``name``),
        # ``shareable`` and ``public`` are emitted at their declaration
        # positions with default value ``False``.
        text = dump_namespace(
            [_team(), _agent("a")],
            name="Tenant A",
        )
        doc = yaml.safe_load(text)
        assert list(doc.keys()) == [
            "namespace",
            "user_id",
            "name",
            "description",
            "properties",
            "shareable",
            "public",
            "entries",
        ]
        assert doc["shareable"] is False
        assert doc["public"] is False

    def test_shareable_true_alone_forces_header(self) -> None:
        # ``shareable=True`` widens the ``has_header`` branch even when
        # ``name`` / ``description`` / ``properties`` / ``public`` /
        # ``external_refs`` are all default. A shareable namespace is
        # structurally meaningful and must surface in the wire shape.
        text = dump_namespace(
            [_team(), _agent("a")],
            shareable=True,
        )
        doc = yaml.safe_load(text)
        assert list(doc.keys()) == [
            "namespace",
            "user_id",
            "name",
            "description",
            "properties",
            "shareable",
            "public",
            "entries",
        ]
        assert doc["shareable"] is True
        assert doc["public"] is False
        assert doc["name"] == ""
        assert doc["description"] == ""
        assert doc["properties"] == {}

    def test_public_true_alone_forces_header(self) -> None:
        # Story 18.2 AC3 — ``public=True`` widens the ``has_header`` branch
        # even when every other header source is at its default. A public
        # namespace is structurally meaningful and must surface in the wire
        # shape (mirrors ``shareable`` from Story 17.7).
        text = dump_namespace(
            [_team(), _agent("a")],
            public=True,
        )
        doc = yaml.safe_load(text)
        assert list(doc.keys()) == [
            "namespace",
            "user_id",
            "name",
            "description",
            "properties",
            "shareable",
            "public",
            "entries",
        ]
        assert doc["public"] is True
        assert doc["shareable"] is False

    def test_public_emitted_immediately_after_shareable(self) -> None:
        # Story 18.2 AC3 — assert ordering by reading the raw YAML text:
        # the ``public:`` line index is greater than ``shareable:``'s and
        # less than ``entries:``'s. This catches accidental reorders that
        # would shift the wire-format key order.
        text = dump_namespace(
            [_team(), _agent("a")],
            name="Tenant A",
            shareable=True,
            public=True,
        )
        # Locate top-level (zero-indent) lines.
        shareable_idx = next(
            i for i, line in enumerate(text.splitlines()) if line.startswith("shareable:")
        )
        public_idx = next(
            i for i, line in enumerate(text.splitlines()) if line.startswith("public:")
        )
        entries_idx = next(
            i for i, line in enumerate(text.splitlines()) if line.startswith("entries:")
        )
        assert shareable_idx < public_idx < entries_idx

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
            _model("id_gpt_41", namespace="global", user_id="anonymous"),
            _model("id_gpt_52", namespace="global", user_id="anonymous"),
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
        external = [_model("m1", namespace="global", user_id="anonymous")]
        text = dump_namespace([_team()], name="X", external_refs=external)
        # The External-ref Models header appears.
        assert _EXTERNAL_KIND_HEADERS["model"] in text

    def test_external_kinds_in_pre_175_order(self) -> None:
        """AC5 — external sections emit team → agent → prompt → tool → model."""
        external = [
            _model("m1", namespace="global", user_id="anonymous"),
            _tool("t1", namespace="global", user_id="anonymous"),
            _agent("ext_a", namespace="global", user_id="anonymous"),
            _prompt("p1", namespace="global", user_id="anonymous"),
            _team(namespace="global", user_id="anonymous"),
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
            _model("zeta", namespace="bbb", user_id="anonymous"),
            _model("alpha", namespace="aaa", user_id="anonymous"),
            _model("alpha", namespace="bbb", user_id="anonymous"),
            _model("zeta", namespace="aaa", user_id="anonymous"),
        ]
        text = dump_namespace([_team()], name="X", external_refs=external)
        doc = yaml.safe_load(text)
        # Expect aaa.alpha, aaa.zeta, bbb.alpha, bbb.zeta.
        composite_keys = [k for k in doc["entries"] if "." in k]
        assert composite_keys == ["aaa.alpha", "aaa.zeta", "bbb.alpha", "bbb.zeta"]

    def test_external_section_header_preceded_by_blank_line(self) -> None:
        """AC5 — every section header is surrounded by a blank line above and below."""
        external = [_model("m1", namespace="global", user_id="anonymous")]
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
            external_refs=[_model("m1", namespace="global", user_id="anonymous")],
        )
        # The local Teams header appears BEFORE the external Models header.
        teams_pos = text.index(_KIND_HEADERS["team"])
        ext_models_pos = text.index(_EXTERNAL_KIND_HEADERS["model"])
        assert teams_pos < ext_models_pos


# --- Story 17.6 — load_namespace for header projection ----------------------


class TestLoadHeaderProjection:
    def test_header_present_when_all_four_set(self) -> None:
        # Story 17.7 — `shareable` joins the projected fields.
        text = dump_namespace(
            [_team(), _agent("a")],
            name="Tenant A",
            description="primary",
            properties={"owner_team": "platform"},
            shareable=True,
        )
        _entries, header = load_namespace(text)
        assert header.present is True
        assert header.name == "Tenant A"
        assert header.description == "primary"
        assert header.properties == {"owner_team": "platform"}
        assert header.shareable is True

    def test_header_absent_for_pre_175_bundle(self) -> None:
        text = dump_namespace([_team(), _agent("a")])
        _entries, header = load_namespace(text)
        assert header.present is False
        assert header.shareable is False

    def test_header_present_when_only_name_set(self) -> None:
        """A bundle carrying just `name` (auto-fills the rest) is still a 17.6 bundle."""
        text = dump_namespace([_team(), _agent("a")], name="Tenant A")
        _entries, header = load_namespace(text)
        assert header.present is True
        assert header.name == "Tenant A"
        # Default ``shareable`` flag.
        assert header.shareable is False

    def test_header_present_when_only_shareable_set(self) -> None:
        """Story 17.7 — ``shareable=True`` alone forces ``present=True``."""
        text = dump_namespace([_team(), _agent("a")], shareable=True)
        _entries, header = load_namespace(text)
        assert header.present is True
        assert header.shareable is True

    def test_legacy_bundle_without_shareable_projects_false(self) -> None:
        """Story 17.7 / AC9 — pre-17.7 bundles (six top-level keys, no ``shareable``)
        parse identically; ``shareable`` defaults to ``False``.
        """
        # Hand-craft the legacy six-key shape (no `shareable` field).
        legacy_yaml = (
            "namespace: ns-1\n"
            "user_id: null\n"
            "name: Old Tenant\n"
            "description: legacy\n"
            "properties:\n"
            "  owner_team: platform\n"
            "entries:\n"
            "  team:\n"
            "    kind: team\n"
            "    model_type: akgentic.team.models.TeamCard\n"
            "    description: ''\n"
            "    payload:\n"
            "      name: T\n"
        )
        _entries, header = load_namespace(legacy_yaml)
        assert header.present is True
        assert header.name == "Old Tenant"
        assert header.shareable is False

    def test_non_bool_shareable_value_projects_false(self) -> None:
        """Defensive parsing — a non-bool ``shareable`` value projects to False."""
        legacy_yaml = (
            "namespace: ns-1\n"
            "user_id: null\n"
            "name: Old Tenant\n"
            "shareable: 'true'\n"  # string, not bool
            "entries:\n"
            "  team:\n"
            "    kind: team\n"
            "    model_type: akgentic.team.models.TeamCard\n"
            "    description: ''\n"
            "    payload:\n"
            "      name: T\n"
        )
        _entries, header = load_namespace(legacy_yaml)
        assert header.present is True
        assert header.shareable is False

    def test_header_projects_public_true(self) -> None:
        # Story 18.2 AC3 — ``public`` round-trips through dump/load.
        text = dump_namespace(
            [_team(), _agent("a")],
            name="Tenant A",
            public=True,
        )
        _entries, header = load_namespace(text)
        assert header.present is True
        assert header.public is True

    def test_header_present_when_only_public_set(self) -> None:
        # Story 18.2 AC3 — ``public=True`` alone forces ``present=True``,
        # mirroring ``shareable``'s shape.
        text = dump_namespace([_team(), _agent("a")], public=True)
        _entries, header = load_namespace(text)
        assert header.present is True
        assert header.public is True
        # Default ``shareable`` is preserved.
        assert header.shareable is False

    def test_legacy_bundle_without_public_projects_false(self) -> None:
        """Story 18.2 AC3 — pre-18.2 bundles (no ``public`` key) load with
        ``BundleHeader.public = False`` and ``present`` driven by the other
        fields only.
        """
        legacy_yaml = (
            "namespace: ns-1\n"
            "user_id: null\n"
            "name: Old Tenant\n"
            "shareable: true\n"
            "entries:\n"
            "  team:\n"
            "    kind: team\n"
            "    model_type: akgentic.team.models.TeamCard\n"
            "    description: ''\n"
            "    payload:\n"
            "      name: T\n"
        )
        _entries, header = load_namespace(legacy_yaml)
        assert header.present is True
        assert header.public is False
        # ``shareable`` still projects through.
        assert header.shareable is True

    def test_non_bool_public_value_projects_false(self) -> None:
        """Defensive parsing — a string ``public`` value projects to False."""
        legacy_yaml = (
            "namespace: ns-1\n"
            "user_id: null\n"
            "name: Old Tenant\n"
            "public: 'true'\n"  # string, not bool
            "entries:\n"
            "  team:\n"
            "    kind: team\n"
            "    model_type: akgentic.team.models.TeamCard\n"
            "    description: ''\n"
            "    payload:\n"
            "      name: T\n"
        )
        _entries, header = load_namespace(legacy_yaml)
        assert header.present is True
        assert header.public is False

    def test_public_explicit_false_present_flips(self) -> None:
        """Story 18.2 AC3 — a YAML doc with ``public: false`` literally
        present projects to ``BundleHeader(public=False, present=True)``
        (the explicit-key presence flips ``present`` even though the value
        is the default).
        """
        legacy_yaml = (
            "namespace: ns-1\n"
            "user_id: null\n"
            "public: false\n"
            "entries:\n"
            "  team:\n"
            "    kind: team\n"
            "    model_type: akgentic.team.models.TeamCard\n"
            "    description: ''\n"
            "    payload:\n"
            "      name: T\n"
        )
        _entries, header = load_namespace(legacy_yaml)
        assert header.present is True
        assert header.public is False


# --- Story 17.6 — round-trip with the new shape -----------------------------


class TestRoundTripNewShape:
    """Two consecutive dumps of the same input produce byte-identical output."""

    def test_byte_identical_two_consecutive_dumps(self) -> None:
        # Story 17.7 — `properties` is fully free-form `str -> str`.
        external = [_model("id_gpt_41", namespace="global", user_id="anonymous")]
        text_a = dump_namespace(
            [_team(), _agent("a"), _prompt("p1")],
            name="Tenant",
            description="primary",
            properties={"owner_team": "platform"},
            shareable=True,
            external_refs=external,
        )
        text_b = dump_namespace(
            [_team(), _agent("a"), _prompt("p1")],
            name="Tenant",
            description="primary",
            properties={"owner_team": "platform"},
            shareable=True,
            external_refs=external,
        )
        assert text_a == text_b

    def test_round_trip_drops_external_entries_on_import(self) -> None:
        """AC10 — composite-keyed entries skip on load, so re-dump without them is asymmetric.

        This pins the documented behaviour: a 17.6 bundle with externals
        round-trips its LOCAL entries verbatim; externals are display
        projection only.
        """
        external = [_model("ext", namespace="global", user_id="anonymous")]
        text = dump_namespace(
            [_team(), _agent("a")],
            name="Tenant",
            external_refs=external,
        )
        local_entries, header = load_namespace(text)
        # Externals dropped — only local entries are reconstructed.
        assert {e.id for e in local_entries} == {"team", "a"}
        assert header.name == "Tenant"


# --- BundleHeader smoke -----------------------------------------------------


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

    def test_bundle_header_carries_every_namespace_meta_field(self) -> None:
        """A field added to the meta model reaches the wire header for free.

        (a) is the load-bearing assertion. Re-declaring the header's fields
        flat still satisfies (b) today — (b) can only compare the fields that
        exist *now*, so a field added later sits at its default on both sides
        and the drift is invisible. Only the inheritance check goes red the
        moment the carrier is split back in two.
        """
        assert issubclass(BundleHeader, NamespaceMeta)  # (a)
        assert set(NamespaceMeta.model_fields) <= set(BundleHeader.model_fields)  # (b)
        assert set(BundleHeader.model_fields) - set(NamespaceMeta.model_fields) == {"present"}

    def test_an_empty_name_is_accepted_by_the_header(self) -> None:
        """The header relaxes the meta model's non-empty name — a bundle may omit it.

        The non-empty contract still holds where it matters: the import path
        re-validates the header through ``NamespaceMeta`` and refuses it.
        """
        assert BundleHeader().name == ""
        with pytest.raises(ValueError, match="name"):
            NamespaceMeta.model_validate({"name": ""})


class TestStaleLineageKeysAreRefused:
    """The stale ``parent_namespace`` / ``parent_id`` lineage keys are now refused.

    These keys used to load and then vanish: ``Entry`` ignores them, so the
    bundle round-tripped one key set shorter than it arrived and nobody was
    told. That silent narrowing is what the closed entry-map set exists to
    stop — a bundle still carrying lineage keys is now a loud import failure
    the author can act on, rather than a quiet loss.
    """

    def test_legacy_bundle_with_lineage_keys_is_refused_naming_every_key(self) -> None:
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
            "    payload:\n"
            "      name: T\n"
            "  legacy-clone:\n"
            "    kind: tool\n"
            "    model_type: akgentic.tool.search.SearchTool\n"
            "    parent_namespace: src-ns\n"
            "    parent_id: src-id\n"
            "    description: ''\n"
            "    payload:\n"
            "      name: legacy\n"
        )
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(legacy_text)
        # One message per key per entry — four in total, accumulated in one pass
        # so the author fixes the whole bundle in a single edit.
        errors = exc_info.value.errors
        assert len(errors) == 4
        joined = " ".join(errors)
        for entry_id in ("team", "legacy-clone"):
            for key in ("parent_namespace", "parent_id"):
                assert f"entry '{entry_id}' has unknown key '{key}'" in joined

    def test_bundle_without_lineage_keys_still_loads_and_redumps(self) -> None:
        """The same bundle minus the stale keys is unaffected by the check."""
        clean_text = (
            "namespace: ns-legacy\n"
            "user_id: alice\n"
            "entries:\n"
            "  team:\n"
            "    kind: team\n"
            "    model_type: akgentic.team.models.TeamCard\n"
            "    description: ''\n"
            "    payload:\n"
            "      name: T\n"
            "  legacy-clone:\n"
            "    kind: tool\n"
            "    model_type: akgentic.tool.search.SearchTool\n"
            "    description: ''\n"
            "    payload:\n"
            "      name: legacy\n"
        )
        entries, _header = load_namespace(clean_text)
        assert {e.id for e in entries} == {"team", "legacy-clone"}
        redumped = dump_namespace(entries)
        assert "parent_namespace" not in redumped
        assert "parent_id" not in redumped


# --- Story 29.2 — the two closed bundle-level key sets ----------------------


def _bundle_text(root_extra: str = "", entry_extra: str = "") -> str:
    """Build a minimal one-entry bundle, optionally injecting a stray key.

    ``root_extra`` is spliced in at the document root, ``entry_extra`` inside
    the ``team`` entry map. Both arrive already indented for their level.
    """
    return (
        "namespace: ns-1\n"
        "user_id: alice\n"
        f"{root_extra}"
        "entries:\n"
        "  team:\n"
        "    kind: team\n"
        "    model_type: akgentic.team.models.TeamCard\n"
        f"{entry_extra}"
        "    description: ''\n"
        "    payload:\n"
        "      name: T\n"
    )


class TestUnknownBundleRootKey:
    """AC12 — a root key outside the closed set is refused."""

    def test_misspelt_shareable_leaves_the_namespace_silently_unshareable(self) -> None:
        """The consequence, not the message: ``sharable:`` reads as correct.

        Nothing rejected it, ``shareable`` stayed ``False``, and a namespace
        the author meant to share simply was not — a permissions surprise
        discovered much later, and by someone else.
        """
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(_bundle_text(root_extra="sharable: true\n"))
        errors = exc_info.value.errors
        assert len(errors) == 1
        assert "bundle root has unknown key 'sharable'" in errors[0]
        # The remedy is spelled out, sorted so the wording is stable.
        assert (
            "expected one of: description, entries, name, namespace, properties, "
            "public, shareable, user_id" in errors[0]
        )

    def test_correctly_spelt_shareable_is_accepted(self) -> None:
        _entries, header = load_namespace(_bundle_text(root_extra="shareable: true\n"))
        assert header.shareable is True

    def test_several_bad_root_keys_accumulate(self) -> None:
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(_bundle_text(root_extra="sharable: true\nnaem: X\n"))
        joined = " | ".join(exc_info.value.errors)
        assert "'sharable'" in joined
        assert "'naem'" in joined

    def test_structural_root_errors_keep_their_wording_and_still_accumulate(self) -> None:
        """AC20 — the pre-existing messages are untouched by the new check."""
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace("user_id: alice\nsharable: true\nentries: {}\n")
        joined = " | ".join(exc_info.value.errors)
        assert "bundle root missing required key 'namespace'" in joined
        assert "bundle root has unknown key 'sharable'" in joined

    def test_entries_as_a_list_keeps_its_own_message(self) -> None:
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace("namespace: ns-1\nuser_id: alice\nentries:\n  - id: team\n")
        joined = " | ".join(exc_info.value.errors)
        assert "bundle 'entries' must be a mapping" in joined
        assert "unknown key" not in joined

    def test_non_mapping_root_is_unchanged(self) -> None:
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace("- just\n- a\n- list\n")
        assert exc_info.value.errors == ["bundle root must be a mapping, got list"]

    def test_the_parse_signal_present_is_not_a_bundle_root_key(self) -> None:
        """``present`` is how the reader records that a header was there at all.

        It is not namespace metadata and has never been legal on the wire.
        Deriving the root key set from the header model rather than from the
        meta model would quietly make it legal — and a bundle could then
        assert its own "presence", overriding the reader's own signal.
        """
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(_bundle_text(root_extra="present: true\n"))
        assert "bundle root has unknown key 'present'" in exc_info.value.errors[0]


class TestUnknownEntryMapKey:
    """AC8-AC11 — a key in a local entry map outside the closed four is refused."""

    def test_misspelt_description_is_named_with_its_entry(self) -> None:
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(_bundle_text(entry_extra="    descriptin: 'oops'\n"))
        errors = exc_info.value.errors
        assert len(errors) == 1
        assert "entry 'team' has unknown key 'descriptin'" in errors[0]
        assert "expected one of: description, kind, model_type, payload" in errors[0]

    def test_two_bad_keys_in_one_entry_map_both_report(self) -> None:
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(_bundle_text(entry_extra="    descriptin: 'a'\n    paylod: {}\n"))
        joined = " | ".join(exc_info.value.errors)
        assert "'descriptin'" in joined
        assert "'paylod'" in joined

    def test_bad_keys_in_two_entries_all_report(self) -> None:
        text = (
            "namespace: ns-1\n"
            "user_id: alice\n"
            "entries:\n"
            "  team:\n"
            "    kind: team\n"
            "    model_type: akgentic.team.models.TeamCard\n"
            "    descriptin: 'a'\n"
            "    payload:\n"
            "      name: T\n"
            "  planner:\n"
            "    kind: agent\n"
            "    model_type: akgentic.core.agent_card.AgentCard\n"
            "    modle_type: 'x'\n"
            "    payload:\n"
            "      role: planner\n"
        )
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(text)
        joined = " | ".join(exc_info.value.errors)
        assert "entry 'team' has unknown key 'descriptin'" in joined
        assert "entry 'planner' has unknown key 'modle_type'" in joined

    def test_external_composite_keys_are_not_swept(self) -> None:
        """AC11 — ``load_namespace`` skips external entries, so a key there is not a loss."""
        text = (
            "namespace: ns-1\n"
            "user_id: alice\n"
            "entries:\n"
            "  team:\n"
            "    kind: team\n"
            "    model_type: akgentic.team.models.TeamCard\n"
            "    description: ''\n"
            "    payload:\n"
            "      name: T\n"
            "  other-ns.shared:\n"
            "    kind: tool\n"
            "    model_type: akgentic.tool.search.SearchTool\n"
            "    whatever_key: 'ignored'\n"
            "    payload: {}\n"
        )
        entries, _header = load_namespace(text)
        assert [e.id for e in entries] == ["team"]

    def test_numeric_entry_key_does_not_crash_the_sweep(self) -> None:
        """A YAML id that is all digits parses as an ``int``, not a ``str``.

        The sweep runs before ``load_namespace`` raises its structural errors,
        so a bundle carrying both a malformed root and a numeric entry id must
        still report the root problem rather than raising ``TypeError`` out of
        the key check — ``validate_namespace_yaml`` only catches
        ``CatalogValidationError``, so anything else escapes as a 500.
        """
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace("user_id: alice\nentries:\n  2024:\n    kind: team\n")
        assert "bundle root missing required key 'namespace'" in " | ".join(exc_info.value.errors)

    def test_non_mapping_entry_value_keeps_the_build_entry_message(self) -> None:
        """AC10 — that error belongs to ``_build_entry``, not to the key sweep."""
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace("namespace: ns-1\nuser_id: alice\nentries:\n  team: 'a string'\n")
        joined = " | ".join(exc_info.value.errors)
        assert "entry 'team' is invalid: expected a mapping, got str" in joined

    def test_missing_required_key_keeps_the_entry_is_invalid_path(self) -> None:
        """AC10 — a missing ``model_type`` still surfaces through the Pydantic wrap."""
        text = (
            "namespace: ns-1\n"
            "user_id: alice\n"
            "entries:\n"
            "  team:\n"
            "    kind: team\n"
            "    description: ''\n"
            "    payload:\n"
            "      name: T\n"
        )
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(text)
        assert "entry 'team' is invalid" in exc_info.value.errors[0]


class TestBundleLevelTyposReportInOnePass:
    """AC14 — a bad root key and a bad entry-map key raise together.

    This is what forces the entry-map sweep to sit *above* ``_build_entry``:
    ``load_namespace`` must raise on root errors before the entry loop can
    start, so a check inside ``_build_entry`` could never be reached here.
    """

    def test_root_and_entry_map_typo_share_one_error_list(self) -> None:
        with pytest.raises(CatalogValidationError) as exc_info:
            load_namespace(
                _bundle_text(root_extra="sharable: true\n", entry_extra="    descriptin: 'a'\n")
            )
        joined = " | ".join(exc_info.value.errors)
        assert "bundle root has unknown key 'sharable'" in joined
        assert "entry 'team' has unknown key 'descriptin'" in joined


class TestLegacyShapesStillLoad:
    """AC16 — closing the sets does not narrow what already loaded."""

    def test_pre_175_three_key_bundle_is_clean(self) -> None:
        entries, header = load_namespace(_bundle_text())
        assert [e.id for e in entries] == ["team"]
        assert header.present is False

    def test_user_id_null_is_still_accepted_and_rewritten(self) -> None:
        text = _bundle_text().replace("user_id: alice\n", "user_id: null\n")
        entries, _header = load_namespace(text)
        assert entries[0].user_id == "anonymous"


class TestEveryEmittedKeyIsAccepted:
    """AC15 — the anti-drift guard between the emit side and the read side.

    ``_BUNDLE_ROOT_KEYS``'s header half is derived from ``NamespaceMeta``, so a
    new meta field is accepted at the root without a second list to edit. That
    is the ONLY half that moves on its own: the emit side (``dump_namespace``'s
    keyword arguments), the read-side projection (``_project_header``), the
    three document-structure root keys and all of ``_ENTRY_MAP_KEYS`` remain
    hand-maintained mirrors. A key added to either emitter without being added
    there would make every exported bundle un-importable; only a round trip
    over the FULL header shape catches it — and it catches only what the
    emitter emits, so a meta field that reaches neither hand-maintained list
    passes here while being dropped on import.
    """

    def test_full_header_plus_external_refs_round_trips(self) -> None:
        text = dump_namespace(
            [_team(), _agent("planner")],
            name="Full Tenant",
            description="every header key populated",
            properties={"owner_team": "platform"},
            shareable=True,
            public=True,
            external_refs=[_agent("shared", namespace="other-ns")],
        )
        # The external section really is on the wire — otherwise the round trip
        # would not exercise the composite-key branch at all.
        assert "other-ns.shared:" in text

        entries, header = load_namespace(text)
        assert header == BundleHeader(
            name="Full Tenant",
            description="every header key populated",
            properties={"owner_team": "platform"},
            shareable=True,
            public=True,
            present=True,
        )
        # External entries are skipped on import; the locals survive intact.
        assert [e.id for e in entries] == ["team", "planner"]


def _distinct_value(field_name: str) -> Any:
    """A value differing from the field's default, derived from its annotation."""
    annotation = NamespaceMeta.model_fields[field_name].annotation
    origin = get_origin(annotation) or annotation
    if origin is bool:
        return True
    if origin is dict:
        return {"probe": "value"}
    if origin is list:
        return ["probe"]
    if origin is int:
        return 4242
    if origin is str:
        return f"probe-{field_name}"
    raise AssertionError(
        f"no probe value for NamespaceMeta.{field_name}: {annotation!r} — "
        "extend _distinct_value so the projection guard can exercise the field"
    )


def _projected_header_fields() -> set[str]:
    """The header keys ``_project_header`` actually reads, derived by probing it.

    Probed rather than listed: a literal set here would be exactly the
    hand-maintained mirror this guard exists to catch, relocated into the test.
    A field counts as projected only if a document carrying it alone both flips
    ``present`` and lands its value on the header — covering the two halves of
    ``_project_header`` (the ``"x" in doc`` presence checks and the ``doc.get``
    reads) independently.
    """
    projected: set[str] = set()
    for field_name in NamespaceMeta.model_fields:
        probe = _distinct_value(field_name)
        header = _project_header({field_name: probe})
        if header.present and getattr(header, field_name) == probe:
            projected.add(field_name)
    return projected


class TestEveryMetaFieldReachesTheImport:
    """The half of the derivation ``_BUNDLE_ROOT_KEYS`` left open.

    Deriving the accepted root keys from ``NamespaceMeta`` moved the *accept*
    side only. ``_project_header`` stays hand-maintained, so a sixth meta field
    would be accepted at the bundle root and then silently discarded on import —
    where before the derivation it was rejected loudly as an unknown key. Loud
    rejection traded for silent loss is the failure this guard closes: adding a
    field without projecting it turns this test red instead of dropping data.
    """

    def test_project_header_reads_every_namespace_meta_field(self) -> None:
        assert _projected_header_fields() == set(NamespaceMeta.model_fields)
