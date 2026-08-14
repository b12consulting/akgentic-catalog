"""Tests for the unknown-key structural diff and the two paths that call it.

Three layers, in the order the story builds them:

* :class:`TestFindUnknownKeys` — the walk itself, against hand-written tree
  pairs (AC1-AC5, AC17).
* :class:`TestSavePathRejectsUnknownKeys` — ``prepare_for_write`` and the
  ``Catalog`` write verbs (AC7-AC11).
* :class:`TestValidateAndSaveParity` and the non-regression classes — the
  validate/save pair and the behaviour that must NOT change (AC16-AC20).
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, ConfigDict

from akgentic.catalog import find_unknown_keys
from akgentic.catalog.catalog import Catalog
from akgentic.catalog.models.entry import Entry
from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.catalog.resolver import REF_KEY, prepare_for_write
from akgentic.catalog.unknown_keys import find_unknown_keys as find_unknown_keys_direct

from .conftest import (
    CatalogFactory,
    CountingEntryRepository,
    FakeEntryRepository,
    make_entry,
    register_akgentic_test_module,
)

_TEAM_TYPE = "akgentic.team.models.TeamCard"


def _team_payload() -> dict[str, Any]:
    return {
        "name": "team",
        "description": "",
        "entry_point": {
            "card": {
                "description": "",
                "skills": [],
                "agent_class": "akgentic.core.agent.Akgent",
                "config": {"name": "entry", "role": "entry"},
            },
            "headcount": 1,
            "members": [],
        },
        "members": [],
        "agent_profiles": [],
    }


class Llm(BaseModel):
    """Nested payload model used by the misprint fixtures."""

    provider: str = "openai"
    temperature: float = 0.7


class Tool(BaseModel):
    """List-element payload model used by the ``tools[1].nmae`` fixture."""

    name: str = ""


class Agent(BaseModel):
    """Root payload model: one nested submodel plus a list of submodels."""

    role: str = "r"
    llm: Llm = Llm()
    tools: list[Tool] = []


class Loose(BaseModel):
    """Payload model that keeps its extras — AC18."""

    model_config = ConfigDict(extra="allow")

    role: str = "r"


def _register_models(monkeypatch: pytest.MonkeyPatch, suffix: str) -> str:
    """Register ``Agent`` / ``Llm`` / ``Tool`` / ``Loose`` under an allowlisted path."""
    return register_akgentic_test_module(
        monkeypatch,
        suffix,
        Agent=Agent,
        Llm=Llm,
        Tool=Tool,
        Loose=Loose,
    )


# --- The walk ---------------------------------------------------------------


class TestFindUnknownKeys:
    """AC1-AC5, AC17 — the structural diff over hand-written tree pairs."""

    def test_top_level_unknown_key_reported_by_bare_name(self) -> None:
        assert find_unknown_keys({"role": "r", "temperatur": 1}, {"role": "r"}) == ["temperatur"]

    def test_nested_unknown_key_reported_with_dotted_path(self) -> None:
        authored = {"llm": {"provider": "openai", "temperatur": 1}}
        dumped = {"llm": {"provider": "openai"}}
        assert find_unknown_keys(authored, dumped) == ["llm.temperatur"]

    def test_unknown_key_inside_list_element_reported_with_index(self) -> None:
        authored = {"tools": [{"name": "a"}, {"name": "b", "nmae": "c"}]}
        dumped = {"tools": [{"name": "a"}, {"name": "b"}]}
        assert find_unknown_keys(authored, dumped) == ["tools[1].nmae"]

    def test_several_unknown_keys_reported_in_document_order(self) -> None:
        authored = {
            "zeta": 1,
            "llm": {"beta": 2, "alpha": 3},
            "alpha": 4,
        }
        dumped: dict[str, Any] = {"llm": {}}
        assert find_unknown_keys(authored, dumped) == ["zeta", "llm.beta", "llm.alpha", "alpha"]

    def test_deeply_nested_all_known_keys_report_nothing(self) -> None:
        tree = {"a": {"b": {"c": [{"d": 1}, {"e": {"f": 2}}]}}}
        assert find_unknown_keys(tree, copy.deepcopy(tree)) == []

    def test_omitted_key_reports_nothing(self) -> None:
        """AC17 — deleting a key stays legal; absence is never a finding.

        The removal guard, deliberately written next to the misprint case: the
        two differ only in whether the key is present in ``authored``.
        """
        authored = {"role": "r"}
        dumped = {"role": "r", "temperature": 0.7}
        assert find_unknown_keys(authored, dumped) == []

    def test_dumped_not_a_dict_makes_every_authored_key_unknown(self) -> None:
        assert find_unknown_keys({"a": 1, "b": 2}, "scalar") == ["a", "b"]

    def test_leaf_and_type_mismatch_contribute_nothing(self) -> None:
        assert find_unknown_keys("scalar", {"a": 1}) == []
        assert find_unknown_keys([1, 2], {"a": 1}) == []

    def test_explicit_path_prefixes_every_finding(self) -> None:
        assert find_unknown_keys({"nmae": 1}, {}, path="tools[0]") == ["tools[0].nmae"]

    @pytest.mark.parametrize("sentinel", ["__ref__", "__type__", "__namespace__", "__model__"])
    def test_exempt_keys_never_reported_at_root(self, sentinel: str) -> None:
        # `__ref__` at the root makes the dict a ref marker, which short-circuits
        # to []; the other three must be skipped key-by-key.
        assert find_unknown_keys({sentinel: "x"}, {}) == []

    @pytest.mark.parametrize("sentinel", ["__type__", "__namespace__", "__model__"])
    def test_exempt_keys_never_reported_when_nested(self, sentinel: str) -> None:
        authored = {"llm": {sentinel: "x", "provider": "openai"}}
        dumped = {"llm": {"provider": "openai"}}
        assert find_unknown_keys(authored, dumped) == []

    def test_ref_marker_interior_not_descended_into(self) -> None:
        """A ref marker's siblings belong to story 29.2, not to this walk."""
        authored = {"llm": {REF_KEY: "shared-llm", "temperatur": 1}}
        dumped = {"llm": {"provider": "openai"}}
        assert find_unknown_keys(authored, dumped) == []

    def test_unset_but_refed_key_not_reported(self) -> None:
        """``_reconcile_dict`` keeps this key verbatim, so nothing is lost."""
        authored = {"llm": {REF_KEY: "shared-llm"}}
        dumped: dict[str, Any] = {}
        assert find_unknown_keys(authored, dumped) == []

    def test_absent_key_whose_value_is_an_ordinary_dict_is_reported(self) -> None:
        assert find_unknown_keys({"llm": {"provider": "openai"}}, {}) == ["llm"]

    def test_mismatched_list_lengths_raise_value_error(self) -> None:
        with pytest.raises(ValueError):
            find_unknown_keys({"tools": [{"a": 1}]}, {"tools": []})

    def test_inputs_are_not_mutated(self) -> None:
        authored = {"llm": {"temperatur": 1}, "tools": [{"nmae": 2}]}
        dumped: dict[str, Any] = {"llm": {}, "tools": [{}]}
        authored_snapshot = copy.deepcopy(authored)
        dumped_snapshot = copy.deepcopy(dumped)
        find_unknown_keys(authored, dumped)
        assert authored == authored_snapshot
        assert dumped == dumped_snapshot

    def test_public_re_export_is_the_module_function(self) -> None:
        """AC6 — ``akgentic.catalog.find_unknown_keys`` is the same object."""
        assert find_unknown_keys is find_unknown_keys_direct


# --- Write path -------------------------------------------------------------


class TestSavePathRejectsUnknownKeys:
    """AC7, AC8, AC11 — ``prepare_for_write`` raises before anything is stored."""

    def test_misprint_raises_with_path_and_model_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module_name = _register_models(monkeypatch, "tests_fixture_29_1_save_misprint")
        model_type = f"{module_name}.Agent"
        entry = make_entry(
            model_type=model_type,
            payload={"role": "r", "llm": {"provider": "openai", "temperatur": 1}},
        )
        with pytest.raises(CatalogValidationError) as exc_info:
            prepare_for_write(entry, FakeEntryRepository())
        assert len(exc_info.value.errors) == 1
        msg = exc_info.value.errors[0]
        assert "unknown key" in msg
        assert "'llm.temperatur'" in msg
        assert model_type in msg

    def test_one_message_per_reported_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module_name = _register_models(monkeypatch, "tests_fixture_29_1_save_multi")
        entry = make_entry(
            model_type=f"{module_name}.Agent",
            payload={"rol": "r", "llm": {"temperatur": 1}},
        )
        with pytest.raises(CatalogValidationError) as exc_info:
            prepare_for_write(entry, FakeEntryRepository())
        assert len(exc_info.value.errors) == 2
        assert "'rol'" in exc_info.value.errors[0]
        assert "'llm.temperatur'" in exc_info.value.errors[1]

    def test_omitted_key_still_stores_with_the_field_at_its_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC17 on the write path — the removal case, next to the misprint case."""
        module_name = _register_models(monkeypatch, "tests_fixture_29_1_save_removal")
        entry = make_entry(
            model_type=f"{module_name}.Agent",
            payload={"role": "r", "llm": {"provider": "openai"}},
        )
        prepared = prepare_for_write(entry, FakeEntryRepository())
        assert prepared.payload == {"role": "r", "llm": {"provider": "openai"}}
        assert Agent.model_validate(prepared.payload).llm.temperature == 0.7

    def test_create_raises_and_writes_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        counting_catalog: tuple[Catalog, CountingEntryRepository],
    ) -> None:
        catalog, counting = counting_catalog
        module_name = _register_models(monkeypatch, "tests_fixture_29_1_create")
        catalog.create(
            Entry(
                id="team",
                kind="team",
                namespace="ns-c",
                model_type=_TEAM_TYPE,
                payload=_team_payload(),
            )
        )
        counting.reset()
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.create(
                Entry(
                    id="agent-1",
                    kind="agent",
                    namespace="ns-c",
                    model_type=f"{module_name}.Agent",
                    payload={"role": "r", "temperatur": 1},
                )
            )
        assert "unknown key" in exc_info.value.errors[0]
        assert counting.count("put") == 0

    def test_update_raises_and_writes_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        counting_catalog: tuple[Catalog, CountingEntryRepository],
    ) -> None:
        catalog, counting = counting_catalog
        module_name = _register_models(monkeypatch, "tests_fixture_29_1_update")
        model_type = f"{module_name}.Agent"
        catalog.create(
            Entry(
                id="team",
                kind="team",
                namespace="ns-u",
                model_type=_TEAM_TYPE,
                payload=_team_payload(),
            )
        )
        catalog.create(
            Entry(
                id="agent-1",
                kind="agent",
                namespace="ns-u",
                model_type=model_type,
                payload={"role": "r"},
            )
        )
        counting.reset()
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.update(
                Entry(
                    id="agent-1",
                    kind="agent",
                    namespace="ns-u",
                    model_type=model_type,
                    payload={"role": "r", "temperatur": 1},
                )
            )
        assert "unknown key" in exc_info.value.errors[0]
        assert counting.count("put") == 0

    def test_import_namespace_yaml_raises_and_writes_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        counting_catalog: tuple[Catalog, CountingEntryRepository],
    ) -> None:
        catalog, counting = counting_catalog
        module_name = _register_models(monkeypatch, "tests_fixture_29_1_import")
        yaml_text = _bundle_yaml(f"{module_name}.Agent", {"role": "r", "temperatur": 1})
        counting.reset()
        with pytest.raises(CatalogValidationError) as exc_info:
            catalog.import_namespace_yaml(yaml_text)
        assert "unknown key" in exc_info.value.errors[0]
        assert counting.count("put") == 0


def _bundle_yaml(
    model_type: str,
    agent_payload: dict[str, Any],
    namespace: str = "ns-b",
) -> str:
    """Build a minimal one-team-one-agent bundle carrying ``agent_payload``."""
    doc = {
        "namespace": namespace,
        "user_id": "anonymous",
        "entries": {
            "team": {
                "kind": "team",
                "model_type": _TEAM_TYPE,
                "description": "",
                "payload": _team_payload(),
            },
            "agent-1": {
                "kind": "agent",
                "model_type": model_type,
                "description": "",
                "payload": agent_payload,
            },
        },
    }
    return yaml.safe_dump(doc, sort_keys=False)


# --- Parity -----------------------------------------------------------------


_MISPRINTS: list[tuple[str, dict[str, Any], str]] = [
    ("top-level", {"role": "r", "temperatur": 1}, "temperatur"),
    ("nested", {"role": "r", "llm": {"temperatur": 1}}, "llm.temperatur"),
    (
        "inside-list-element",
        {"role": "r", "tools": [{"name": "a"}, {"name": "b", "nmae": "c"}]},
        "tools[1].nmae",
    ),
]


class TestValidateAndSaveParity:
    """AC16 — validate-fails iff save-raises, asserted as a pair in one body."""

    @pytest.mark.parametrize(
        ("payload", "expected_path"),
        [(payload, path) for _id, payload, path in _MISPRINTS],
        ids=[case_id for case_id, _payload, _path in _MISPRINTS],
    )
    def test_misprint_fails_validate_and_save_alike(
        self,
        monkeypatch: pytest.MonkeyPatch,
        catalog_factory: CatalogFactory,
        payload: dict[str, Any],
        expected_path: str,
    ) -> None:
        catalog, _repo = catalog_factory()
        module_name = _register_models(monkeypatch, "tests_fixture_29_1_parity")
        model_type = f"{module_name}.Agent"

        report = catalog.validate_namespace_yaml(_bundle_yaml(model_type, payload))
        validate_findings = [err for issue in report.entry_issues for err in issue.errors]

        entry = make_entry(model_type=model_type, payload=payload)
        with pytest.raises(CatalogValidationError) as exc_info:
            prepare_for_write(entry, FakeEntryRepository())

        assert report.ok is False
        assert any(f"'{expected_path}'" in err for err in validate_findings)
        assert any(f"'{expected_path}'" in err for err in exc_info.value.errors)
        # Same helper, same template — the two paths report the same sentence.
        assert set(validate_findings) == set(exc_info.value.errors)

    @pytest.mark.parametrize(
        ("payload", "expected_path"),
        [(payload, path) for _id, payload, path in _MISPRINTS],
        ids=[case_id for case_id, _payload, _path in _MISPRINTS],
    )
    def test_clean_payload_passes_validate_and_save_alike(
        self,
        monkeypatch: pytest.MonkeyPatch,
        catalog_factory: CatalogFactory,
        payload: dict[str, Any],
        expected_path: str,
    ) -> None:
        """The other half of the iff: no misprint ⇒ neither path complains."""
        del expected_path
        catalog, _repo = catalog_factory()
        module_name = _register_models(monkeypatch, "tests_fixture_29_1_parity_clean")
        model_type = f"{module_name}.Agent"
        clean = _strip_misprints(payload)

        report = catalog.validate_namespace_yaml(_bundle_yaml(model_type, clean))
        prepared = prepare_for_write(
            make_entry(model_type=model_type, payload=clean), FakeEntryRepository()
        )

        assert report.ok is True, f"unexpected findings: {report.entry_issues!r}"
        assert prepared.payload == clean


def _strip_misprints(payload: dict[str, Any]) -> dict[str, Any]:
    """Return ``payload`` with every key unknown to ``Agent`` / ``Llm`` / ``Tool`` removed."""
    known = {"role", "llm", "tools"}
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in known:
            continue
        if key == "llm":
            clean[key] = {k: v for k, v in value.items() if k in Llm.model_fields}
        elif key == "tools":
            clean[key] = [{k: v for k, v in t.items() if k in Tool.model_fields} for t in value]
        else:
            clean[key] = value
    return clean


# --- Behaviour that must not change -----------------------------------------


class TestBehaviourThatMustNotChange:
    """AC18-AC20 — extras, value-level errors, and namespace round-trips."""

    def test_extra_allow_model_keeps_its_extras(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC18 — ``extra="allow"`` is inherited, not overridden."""
        module_name = _register_models(monkeypatch, "tests_fixture_29_1_extra_allow")
        entry = make_entry(
            model_type=f"{module_name}.Loose",
            payload={"role": "r", "whatever": 1},
        )
        prepared = prepare_for_write(entry, FakeEntryRepository())
        assert prepared.payload == {"role": "r", "whatever": 1}

    def test_extra_allow_model_produces_no_findings_on_the_validate_path(
        self, monkeypatch: pytest.MonkeyPatch, catalog_factory: CatalogFactory
    ) -> None:
        """AC18 on the validate half — the write-path case above is only half the pair.

        ``extra="allow"`` is inherited from the model, so the two paths must
        agree here for the same reason they agree on a misprint: one helper,
        one dump, taken with the same flags.
        """
        catalog, _repo = catalog_factory()
        module_name = _register_models(monkeypatch, "tests_fixture_29_1_extra_allow_validate")
        report = catalog.validate_namespace_yaml(
            _bundle_yaml(f"{module_name}.Loose", {"role": "r", "whatever": 1})
        )
        assert report.ok is True, f"unexpected findings: {report.entry_issues!r}"

    def test_wrong_typed_value_keeps_the_existing_message_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC19 — a value-level error is unchanged and produces no unknown-key noise."""
        module_name = _register_models(monkeypatch, "tests_fixture_29_1_value_error")
        entry = make_entry(
            model_type=f"{module_name}.Agent",
            payload={"role": "r", "llm": {"temperature": "hot"}},
        )
        with pytest.raises(CatalogValidationError) as exc_info:
            prepare_for_write(entry, FakeEntryRepository())
        joined = " | ".join(exc_info.value.errors)
        assert "Payload does not validate against" in joined
        assert "unknown key" not in joined

    def test_wrong_typed_value_on_validate_path_keeps_its_message_only(
        self, monkeypatch: pytest.MonkeyPatch, catalog_factory: CatalogFactory
    ) -> None:
        catalog, _repo = catalog_factory()
        module_name = _register_models(monkeypatch, "tests_fixture_29_1_value_error_validate")
        yaml_text = _bundle_yaml(f"{module_name}.Agent", {"llm": {"temperature": "hot"}})
        report = catalog.validate_namespace_yaml(yaml_text)
        assert report.ok is False
        joined = " | ".join(err for issue in report.entry_issues for err in issue.errors)
        assert "payload does not validate against" in joined
        assert "unknown key" not in joined

    def test_export_then_reimport_produces_no_findings(
        self, catalog_factory: CatalogFactory
    ) -> None:
        """AC20 — round-tripping an existing namespace is unaffected."""
        catalog, _repo = catalog_factory()
        catalog.create(
            Entry(
                id="team",
                kind="team",
                namespace="ns-rt",
                model_type=_TEAM_TYPE,
                payload=_team_payload(),
            )
        )
        catalog.create(
            Entry(
                id="agent-1",
                kind="agent",
                namespace="ns-rt",
                model_type="akgentic.core.agent_card.AgentCard",
                payload={
                    "description": "",
                    "skills": [],
                    "agent_class": "akgentic.core.agent.Akgent",
                    "config": {"name": "a", "role": "r"},
                },
            )
        )
        exported = catalog.export_namespace_yaml("ns-rt")
        report = catalog.validate_namespace_yaml(exported.replace("ns-rt", "ns-rt2"))
        findings = [err for issue in report.entry_issues for err in issue.errors]
        assert not any("unknown key" in err for err in findings), findings
        catalog.import_namespace_yaml(exported.replace("ns-rt", "ns-rt2"))
        assert catalog.validate_namespace("ns-rt2").ok is True
