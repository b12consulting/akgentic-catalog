"""A ``__ref__`` marker carries the sentinels and nothing else.

The marker used to accept non-reserved siblings, shallow-merged onto the target
payload before validation. Removing that interior is what these tests pin, and
the last two classes pin *why* it was removed: with an interior, the package's
payload walkers disagreed about whether to descend into a marker, and two of
them were wrong in a way that no test caught.

Both defect tests are named for the defect rather than for the mechanism —
the mechanism is gone, the defects are what must never come back.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.catalog.repositories.yaml import _payload_has_cross_ns_ref, _payload_has_ref
from akgentic.catalog.resolver import populate_refs, prepare_for_write, validate_delete

from .conftest import FakeEntryRepository, make_entry, register_akgentic_test_module


class Anything(BaseModel):
    """Permissive test model — ``extra='allow'`` so any payload validates."""

    model_config = ConfigDict(extra="allow")


class Strict(BaseModel):
    """Restrictive test model declaring a real field and a required one.

    ``template`` is a genuine field, so a marker carrying ``template:`` is the
    case the removed mechanism was *sanctioned* for; ``mandatory`` lets a target
    payload be built that fails its own validation.
    """

    template: str = "base"
    mandatory: str


@pytest.fixture
def anything_model_type(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register ``Anything`` under an ``akgentic.*`` module and return its FQCN."""
    module_name = register_akgentic_test_module(
        monkeypatch,
        "tests_fixture_pure_pointer_anything",
        Anything=Anything,
    )
    return f"{module_name}.Anything"


@pytest.fixture
def strict_model_type(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register ``Strict`` under an ``akgentic.*`` module and return its FQCN."""
    module_name = register_akgentic_test_module(
        monkeypatch,
        "tests_fixture_pure_pointer_strict",
        Strict=Strict,
    )
    return f"{module_name}.Strict"


@pytest.fixture
def repo_with_target(anything_model_type: str) -> FakeEntryRepository:
    """A repository holding one resolvable target entry in ``ns-1``."""
    repo = FakeEntryRepository()
    repo.put(
        make_entry(
            id="target",
            namespace="ns-1",
            model_type=anything_model_type,
            payload={"template": "base", "params": {"role": "assistant"}},
        )
    )
    return repo


class TestMarkerTakesOnlySentinels:
    """Any key beyond ``__ref__`` / ``__type__`` / ``__namespace__`` is refused.

    The target here declares ``extra='allow'``, which the removed check skipped
    entirely — a marker's shape is wrong regardless of what the target model
    happens to tolerate.
    """

    @pytest.mark.parametrize(
        "sibling",
        [
            {"params": {"role": "Manager"}},  # the shape the feature existed for
            {"temperatur": 0.7},  # a misprint
            {"anything": None},
        ],
        ids=["params", "misprint", "arbitrary"],
    )
    def test_sibling_on_an_extra_allow_target_is_rejected(
        self, repo_with_target: FakeEntryRepository, sibling: dict[str, Any]
    ) -> None:
        marker = {"__ref__": "target", **sibling}
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs(marker, repo_with_target, "ns-1")
        key = next(iter(sibling))
        assert any(f"'{key}'" in e for e in exc_info.value.errors)

    def test_a_genuine_field_of_the_target_model_is_rejected_too(
        self, strict_model_type: str
    ) -> None:
        """The sanctioned use of the old mechanism is now an error like any other."""
        repo = FakeEntryRepository()
        repo.put(
            make_entry(
                id="strict-target",
                namespace="ns-1",
                model_type=strict_model_type,
                payload={"template": "base", "mandatory": "set"},
            )
        )
        marker = {"__ref__": "strict-target", "template": "inline"}
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs(marker, repo, "ns-1")
        assert any("'template'" in e and "pure pointer" in e for e in exc_info.value.errors)

    def test_sibling_on_a_cross_ns_marker_behaves_identically(
        self, anything_model_type: str
    ) -> None:
        repo = FakeEntryRepository()
        repo.put(
            make_entry(
                id="shared",
                namespace="other-ns",
                model_type=anything_model_type,
                payload={"template": "from-other"},
            )
        )
        marker = {"__ref__": "shared", "__namespace__": "other-ns", "params": {"role": "M"}}
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs(marker, repo, "ns-1", is_namespace_shareable=lambda _ns: True)
        assert any("'params'" in e and "pure pointer" in e for e in exc_info.value.errors)

    def test_message_names_the_replacement(self, repo_with_target: FakeEntryRepository) -> None:
        """The teaching half of the message is asserted, so a reword cannot drop it.

        Authors hit this while editing a bundle with no ADR at hand; an error
        that only forbids leaves them stuck.
        """
        marker = {"__ref__": "target", "params": {"role": "Manager"}}
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs(marker, repo_with_target, "ns-1")
        joined = " ".join(exc_info.value.errors)
        assert "NativeValue" in joined
        assert "pure pointer" in joined

    def test_one_message_per_key_in_author_order(
        self, repo_with_target: FakeEntryRepository
    ) -> None:
        marker = {"__ref__": "target", "zeta": 1, "alpha": 2}
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs(marker, repo_with_target, "ns-1")
        assert len(exc_info.value.errors) == 2
        assert "'zeta'" in exc_info.value.errors[0]
        assert "'alpha'" in exc_info.value.errors[1]


class TestErrorPrecedenceIsUnchanged:
    """Every check that existed before keeps the precedence it already had.

    The rejection slots in after the ``__type__`` mismatch check and before the
    target's payload is walked. Everything upstream of it therefore still wins,
    and nothing downstream of it can mask it.
    """

    def test_cycle_wins(
        self, repo_with_target: FakeEntryRepository, anything_model_type: str
    ) -> None:
        repo_with_target.put(
            make_entry(
                id="loop",
                namespace="ns-1",
                model_type=anything_model_type,
                payload={"x": {"__ref__": "loop", "params": {}}},
            )
        )
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs({"__ref__": "loop"}, repo_with_target, "ns-1")
        assert any("cycle" in e.lower() for e in exc_info.value.errors)

    def test_not_shareable_wins(self, repo_with_target: FakeEntryRepository) -> None:
        marker = {"__ref__": "shared", "__namespace__": "locked-ns", "params": {}}
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs(marker, repo_with_target, "ns-1")
        assert any("is not shareable" in e for e in exc_info.value.errors)

    def test_missing_target_wins(self, repo_with_target: FakeEntryRepository) -> None:
        marker = {"__ref__": "no-such-entry", "params": {}}
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs(marker, repo_with_target, "ns-1")
        assert any("not found" in e for e in exc_info.value.errors)

    def test_type_mismatch_wins(self, repo_with_target: FakeEntryRepository) -> None:
        marker = {
            "__ref__": "target",
            "__type__": "akgentic.llm.prompts.PromptTemplate",
            "params": {},
        }
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs(marker, repo_with_target, "ns-1")
        assert any("expected" in e for e in exc_info.value.errors)

    def test_sibling_wins_over_a_target_payload_that_does_not_validate(
        self, strict_model_type: str
    ) -> None:
        """The marker is refused before its target's payload is ever walked."""
        repo = FakeEntryRepository()
        repo.put(
            make_entry(
                id="invalid-target",
                namespace="ns-1",
                model_type=strict_model_type,
                payload={"template": "base"},  # ``mandatory`` missing
            )
        )
        marker = {"__ref__": "invalid-target", "params": {}}
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs(marker, repo, "ns-1")
        assert any("pure pointer" in e for e in exc_info.value.errors)
        assert not any("does not validate" in e for e in exc_info.value.errors)

    def test_sibling_wins_over_a_dangling_ref_in_its_own_value(
        self, repo_with_target: FakeEntryRepository
    ) -> None:
        marker = {"__ref__": "target", "params": {"x": {"__ref__": "does-not-exist"}}}
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs(marker, repo_with_target, "ns-1")
        assert any("pure pointer" in e for e in exc_info.value.errors)
        assert not any("not found" in e for e in exc_info.value.errors)

    def test_sibling_wins_over_a_cyclic_ref_in_its_own_value(
        self, repo_with_target: FakeEntryRepository
    ) -> None:
        marker = {"__ref__": "target", "params": {"x": {"__ref__": "target"}}}
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs(marker, repo_with_target, "ns-1")
        assert any("pure pointer" in e for e in exc_info.value.errors)
        assert not any("cycle" in e.lower() for e in exc_info.value.errors)


class TestSentinelsStillWork:
    """The reserved keys keep their ADR-008 roles — this is not a blanket ban."""

    def test_bare_marker_resolves(self, repo_with_target: FakeEntryRepository) -> None:
        result = populate_refs({"__ref__": "target"}, repo_with_target, "ns-1")
        assert result.template == "base"

    def test_type_pin_resolves(
        self, repo_with_target: FakeEntryRepository, anything_model_type: str
    ) -> None:
        marker = {"__ref__": "target", "__type__": anything_model_type}
        result = populate_refs(marker, repo_with_target, "ns-1")
        assert result.template == "base"

    def test_type_mismatch_still_reports_the_type_error(
        self, repo_with_target: FakeEntryRepository
    ) -> None:
        marker = {"__ref__": "target", "__type__": "akgentic.llm.prompts.PromptTemplate"}
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs(marker, repo_with_target, "ns-1")
        assert any("expected" in e for e in exc_info.value.errors)

    def test_namespace_sentinel_is_not_a_sibling(self, anything_model_type: str) -> None:
        """``__namespace__`` is pointer metadata, not payload content."""
        repo = FakeEntryRepository()
        repo.put(
            make_entry(
                id="shared",
                namespace="other-ns",
                model_type=anything_model_type,
                payload={"template": "from-other"},
            )
        )
        marker = {"__ref__": "shared", "__namespace__": "other-ns"}
        result = populate_refs(marker, repo, "ns-1", is_namespace_shareable=lambda _ns: True)
        assert result.template == "from-other"

    def test_all_three_sentinels_together_resolve(self, anything_model_type: str) -> None:
        """``__ref__`` + ``__namespace__`` + ``__type__`` is a legal marker."""
        repo = FakeEntryRepository()
        repo.put(
            make_entry(
                id="shared",
                namespace="other-ns",
                model_type=anything_model_type,
                payload={"template": "from-other"},
            )
        )
        marker = {
            "__ref__": "shared",
            "__namespace__": "other-ns",
            "__type__": anything_model_type,
        }
        result = populate_refs(marker, repo, "ns-1", is_namespace_shareable=lambda _ns: True)
        assert result.template == "from-other"

    def test_shorthand_resolves(self, anything_model_type: str) -> None:
        """The ``<ns>.<id>`` shorthand is pointer syntax, not a sibling."""
        repo = FakeEntryRepository()
        repo.put(
            make_entry(
                id="shared",
                namespace="other-ns",
                model_type=anything_model_type,
                payload={"template": "from-other"},
            )
        )
        result = populate_refs(
            {"__ref__": "other-ns.shared"}, repo, "ns-1", is_namespace_shareable=lambda _ns: True
        )
        assert result.template == "from-other"


class TestBothPathsAgree:
    """Validate and write must refuse identically — a half-fix is worse than the bug."""

    def test_write_path_refuses_and_writes_nothing(
        self, repo_with_target: FakeEntryRepository, anything_model_type: str
    ) -> None:
        entry = make_entry(
            id="consumer",
            namespace="ns-1",
            model_type=anything_model_type,
            payload={"prompt": {"__ref__": "target", "params": {"role": "Manager"}}},
        )
        with pytest.raises(CatalogValidationError) as exc_info:
            prepare_for_write(entry, repo_with_target)
        assert any("pure pointer" in e for e in exc_info.value.errors)
        # Refused before anything reached the repository.
        assert repo_with_target.get("ns-1", "consumer") is None


class TestDanglingRefCannotHideInsideAMarker:
    """Defect: a ref nested in a marker escaped the bundle dangling-ref check.

    ``_iter_ref_targets`` treated a marker as a leaf while ``_walk_for_cross_ns``,
    over the same tree, descended into it — so a dangling ref written inside a
    marker was invisible to the check that exists to catch exactly that.

    Pinned here at the resolver, and on the bundle path itself by
    ``test_catalog_namespace_bundle.py::TestImportRejectsMarkerSiblings::
    test_a_dangling_ref_cannot_hide_inside_a_marker`` — the blind spot lived in
    the bundle check, so the refusal is asserted where the check runs.
    """

    def test_a_ref_nested_in_a_marker_is_unauthorable(
        self, repo_with_target: FakeEntryRepository
    ) -> None:
        marker = {"__ref__": "target", "params": {"x": {"__ref__": "does-not-exist"}}}
        with pytest.raises(CatalogValidationError) as exc_info:
            populate_refs(marker, repo_with_target, "ns-1")
        assert any("pure pointer" in e for e in exc_info.value.errors)


class TestCrossNsRefCannotHideFromTheDeleteGuard:
    """Defect: a cross-ns ref nested in a marker was invisible to the delete guard.

    ``_payload_has_cross_ns_ref`` treats a marker as a leaf, so a cross-namespace
    referrer written inside one never reached ``find_references_global`` — and
    ``validate_delete`` would green-light deleting a still-referenced entry.
    All three backends share that walker, so all three inherited it.
    """

    def test_marker_is_a_leaf_to_the_referrer_walker(self) -> None:
        """``_payload_has_ref`` terminates at a marker rather than descending.

        Asserted on the shared walker itself: YAML, Mongo and Postgres all
        import this function, so pinning it here pins every backend.
        """
        payload = {"prompt": {"__ref__": "target", "params": {"x": {"__ref__": "hidden"}}}}
        assert _payload_has_ref(payload, "target") is True
        assert _payload_has_ref(payload, "hidden") is False

    def test_marker_is_a_leaf_to_the_cross_ns_walker(self) -> None:
        """``_payload_has_cross_ns_ref`` is the walker the delete guard runs.

        Mongo and Postgres import it verbatim from the YAML module, so pinning
        it here pins the guarantee for all three tiers without a Docker-backed
        test.
        """
        payload = {
            "prompt": {
                "__ref__": "global.shared",
                "params": {"x": {"__ref__": "global.hidden"}},
            }
        }
        assert _payload_has_cross_ns_ref(payload, "global", "shared") is True
        assert _payload_has_cross_ns_ref(payload, "global", "hidden") is False

    def test_delete_guard_sees_a_plain_marker_referrer(
        self, repo_with_target: FakeEntryRepository, anything_model_type: str
    ) -> None:
        repo_with_target.put(
            make_entry(
                id="consumer",
                namespace="ns-1",
                model_type=anything_model_type,
                payload={"prompt": {"__ref__": "target"}},
            )
        )
        blockers = validate_delete("ns-1", "target", repo_with_target)
        assert len(blockers) == 1
        assert "consumer" in blockers[0]
