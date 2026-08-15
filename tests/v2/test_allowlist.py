"""Tests for the configurable ``model_type`` prefix allowlist — Story 28.1.

Split in two halves: unit tests for the policy module itself (parsing,
normalisation, validation, caching, permanence of ``akgentic.``), then
integration tests proving the four call sites — the two enforcement points and
the two enumeration helpers — read that one policy.

Every test in this directory runs under the autouse ``_isolate_allowlist_policy``
fixture in ``conftest.py``, which clears the environment variable and resets the
module global before and after each test.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

from akgentic.catalog.allowlist import (
    BASE_PREFIX,
    ENV_VAR,
    allowed_prefixes,
    parse_prefixes,
    prefix_violation,
    reset_allowed_prefixes,
    set_allowed_prefixes,
)
from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.catalog.resolver import (
    REF_KEY,
    enumerate_allowlisted_model_types,
    load_model_type,
    resolve,
)

from .conftest import FakeEntryRepository, make_entry, register_test_module

_CUSTOMER_PREFIX = "acme."
_CUSTOMER_MODULE = "acme.core.models"
_ENTRY_PATH = "akgentic.catalog.models.entry.Entry"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _customer_holder(leaf_cls: type[BaseModel], module_name: str) -> type[BaseModel]:
    """Build a deployment-owned holder model reporting ``module_name`` as its home.

    ``__module__`` is set explicitly because enumeration derives a class path
    from ``value.__module__``, and a class declared inside a test function
    would otherwise report the test module.
    """

    class Holder(BaseModel):
        case_id: str = ""
        leaf: leaf_cls | None = None  # type: ignore[valid-type]

    Holder.__module__ = module_name
    return Holder


def _model_with_reserved_field(reserved_name: str) -> type[BaseModel]:
    """Return a BaseModel subclass carrying ``reserved_name`` in ``model_fields``."""

    class Host(BaseModel):
        placeholder: str = ""

    Host.model_fields[reserved_name] = FieldInfo(annotation=str, default="")
    return Host


# --------------------------------------------------------------------------- #
# AC 25, 27, 28 — resolution order and caching
# --------------------------------------------------------------------------- #


class TestResolutionOrder:
    """Lazy env resolution, setter override, and reset semantics."""

    def test_unset_default_is_base_prefix_only(self) -> None:
        """AC 25 — with the variable absent the policy is exactly ``akgentic.``."""
        assert allowed_prefixes() == (BASE_PREFIX,)

    def test_env_var_widens_the_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_VAR, "acme.,contoso.models.")
        assert allowed_prefixes() == (BASE_PREFIX, "acme.", "contoso.models.")

    def test_setter_overrides_env_and_stops_it_being_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC 27 — once set explicitly, ``ENV_VAR`` is never consulted again."""
        monkeypatch.setenv(ENV_VAR, "a.")
        set_allowed_prefixes(["b."])
        assert allowed_prefixes() == (BASE_PREFIX, "b.")
        assert allowed_prefixes() == (BASE_PREFIX, "b.")

    def test_result_is_cached_until_reset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC 28 — a changed variable only takes effect after an explicit reset."""
        monkeypatch.setenv(ENV_VAR, "a.")
        assert allowed_prefixes() == (BASE_PREFIX, "a.")

        monkeypatch.setenv(ENV_VAR, "c.")
        assert allowed_prefixes() == (BASE_PREFIX, "a.")

        reset_allowed_prefixes()
        assert allowed_prefixes() == (BASE_PREFIX, "c.")

    def test_explicit_empty_does_not_fall_back_to_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC 5 — ``set_allowed_prefixes(None)`` means "no extras", not "read the env"."""
        monkeypatch.setenv(ENV_VAR, "a.")
        set_allowed_prefixes(None)
        assert allowed_prefixes() == (BASE_PREFIX,)
        set_allowed_prefixes([])
        assert allowed_prefixes() == (BASE_PREFIX,)


# --------------------------------------------------------------------------- #
# AC 5 — akgentic. is permanent
# --------------------------------------------------------------------------- #


class TestBasePrefixPermanence:
    """``akgentic.`` can never be removed from, or displaced within, the policy."""

    @pytest.mark.parametrize(
        "configured",
        [None, [], ["akgentic."], ["akgentic"], "akgentic.", "akgentic"],
        ids=["none", "empty", "base-dotted", "base-bare", "raw-dotted", "raw-bare"],
    )
    def test_base_prefix_survives(self, configured: list[str] | str | None) -> None:
        set_allowed_prefixes(configured)
        assert allowed_prefixes() == (BASE_PREFIX,)

    def test_base_prefix_stays_first_when_widened(self) -> None:
        set_allowed_prefixes(["acme.", "akgentic.", "contoso."])
        assert allowed_prefixes() == (BASE_PREFIX, "acme.", "contoso.")

    def test_parse_prefixes_does_not_strip_base_prefix(self) -> None:
        """AC 4 — stripping is ``allowed_prefixes``' job; the parser stays pure."""
        assert parse_prefixes(["akgentic.", "acme."]) == ("akgentic.", "acme.")


# --------------------------------------------------------------------------- #
# AC 6, 7, 8, 9, 10, 11 — parse_prefixes
# --------------------------------------------------------------------------- #


class TestParsePrefixes:
    """Format detection, normalisation, validation, and deduplication."""

    def test_both_env_formats_parse_identically(self) -> None:
        """AC 6 — comma-separated and JSON forms produce the same tuple."""
        expected = ("acme.", "contoso.models.")
        assert parse_prefixes("acme.,contoso.models.") == expected
        assert parse_prefixes('["acme.","contoso.models."]') == expected

    def test_whitespace_is_stripped_in_both_forms(self) -> None:
        expected = ("acme.", "contoso.models.")
        assert parse_prefixes("  acme. , contoso.models.  ") == expected
        assert parse_prefixes('  [" acme. ", "contoso.models."]  ') == expected

    @pytest.mark.parametrize(
        "raw",
        ['["a", 1]', '[{"a": 1}]', '["a"', "[", "[1]"],
        ids=["mixed-types", "list-of-dicts", "truncated", "bare-bracket", "int-list"],
    )
    def test_malformed_json_never_falls_back_to_comma_splitting(self, raw: str) -> None:
        """AC 7 — a value opening with ``[`` is JSON or it is an error."""
        with pytest.raises(ValueError, match="invalid model_type prefix"):
            parse_prefixes(raw)

    @pytest.mark.parametrize(
        "raw",
        [None, "", "   ", "[]"],
        ids=["none", "empty", "blank", "empty-json-list"],
    )
    def test_unset_means_unset(self, raw: str | None) -> None:
        """AC 8 — a declared-but-empty variable must not brick the process."""
        assert parse_prefixes(raw) == ()

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("acme", ("acme.",)),
            ("acme.core.models", ("acme.core.models.",)),
            ("acme.", ("acme.",)),
            (["acme", "contoso."], ("acme.", "contoso.")),
        ],
        ids=["bare", "bare-dotted-path", "already-dotted", "sequence-mixed"],
    )
    def test_trailing_dot_is_appended_when_missing(
        self, raw: str | list[str], expected: tuple[str, ...]
    ) -> None:
        """AC 9 — normalisation makes ``acme`` and ``acme.`` the same policy."""
        assert parse_prefixes(raw) == expected

    def test_a_plain_string_is_not_iterated_character_by_character(self) -> None:
        """A ``str`` is itself a ``Sequence[str]`` — the isinstance order matters."""
        assert parse_prefixes("acme.") == ("acme.",)

    def test_order_preserved_duplicates_dropped(self) -> None:
        """AC 10 — including duplicates that only collide after normalisation."""
        assert parse_prefixes("b.,a.,b.,a") == ("b.", "a.")

    @pytest.mark.parametrize(
        "prefixes",
        [
            [""],
            ["   "],
            ["."],
            [".."],
            ["*"],
            ["acme models"],
            ["acme-core."],
            ["1acme."],
        ],
        ids=[
            "empty-token",
            "blank-token",
            "single-dot",
            "double-dot",
            "star",
            "space-in-segment",
            "hyphen-in-segment",
            "leading-digit",
        ],
    )
    def test_rejected_inputs(self, prefixes: list[str]) -> None:
        """AC 11 — every rejection carries the one grep-stable substring."""
        with pytest.raises(ValueError, match="invalid model_type prefix"):
            parse_prefixes(prefixes)

    def test_stray_comma_producing_an_empty_token_is_an_error(self) -> None:
        """AC 9 — an empty *token* is a typo; only a wholly empty *value* is "unset"."""
        with pytest.raises(ValueError, match="invalid model_type prefix"):
            parse_prefixes("acme.,,contoso.")

    def test_set_allowed_prefixes_rejects_the_same_inputs(self) -> None:
        with pytest.raises(ValueError, match="invalid model_type prefix"):
            set_allowed_prefixes(["acme-core."])

    def test_allowed_prefixes_surfaces_a_malformed_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_VAR, "acme-core.")
        with pytest.raises(ValueError, match="invalid model_type prefix"):
            allowed_prefixes()


# --------------------------------------------------------------------------- #
# AC 29, 30, 35 — the two enforcement points
# --------------------------------------------------------------------------- #


class TestEnforcement:
    """``Entry`` construction and ``load_model_type`` read the same policy."""

    def test_prefix_violation_is_the_shared_check(self) -> None:
        """The one predicate-and-message both enforcement points call, both branches."""
        set_allowed_prefixes([_CUSTOMER_PREFIX])

        assert prefix_violation(_ENTRY_PATH) is None
        assert prefix_violation(f"{_CUSTOMER_MODULE}.CaseIngestionConfig") is None
        assert prefix_violation("contoso.models.Thing") == (
            "model_type 'contoso.models.Thing' outside allowlist ('akgentic.', 'acme.')"
        )

    def test_entry_construction_widens_with_the_policy(self) -> None:
        """AC 29 — the identical construction is accepted or rejected by policy alone."""
        path = f"{_CUSTOMER_MODULE}.CaseIngestionConfig"

        with pytest.raises(ValidationError) as unconfigured:
            make_entry(model_type=path)
        assert "outside allowlist" in str(unconfigured.value)

        set_allowed_prefixes([_CUSTOMER_PREFIX])
        entry = make_entry(model_type=path)
        assert entry.model_type == path

    def test_both_layers_agree(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC 30 — the split-brain guard: assert both layers as a pair, in one test."""

        class Leaf(BaseModel):
            provider: str = "openai"

        holder = _customer_holder(Leaf, _CUSTOMER_MODULE)
        module_name = register_test_module(monkeypatch, _CUSTOMER_MODULE, Holder=holder)
        configured = f"{module_name}.Holder"
        unconfigured = "contoso.models.Thing"

        set_allowed_prefixes([_CUSTOMER_PREFIX])

        # Configured prefix — both layers accept the same path.
        assert make_entry(model_type=configured).model_type == configured
        assert load_model_type(configured) is holder

        # Unconfigured prefix — both layers reject the same path.
        with pytest.raises(ValidationError) as entry_exc:
            make_entry(model_type=unconfigured)
        assert "outside allowlist" in str(entry_exc.value)
        with pytest.raises(CatalogValidationError) as loader_exc:
            load_model_type(unconfigured)
        assert "outside allowlist" in loader_exc.value.errors[0]

    def test_error_message_names_the_configured_set(self) -> None:
        """AC 15 + AC 35 — the rejection renders the live tuple, both layers."""
        set_allowed_prefixes(["acme."])
        rendered = "outside allowlist ('akgentic.', 'acme.')"

        with pytest.raises(ValidationError) as entry_exc:
            make_entry(model_type="contoso.models.Thing")
        assert rendered in str(entry_exc.value)

        with pytest.raises(CatalogValidationError) as loader_exc:
            load_model_type("contoso.models.Thing")
        assert rendered in loader_exc.value.errors[0]

    def test_error_message_is_unchanged_when_unset(self) -> None:
        """AC 15 — byte-identical to the pre-story rendering."""
        with pytest.raises(CatalogValidationError) as loader_exc:
            load_model_type("os.system")
        assert "outside allowlist ('akgentic.',)" in loader_exc.value.errors[0]


class TestOtherGatesStillFireInsideAWidenedPrefix:
    """AC 32 — widening the prefix must not weaken the other two checks."""

    def test_non_basemodel_inside_a_configured_prefix_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class PlainClass:
            """Deployment-owned, but not a Pydantic model."""

        module_name = register_test_module(monkeypatch, _CUSTOMER_MODULE, PlainClass=PlainClass)
        set_allowed_prefixes([_CUSTOMER_PREFIX])
        with pytest.raises(CatalogValidationError) as exc_info:
            load_model_type(f"{module_name}.PlainClass")
        assert "is not a Pydantic BaseModel subclass" in exc_info.value.errors[0]

    def test_reserved_ref_sentinel_fields_inside_a_configured_prefix_are_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        colliding = _model_with_reserved_field(REF_KEY)
        module_name = register_test_module(monkeypatch, _CUSTOMER_MODULE, CollidingModel=colliding)
        set_allowed_prefixes([_CUSTOMER_PREFIX])
        with pytest.raises(CatalogValidationError) as exc_info:
            load_model_type(f"{module_name}.CollidingModel")
        assert "reserved ref-sentinel fields" in exc_info.value.errors[0]


# --------------------------------------------------------------------------- #
# AC 31 — end-to-end resolution across the two namespaces
# --------------------------------------------------------------------------- #


class TestEndToEndResolution:
    """A deployment-typed entry resolves, and can reference a framework-typed one."""

    def test_customer_entry_resolves_with_a_ref_to_a_framework_typed_sibling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Leaf(BaseModel):
            provider: str = "openai"

        framework_module = register_test_module(
            monkeypatch, "akgentic.tests_fixture_28_1_leaf", Leaf=Leaf
        )
        holder = _customer_holder(Leaf, _CUSTOMER_MODULE)
        customer_module = register_test_module(monkeypatch, _CUSTOMER_MODULE, Holder=holder)
        set_allowed_prefixes([_CUSTOMER_PREFIX])

        repo = FakeEntryRepository()
        repo.put(
            make_entry(
                id="leaf-1",
                kind="model",
                namespace="ns-1",
                model_type=f"{framework_module}.Leaf",
                payload={"provider": "anthropic"},
            )
        )
        entry = make_entry(
            id="case-1",
            namespace="ns-1",
            model_type=f"{customer_module}.Holder",
            payload={"case_id": "c-1", "leaf": {REF_KEY: "leaf-1"}},
        )

        result = resolve(entry, repo)
        assert isinstance(result, holder)
        assert result.case_id == "c-1"
        assert result.leaf is not None
        assert result.leaf.provider == "anthropic"


# --------------------------------------------------------------------------- #
# AC 33, 34 — enumeration
# --------------------------------------------------------------------------- #


class TestEnumeration:
    """``enumerate_allowlisted_model_types`` widens with the policy, imports nothing."""

    def test_unchanged_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC 34 — an unconfigured process enumerates exactly what it did before."""
        import akgentic.catalog.models.entry  # noqa: F401 — populate sys.modules

        holder = _customer_holder(BaseModel, _CUSTOMER_MODULE)
        register_test_module(monkeypatch, _CUSTOMER_MODULE, Holder=holder)

        paths = enumerate_allowlisted_model_types()
        assert _ENTRY_PATH in paths
        assert all(p.startswith(BASE_PREFIX) for p in paths)
        assert f"{_CUSTOMER_MODULE}.Holder" not in paths

    def test_widens_with_a_configured_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC 33 — the deployment class appears alongside the framework ones."""
        import akgentic.catalog.models.entry  # noqa: F401 — populate sys.modules

        holder = _customer_holder(BaseModel, _CUSTOMER_MODULE)
        register_test_module(monkeypatch, _CUSTOMER_MODULE, Holder=holder)
        set_allowed_prefixes([_CUSTOMER_PREFIX])

        paths = enumerate_allowlisted_model_types()
        assert f"{_CUSTOMER_MODULE}.Holder" in paths
        assert _ENTRY_PATH in paths

    def test_matches_a_module_named_exactly_the_prefix_minus_its_dot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC 16 + AC 33 — ``"a.b".startswith("a.b.")`` is False; the exact arm covers it."""
        holder = _customer_holder(BaseModel, _CUSTOMER_MODULE)
        register_test_module(monkeypatch, _CUSTOMER_MODULE, Holder=holder)
        set_allowed_prefixes([f"{_CUSTOMER_MODULE}."])

        assert f"{_CUSTOMER_MODULE}.Holder" in enumerate_allowlisted_model_types()

    def test_absent_module_yields_nothing_and_does_not_raise(self) -> None:
        """AC 33 + AC 18 — enumeration never imports; a configured-but-unloaded
        namespace simply contributes nothing."""
        set_allowed_prefixes([_CUSTOMER_PREFIX])

        paths = enumerate_allowlisted_model_types()
        assert not any(p.startswith(_CUSTOMER_PREFIX) for p in paths)
        assert _ENTRY_PATH in paths
