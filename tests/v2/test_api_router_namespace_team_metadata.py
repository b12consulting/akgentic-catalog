"""``GET /catalog/namespaces`` — the row carries the team's metadata contract.

Story 37.1. The projection, the two DTOs, and the failure matrix that keeps a
single bad declaration from blanking the team-creation picker.

Every fixture metadata model is registered under an ``akgentic.`` module name
via ``register_akgentic_test_module``: a class declared in a test module has
the dotted path ``tests.v2.<module>.<Class>``, which the default prefix
allowlist refuses — so a happy-path test written that way would silently
exercise the *failure* branch and pass for the wrong reason.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Annotated, Any

import pytest

pytest.importorskip("fastapi")

from akgentic.team import TeamMetadata  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import BaseModel, Field, StringConstraints  # noqa: E402

from akgentic.catalog.api import router as router_module  # noqa: E402
from akgentic.catalog.api.router import (  # noqa: E402
    MetadataFieldDescriptor,
    NamespaceSummary,
    TeamMetadataContract,
    _build_namespace_summary,
    _describe_metadata_fields,
    _project_team_metadata,
    _zero_counts,
)
from akgentic.catalog.catalog import Catalog  # noqa: E402
from akgentic.catalog.models.entry import Entry  # noqa: E402

from ..conftest import team_payload  # noqa: E402
from .conftest import (  # noqa: E402
    _TEAM_TYPE,
    _seed_team,
    make_meta_entry,
    register_akgentic_test_module,
    register_test_module,
)

_ROUTER_LOGGER = "akgentic.catalog.api.router"


# --- Fixture metadata models -------------------------------------------------
#
# Field names are deliberately NOT in alphabetical order, so an implementation
# that sorts instead of preserving declaration order fails.


class DeclarationOrderMeta(TeamMetadata):
    """Three fields whose declaration order is not their alphabetical order."""

    zone: str = Field(description="Delivery zone", json_schema_extra={"indexed": True})
    account: str = Field(default="", description="Billing account code")
    beta: bool = False


class EmptyMeta(TeamMetadata):
    """A declared contract with no fields at all — a real, reachable state."""


class MandatoryCasesMeta(BaseModel):
    """The four shapes ``mandatory`` has to tell apart."""

    plain_required: str
    with_default: str = "d"
    nullable_with_default: str | None = None
    nullable_no_default: str | None


# The two ordinary spellings of a pattern declaration share one regex, so a
# test can assert they report the identical string rather than two literals
# that only look alike.
_SHARED_PATTERN = r"^acme-[0-9]{3}$"
_ALIASED_PATTERN = r"^[A-Z]{2}$"
_NULLABLE_PATTERN = r"^[0-9]{4}$"


class NestedValue(BaseModel):
    """A field type that schematises to a ``$ref`` rather than to a scalar."""

    inner: str = ""


class PatternMeta(BaseModel):
    """The patterned declaration shapes a model author actually writes, on one model.

    ``plain``, ``constrained`` and ``bounded`` spell the same constraint
    three ways: the ordinary ``Field(pattern=...)``, the
    ``StringConstraints`` form, and a pattern alongside a second constraint.
    Only the third has anything to say about the *source* — its
    ``FieldInfo.metadata`` is ``[MinLen(3), <private object>]``, so the
    pattern is not at index 0. ``constrained`` deliberately does **not**
    discriminate: it collapses to one public ``StringConstraints`` object
    that carries ``.pattern``, which is why ``bounded`` is the field the
    source test pins. ``aliased`` is keyed by its alias in the default
    schema and ``nullable`` has its constraint pushed into an ``anyOf``
    branch; those two are the shapes a naive lookup drops. ``nested``
    schematises to a ``$ref`` and declares no pattern at all.
    """

    plain: str = Field(pattern=_SHARED_PATTERN)
    constrained: Annotated[str, StringConstraints(min_length=3, pattern=_SHARED_PATTERN)]
    bounded: str = Field(min_length=3, pattern=_SHARED_PATTERN)
    aliased: str = Field(alias="aliasedName", pattern=_ALIASED_PATTERN)
    nullable: str | None = Field(default=None, pattern=_NULLABLE_PATTERN)
    nested: NestedValue = Field(default_factory=NestedValue)


def _mutate_schema(schema: dict[str, Any]) -> None:
    """A ``json_schema_extra`` **callable** — legal, and never an index marker."""
    schema["x-acme-hint"] = "free text"


class BaseIndexedMeta(TeamMetadata):
    """An intermediate subclass whose own field is marked indexed."""

    tenant: str = Field(description="Owning tenant", json_schema_extra={"indexed": True})


class InheritedIndexMeta(BaseIndexedMeta):
    """Inherits a marked field, adds a marked one and a callable-extra one.

    ``indexed_fields()`` reports ``tenant`` **first** (Pydantic collects
    base-class fields before the subclass's own) and skips ``region``
    entirely, because a callable ``json_schema_extra`` is not a dict and
    therefore carries no marker. Both are shapes a re-derived predicate
    gets wrong.
    """

    region: str = Field(default="", json_schema_extra=_mutate_schema)
    tier: str = Field(default="", json_schema_extra={"indexed": True})


class PlainMeta(BaseModel):
    """Not a ``TeamMetadata`` subclass — legal, and carries no index contract.

    ``label`` is marked with the very same ``json_schema_extra`` payload a
    ``TeamMetadata`` subclass would use, so a projection that read the
    marker directly instead of calling ``indexed_fields()`` would report
    ``index=True`` here and be wrong: the write path indexes nothing for a
    plain model.
    """

    label: str = Field(default="", json_schema_extra={"indexed": True})


class NotAModel:
    """A resolvable class that is not a ``BaseModel`` — the last resolution gate."""


class UndescribableMeta(BaseModel):
    """Resolves cleanly, then breaks *introspection*: a non-string description.

    ``Field(description=...)`` is not validated at class-definition time, so
    this class is legal and ``load_model_type`` accepts it — the failure lands
    one step later, building the descriptor. It stands in for every way a
    declared model can pass the resolution gates and still defeat the
    projection.
    """

    broken: str = Field(default="", description=123)  # type: ignore[arg-type]


def _noop() -> None:
    """A default for a field pydantic validates happily and refuses to schematise."""


class UnschematisableMeta(BaseModel):
    """Resolves and introspects cleanly, then defeats *schema generation*.

    Pydantic accepts a ``Callable`` annotation and validates it, but there is
    no JSON Schema for a function — ``model_json_schema()`` raises
    ``PydanticInvalidForJsonSchema``. It stands in for the second way a
    declared model can clear every resolution gate and still defeat the
    projection: the failure lands on the schema call rather than on a
    ``FieldInfo`` read.
    """

    hook: Callable[[], None] = _noop


@pytest.fixture
def meta_module(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register every fixture model under one allowlisted ``akgentic.`` module."""
    return register_akgentic_test_module(
        monkeypatch,
        "meta_fixtures_37",
        DeclarationOrderMeta=DeclarationOrderMeta,
        EmptyMeta=EmptyMeta,
        MandatoryCasesMeta=MandatoryCasesMeta,
        InheritedIndexMeta=InheritedIndexMeta,
        PlainMeta=PlainMeta,
        PatternMeta=PatternMeta,
        NotAModel=NotAModel,
        UndescribableMeta=UndescribableMeta,
        UnschematisableMeta=UnschematisableMeta,
    )


def _seed_declaring_team(
    catalog: Catalog,
    namespace: str,
    declared: Any,
    *,
    user_id: str = "anonymous",
    description: str = "",
) -> Entry:
    """Seed a team entry whose payload carries ``metadata_type`` verbatim.

    Goes through the repository rather than ``Catalog.create``: a payload key
    the resolved ``TeamCard`` does not declare is a validation error, and
    whether the *installed* ``akgentic-team`` declares ``metadata_type``
    depends on which version resolved (workspace checkout vs. the PyPI floor).
    Seeding out of band sidesteps that question entirely — the same idiom the
    sibling namespace module uses for out-of-band states.
    """
    payload = team_payload()
    payload["metadata_type"] = declared
    return catalog._repository.put(
        Entry(
            id="team",
            kind="team",
            namespace=namespace,
            user_id=user_id,
            model_type=_TEAM_TYPE,
            description=description,
            payload=payload,
        )
    )


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return the captured records at ``WARNING`` or above."""
    return [record for record in caplog.records if record.levelno >= logging.WARNING]


# --- The DTOs ----------------------------------------------------------------


class TestContractDtoShape:
    """The two appended DTOs: their fields, their order, their export."""

    def test_descriptor_ships_exactly_five_fields_in_order(self) -> None:
        assert list(MetadataFieldDescriptor.model_fields.keys()) == [
            "key",
            "description",
            "index",
            "mandatory",
            "pattern",
        ]

    def test_pattern_defaults_to_none_on_the_descriptor(self) -> None:
        """Appended with a default — a construction site predating it keeps working.

        The server nonetheless passes ``pattern`` explicitly on every
        descriptor it builds, so the key is on the wire for every field.
        """
        descriptor = MetadataFieldDescriptor(
            key="zone", description="Delivery zone", index=True, mandatory=True
        )
        assert descriptor.pattern is None

    def test_contract_ships_exactly_two_fields_in_order(self) -> None:
        assert list(TeamMetadataContract.model_fields.keys()) == ["type", "fields"]

    def test_both_are_exported_alongside_namespace_summary(self) -> None:
        from akgentic.catalog.api import router

        assert "MetadataFieldDescriptor" in router.__all__
        assert "TeamMetadataContract" in router.__all__
        assert "NamespaceSummary" in router.__all__
        assert router.__all__ == sorted(router.__all__)

    def test_team_metadata_defaults_to_none_on_the_row(self) -> None:
        """The appended field is optional — an omitted contract is the default."""
        row = NamespaceSummary(
            namespace="ns",
            name="ns",
            description="",
            team=True,
            shareable=False,
            public=False,
            owner="anonymous",
            counts=_zero_counts(),
        )
        assert row.team_metadata is None


class TestDescribeMetadataFields:
    """``_describe_metadata_fields`` — order, descriptions, and the empty case."""

    def test_fields_come_back_in_declaration_order(self) -> None:
        """Not alphabetical: a sorted implementation returns ``account`` first."""
        descriptors = _describe_metadata_fields(DeclarationOrderMeta)
        assert [d.key for d in descriptors] == ["zone", "account", "beta"]

    def test_description_is_the_declared_one(self) -> None:
        by_key = {d.key: d for d in _describe_metadata_fields(DeclarationOrderMeta)}
        assert by_key["zone"].description == "Delivery zone"
        assert by_key["account"].description == "Billing account code"

    def test_description_is_empty_string_when_undeclared(self) -> None:
        """``""``, never ``None`` — matching ``NamespaceSummary.description``."""
        by_key = {d.key: d for d in _describe_metadata_fields(DeclarationOrderMeta)}
        assert by_key["beta"].description == ""

    def test_a_model_with_no_fields_yields_an_empty_list(self) -> None:
        assert _describe_metadata_fields(EmptyMeta) == []


class TestIndexIsReadFromTeamMetadata:
    """``index`` is ``TeamMetadata.indexed_fields()``, never a second derivation."""

    def test_marked_fields_are_indexed_and_unmarked_are_not(self) -> None:
        by_key = {d.key: d for d in _describe_metadata_fields(DeclarationOrderMeta)}
        assert by_key["zone"].index is True
        assert by_key["account"].index is False
        assert by_key["beta"].index is False

    def test_projection_agrees_with_indexed_fields_on_the_awkward_shapes(self) -> None:
        """The two shapes a re-implemented marker predicate gets wrong.

        ``tenant`` is inherited from an intermediate subclass — indexed, and
        **first** in declaration order. ``region``'s ``json_schema_extra`` is
        a callable, which carries no marker at all. Asserting equality with
        ``indexed_fields()`` rather than a hand-written expectation is the
        point: the two can only agree if there is one derivation.
        """
        descriptors = _describe_metadata_fields(InheritedIndexMeta)
        assert [d.key for d in descriptors] == ["tenant", "region", "tier"]
        assert {d.key for d in descriptors if d.index} == set(InheritedIndexMeta.indexed_fields())
        assert InheritedIndexMeta.indexed_fields() == ["tenant", "tier"]

    def test_a_non_team_metadata_model_is_legal_and_indexes_nothing(self) -> None:
        """Mirrors ``derive_metadata_indexes``'s own non-``TeamMetadata`` branch.

        ``PlainMeta.label`` carries the identical marker payload; the write
        path still indexes nothing for a plain model, so the descriptor must
        not claim otherwise.
        """
        descriptors = _describe_metadata_fields(PlainMeta)
        assert [d.key for d in descriptors] == ["label"]
        assert all(d.index is False for d in descriptors)


class TestMandatoryIsRequiredAndNotNullable:
    """``mandatory`` is *required and not nullable* — not ``is_required()``."""

    def _by_key(self) -> dict[str, MetadataFieldDescriptor]:
        return {d.key: d for d in _describe_metadata_fields(MandatoryCasesMeta)}

    def test_required_field_is_mandatory(self) -> None:
        """``x: str`` → ``True``."""
        assert self._by_key()["plain_required"].mandatory is True

    def test_field_with_a_default_is_not_mandatory(self) -> None:
        """``x: str = "d"`` → ``False``."""
        assert self._by_key()["with_default"].mandatory is False

    def test_nullable_field_with_a_default_is_not_mandatory(self) -> None:
        """``x: str | None = None`` → ``False``."""
        assert self._by_key()["nullable_with_default"].mandatory is False

    def test_required_nullable_field_without_a_default_is_not_mandatory(self) -> None:
        """``x: str | None`` with no default → ``False``. The one that trips people.

        Pydantic calls it required — a key must be present in the input. A
        user may legitimately answer ``None``, so a form must not star it.
        ``is_required()`` alone reports ``True`` here.
        """
        field = MandatoryCasesMeta.model_fields["nullable_no_default"]
        assert field.is_required() is True
        assert self._by_key()["nullable_no_default"].mandatory is False


class TestPatternComesFromJsonSchema:
    """``pattern`` is the field's regex, read from the model's **JSON Schema**.

    The source route is the whole point. JSON Schema is public API and
    defines ``pattern`` as ECMA-262 — the dialect a browser's ``RegExp``
    parses — where ``FieldInfo.metadata`` exposes a private Pydantic class
    in a dialect-neutral wrapper.
    """

    def _schema(self) -> dict[str, Any]:
        return PatternMeta.model_json_schema(by_alias=False)

    def _by_key(self) -> dict[str, MetadataFieldDescriptor]:
        return {d.key: d for d in _describe_metadata_fields(PatternMeta)}

    def test_an_ordinary_field_pattern_is_reported(self) -> None:
        """``Field(pattern=...)`` — what a model author actually writes."""
        expected = self._schema()["properties"]["plain"]["pattern"]
        assert self._by_key()["plain"].pattern == expected

    def test_string_constraints_report_the_same_pattern(self) -> None:
        """The second spelling of the same constraint agrees with the first."""
        by_key = self._by_key()
        expected = self._schema()["properties"]["constrained"]["pattern"]
        assert by_key["constrained"].pattern == expected
        assert by_key["constrained"].pattern == by_key["plain"].pattern

    def test_the_private_field_info_route_could_not_produce_this(self) -> None:
        """The discriminator: ``metadata[0]`` here is ``MinLen``, not a pattern carrier.

        ``Field(min_length=3, pattern=...)`` — an ordinary declaration, not a
        contrivance — expands to ``[MinLen(3), <private general metadata>]``.
        A projection reading ``field.metadata[0].pattern`` raises
        ``AttributeError`` on this field and could not produce the value
        asserted below; the object at index 1 that *does* carry it is
        precisely the private Pydantic class this projection refuses to
        depend on.
        """
        first = PatternMeta.model_fields["bounded"].metadata[0]
        assert not hasattr(first, "pattern")

        expected = self._schema()["properties"]["bounded"]["pattern"]
        assert self._by_key()["bounded"].pattern == expected

    def test_an_aliased_field_matches_its_own_descriptor(self) -> None:
        """``by_alias=False`` keys ``properties`` by field name — the descriptor's ``key``."""
        expected = self._schema()["properties"]["aliased"]["pattern"]
        assert self._by_key()["aliased"].pattern == expected

    def test_the_alias_keyed_schema_is_the_wrong_one_to_read(self) -> None:
        """Why ``by_alias=False`` is load-bearing rather than decorative.

        ``model_json_schema()`` defaults to ``by_alias=True``, and the
        aliased field is then absent under its own name — a lookup keyed by
        ``model_fields`` order would report ``None`` for a field that plainly
        declares a pattern.
        """
        by_alias = PatternMeta.model_json_schema()["properties"]
        assert "aliased" not in by_alias
        assert "aliasedName" in by_alias

    def test_a_nullable_fields_pattern_lives_under_any_of_and_is_still_found(self) -> None:
        """``x: str | None`` is the declaration shape model authors are told to write."""
        prop = self._schema()["properties"]["nullable"]
        assert "pattern" not in prop
        expected = next(
            branch["pattern"] for branch in prop["anyOf"] if isinstance(branch.get("pattern"), str)
        )
        assert self._by_key()["nullable"].pattern == expected

    def test_a_ref_valued_property_carries_no_pattern(self) -> None:
        """A nested model schematises to a ``$ref`` — no ``pattern`` key, and no crash.

        The helper documents this shape as coming back through the same
        single ``None`` path as an unpatterned scalar; nothing else in the
        suite exercises a property that is neither a scalar nor an
        ``anyOf``.
        """
        prop = self._schema()["properties"]["nested"]
        assert "pattern" not in prop
        assert self._by_key()["nested"].pattern is None

    def test_a_field_with_no_pattern_and_a_non_str_field_both_report_none(self) -> None:
        """One helper, one answer — no special-casing, and never an empty string."""
        by_key = {d.key: d for d in _describe_metadata_fields(DeclarationOrderMeta)}
        assert by_key["zone"].pattern is None
        assert by_key["account"].pattern is None
        assert by_key["beta"].pattern is None

    def test_the_reference_metadata_model_declares_no_pattern_today(self) -> None:
        """The shipped worked example: four fields, none constrained. Correct state.

        Guarded — whether this class resolves depends on which
        ``akgentic-team`` the environment picked up, and a hard import would
        turn a floor-version run red for no behavioural reason.
        """
        try:
            from akgentic.team import ReferenceTeamMetadata
        except ImportError:  # pragma: no cover — depends on the resolved akgentic-team
            pytest.skip("the resolved akgentic-team ships no ReferenceTeamMetadata")

        descriptors = _describe_metadata_fields(ReferenceTeamMetadata)
        assert descriptors != []
        assert all(d.pattern is None for d in descriptors)


# --- Resolution --------------------------------------------------------------


class TestResolutionGoesThroughTheAllowlist:
    """The declared path is resolved by ``load_model_type``, never a bare import."""

    def test_load_model_type_is_the_call_that_fires(
        self,
        api_client: tuple[TestClient, Catalog],
        meta_module: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, catalog = api_client
        path = f"{meta_module}.DeclarationOrderMeta"
        _seed_declaring_team(catalog, "ns-declaring", {"__type__": path})

        seen: list[str] = []
        original = router_module.load_model_type

        def _spy(dotted: str) -> type[BaseModel]:
            seen.append(dotted)
            return original(dotted)

        monkeypatch.setattr(router_module, "load_model_type", _spy)

        rows = client.get("/catalog/namespaces").json()
        assert seen == [path]
        assert rows[0]["team_metadata"]["type"] == path

    def test_a_non_allowlisted_path_is_refused_without_importing_it(
        self,
        api_client: tuple[TestClient, Catalog],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The prefix gate runs *before* the import — that is the whole point.

        The fixture model is registered under a non-allowlisted module that
        is already in ``sys.modules``, so it *would* import cleanly. The row
        still carries ``None``, and ``import_class`` is never reached: an
        import executes the target module's top-level code, which is exactly
        what the allowlist keeps off an HTTP read path.
        """
        from akgentic.catalog import model_types

        client, catalog = api_client
        module = register_test_module(monkeypatch, "acme.meta_37", AcmeMeta=DeclarationOrderMeta)
        path = f"{module}.AcmeMeta"

        imported: list[str] = []
        original = model_types.import_class

        def _spy(dotted: str) -> type[Any]:
            imported.append(dotted)
            return original(dotted)

        monkeypatch.setattr(model_types, "import_class", _spy)

        _seed_declaring_team(catalog, "ns-outside", {"__type__": path})
        with caplog.at_level(logging.WARNING, logger=_ROUTER_LOGGER):
            response = client.get("/catalog/namespaces")

        assert response.status_code == 200
        assert response.json()[0]["team_metadata"] is None
        assert path not in imported
        assert len(_warnings(caplog)) == 1


class TestContractProjectedOntoTheRow:
    """The happy paths, end to end through the route."""

    def test_a_declaring_namespace_carries_its_contract(
        self, api_client: tuple[TestClient, Catalog], meta_module: str
    ) -> None:
        client, catalog = api_client
        path = f"{meta_module}.DeclarationOrderMeta"
        _seed_declaring_team(catalog, "ns-declaring", {"__type__": path})

        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        assert response.json()[0]["team_metadata"] == {
            "type": path,
            "fields": [
                {
                    "key": "zone",
                    "description": "Delivery zone",
                    "index": True,
                    "mandatory": True,
                    "pattern": None,
                },
                {
                    "key": "account",
                    "description": "Billing account code",
                    "index": False,
                    "mandatory": False,
                    "pattern": None,
                },
                {
                    "key": "beta",
                    "description": "",
                    "index": False,
                    "mandatory": False,
                    "pattern": None,
                },
            ],
        }

    def test_a_declared_pattern_reaches_the_response_body(
        self, api_client: tuple[TestClient, Catalog], meta_module: str
    ) -> None:
        """The regex arrives at the client, not merely at the helper that builds it.

        Every other wire-level expectation in this module pins ``pattern:
        null``, which a route that dropped the value on the way out would
        still satisfy. This one declares a model that carries patterns and
        reads them back off the HTTP response — including the two shapes
        the projection has to work for, an aliased field and a nullable one.
        """
        client, catalog = api_client
        path = f"{meta_module}.PatternMeta"
        _seed_declaring_team(catalog, "ns-patterned", {"__type__": path})

        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        contract = response.json()[0]["team_metadata"]
        on_the_wire = {field["key"]: field["pattern"] for field in contract["fields"]}

        properties = PatternMeta.model_json_schema(by_alias=False)["properties"]
        assert on_the_wire["plain"] == properties["plain"]["pattern"]
        assert on_the_wire["aliased"] == properties["aliased"]["pattern"]
        assert on_the_wire["nullable"] == next(
            branch["pattern"]
            for branch in properties["nullable"]["anyOf"]
            if isinstance(branch.get("pattern"), str)
        )
        assert on_the_wire["nested"] is None

    def test_a_declared_type_with_no_fields_is_a_contract_not_a_none(
        self, api_client: tuple[TestClient, Catalog], meta_module: str
    ) -> None:
        """The two states are distinct: ``None`` vs. a contract with no fields."""
        client, catalog = api_client
        path = f"{meta_module}.EmptyMeta"
        _seed_declaring_team(catalog, "ns-empty-contract", {"__type__": path})
        _seed_team(catalog, "ns-no-contract")

        by_ns = {r["namespace"]: r for r in client.get("/catalog/namespaces").json()}
        assert by_ns["ns-empty-contract"]["team_metadata"] == {"type": path, "fields": []}
        assert by_ns["ns-no-contract"]["team_metadata"] is None

    def test_the_type_is_the_declared_path_verbatim_not_a_re_derived_one(
        self, api_client: tuple[TestClient, Catalog], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Re-exports and aliases mean the two can differ; the client needs the
        string that is actually in the catalog."""
        client, catalog = api_client
        alias = register_akgentic_test_module(
            monkeypatch, "meta_alias_37", Aliased=DeclarationOrderMeta
        )
        path = f"{alias}.Aliased"
        _seed_declaring_team(catalog, "ns-aliased", {"__type__": path})

        contract = client.get("/catalog/namespaces").json()[0]["team_metadata"]
        assert contract["type"] == path
        # The re-derived path would name the class's real home instead.
        assert path != f"{DeclarationOrderMeta.__module__}.{DeclarationOrderMeta.__qualname__}"

    def test_a_meta_only_namespace_carries_no_contract(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """No team entry, no card to read."""
        client, catalog = api_client
        catalog._repository.put(make_meta_entry("ns-library", name="Library"))

        row = client.get("/catalog/namespaces").json()[0]
        assert row["team"] is False
        assert row["team_metadata"] is None


# --- Failure --------------------------------------------------------------


def _unusable_cases() -> list[Any]:
    """The present-but-unusable declarations, with the token each warning must name."""
    return [
        pytest.param("akgentic.team.TeamMetadata", "akgentic.team.TeamMetadata", id="not-a-dict"),
        pytest.param({}, "{}", id="no-type-key"),
        pytest.param({"__type__": 123}, "123", id="type-is-an-int"),
        pytest.param({"__type__": None}, "None", id="type-is-none"),
        pytest.param({"__type__": ["a"]}, "'a'", id="type-is-a-list"),
        pytest.param(
            {"__type__": "acme.not_allowlisted.Meta"},
            "acme.not_allowlisted.Meta",
            id="outside-allowlist",
        ),
        pytest.param(
            {"__type__": "akgentic.does.not.Exist"},
            "akgentic.does.not.Exist",
            id="unimportable",
        ),
        pytest.param({"__type__": "NotDotted"}, "NotDotted", id="bare-undotted-path"),
    ]


class TestUnusableDeclarationDegradesToNone:
    """Every present-but-unusable declaration: ``None``, no raise, one ``WARNING``."""

    @pytest.mark.parametrize(("declared", "token"), _unusable_cases())
    def test_row_survives_with_a_null_contract_and_one_warning(
        self,
        api_client: tuple[TestClient, Catalog],
        caplog: pytest.LogCaptureFixture,
        declared: Any,
        token: str,
    ) -> None:
        client, catalog = api_client
        _seed_declaring_team(catalog, "ns-broken", declared, description="still described")

        with caplog.at_level(logging.WARNING, logger=_ROUTER_LOGGER):
            response = client.get("/catalog/namespaces")

        assert response.status_code == 200
        row = response.json()[0]
        assert row["team_metadata"] is None
        # The rest of the row is fully populated — a bad declaration costs
        # the contract and nothing else.
        assert row["namespace"] == "ns-broken"
        assert row["name"] == "ns-broken"
        assert row["description"] == "still described"
        assert row["team"] is True
        assert row["shareable"] is False
        assert row["public"] is False
        assert row["owner"] == "anonymous"
        assert row["counts"]["team"]["total"] == 1

        records = _warnings(caplog)
        assert len(records) == 1
        message = records[0].getMessage()
        assert "ns-broken" in message
        assert token in message

    def test_a_type_resolving_to_a_non_basemodel_is_refused(
        self,
        api_client: tuple[TestClient, Catalog],
        meta_module: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Importable and allowlisted, but not a Pydantic model — still ``None``."""
        client, catalog = api_client
        path = f"{meta_module}.NotAModel"
        _seed_declaring_team(catalog, "ns-not-a-model", {"__type__": path})

        with caplog.at_level(logging.WARNING, logger=_ROUTER_LOGGER):
            response = client.get("/catalog/namespaces")

        assert response.status_code == 200
        assert response.json()[0]["team_metadata"] is None
        records = _warnings(caplog)
        assert len(records) == 1
        assert "ns-not-a-model" in records[0].getMessage()
        assert path in records[0].getMessage()


class TestAbsentDeclarationIsSilent:
    """No ``metadata_type`` key at all → ``None``, and **no log record**."""

    def test_the_projection_emits_nothing_for_a_non_declaring_team(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Asserted at the helper, where silence means literally no record.

        This is the state of every namespace shipped today. A ``WARNING``
        here would fire once per non-declaring namespace per request and
        make the log useless on the first deployment that read it.
        """
        entry = Entry(
            id="team",
            kind="team",
            namespace="ns-quiet",
            user_id="anonymous",
            model_type=_TEAM_TYPE,
            payload=team_payload(),
        )
        with caplog.at_level(logging.DEBUG, logger=_ROUTER_LOGGER):
            assert _project_team_metadata("ns-quiet", entry) is None
        assert caplog.records == []

    def test_the_route_logs_no_warning_for_a_non_declaring_namespace(
        self, api_client: tuple[TestClient, Catalog], caplog: pytest.LogCaptureFixture
    ) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-quiet")

        with caplog.at_level(logging.WARNING, logger=_ROUTER_LOGGER):
            response = client.get("/catalog/namespaces")

        assert response.status_code == 200
        assert response.json()[0]["team_metadata"] is None
        assert _warnings(caplog) == []


class TestAnExplicitNullDeclaresNothing:
    """``metadata_type: null`` is *declares none*, spelled out — not a failure.

    This is the spelling a stored payload actually carries once anything
    writes the key: ``TeamCard.metadata_type`` defaults to ``None``, and the
    card's serializer emits every declared field, so a client that builds its
    catalog payload from ``TeamCard.model_dump()`` persists
    ``metadata_type: None``. Treating that as *present but unusable* would put
    one ``WARNING`` per such namespace on every listing request — the very
    flood the silent-absent ruling exists to prevent, arrived at by a
    different door.
    """

    def test_the_projection_emits_nothing_for_an_explicit_null(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        payload = team_payload()
        payload["metadata_type"] = None
        entry = Entry(
            id="team",
            kind="team",
            namespace="ns-null",
            user_id="anonymous",
            model_type=_TEAM_TYPE,
            payload=payload,
        )
        with caplog.at_level(logging.DEBUG, logger=_ROUTER_LOGGER):
            assert _project_team_metadata("ns-null", entry) is None
        assert caplog.records == []

    def test_the_route_logs_no_warning_for_an_explicit_null(
        self, api_client: tuple[TestClient, Catalog], caplog: pytest.LogCaptureFixture
    ) -> None:
        client, catalog = api_client
        _seed_declaring_team(catalog, "ns-null", None)

        with caplog.at_level(logging.WARNING, logger=_ROUTER_LOGGER):
            response = client.get("/catalog/namespaces")

        assert response.status_code == 200
        assert response.json()[0]["team_metadata"] is None
        assert _warnings(caplog) == []


class TestIntrospectionFailureDegradesToo:
    """A model that resolves and *then* defeats the projection still yields ``None``.

    The never-raises rule covers the whole projection, not just the import.
    Resolution is only the first half of the work, and there are two ways to
    fail the second half: building the descriptors reads ``FieldInfo`` values
    a model author is free to set to anything, and generating the JSON Schema
    the ``pattern`` lookup reads can refuse outright. Either raising through
    to the handler would 500 the listing for every other namespace in the
    deployment, so both land in the one existing guard.
    """

    @pytest.mark.parametrize("class_name", ["UndescribableMeta", "UnschematisableMeta"])
    def test_a_model_that_breaks_the_projection_yields_none_not_a_500(
        self,
        api_client: tuple[TestClient, Catalog],
        meta_module: str,
        caplog: pytest.LogCaptureFixture,
        class_name: str,
    ) -> None:
        client, catalog = api_client
        path = f"{meta_module}.{class_name}"
        _seed_declaring_team(catalog, "ns-a-healthy", {"__type__": f"{meta_module}.EmptyMeta"})
        _seed_declaring_team(catalog, "ns-b-unprojectable", {"__type__": path})

        with caplog.at_level(logging.WARNING, logger=_ROUTER_LOGGER):
            response = client.get("/catalog/namespaces")

        assert response.status_code == 200
        by_ns = {r["namespace"]: r for r in response.json()}
        assert by_ns["ns-b-unprojectable"]["team_metadata"] is None
        # The healthy namespace alongside it is unaffected.
        assert by_ns["ns-a-healthy"]["team_metadata"] == {
            "type": f"{meta_module}.EmptyMeta",
            "fields": [],
        }
        records = _warnings(caplog)
        assert len(records) == 1
        assert "ns-b-unprojectable" in records[0].getMessage()
        assert path in records[0].getMessage()


class TestOneBadDeclarationDoesNotBlankThePicker:
    """The criterion that matters most: a typo costs one row's contract, not the listing."""

    def test_healthy_and_broken_namespaces_are_all_returned(
        self,
        api_client: tuple[TestClient, Catalog],
        meta_module: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        client, catalog = api_client
        path = f"{meta_module}.DeclarationOrderMeta"
        _seed_declaring_team(catalog, "ns-a-healthy", {"__type__": path})
        _seed_declaring_team(catalog, "ns-b-broken", {"__type__": "akgentic.nope.Missing"})
        _seed_team(catalog, "ns-c-silent")

        with caplog.at_level(logging.WARNING, logger=_ROUTER_LOGGER):
            response = client.get("/catalog/namespaces")

        assert response.status_code == 200
        rows = response.json()
        assert [r["namespace"] for r in rows] == ["ns-a-healthy", "ns-b-broken", "ns-c-silent"]
        by_ns = {r["namespace"]: r for r in rows}
        assert by_ns["ns-a-healthy"]["team_metadata"]["type"] == path
        assert by_ns["ns-b-broken"]["team_metadata"] is None
        assert by_ns["ns-c-silent"]["team_metadata"] is None
        # Exactly one namespace earned a warning — the broken one.
        records = _warnings(caplog)
        assert len(records) == 1
        assert "ns-b-broken" in records[0].getMessage()


# --- The handler -------------------------------------------------------------


class TestResolutionIsAnImportNotAQuery:
    """Adding the contract adds no repository traffic."""

    def test_six_list_calls_with_a_declaring_namespace_in_the_fixture(
        self, counting_catalog: tuple[Catalog, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from akgentic.catalog.api._errors import add_exception_handlers
        from akgentic.catalog.api._settings import CatalogRouterSettings
        from akgentic.catalog.api.router import _ENTRY_KINDS, build_router, set_catalog

        catalog, counting = counting_catalog
        module = register_akgentic_test_module(
            monkeypatch, "meta_counting_37", CountingMeta=DeclarationOrderMeta
        )
        path = f"{module}.CountingMeta"
        _seed_team(catalog, "ns-plain")
        _seed_declaring_team(catalog, "ns-declares", {"__type__": path})
        counting.reset()

        from fastapi import FastAPI

        app = FastAPI(title="counting")
        app.include_router(build_router(CatalogRouterSettings(expose_generic_kind_crud=True)))
        set_catalog(catalog)
        add_exception_handlers(app)
        client = TestClient(app)

        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        by_ns = {r["namespace"]: r for r in response.json()}
        # The declaring namespace really did resolve — otherwise this test
        # would pass while exercising the failure branch.
        assert by_ns["ns-declares"]["team_metadata"]["type"] == path

        assert counting.count("list") == len(_ENTRY_KINDS)
        assert counting.count("get") == 0
        assert counting.count("list_by_namespace") == 0


class TestBuildNamespaceSummaryTakesTheContract:
    """The helper sets the contract verbatim and performs no repository I/O."""

    def test_the_contract_is_set_verbatim_with_no_catalog_in_hand(self) -> None:
        contract = TeamMetadataContract(
            type="akgentic.meta.Whatever",
            fields=[
                MetadataFieldDescriptor(
                    key="zone", description="Delivery zone", index=True, mandatory=True
                )
            ],
        )
        row = _build_namespace_summary(
            "ns-helper",
            None,
            None,
            owner=None,
            counts=_zero_counts(),
            team_metadata=contract,
        )
        assert row.team_metadata is contract

    def test_none_is_passed_through_unchanged(self) -> None:
        row = _build_namespace_summary(
            "ns-helper",
            None,
            None,
            owner=None,
            counts=_zero_counts(),
            team_metadata=None,
        )
        assert row.team_metadata is None
