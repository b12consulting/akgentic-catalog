"""Shared fixtures and factories for the v2 test suite.

Factory functions are plain helpers (not pytest fixtures) per the project
convention for stateless constructions. The one fixture-like helper
(``register_akgentic_test_module``) is a plain function that accepts a
``monkeypatch`` argument so test cleanup is handled by pytest's built-in
fixture teardown without additional bookkeeping in the test body.

The ``FakeEntryRepository`` class is a stateful, in-memory
``EntryRepository`` implementation used exclusively by Story 15.2 resolver
tests. It is a testing utility — concrete production repositories ship in
Stories 15.3 (YAML) and 15.4 (Mongo).
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from akgentic.catalog.allowlist import ENV_VAR, reset_allowed_prefixes
from akgentic.catalog.catalog import Catalog
from akgentic.catalog.models.entry import Entry, EntryKind
from akgentic.catalog.models.queries import EntryQuery
from akgentic.catalog.repositories.base import EntryRepository
from akgentic.catalog.repositories.yaml import (
    YamlEntryRepository,
)
from akgentic.catalog.repositories.yaml import (
    _payload_has_cross_ns_ref as _payload_has_cross_ns_ref,
)
from akgentic.catalog.repositories.yaml import (
    _payload_has_ref as _payload_has_ref,
)

from ..conftest import team_payload

if TYPE_CHECKING:
    import pymongo.collection
    from typer.testing import CliRunner


_NAMESPACE_META_TYPE = "akgentic.catalog.models.namespace_meta.NamespaceMeta"


@pytest.fixture(autouse=True)
def _isolate_allowlist_policy(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give every v2 test a pristine, unconfigured ``model_type`` prefix policy.

    ``akgentic.catalog.allowlist`` caches the resolved policy in a module
    global, so without this fixture the first test that calls
    ``set_allowed_prefixes`` would poison every test running later in the same
    process — and the damage would surface as unrelated tests going red rather
    than as a failure in the test that caused it.

    Clears the environment variable, resets the cache before the test, and
    resets it again afterwards so a test that leaves a prefix configured
    cannot leak it forward.
    """
    monkeypatch.delenv(ENV_VAR, raising=False)
    reset_allowed_prefixes()
    yield
    reset_allowed_prefixes()


def make_meta_entry(
    namespace: str,
    *,
    shareable: bool = True,
    public: bool = False,
    name: str | None = None,
    description: str = "",
    extra_properties: dict[str, str] | None = None,
    user_id: str = "anonymous",
) -> Entry:
    """Build a ``kind="meta"`` Entry with the canonical id ``"_meta"``.

    Used by cross-ns shareable-flag tests to opt the target namespace into
    being a cross-ns ref target. Story 17.7 / AC1 — ``shareable`` is a typed
    bool at the root of the meta payload. Story 18.2 — ``public`` is a typed
    bool at the root of the meta payload (default ``False``). The factory
    always emits the typed-bool shape; legacy-shape fixtures construct the
    payload by hand.
    """
    properties: dict[str, str] = {}
    if extra_properties:
        properties.update(extra_properties)
    payload: dict[str, Any] = {
        "name": name if name is not None else namespace,
        "description": description,
        "properties": properties,
        "shareable": shareable,
        "public": public,
    }
    return Entry(
        id="_meta",
        kind="meta",
        namespace=namespace,
        user_id=user_id,
        model_type=_NAMESPACE_META_TYPE,
        payload=payload,
    )


def make_entry(**overrides: Any) -> Entry:
    """Build a minimal valid ``Entry`` with sensible defaults, overridable by kwargs.

    Defaults model a fresh, community-tier (``user_id="anonymous"``) entry of
    kind ``tool`` pointing at a known-valid ``akgentic.*`` class. Tests pass
    keyword overrides for the attribute under test.
    """
    base: dict[str, Any] = {
        "id": "entry-1",
        "kind": "tool",
        "namespace": "ns-1",
        "model_type": "akgentic.core.agent_card.AgentCard",
        "description": "",
        "payload": {},
    }
    base.update(overrides)
    return Entry(**base)


# --------------------------------------------------------------------------- #
# Shared seeding preamble — the ``test_api_router_*`` modules
# --------------------------------------------------------------------------- #


_TEAM_TYPE = "akgentic.team.models.TeamCard"
_AGENT_TYPE = "akgentic.core.agent_card.AgentCard"


def _agent_payload(name: str = "a") -> dict[str, Any]:
    """Return a minimal valid ``AgentCard`` payload."""
    return {
        "description": "",
        "skills": [],
        "agent_class": "akgentic.core.agent.Akgent",
        "config": {"name": name, "role": "r"},
        "routes_to": [],
        "metadata": {},
    }


def _seed_team(catalog: Catalog, namespace: str, user_id: str = "anonymous") -> Entry:
    """Seed a team entry in ``namespace``."""
    return catalog.create(
        Entry(
            id="team",
            kind="team",
            namespace=namespace,
            user_id=user_id,
            model_type=_TEAM_TYPE,
            payload=team_payload(),
        )
    )


def _seed_agent(
    catalog: Catalog,
    namespace: str,
    id: str = "agent-a",
    user_id: str = "anonymous",
    payload: dict[str, Any] | None = None,
) -> Entry:
    """Seed a minimal agent entry sharing the team's ownership."""
    return catalog.create(
        Entry(
            id=id,
            kind="agent",
            namespace=namespace,
            user_id=user_id,
            model_type=_AGENT_TYPE,
            payload=payload if payload is not None else _agent_payload(id),
        )
    )


def _seed_meta_entry(
    catalog: Catalog,
    namespace: str,
    user_id: str = "anonymous",
    name: str = "Tenant 42",
    description: str = "primary tenant",
) -> Entry:
    """Seed a kind=meta entry whose payload omits ``shareable`` entirely (Story 17.2)."""
    return catalog.create(
        Entry(
            id="_meta",
            kind="meta",
            namespace=namespace,
            user_id=user_id,
            model_type=_NAMESPACE_META_TYPE,
            description=description,
            payload={"name": name, "description": description, "properties": {}},
        )
    )


def _seed_meta(
    catalog: Catalog,
    namespace: str,
    *,
    name: str = "Display",
    description: str = "",
    shareable: bool = False,
    user_id: str = "anonymous",
) -> Entry:
    """Seed a kind=meta entry whose payload carries a typed-bool ``shareable`` (Story 17.7)."""
    return catalog.create(
        Entry(
            id="_meta",
            kind="meta",
            namespace=namespace,
            user_id=user_id,
            model_type=_NAMESPACE_META_TYPE,
            description=description,
            payload={
                "name": name,
                "description": description,
                "properties": {},
                "shareable": shareable,
            },
        )
    )


def register_test_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    **attributes: Any,
) -> str:
    """Register a throwaway module under ``sys.modules[module_name]``.

    Builds a ``types.ModuleType`` carrying every attribute passed as a kwarg,
    then installs it via ``monkeypatch.setitem`` so pytest's fixture teardown
    un-registers it after the test finishes.

    Takes a **fully-qualified** module name so tests can stand up modules
    outside the ``akgentic.`` namespace — the allowlist-policy tests register
    fake customer modules such as ``acme.core.models``.
    ``register_akgentic_test_module`` is the ``akgentic.``-relative shorthand
    over this function.

    Args:
        monkeypatch: Pytest's ``monkeypatch`` fixture.
        module_name: Fully-qualified module name to register.
        **attributes: Names to attach to the module (classes, functions, …).

    Returns:
        ``module_name`` unchanged, so tests can build class paths off the
        call expression.
    """
    module = types.ModuleType(module_name)
    for name, value in attributes.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module_name


def register_akgentic_test_module(
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    **attributes: Any,
) -> str:
    """Register a throwaway module under ``sys.modules["akgentic.<suffix>"]``.

    Thin shorthand over :func:`register_test_module` for the common case of a
    module inside the framework namespace.

    Args:
        monkeypatch: Pytest's ``monkeypatch`` fixture.
        suffix: The portion after ``"akgentic."`` used as the module name.
        **attributes: Names to attach to the module (classes, functions, …).

    Returns:
        The fully-qualified module name (``"akgentic.<suffix>"``) so tests can
        build class paths off it.
    """
    return register_test_module(monkeypatch, f"akgentic.{suffix}", **attributes)


# --------------------------------------------------------------------------- #
# Shared ``ak-catalog`` CLI preamble — test_cli_foundation / _graph_schema /
# _validate / _bundle
# --------------------------------------------------------------------------- #


_CLI_NAMESPACE = "ns-a"
_CLI_USER_ID = "alice"


def _build_cli_fixture_models(owning_module: str) -> dict[str, type[BaseModel]]:
    """Build a fresh set of throwaway model classes stamped with ``owning_module``.

    Two properties are load-bearing on rendered CLI output, so leave both alone.
    ``__module__``: ``resolve`` prints ``f"{cls.__module__}.{cls.__name__}"``,
    which is why the classes are built per requesting module rather than shared
    from this conftest. Docstrings: pydantic copies one into the JSON schema's
    ``description``, which the ``schema`` verb prints — these classes carry none
    and must not gain one.
    """

    class LeafModel(BaseModel):
        provider: str = "openai"
        temperature: float = 0.0

    class AgentModel(BaseModel):
        provider: str = "openai"
        temperature: float = 0.0
        linked: LeafModel | None = None

    class RequiredFieldModel(BaseModel):
        # No defaults — forces transient validation failure when required fields are absent.
        must_be_present: str

    models: dict[str, type[BaseModel]] = {
        "LeafModel": LeafModel,
        "AgentModel": AgentModel,
        "RequiredFieldModel": RequiredFieldModel,
    }
    for model in models.values():
        model.__module__ = owning_module
    return models


def cli_team_payload() -> dict[str, Any]:
    """``team_payload`` with the card description the CLI modules have always used."""
    return team_payload(card_description="entry")


def base_args(catalog_root: Path) -> list[str]:
    """The backend/root prefix every ``ak-catalog`` invocation in these tests carries."""
    return ["--backend", "yaml", "--root", str(catalog_root)]


@pytest.fixture
def runner() -> CliRunner:
    """CliRunner with stderr separated from stdout so we can pin both.

    ``typer`` ships under the optional ``cli`` extra, so the import lives in the
    fixture body — same discipline ``api_client`` applies to ``fastapi``, which
    keeps this conftest importable when the extra is absent.
    """
    from typer.testing import CliRunner

    return CliRunner()


@pytest.fixture
def cli_fixture_models(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Register the throwaway model classes the requesting module's seed points at.

    Reads ``_FIXTURE_MODULE`` off the requesting module: the four CLI modules
    keep distinct fixture-module names because every seeded entry's
    ``model_type`` embeds one and the CLI renders it verbatim.
    ``RequiredFieldModel`` is registered only where the module declares
    ``_REQUIRED_TYPE``.

    Deliberately NOT autouse: this conftest reaches all 40 v2 modules and
    ``enumerate_allowlisted_model_types()`` walks ``sys.modules`` live, so an
    autouse registration would change what unrelated modules observe. Modules
    opt in with ``pytestmark = pytest.mark.usefixtures("cli_fixture_models")``,
    and each module's ``catalog_root`` takes this fixture as a parameter so
    registration is ordered before the seed explicitly rather than by accident.
    """
    module_name: str = request.module._FIXTURE_MODULE
    models = _build_cli_fixture_models(request.module.__name__)
    attributes: dict[str, Any] = {
        "LeafModel": models["LeafModel"],
        "AgentModel": models["AgentModel"],
    }
    if hasattr(request.module, "_REQUIRED_TYPE"):
        attributes["RequiredFieldModel"] = models["RequiredFieldModel"]
    return register_test_module(monkeypatch, module_name, **attributes)


def _cli_entry(
    entry_id: str,
    kind: EntryKind,
    model_type: str,
    description: str,
    payload: dict[str, Any],
) -> Entry:
    """Build one seed entry — every seeded entry shares namespace and user_id."""
    return Entry(
        id=entry_id,
        kind=kind,
        namespace=_CLI_NAMESPACE,
        user_id=_CLI_USER_ID,
        model_type=model_type,
        description=description,
        payload=payload,
    )


def seed_namespace(
    root: Path,
    *,
    fixture_module: str,
    with_tool: bool = True,
    model_description: str | None = None,
    model_temperature: float = 0.1,
    agent_description: str = "agent referencing tool",
    agent_ref: str | None = "tool-a",
) -> None:
    """Seed ``root`` with the ``ns-a`` namespace one CLI module expects.

    The three resulting shapes are deliberately different and are NOT
    interchangeable: ``test_cli_foundation.py`` seeds a namespace with no
    ``__ref__`` and no tool at all — the premise of its plain CRUD/rendering
    assertions — while the other modules need a ref graph. These tests assert
    on *rendered* CLI stdout, so a flattened seed would change what each module
    tests without failing an assertion.
    """
    catalog = Catalog(YamlEntryRepository(root))
    # Team must land first — it is the bootstrap for the namespace.
    catalog.create(_cli_entry("team-a", "team", _TEAM_TYPE, "primary team", cli_team_payload()))
    if with_tool:
        catalog.create(
            _cli_entry(
                "tool-a",
                "tool",
                f"{fixture_module}.LeafModel",
                "the tool",
                {"provider": "openai", "temperature": 0.0},
            )
        )
    if model_description is not None:
        catalog.create(
            _cli_entry(
                "model-a",
                "model",
                f"{fixture_module}.LeafModel",
                model_description,
                {"provider": "openai", "temperature": model_temperature},
            )
        )
    agent_payload: dict[str, Any] = {"provider": "openai", "temperature": 0.0}
    if agent_ref is not None:
        agent_payload["linked"] = {"__ref__": agent_ref}
    catalog.create(
        _cli_entry(
            "agent-a", "agent", f"{fixture_module}.AgentModel", agent_description, agent_payload
        )
    )


@pytest.fixture
def entries_collection() -> pymongo.collection.Collection:  # type: ignore[type-arg]
    """Provide a fresh mongomock-backed ``catalog_entries`` collection per test.

    Builds an in-memory ``mongomock.MongoClient`` on demand so tests that do
    not touch Mongo pay no import cost. Each test gets an isolated collection
    — no cross-test state. ``pymongo`` is an optional dep per the package
    ``pyproject.toml``; ``mongomock`` ships under the ``dev`` extra.
    """
    import mongomock

    client = mongomock.MongoClient()
    return client["test_catalog"]["catalog_entries"]


class FakeEntryRepository:
    """In-memory ``EntryRepository`` for resolver + write-pipeline tests.

    Stateful by design — each test that mutates the store instantiates its
    own ``FakeEntryRepository()``. Satisfies the ``EntryRepository``
    structural protocol for every method the Story 15.2 resolver pipeline
    touches (``get``, ``put``, ``delete``, ``list_by_namespace``,
    ``get_by_kind``, ``find_references``). ``list`` raises
    ``NotImplementedError`` — no 15.2 code path consumes it.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], Entry] = {}

    def get(self, namespace: str, id: str) -> Entry | None:
        return self._store.get((namespace, id))

    def put(self, entry: Entry) -> Entry:
        self._store[(entry.namespace, entry.id)] = entry
        return entry

    def delete(self, namespace: str, id: str) -> None:
        self._store.pop((namespace, id), None)

    def list(self, query: EntryQuery) -> list[Entry]:
        # Story 15.5 needs a minimal implementation so the Catalog service's
        # list() pass-through can be counted. Applies AND semantics over the
        # EntryQuery fields used by current tests; ignores filters not yet
        # exercised to keep the fake honest about its minimality.
        out: list[Entry] = list(self._store.values())
        if query.namespace is not None:
            out = [e for e in out if e.namespace == query.namespace]
        if query.kind is not None:
            out = [e for e in out if e.kind == query.kind]
        if query.id is not None:
            out = [e for e in out if e.id == query.id]
        if query.user_id:
            out = [e for e in out if e.user_id == query.user_id]
        if query.user_id_set is True:
            out = [e for e in out if e.user_id != "anonymous"]
        elif query.user_id_set is False:
            out = [e for e in out if e.user_id == "anonymous"]
        if query.description_contains is not None:
            out = [e for e in out if query.description_contains in e.description]
        return out

    def list_by_namespace(self, namespace: str) -> list[Entry]:
        return [e for (ns, _), e in self._store.items() if ns == namespace]

    def get_by_kind(self, namespace: str, kind: EntryKind) -> Entry | None:
        for (ns, _), e in self._store.items():
            if ns == namespace and e.kind == kind:
                return e
        return None

    def find_references(self, namespace: str, target_id: str) -> list[Entry]:
        out: list[Entry] = []
        for (ns, _), e in self._store.items():
            if ns == namespace and _payload_has_ref(e.payload, target_id):
                out.append(e)
        return out

    def find_references_global(self, namespace: str, target_id: str) -> list[Entry]:
        out: list[Entry] = []
        for _key, e in self._store.items():
            if _payload_has_cross_ns_ref(e.payload, namespace, target_id):
                out.append(e)
        return out


class CountingEntryRepository:
    """Decorator repository recording every method invocation.

    Wraps any ``EntryRepository`` (in practice a ``FakeEntryRepository`` or one
    of the production backends) and records each call into a public ``calls``
    list as ``(method_name, args, kwargs)`` tuples. Tests use this to assert
    "repository method X was called exactly once with arg Y" without touching
    the production repositories.

    The inner repository is accessible via ``inner`` for tests that need to
    seed state directly without polluting the call log. Call ``reset()`` to
    clear the recorded history (e.g. after seeding).
    """

    def __init__(self, inner: EntryRepository) -> None:
        self.inner: EntryRepository = inner
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def reset(self) -> None:
        """Clear the recorded call log."""
        self.calls = []

    def _record(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.calls.append((name, args, kwargs))

    def get(self, namespace: str, id: str) -> Entry | None:
        self._record("get", (namespace, id), {})
        return self.inner.get(namespace, id)

    def put(self, entry: Entry) -> Entry:
        self._record("put", (entry,), {})
        return self.inner.put(entry)

    def delete(self, namespace: str, id: str) -> None:
        self._record("delete", (namespace, id), {})
        self.inner.delete(namespace, id)

    def list(self, query: EntryQuery) -> list[Entry]:
        self._record("list", (query,), {})
        return self.inner.list(query)

    def list_by_namespace(self, namespace: str) -> list[Entry]:
        self._record("list_by_namespace", (namespace,), {})
        return self.inner.list_by_namespace(namespace)

    def get_by_kind(self, namespace: str, kind: EntryKind) -> Entry | None:
        self._record("get_by_kind", (namespace, kind), {})
        return self.inner.get_by_kind(namespace, kind)

    def find_references(self, namespace: str, target_id: str) -> list[Entry]:
        self._record("find_references", (namespace, target_id), {})
        return self.inner.find_references(namespace, target_id)

    def find_references_global(self, namespace: str, target_id: str) -> list[Entry]:
        self._record("find_references_global", (namespace, target_id), {})
        return self.inner.find_references_global(namespace, target_id)

    def count(self, method_name: str) -> int:
        """Return the number of recorded calls to ``method_name``."""
        return sum(1 for name, _, _ in self.calls if name == method_name)


CatalogFactory = Callable[[], tuple[Catalog, EntryRepository]]


@pytest.fixture(params=["yaml", "mongo"], ids=["yaml", "mongo"])
def catalog_factory(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    entries_collection: pymongo.collection.Collection,  # type: ignore[type-arg]
) -> CatalogFactory:
    """Yield a factory producing ``(Catalog, repository)`` for both backends.

    Parametrised with ids ``yaml`` and ``mongo`` so pytest reports make the
    backend obvious. Every catalog-service behavioural test runs against both
    backends. The factory shape (callable rather than direct tuple) lets
    individual tests build multiple repositories if they ever need to (e.g. a
    src vs dst backend — though the current story does not need this).
    """

    def _make() -> tuple[Catalog, EntryRepository]:
        repo: EntryRepository
        if request.param == "yaml":
            repo = YamlEntryRepository(tmp_path)
        elif request.param == "mongo":
            pytest.importorskip("pymongo")
            from akgentic.catalog.repositories.mongo import (
                MongoEntryRepository,
            )

            repo = MongoEntryRepository(entries_collection)
        else:  # pragma: no cover — guarded by pytest.fixture params
            raise AssertionError(f"Unexpected backend param: {request.param}")
        return Catalog(repo), repo

    return _make


@pytest.fixture
def api_client(tmp_path: Path) -> tuple[Any, Catalog]:
    """Yield a ``(TestClient, Catalog)`` pair wired to a YAML-backed v2 router.

    The fixture is function-scoped; ``set_catalog`` is called fresh per test so
    the module-level ``_catalog`` in ``api/router.py`` cannot leak between
    tests. ``fastapi`` is guarded via ``importorskip`` inside the fixture body
    so this conftest module stays importable when the ``api`` extra is absent.

    Story 16.7: the generic ``/catalog/{kind}`` CRUD family is gated behind
    the ``expose_generic_kind_crud`` router setting (default ``False``).
    This fixture opts **in** to keep the existing integration tests that
    drive every route exercising the full surface. Tests that need to
    verify the default-off behaviour use ``api_client_kind_crud_hidden``.
    """
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from akgentic.catalog.api._errors import add_exception_handlers
    from akgentic.catalog.api._settings import CatalogRouterSettings
    from akgentic.catalog.api.router import build_router, set_catalog

    repo = YamlEntryRepository(tmp_path)
    catalog = Catalog(repo)

    app = FastAPI(title="Akgentic Catalog")
    app.include_router(build_router(CatalogRouterSettings(expose_generic_kind_crud=True)))
    set_catalog(catalog)
    add_exception_handlers(app)

    return TestClient(app), catalog


@pytest.fixture
def api_client_kind_crud_hidden(tmp_path: Path) -> tuple[Any, Catalog]:
    """Same as ``api_client`` but with ``expose_generic_kind_crud=False``.

    Used by Story 16.7 tests that assert the kind-generic CRUD family is
    hidden (404 / absent from OpenAPI) in the default community-tier build.
    """
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from akgentic.catalog.api._errors import add_exception_handlers
    from akgentic.catalog.api._settings import CatalogRouterSettings
    from akgentic.catalog.api.router import build_router, set_catalog

    repo = YamlEntryRepository(tmp_path)
    catalog = Catalog(repo)

    app = FastAPI(title="Akgentic Catalog")
    app.include_router(build_router(CatalogRouterSettings(expose_generic_kind_crud=False)))
    set_catalog(catalog)
    add_exception_handlers(app)

    return TestClient(app), catalog


@pytest.fixture
def counting_catalog() -> tuple[Catalog, CountingEntryRepository]:
    """Build a ``Catalog`` backed by a ``CountingEntryRepository`` around a Fake.

    Backend-agnostic — used by tests that need to assert on repository call
    counts (AC5 pass-throughs, AC33 load_team single-query, AC41 clone
    atomicity). Returns ``(Catalog, CountingEntryRepository)`` so tests can
    reach into ``.calls`` directly.
    """
    fake = FakeEntryRepository()
    # FakeEntryRepository.list raises NotImplementedError by design; extend it
    # here for the counting double by delegating to list_by_namespace when a
    # namespace filter is set. Tests that use counting_catalog only exercise
    # methods Fake actually supports, so we do not wire list semantics.
    counting = CountingEntryRepository(fake)
    return Catalog(counting), counting
