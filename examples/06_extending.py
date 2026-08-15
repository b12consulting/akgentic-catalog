"""Plugging a deployment's own models in — the prefix allowlist and a custom ``ToolCard``.

Run it directly::

    python examples/06_extending.py

Every ``model_type`` in examples ``00``-``05`` began with ``akgentic.``. That prefix is
not a naming convention: a ``model_type`` is an instruction to **import a module**, so
the set of paths an entry may name is a security boundary. Widening it is a deployment
decision, made in startup wiring or in the environment, and never reachable from the
HTTP surface.

This example widens the policy for one fictional customer — ``acme`` — stores two of its
own models, and then puts the policy back. It is the executable form of the README's
*Registering customer model types*.

The two models are declared here and installed into ``sys.modules`` under
``acme.core.models``. That scaffolding exists only because an example cannot ship a
second distribution: in a deployment ``acme-core`` is an installed package and the
module is simply importable.

The narrative half is ``06-extending.md``.
"""

from __future__ import annotations

import sys
import tempfile
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

from akgentic.tool import ToolCard
from pydantic import BaseModel, Field, PrivateAttr, ValidationError

from akgentic.catalog import (
    Catalog,
    Entry,
    YamlEntryRepository,
    allowed_prefixes,
    reset_allowed_prefixes,
    set_allowed_prefixes,
)

REQUIRES: tuple[str, ...] = ()
"""Importable module names this example needs beyond the base install — none.

``akgentic-tool`` is a hard runtime dependency of ``akgentic-catalog``, so importing
``ToolCard`` at module level needs no declaration here.
"""

CUSTOMER_MODULE = "acme.core.models"
CUSTOMER_PREFIX = f"{CUSTOMER_MODULE}."
PROFILE_TYPE = f"{CUSTOMER_MODULE}.EngagementProfile"
TOOL_TYPE = f"{CUSTOMER_MODULE}.ArchiveSearchTool"

NAMESPACE = "acme-prod"
OWNER = "u1"
META_ID = "_meta"
BATCH_SIZE_ID = "default-batch-size"
PROFILE_ID = "case-ingestion"
TOOL_ID = "archive-search"

SHARED_BATCH_SIZE = 200
PROFILE_SOURCE = "sftp"

BATCH_SIZE_REF: dict[str, Any] = {"__ref__": BATCH_SIZE_ID}

TOOL_DUMP_KEYS = frozenset(
    {"__model__", "endpoint", "max_results", "include_attachments", "regions"}
)
"""Every key a dumped ``ArchiveSearchTool`` may carry — a **closed** set, deliberately.

Asserted as an equality rather than as "``_client`` is absent": promoting the runtime
attribute to a Pydantic field would rename it, and a check for one spelling would miss
the very regression Golden Rule #1b exists to prevent.
"""


class EngagementProfile(BaseModel):
    """A customer-owned configuration model, declared outside the ``akgentic.`` namespace.

    Nothing about it is catalog-specific — it is an ordinary Pydantic model. That is the
    whole contract: your model is the schema, the entry is the row.
    """

    source: str
    batch_size: int = 100


class _SearchClient:
    """Stand-in for a live search connection — the kind of thing that is never a field."""


class ArchiveSearchTool(ToolCard):
    """A customer-owned ``ToolCard``, honouring the framework's serialisation rules.

    Three rules, each of them checked by an assertion further down rather than merely
    stated here:

    * **Only serialisable fields** — plain ``str`` / ``int`` / ``bool`` / ``list[str]``.
    * **No ``model_config = ConfigDict(arbitrary_types_allowed=True)`` on the subclass.**
      ``ToolCard`` already inherits it from ``SerializableBaseModel``; the rule forbids a
      subclass *adding* it, because that is the signal that a non-serialisable type has
      leaked into a field. Grep for it here and you will not find it.
    * **Runtime state in a ``PrivateAttr``** — ``_client`` never appears in a dump and
      comes back at its default on the far side of a round trip. ``ToolCard`` uses the
      same pattern itself, for its ``_observer_ref``.

    ``get_tools`` is abstract on ``ToolCard``: a subclass that does not implement it
    cannot be instantiated at all.
    """

    endpoint: str
    max_results: int = 20
    include_attachments: bool = False
    regions: list[str] = Field(default_factory=list)

    _client: _SearchClient | None = PrivateAttr(default=None)

    def connect(self, client: _SearchClient) -> None:
        """Attach a live client. Runtime state — it is not part of the configuration."""
        self._client = client

    @property
    def client(self) -> _SearchClient | None:
        """The attached client, or ``None`` when nothing has been attached."""
        return self._client

    def get_tools(self) -> list[Callable[..., Any]]:
        """Return the LLM-callable functions this card exposes — none, in an example.

        Written ``list[Callable[..., Any]]`` and not ``list[Callable]``: ``mypy --strict``
        forbids a bare generic, and CI type-checks this directory.
        """
        return []


def main() -> None:
    """Refuse a customer type, widen the policy, use it, and put the policy back."""
    _assert_customer_type_is_refused()
    print("1. under the default policy, an entry naming an acme class is refused")

    try:
        _widen_the_allowlist()
        _register_customer_module()
        print(f"2. the policy is now {allowed_prefixes()}, and acme.core.models is importable")

        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = Catalog(YamlEntryRepository(Path(tmpdir)))
            catalog.create(_meta_entry())

            _assert_customer_model_resolves(catalog)
            print("3. the catalog built an acme class, with a shared scalar spliced in")

            _assert_tool_card_round_trips(catalog)
            print("4. the ToolCard round-tripped by FQCN; its runtime state did not")
    finally:
        # In a ``finally`` so a failure above still leaves the process as it found it:
        # the allowlist is a module-level global and the harness runs every example in
        # one pytest process.
        reset_allowed_prefixes()
        sys.modules.pop(CUSTOMER_MODULE, None)

    _assert_policy_is_restored()
    print("5. the default policy is back, and the same construction is refused again")


# --- The policy -------------------------------------------------------------------


def _assert_customer_type_is_refused() -> None:
    """Require an ``Entry`` naming an un-allowlisted class to be refused at construction.

    Note *which* error: Pydantic's ``ValidationError``, not ``CatalogValidationError``.
    The check lives on the ``Entry.model_type`` annotation, so it fires the moment the
    entry is built — before any catalog, any repository, and above all before any import.

    The rendering of the allowed tuple moves with the policy, so the assertion pins the
    two stable halves: the ``outside allowlist`` marker and the rejected path.
    """
    try:
        Entry(
            id=PROFILE_ID,
            kind="model",
            namespace=NAMESPACE,
            user_id=OWNER,
            model_type=PROFILE_TYPE,
            payload={},
        )
    except ValidationError as exc:
        message = str(exc)
        assert "outside allowlist" in message, message
        assert PROFILE_TYPE in message, message
    else:
        raise AssertionError(f"an Entry naming {PROFILE_TYPE} was built under the base policy")


def _widen_the_allowlist() -> None:
    """Authorise exactly ``acme.core.models.``, and assert the resulting policy.

    ``acme.core.models.`` and not ``acme.``: every module under an allowed prefix becomes
    something a catalog entry can cause to be imported, so a prefix is a blast radius
    rather than a gate.

    The equality holds whatever the environment contains. ``set_allowed_prefixes``
    replaces the cached policy *and* stops ``AKGENTIC_CATALOG_MODEL_TYPE_PREFIXES`` being
    consulted afterwards, and ``akgentic.`` is always first and never removable.
    """
    set_allowed_prefixes([CUSTOMER_PREFIX])
    assert allowed_prefixes() == ("akgentic.", CUSTOMER_PREFIX), allowed_prefixes()


def _register_customer_module() -> None:
    """Install ``acme.core.models`` in ``sys.modules`` and make both classes report it.

    Both halves are needed, for different reasons. ``import_class`` resolves
    ``acme.core.models.X`` out of ``sys.modules``, so without the module the resolver
    cannot find the class at all. And ``serialize_type`` reads ``__module__`` to stamp
    ``__model__`` on every dump, so without the reassignment the FQCN in a dump would
    read ``__main__.ArchiveSearchTool`` when run standalone and
    ``akgentic_catalog_example_06_extending.ArchiveSearchTool`` under pytest.

    Reassigning ``__module__`` after class creation is safe here because Pydantic has
    already built both schemas by this point.
    """
    module = types.ModuleType(CUSTOMER_MODULE)
    for cls in (EngagementProfile, ArchiveSearchTool):
        cls.__module__ = CUSTOMER_MODULE
        setattr(module, cls.__name__, cls)
    sys.modules[CUSTOMER_MODULE] = module


def _assert_policy_is_restored() -> None:
    """Require the process-wide policy to be back where it started — behaviourally.

    ``reset_allowed_prefixes`` returns the module to its **unresolved** state, so the next
    read consults ``AKGENTIC_CATALOG_MODEL_TYPE_PREFIXES`` again. It does not restore a
    previously-set tuple, and there is no API that does.

    This is more than housekeeping. The allowlist is a module-level global, the harness
    runs every example in one pytest process, and the autouse fixture that isolates the
    policy covers ``tests/v2/`` only. An example that leaked its widening would silently
    change the policy for whatever ran next — and so would one of your own tests.
    """
    assert CUSTOMER_PREFIX not in allowed_prefixes(), allowed_prefixes()
    _assert_customer_type_is_refused()


# --- The customer's models, in the catalog ----------------------------------------


def _assert_customer_model_resolves(catalog: Catalog) -> None:
    """Store a customer model whose payload refs a shared scalar, and resolve it.

    The shape is the README's: one ``NativeValue`` entry holding the number, and one
    entry whose ``model_type`` names the customer's class, pulling that number in through
    a marker.

    Three claims, none of them a restatement of an input. The resolved object is an
    ``EngagementProfile`` — the catalog imported and built the customer's own class, not
    a dict. ``batch_size`` holds a value written into a *different* entry, and since the
    field is declared ``int`` the ``{"value": 200}`` wrapper could not have been assigned
    to it. And what is written down is still the marker: resolution happens on the way
    out, which is what keeps the two entries independent.
    """
    catalog.create(_native_value_entry())
    catalog.create(_profile_entry())

    resolved = catalog.resolve_by_id(NAMESPACE, PROFILE_ID)
    assert isinstance(resolved, EngagementProfile), f"resolved to {type(resolved).__name__}"
    assert resolved.batch_size == SHARED_BATCH_SIZE, resolved.batch_size
    assert resolved.source == PROFILE_SOURCE, resolved.source

    stored = catalog.get(NAMESPACE, PROFILE_ID).payload["batch_size"]
    assert stored == BATCH_SIZE_REF, f"the stored payload was flattened: {stored}"


def _assert_tool_card_round_trips(catalog: Catalog) -> None:
    """Dump a custom ``ToolCard``, store it, resolve it, and check what did *not* survive.

    **This is the load-bearing pair**, and what it protects is Golden Rule #1b. The dump
    is compared against a closed key set, so a runtime attribute promoted to a Pydantic
    field shows up at once. On the far side of the trip through YAML every declared field
    is intact and the private attribute is back at its default — the value attached below
    did not travel, because it is not configuration.

    ``__model__`` is what makes the round trip work at all: it carries the concrete FQCN,
    which is how a polymorphic field recovers ``ArchiveSearchTool`` rather than some base
    class. The catalog's unknown-key check exempts that key by name, so a payload pasted
    straight from a dump is accepted rather than reported as a misprint.
    """
    card = ArchiveSearchTool(
        endpoint="https://archive.acme.internal/search",
        max_results=50,
        include_attachments=True,
        regions=["eu-west", "eu-central"],
    )
    card.connect(_SearchClient())
    assert card.client is not None, "the client was not attached"

    dumped = card.model_dump()
    assert set(dumped) == TOOL_DUMP_KEYS, sorted(dumped)
    assert dumped["__model__"] == TOOL_TYPE, dumped["__model__"]

    catalog.create(_tool_entry(dumped))

    restored = catalog.resolve_by_id(NAMESPACE, TOOL_ID)
    assert isinstance(restored, ArchiveSearchTool), f"resolved to {type(restored).__name__}"
    assert restored.endpoint == card.endpoint, restored.endpoint
    assert restored.max_results == card.max_results, restored.max_results
    assert restored.include_attachments == card.include_attachments, restored.include_attachments
    assert restored.regions == card.regions, restored.regions
    assert restored.client is None, restored.client


# --- Building the namespace -------------------------------------------------------


def _meta_entry() -> Entry:
    """Build the ``_meta`` anchor.

    A ``kind="meta"`` entry *is* an anchor — it skips the initialisation check the way a
    team entry does. Without one, the first ordinary ``create`` fails with "has no team
    entry and no meta entry".
    """
    return Entry(
        id=META_ID,
        kind="meta",
        namespace=NAMESPACE,
        user_id=OWNER,
        model_type="akgentic.catalog.models.namespace_meta.NamespaceMeta",
        description="Namespace metadata",
        payload={
            "name": "Acme production",
            "description": "A namespace holding acme's own models",
            "properties": {},
            "shareable": False,
            "public": False,
        },
    )


def _native_value_entry() -> Entry:
    """Build the entry holding the one number every acme profile reads."""
    return Entry(
        id=BATCH_SIZE_ID,
        kind="model",
        namespace=NAMESPACE,
        user_id=OWNER,
        model_type="akgentic.catalog.NativeValue",
        description="Batch size shared across acme's ingestion profiles",
        payload={"value": SHARED_BATCH_SIZE},
    )


def _profile_entry() -> Entry:
    """Build the customer-model entry — an ``akgentic.``-free ``model_type``."""
    return Entry(
        id=PROFILE_ID,
        kind="model",
        namespace=NAMESPACE,
        user_id=OWNER,  # sub-entries must match the anchor entry's owner
        model_type=PROFILE_TYPE,
        description="Case ingestion settings",
        payload={"source": PROFILE_SOURCE, "batch_size": dict(BATCH_SIZE_REF)},
    )


def _tool_entry(payload: dict[str, Any]) -> Entry:
    """Build the ``kind="tool"`` entry carrying a dumped ``ArchiveSearchTool``."""
    return Entry(
        id=TOOL_ID,
        kind="tool",
        namespace=NAMESPACE,
        user_id=OWNER,
        model_type=TOOL_TYPE,
        description="Full-text search over the acme document archive",
        payload=payload,
    )


if __name__ == "__main__":
    main()
