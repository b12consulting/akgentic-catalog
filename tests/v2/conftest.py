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
from typing import Any

import pytest

from akgentic.catalog.models.entry import Entry, EntryKind
from akgentic.catalog.models.queries import EntryQuery
from akgentic.catalog.repositories.yaml_entry_repo import (
    _payload_has_ref as _payload_has_ref,
)


def make_entry(**overrides: Any) -> Entry:
    """Build a minimal valid ``Entry`` with sensible defaults, overridable by kwargs.

    Defaults model a fresh, global (no user_id), freshly minted (no lineage)
    entry of kind ``tool`` pointing at a known-valid ``akgentic.*`` class.
    Tests pass keyword overrides for the attribute under test.
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


def register_akgentic_test_module(
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    **attributes: Any,
) -> str:
    """Register a throwaway module under ``sys.modules["akgentic.<suffix>"]``.

    Builds a ``types.ModuleType`` carrying every attribute passed as a kwarg,
    then installs it via ``monkeypatch.setitem`` so pytest's fixture teardown
    un-registers it after the test finishes.

    Args:
        monkeypatch: Pytest's ``monkeypatch`` fixture.
        suffix: The portion after ``"akgentic."`` used as the module name.
        **attributes: Names to attach to the module (classes, functions, …).

    Returns:
        The fully-qualified module name (``"akgentic.<suffix>"``) so tests can
        build class paths off it.
    """
    module_name = f"akgentic.{suffix}"
    module = types.ModuleType(module_name)
    for name, value in attributes.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module_name


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

    def list(self, query: EntryQuery) -> list[Entry]:  # noqa: ARG002 — not needed for 15.2
        raise NotImplementedError("FakeEntryRepository.list is not required for Story 15.2")

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
