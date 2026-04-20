"""Cross-backend parity tests for the v2 ``EntryRepository`` protocol (AC21).

Parametrised on a ``backend`` fixture that yields one instance of each
concrete implementation: ``YamlEntryRepository(tmp_path)`` and
``MongoEntryRepository(entries_collection)``. Every parity scenario is
backend-agnostic in its assertions — no ``if backend is ...`` branching in the
test body. Scenarios that legitimately differ between the two backends live in
their respective backend-specific test files.

If ``pymongo`` is not installed in the test environment, the ``mongo`` branch
of the parametrisation is skipped via ``pytest.importorskip`` inside the
fixture; the YAML branch continues to run.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from akgentic.catalog.models.queries import EntryQuery
from akgentic.catalog.repositories.base import EntryRepository
from akgentic.catalog.repositories.yaml_entry_repo import YamlEntryRepository

from .conftest import make_entry

if TYPE_CHECKING:
    import pymongo.collection


@pytest.fixture(params=["yaml", "mongo"])
def backend(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    entries_collection: pymongo.collection.Collection,  # type: ignore[type-arg]
) -> EntryRepository:
    """Yield a fresh ``EntryRepository`` for each parametrised backend.

    The ``yaml`` branch builds a ``YamlEntryRepository`` over pytest's
    ``tmp_path``. The ``mongo`` branch imports ``pymongo`` via
    ``pytest.importorskip`` so the test is skipped gracefully if the optional
    dep is missing (the package does carry it under the ``dev`` extras, but
    belt-and-braces); then it returns a ``MongoEntryRepository`` over the
    mongomock-backed ``entries_collection`` fixture.
    """
    if request.param == "yaml":
        return YamlEntryRepository(tmp_path)
    if request.param == "mongo":
        pytest.importorskip("pymongo")
        from akgentic.catalog.repositories.mongo_entry_repo import MongoEntryRepository

        return MongoEntryRepository(entries_collection)
    raise AssertionError(f"Unexpected backend parameter: {request.param}")


class TestEntryRepositoryParity:
    """Shared acceptance scenarios pinned by AC21 against both backends."""

    def test_put_get_round_trip(self, backend: EntryRepository) -> None:
        """AC11, AC18: put → get returns an equal entry."""
        entry = make_entry(
            id="a",
            kind="agent",
            namespace="ns-1",
            description="desc",
            payload={"k": "v"},
        )
        backend.put(entry)
        assert backend.get("ns-1", "a") == entry

    def test_upsert_overwrites_existing(self, backend: EntryRepository) -> None:
        """AC8: second put with same (namespace, id) replaces the first."""
        first = make_entry(id="a", kind="agent", namespace="ns-1", description="v1")
        second = make_entry(id="a", kind="agent", namespace="ns-1", description="v2")

        backend.put(first)
        backend.put(second)

        got = backend.get("ns-1", "a")
        assert got is not None
        assert got.description == "v2"

    def test_list_by_namespace_returns_every_entry(self, backend: EntryRepository) -> None:
        """AC14: list_by_namespace returns every entry regardless of kind."""
        backend.put(make_entry(id="t1", kind="team", namespace="team-abc"))
        backend.put(make_entry(id="a1", kind="agent", namespace="team-abc"))
        backend.put(make_entry(id="a2", kind="agent", namespace="team-abc"))
        backend.put(make_entry(id="tool1", kind="tool", namespace="team-abc"))
        # Cross-namespace noise.
        backend.put(make_entry(id="other", kind="agent", namespace="team-xyz"))

        got = backend.list_by_namespace("team-abc")

        assert {e.id for e in got} == {"t1", "a1", "a2", "tool1"}

    def test_get_by_kind_returns_singleton(self, backend: EntryRepository) -> None:
        """AC15: get_by_kind returns the only entry of that kind in the namespace."""
        team = make_entry(id="team", kind="team", namespace="team-abc")
        backend.put(team)
        backend.put(make_entry(id="a1", kind="agent", namespace="team-abc"))

        got = backend.get_by_kind("team-abc", "team")

        assert got == team

    def test_find_references_at_arbitrary_depth(self, backend: EntryRepository) -> None:
        """AC16: find_references walks payloads at arbitrary depth."""
        backend.put(
            make_entry(
                id="a",
                kind="agent",
                namespace="ns-1",
                payload={"config": {"model_cfg": {"__ref__": "target"}}},
            )
        )
        backend.put(
            make_entry(
                id="b",
                kind="agent",
                namespace="ns-1",
                payload={"config": {"tools": [{"__ref__": "target"}]}},
            )
        )
        backend.put(
            make_entry(
                id="c",
                kind="agent",
                namespace="ns-1",
                payload={"config": {"model_cfg": {"__ref__": "other"}}},
            )
        )

        got = backend.find_references("ns-1", "target")

        assert {e.id for e in got} == {"a", "b"}

    def test_namespace_isolation_same_id_different_namespace(
        self,
        backend: EntryRepository,
    ) -> None:
        """AC17: same id in two namespaces coexists; no cross-namespace bleed."""
        a = make_entry(id="shared", kind="agent", namespace="ns-a", payload={"which": "a"})
        b = make_entry(id="shared", kind="agent", namespace="ns-b", payload={"which": "b"})

        backend.put(a)
        backend.put(b)

        assert backend.get("ns-a", "shared") == a
        assert backend.get("ns-b", "shared") == b
        assert [e.id for e in backend.list_by_namespace("ns-a")] == ["shared"]
        assert [e.id for e in backend.list_by_namespace("ns-b")] == ["shared"]

    def test_nested_ref_markers_round_trip(self, backend: EntryRepository) -> None:
        """AC20: nested dicts and list elements with ``__ref__`` / ``__type__`` round-trip."""
        payload = {
            "config": {
                "tools": [
                    {"__ref__": "id_tool_a"},
                    {"__ref__": "id_tool_b", "__type__": "akgentic.tool.ToolCard"},
                ]
            }
        }
        entry = make_entry(id="agent", kind="agent", namespace="ns-1", payload=payload)

        backend.put(entry)
        got = backend.get("ns-1", "agent")

        assert got is not None
        assert got.payload == payload

    def test_user_id_and_user_id_set_combine_conjunctively(
        self,
        backend: EntryRepository,
    ) -> None:
        """AC13/AC21: user_id + user_id_set evaluate as AND on both backends."""
        backend.put(
            make_entry(
                id="alice", kind="agent", namespace="ns-1", user_id="alice", payload={}
            )
        )
        backend.put(
            make_entry(
                id="bob", kind="agent", namespace="ns-1", user_id=None, payload={}
            )
        )

        # user_id="alice" AND user_id_set=True → alice satisfies both.
        got = backend.list(EntryQuery(namespace="ns-1", user_id="alice", user_id_set=True))
        assert {e.id for e in got} == {"alice"}
        # user_id="alice" AND user_id_set=False → contradiction; zero entries.
        got = backend.list(EntryQuery(namespace="ns-1", user_id="alice", user_id_set=False))
        assert got == []
