"""``GET /catalog/namespaces`` — namespace discovery and summary projection."""

from __future__ import annotations

from typing import Any, get_args

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from akgentic.catalog.api.router import (  # noqa: E402
    _ENTRY_KINDS,
    _build_namespace_summary,
    _count_by_namespace,
    _derive_owner,
    _zero_counts,
    list_entries,
    list_namespaces,
)
from akgentic.catalog.catalog import Catalog  # noqa: E402
from akgentic.catalog.models.entry import Entry, EntryKind  # noqa: E402
from akgentic.catalog.models.queries import EntryQuery  # noqa: E402
from akgentic.catalog.serialization import dump_namespace  # noqa: E402

from ..conftest import team_payload  # noqa: E402
from .conftest import (  # noqa: E402
    _AGENT_TYPE,
    _NAMESPACE_META_TYPE,
    _TEAM_TYPE,
    _agent_payload,
    _seed_agent,
    _seed_meta,
    _seed_meta_entry,
    _seed_team,
    make_meta_entry,
)

_ENTRY_KIND_NAMES: tuple[str, ...] = get_args(EntryKind)


def expected_summary(
    namespace: str,
    *,
    name: str | None = None,
    description: str = "",
    team: bool = True,
    shareable: bool = False,
    public: bool = False,
    owner: str | None = "anonymous",
    meta: bool = False,
    counts: dict[str, int] | None = None,
    team_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one expected ``NamespaceSummary`` row; ``name`` defaults to ``namespace``.

    ``owner`` defaults to ``"anonymous"`` — every fixture in this module
    seeds the community-tier default. ``counts`` takes a **sparse** map of
    kind → tally and expands it into the full six-key shape the DTO
    always emits; the ``team`` and ``meta`` tallies default from the
    ``team`` flag and the ``meta`` argument, so a row that holds exactly
    one team entry needs no ``counts`` at all. ``team_metadata`` defaults
    to ``None`` — no fixture in this module declares a ``metadata_type``,
    so every whole-row assertion here reads the no-contract state.
    """
    tallies: dict[str, int] = {"team": int(team), "meta": int(meta)}
    if counts is not None:
        tallies.update(counts)
    return {
        "namespace": namespace,
        "name": namespace if name is None else name,
        "description": description,
        "team": team,
        "shareable": shareable,
        "public": public,
        "owner": owner,
        "counts": {kind: {"total": tallies.get(kind, 0)} for kind in _ENTRY_KIND_NAMES},
        "team_metadata": team_metadata,
    }


# --- List namespaces (Story 16.6) ------------------------------------------


class TestListNamespaces:
    """``GET /catalog/namespaces`` — Story 16.6 ACs 1-5."""

    def test_empty_catalog_returns_empty_list(self, api_client: tuple[TestClient, Catalog]) -> None:
        """AC #2 — empty catalog → HTTP 200, ``[]``."""
        client, _ = api_client
        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        assert response.json() == []

    def test_returns_summaries_sorted_by_namespace(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """AC #1 — two namespaces with team entries project to sorted summaries.

        Story 17.8 — ``name`` falls back to the namespace identifier (NOT
        the team's payload ``name``) when no meta entry exists for the
        namespace. Both fixtures here are team-only, so each row's ``name``
        is the namespace identifier.
        """
        client, catalog = api_client
        # Seed namespaces out of alphabetical order and with distinct team
        # names + descriptions to verify the projection. Team payload
        # ``name`` is set to a deliberately misleading sentinel — Story 17.8
        # demands the picker NEVER read team.payload["name"].
        team_b = team_payload()
        team_b["name"] = "TeamPayloadIgnored-B"
        catalog.create(
            Entry(
                id="team",
                kind="team",
                namespace="ns-b",
                model_type=_TEAM_TYPE,
                description="beta team",
                payload=team_b,
            )
        )
        team_a = team_payload()
        team_a["name"] = "TeamPayloadIgnored-A"
        catalog.create(
            Entry(
                id="team",
                kind="team",
                namespace="ns-a",
                model_type=_TEAM_TYPE,
                description="alpha team",
                payload=team_a,
            )
        )
        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        data = response.json()
        assert data == [
            expected_summary("ns-a", description="alpha team"),
            expected_summary("ns-b", description="beta team"),
        ]

    def test_missing_name_in_payload_falls_back_to_namespace_identifier(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """Story 17.8 — defensive: missing/absent team-payload ``name`` is irrelevant.

        Bypasses the service to seed a team entry whose payload lacks the
        ``name`` key (a state ``create`` would reject). Pre-17.8 the row's
        ``name`` was ``""``; the new two-rung rule projects the namespace
        identifier instead. The team payload's ``name`` is never read either
        way.
        """
        client, catalog = api_client
        payload_without_name = team_payload()
        payload_without_name.pop("name")
        catalog._repository.put(
            Entry(
                id="team",
                kind="team",
                namespace="ns-no-name",
                model_type=_TEAM_TYPE,
                description="",
                payload=payload_without_name,
            )
        )
        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        data = response.json()
        assert data == [expected_summary("ns-no-name")]

    def test_namespace_without_team_entry_is_skipped(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """AC #3 — namespace with sub-entries but no team entry is omitted.

        The service's ``create`` invariants forbid seeding an agent without a
        team (tested via 409 in :class:`TestCreate` in
        ``test_api_router_crud.py``), so we bypass via the repository to
        synthesize the corrupted state, then verify the endpoint skips it
        silently (no error, no partial row).
        """
        client, catalog = api_client
        catalog._repository.put(
            Entry(
                id="orphan-agent",
                kind="agent",
                namespace="ns-no-team",
                model_type=_AGENT_TYPE,
                payload=_agent_payload("orphan"),
            )
        )
        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        namespaces = [row["namespace"] for row in response.json()]
        assert "ns-no-team" not in namespaces

    def test_includes_entries_across_user_ids(self, api_client: tuple[TestClient, Catalog]) -> None:
        """AC #4 — the endpoint does not filter by ``user_id``.

        Community-tier (``user_id="anonymous"``) and multi-tenant
        (``user_id="alice"``, ``user_id="bob"``) namespaces must all appear.
        Tenancy filtering is a caller concern.
        """
        client, catalog = api_client
        _seed_team(catalog, "ns-public", user_id="anonymous")
        _seed_team(catalog, "ns-alice", user_id="alice")
        _seed_team(catalog, "ns-bob", user_id="bob")
        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        namespaces = [row["namespace"] for row in response.json()]
        assert namespaces == ["ns-alice", "ns-bob", "ns-public"]

    def test_openapi_declares_response_model(self, api_client: tuple[TestClient, Catalog]) -> None:
        """AC #5 — ``/openapi.json`` exposes the operation with the right shape."""
        client, _ = api_client
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        op = spec["paths"]["/catalog/namespaces"]["get"]
        # Response schema is declared as ``list[NamespaceSummary]``.
        content = op["responses"]["200"]["content"]["application/json"]["schema"]
        assert content["type"] == "array"
        # Resolve the referenced component name.
        ref = content["items"]["$ref"]
        assert ref.endswith("/NamespaceSummary")
        component = spec["components"]["schemas"]["NamespaceSummary"]
        # Story 37.1 — nine pinned fields in declaration order (the six of
        # Story 18.2, ``owner`` and ``counts`` appended by Story 36.1, then
        # ``team_metadata``).
        assert set(component["properties"].keys()) == {
            "namespace",
            "name",
            "description",
            "team",
            "shareable",
            "public",
            "owner",
            "counts",
            "team_metadata",
        }
        # AC5 — declaration order pinned via OpenAPI's required-list ordering
        # (FastAPI emits declaration order in ``required:`` for required
        # fields). The original eight stay required; ``team_metadata``
        # carries a default, so FastAPI does not mark it required and the
        # list is unchanged.
        assert component["required"] == [
            "namespace",
            "name",
            "description",
            "team",
            "shareable",
            "public",
            "owner",
            "counts",
        ]
        # ``counts`` values are a component of their own, not an inline
        # integer — the shape that makes a second tally additive later.
        assert "NamespaceKindCount" in spec["components"]["schemas"]
        # The contract DTOs are emitted as components of their own too, so a
        # generated client gets named types rather than inline objects.
        assert "TeamMetadataContract" in spec["components"]["schemas"]
        assert "MetadataFieldDescriptor" in spec["components"]["schemas"]


# --- Story 17.2 — meta-then-team fallback ----------------------------------


class TestListNamespacesMetaFallback:
    """Story 17.2 AC6 — meta-then-team fallback for ``GET /catalog/namespaces``."""

    def test_meta_entry_takes_precedence_over_team(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, catalog = api_client
        team_entry_payload = team_payload()
        team_entry_payload["name"] = "Team Display Name"
        catalog.create(
            Entry(
                id="team",
                kind="team",
                namespace="tenant-42",
                model_type=_TEAM_TYPE,
                description="team description",
                payload=team_entry_payload,
            )
        )
        _seed_meta_entry(
            catalog,
            namespace="tenant-42",
            user_id="anonymous",
            name="Friendly Display",
            description="meta description",
        )
        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        rows = response.json()
        assert rows == [
            expected_summary(
                "tenant-42",
                name="Friendly Display",
                description="meta description",
                meta=True,
            )
        ]

    def test_namespace_identifier_fallback_when_no_meta_entry(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """Story 17.8 — when no meta entry exists, the row's ``name`` is the namespace identifier.

        The team's ``payload["name"]`` is set to a deliberately misleading
        sentinel and the assertion isolates the rule that the picker never
        reads it.
        """
        client, catalog = api_client
        team_entry_payload = team_payload()
        team_entry_payload["name"] = "TeamPayloadIgnored"
        catalog.create(
            Entry(
                id="team",
                kind="team",
                namespace="tenant-42",
                model_type=_TEAM_TYPE,
                description="team description",
                payload=team_entry_payload,
            )
        )
        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        rows = response.json()
        assert rows == [expected_summary("tenant-42", description="team description")]

    def test_meta_with_empty_name_falls_back_to_namespace_identifier(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """Story 17.8 — meta exists but ``name`` is missing/empty → namespace identifier wins.

        Pre-17.8 (Story 17.7): a graceful-degradation rung re-anchored the
        row's ``name`` on the team payload. Post-17.8: the fallback is the
        namespace identifier itself; the team payload's ``name`` is never
        read.
        """
        client, catalog = api_client
        team_entry_payload = team_payload()
        team_entry_payload["name"] = "TeamPayloadIgnored"
        catalog.create(
            Entry(
                id="team",
                kind="team",
                namespace="tenant-42",
                model_type=_TEAM_TYPE,
                description="team description",
                payload=team_entry_payload,
            )
        )
        # Bypass NamespaceMeta validation — directly seed a meta entry whose
        # payload lacks ``name`` (a state put cannot reject because the entry
        # is structurally valid; only ``NamespaceMeta`` validation enforces
        # name presence at the route layer).
        catalog._repository.put(
            Entry(
                id="_meta",
                kind="meta",
                namespace="tenant-42",
                model_type=_NAMESPACE_META_TYPE,
                description="meta description",
                payload={"description": "meta description", "properties": {}},
            )
        )
        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        rows = response.json()
        # name falls back to the namespace identifier; description comes
        # from meta.
        assert rows == [expected_summary("tenant-42", description="meta description", meta=True)]


# --- Story 17.7 — union discovery (team + meta) for /catalog/namespaces ---


class TestListNamespacesUnionDiscovery:
    """Story 17.7 / AC19 — union discovery surfaces team-only, team+meta, meta-only namespaces."""

    def test_three_namespace_fixture_team_meta_union(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """Three-namespace fixture with all three row classes per AC19.

        * ``ns-team-only`` — team entry, no meta. Expect ``team=True, shareable=False``.
        * ``ns-team-meta-shared`` — team + meta (shareable=True). Expect ``team=True,
          shareable=True``.
        * ``ns-meta-only`` — meta entry only (no team), shareable=True. Expect
          ``team=False, shareable=True``. This is the regression guard for the
          union-discovery widening — pre-17.7 this row was invisible.
        """
        client, catalog = api_client

        # Case 1: team-only.
        # Story 17.8 — team payload ``name`` is a deliberately misleading
        # sentinel (NoLongerAuthoritative) so the assertion below clearly
        # demonstrates that the team's payload ``name`` is NOT being read.
        team_entry_payload = team_payload()
        team_entry_payload["name"] = "NoLongerAuthoritative"
        catalog.create(
            Entry(
                id="team",
                kind="team",
                namespace="ns-team-only",
                model_type=_TEAM_TYPE,
                description="team only desc",
                payload=team_entry_payload,
            )
        )

        # Case 2: team + meta with shareable=True.
        team_payload_2 = team_payload()
        team_payload_2["name"] = "Team Plus Shared Meta"
        catalog.create(
            Entry(
                id="team",
                kind="team",
                namespace="ns-team-meta-shared",
                model_type=_TEAM_TYPE,
                description="team team desc",
                payload=team_payload_2,
            )
        )
        _seed_meta(
            catalog,
            "ns-team-meta-shared",
            name="Friendly Display",
            description="meta description",
            shareable=True,
        )

        # Case 3: meta-only (no team), shareable=True. Bypass create to avoid the
        # bootstrap invariant — meta-only library namespaces are out-of-band
        # for ``Catalog.create`` but legitimate state for the discovery query.
        catalog._repository.put(
            Entry(
                id="_meta",
                kind="meta",
                namespace="ns-meta-only",
                user_id="anonymous",
                model_type=_NAMESPACE_META_TYPE,
                description="meta-only desc",
                payload={
                    "name": "Library NS",
                    "description": "meta-only desc",
                    "properties": {},
                    "shareable": True,
                },
            )
        )

        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        rows = response.json()

        # Sorted alphabetically by namespace.
        assert [r["namespace"] for r in rows] == [
            "ns-meta-only",
            "ns-team-meta-shared",
            "ns-team-only",
        ]

        by_ns = {r["namespace"]: r for r in rows}
        # Story 17.8 — ``name`` is the namespace identifier (NOT
        # ``"NoLongerAuthoritative"`` — the team payload's ``name`` is no
        # longer a picker display source).
        assert by_ns["ns-team-only"] == expected_summary(
            "ns-team-only", description="team only desc"
        )
        assert by_ns["ns-team-meta-shared"] == expected_summary(
            "ns-team-meta-shared",
            name="Friendly Display",
            description="meta description",
            shareable=True,
            meta=True,
        )
        assert by_ns["ns-meta-only"] == expected_summary(
            "ns-meta-only",
            name="Library NS",
            description="meta-only desc",
            team=False,
            shareable=True,
            meta=True,
        )


class TestListNamespacesIgnoresTeamPayloadName:
    """Story 17.8 — regression guard: ``team.payload["name"]`` is never read by the picker.

    Pre-17.8 the projection chain re-anchored the row's ``name`` on the
    team's payload as the second rung of three. Post-17.8 the chain is
    two-rung (``meta.name`` → namespace identifier); the team payload's
    ``name`` is preserved on disk but is no longer authoritative for the
    picker. Each test below sets the team payload's ``name`` to a
    deliberately distinct sentinel so a regression that re-introduces the
    rung-2 fallback would surface a sentinel string in the assertion
    failure — making the regression visible at a glance.
    """

    def test_team_only_with_nonempty_payload_name_yields_namespace_identifier(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """(a) team-only with non-empty payload ``name`` → row ``name`` is namespace identifier."""
        client, catalog = api_client
        team_entry_payload = team_payload()
        team_entry_payload["name"] = "NoLongerAuthoritative"
        catalog.create(
            Entry(
                id="team",
                kind="team",
                namespace="ns-team-payload-ignored",
                model_type=_TEAM_TYPE,
                description="",
                payload=team_entry_payload,
            )
        )
        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        rows = response.json()
        by_ns = {r["namespace"]: r for r in rows}
        assert by_ns["ns-team-payload-ignored"]["name"] == "ns-team-payload-ignored"

    def test_team_only_with_empty_payload_name_yields_namespace_identifier(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """(b) team-only with empty payload ``name`` → row ``name`` is namespace identifier.

        Pre-17.8 (rung-2 path): an empty team-payload ``name`` projected
        as ``""``. Post-17.8: the namespace identifier is used in BOTH
        cases (a) and (b) — the team payload's ``name`` is never consulted.
        """
        client, catalog = api_client
        team_entry_payload = team_payload()
        team_entry_payload["name"] = ""
        catalog.create(
            Entry(
                id="team",
                kind="team",
                namespace="ns-empty-team-name",
                model_type=_TEAM_TYPE,
                description="",
                payload=team_entry_payload,
            )
        )
        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        rows = response.json()
        by_ns = {r["namespace"]: r for r in rows}
        assert by_ns["ns-empty-team-name"]["name"] == "ns-empty-team-name"

    def test_team_plus_meta_with_nonempty_meta_name_meta_wins(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """(c) team + meta with non-empty meta ``name`` → meta wins; team payload irrelevant.

        The team payload's ``name`` is set to a value distinct from BOTH
        the meta name AND the namespace identifier so the assertion
        verifies the rung-1 win, not happenstance.
        """
        client, catalog = api_client
        team_entry_payload = team_payload()
        team_entry_payload["name"] = "ShouldBeIgnored"
        catalog.create(
            Entry(
                id="team",
                kind="team",
                namespace="ns-meta-wins",
                model_type=_TEAM_TYPE,
                description="",
                payload=team_entry_payload,
            )
        )
        _seed_meta(catalog, "ns-meta-wins", name="MetaWins")
        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        rows = response.json()
        by_ns = {r["namespace"]: r for r in rows}
        assert by_ns["ns-meta-wins"]["name"] == "MetaWins"

    def test_empty_meta_name_with_team_falls_back_to_namespace_identifier(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """Story 17.8 — empty meta ``name`` with a team falls back to namespace identifier.

        Pre-17.8: an empty meta ``name`` triggered the graceful-degradation
        fallback to ``team.payload["name"]``. Post-17.8: the fallback is the
        namespace identifier. The team payload's ``name`` is set to a
        deliberately distinct sentinel so the test verifies that even with
        both a team AND a (name-empty) meta in hand, the team payload is
        bypassed.
        """
        client, catalog = api_client
        team_entry_payload = team_payload()
        team_entry_payload["name"] = "WouldHaveBeenUsed"
        catalog.create(
            Entry(
                id="team",
                kind="team",
                namespace="ns-team-only-with-empty-meta",
                model_type=_TEAM_TYPE,
                description="",
                payload=team_entry_payload,
            )
        )
        # Bypass NamespaceMeta validation — directly seed a meta entry with
        # an empty ``name`` (a state the route layer would reject but the
        # repository accepts).
        catalog._repository.put(
            Entry(
                id="_meta",
                kind="meta",
                namespace="ns-team-only-with-empty-meta",
                model_type=_NAMESPACE_META_TYPE,
                description="meta description",
                payload={
                    "name": "",
                    "description": "meta description",
                    "properties": {},
                    "shareable": False,
                },
            )
        )
        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        rows = response.json()
        by_ns = {r["namespace"]: r for r in rows}
        # NOT ``"WouldHaveBeenUsed"`` (team payload — never read), NOT ``""``
        # (the empty meta name) — the namespace identifier wins.
        assert by_ns["ns-team-only-with-empty-meta"]["name"] == "ns-team-only-with-empty-meta"


class TestNamespaceTeamPayloadRoundTrip:
    """Story 17.8 — picker decoupling does NOT mutate stored team payload data."""

    def test_team_payload_name_preserved_on_disk(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """``team.payload["name"]`` round-trips byte-identically through the picker.

        Create a team with ``payload["name"]="Operations"``; list namespaces
        (assert the row's ``name`` equals the namespace identifier — NOT
        ``"Operations"``); then ``Catalog.get(ns, "team")`` and assert
        ``team.payload["name"] == "Operations"`` unchanged. This pins that
        the picker decoupling does not mutate stored data.
        """
        client, catalog = api_client
        team_entry_payload = team_payload()
        team_entry_payload["name"] = "Operations"
        catalog.create(
            Entry(
                id="team",
                kind="team",
                namespace="ns-roundtrip",
                model_type=_TEAM_TYPE,
                description="",
                payload=team_entry_payload,
            )
        )
        # Picker: the row's ``name`` is the namespace identifier — not
        # ``"Operations"`` — confirming the team payload is bypassed.
        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        rows = response.json()
        by_ns = {r["namespace"]: r for r in rows}
        assert by_ns["ns-roundtrip"]["name"] == "ns-roundtrip"
        # On-disk: the team payload's ``name`` is preserved byte-identically.
        stored = catalog.get("ns-roundtrip", "team")
        assert isinstance(stored.payload, dict)
        assert stored.payload["name"] == "Operations"


class TestListNamespacesRepositoryCallsAreConstant:
    """Counting-spy guard: the handler's repository traffic does not grow with N.

    Story 17.7 pinned two ``list`` calls; Story 36.1 widened the query to
    one per ``EntryKind``. The number changed — the invariant did not:
    the count is a constant, never ``k + N``, and no per-row
    ``catalog.get(*, "_meta")`` round-trip fires.
    """

    def test_one_listing_per_entry_kind_independent_of_n(
        self, counting_catalog: tuple[Catalog, Any]
    ) -> None:
        """For N namespaces, exactly 6 repository ``list`` calls fire — not 6 + N.

        Wraps the underlying repository with ``CountingEntryRepository`` and
        seeds N=3 namespaces (matching AC20's "N >= 3" minimum so a per-row
        round-trip would produce >= 9 calls vs the asserted 6).
        """
        from akgentic.catalog.api._errors import add_exception_handlers
        from akgentic.catalog.api._settings import CatalogRouterSettings
        from akgentic.catalog.api.router import build_router, set_catalog

        catalog, counting = counting_catalog

        # Seed three namespaces — each with a team to satisfy bootstrap.
        for ns_name in ("ns-1", "ns-2", "ns-3"):
            _seed_team(catalog, ns_name)
        # Add a meta to ns-2 to broaden coverage of the union path.
        _seed_meta(catalog, "ns-2", name="N2", shareable=True)

        # Reset call counts AFTER seeding (we only care about counts during
        # the route handler dispatch).
        counting.reset()

        pytest.importorskip("fastapi")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI(title="counting")
        app.include_router(build_router(CatalogRouterSettings(expose_generic_kind_crud=True)))
        set_catalog(catalog)
        add_exception_handlers(app)
        client = TestClient(app)

        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        rows = response.json()
        # All three namespaces surface (regression guard for union widening).
        assert sorted(r["namespace"] for r in rows) == ["ns-1", "ns-2", "ns-3"]

        # Exactly six ``list`` calls hit the repository — one per
        # ``EntryKind`` — and the number is a constant, not 6 + N (= 9).
        list_calls = [c for c in counting.calls if c[0] == "list"]
        assert counting.count("list") == len(_ENTRY_KIND_NAMES), (
            f"expected exactly {len(_ENTRY_KIND_NAMES)} list() calls, "
            f"got {counting.count('list')}: {list_calls}"
        )
        # One query per kind, derived from the alias rather than spelled out.
        kinds = sorted(c[1][0].kind for c in list_calls)
        assert kinds == sorted(_ENTRY_KIND_NAMES)

        # Counts never go through the unfiltered ``list_by_namespace``
        # pass-through — doing so would report entries the caller cannot see.
        assert counting.count("list_by_namespace") == 0, (
            "the handler must count through the visibility-filtered "
            "catalog.list, never list_by_namespace"
        )

        # No per-row ``catalog.get(ns, "_meta")`` round-trip during the
        # handler — this is the regression we are guarding against.
        meta_gets = [
            c for c in counting.calls if c[0] == "get" and len(c[1]) == 2 and c[1][1] == "_meta"
        ]
        assert meta_gets == [], (
            f"expected 0 catalog.get(*, _meta) calls during handler dispatch, got {meta_gets}"
        )


# --- Story 18.2 — NamespaceSummary.public projection ----------------------------


class TestNamespaceSummaryPublicField:
    """Story 18.2 AC2 — ``NamespaceSummary.public`` projection from ``_meta`` payload."""

    def test_namespace_summary_field_order(self) -> None:
        # Nine pinned fields in declaration order — the six of Story 18.2,
        # ``owner`` and ``counts`` appended by Story 36.1, then
        # ``team_metadata`` appended by Story 37.1. The lockdown catches
        # accidental reorders that would shift the OpenAPI / wire-format key
        # order downstream.
        from akgentic.catalog.api.router import NamespaceSummary

        assert list(NamespaceSummary.model_fields.keys()) == [
            "namespace",
            "name",
            "description",
            "team",
            "shareable",
            "public",
            "owner",
            "counts",
            "team_metadata",
        ]

    def test_namespace_summary_has_public_field_when_meta_public_true(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-public-true")
        catalog._repository.put(
            Entry(
                id="_meta",
                kind="meta",
                namespace="ns-public-true",
                user_id="anonymous",
                model_type=_NAMESPACE_META_TYPE,
                description="",
                payload={
                    "name": "Forkable Library",
                    "description": "",
                    "properties": {},
                    "shareable": False,
                    "public": True,
                },
            )
        )
        response = client.get("/catalog/namespaces")
        assert response.status_code == 200
        rows = response.json()
        by_ns = {r["namespace"]: r for r in rows}
        assert by_ns["ns-public-true"]["public"] is True
        assert by_ns["ns-public-true"]["shareable"] is False

    def test_namespace_summary_public_strict_bool(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        # AC2 — truthy strings do NOT flip the flag; the projection uses
        # ``is True`` (strict-bool comparison), mirroring ``shareable``.
        client, catalog = api_client
        _seed_team(catalog, "ns-public-string")
        catalog._repository.put(
            Entry(
                id="_meta",
                kind="meta",
                namespace="ns-public-string",
                user_id="anonymous",
                model_type=_NAMESPACE_META_TYPE,
                description="",
                payload={
                    "name": "Confused Public",
                    "description": "",
                    "properties": {},
                    "shareable": False,
                    "public": "yes",  # NOT a real bool — defensive projection.
                },
            )
        )
        response = client.get("/catalog/namespaces")
        rows = response.json()
        by_ns = {r["namespace"]: r for r in rows}
        assert by_ns["ns-public-string"]["public"] is False

    def test_namespace_summary_public_default_false_when_no_meta(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-no-meta")
        response = client.get("/catalog/namespaces")
        rows = response.json()
        by_ns = {r["namespace"]: r for r in rows}
        assert by_ns["ns-no-meta"]["public"] is False

    def test_namespace_summary_public_false_when_meta_payload_omits_key(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        # A meta entry whose payload omits the ``public`` key projects to
        # ``public=False`` — no AttributeError, no truthy fallback.
        client, catalog = api_client
        _seed_team(catalog, "ns-meta-no-public-key")
        catalog._repository.put(
            Entry(
                id="_meta",
                kind="meta",
                namespace="ns-meta-no-public-key",
                user_id="anonymous",
                model_type=_NAMESPACE_META_TYPE,
                description="",
                payload={"name": "Pre-18.2 meta", "description": "", "properties": {}},
            )
        )
        response = client.get("/catalog/namespaces")
        rows = response.json()
        by_ns = {r["namespace"]: r for r in rows}
        assert by_ns["ns-meta-no-public-key"]["public"] is False


# --- Story 34.2 — a bundle-imported namespace shows its description -------------


class TestBundleImportedNamespaceDescription:
    """A namespace imported from a bundle shows its header description in the picker."""

    def test_bundle_imported_namespace_row_shows_the_header_description(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        # The route path (PUT /catalog/namespace/{ns}/meta) has always set the
        # meta entry's ``description``; the bundle-import path did not, so the
        # same namespace showed a blank row depending on how it was created.
        client, catalog = api_client
        bundle = dump_namespace(
            [
                Entry(
                    id="team",
                    kind="team",
                    namespace="ns-from-bundle",
                    user_id="anonymous",
                    model_type=_TEAM_TYPE,
                    payload=team_payload(),
                )
            ],
            name="Imported tenant",
            description="described in the bundle header",
        )
        catalog.import_namespace_yaml(bundle)

        rows = client.get("/catalog/namespaces").json()
        by_ns = {r["namespace"]: r for r in rows}
        assert by_ns["ns-from-bundle"]["description"] == "described in the bundle header"


# --- Story 36.1 — the row names its owner and tallies its entries ---------------


def _put_entry(
    catalog: Catalog,
    namespace: str,
    id: str,
    kind: EntryKind,
    user_id: str = "anonymous",
) -> Entry:
    """Seed one entry of any kind straight through the repository.

    The shared conftest has no seeder for ``tool`` / ``model`` / ``prompt``,
    and ``Catalog.create``'s bootstrap invariant forbids a sub-entry in a
    namespace holding neither a team nor a meta — which is exactly the
    state the tool-only regression guard needs. Kept local rather than
    widening the shared conftest for one module.
    """
    return catalog._repository.put(
        Entry(
            id=id,
            kind=kind,
            namespace=namespace,
            user_id=user_id,
            model_type=_AGENT_TYPE,
            payload={},
        )
    )


class TestNamespaceKindCountShape:
    """The tally model ships one field, and only one."""

    def test_ships_exactly_one_field(self) -> None:
        """``total`` and nothing else.

        A second tally is wanted later and is deliberately not here: a
        placeholder that is always zero cannot be told apart from a real
        zero, so a consumer would render it as fact. This assertion fails
        loudly the moment one is added.
        """
        from akgentic.catalog.api.router import NamespaceKindCount

        assert list(NamespaceKindCount.model_fields.keys()) == ["total"]

    def test_is_exported_alongside_namespace_summary(self) -> None:
        from akgentic.catalog.api import router

        assert "NamespaceKindCount" in router.__all__
        assert "NamespaceSummary" in router.__all__


class TestNamespaceSummaryOwner:
    """``owner`` — the team entry's ``user_id``, else the meta's, else ``None``."""

    def test_team_entry_supplies_the_owner(self, api_client: tuple[TestClient, Catalog]) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-team-owned", user_id="alice")
        rows = client.get("/catalog/namespaces").json()
        assert {r["namespace"]: r["owner"] for r in rows} == {"ns-team-owned": "alice"}

    def test_meta_entry_supplies_the_owner_when_there_is_no_team(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, catalog = api_client
        catalog._repository.put(make_meta_entry("ns-library", name="Library", user_id="carol"))
        rows = client.get("/catalog/namespaces").json()
        assert {r["namespace"]: r["owner"] for r in rows} == {"ns-library": "carol"}

    def test_team_wins_when_team_and_meta_name_different_owners(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """The two entries deliberately disagree, so this proves precedence.

        With both owners equal the assertion would pass on either rung and
        prove nothing.
        """
        client, catalog = api_client
        _seed_team(catalog, "ns-two-owners", user_id="alice")
        catalog._repository.put(make_meta_entry("ns-two-owners", name="Shared", user_id="bob"))
        rows = client.get("/catalog/namespaces").json()
        by_ns = {r["namespace"]: r for r in rows}
        assert by_ns["ns-two-owners"]["owner"] == "alice"

    def test_owner_is_none_when_neither_entry_is_present(self) -> None:
        """The bottom rung, reachable only at the helper level.

        Every listed namespace has a team or a meta entry by construction,
        and ``Entry.user_id`` is non-empty by construction, so no request
        can produce this row. The helper takes no catalog and performs no
        repository I/O, which is what makes the rung testable at all.
        """
        assert _derive_owner(None, None) is None
        row = _build_namespace_summary(
            "ns-nobody",
            None,
            None,
            owner=_derive_owner(None, None),
            counts=_zero_counts(),
            team_metadata=None,
        )
        assert row.owner is None
        assert row.team is False
        assert set(row.counts) == set(get_args(EntryKind))
        assert all(count.total == 0 for count in row.counts.values())

    def test_rows_with_different_owners_are_all_returned_to_one_caller(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """``owner`` is reported, never enforced — it filters nothing.

        Two namespaces with different owners both reach the same caller;
        adding the field introduced no gate.
        """
        client, catalog = api_client
        _seed_team(catalog, "ns-alice", user_id="alice")
        _seed_team(catalog, "ns-bob", user_id="bob")
        rows = client.get("/catalog/namespaces").json()
        assert {r["namespace"]: r["owner"] for r in rows} == {
            "ns-alice": "alice",
            "ns-bob": "bob",
        }


class TestNamespaceSummaryCounts:
    """``counts`` — every kind on every row, tallied from the widened query."""

    def test_every_row_carries_a_key_for_every_entry_kind(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """Pinned against the ``EntryKind`` alias, not a hand-written list.

        A seventh kind added to the alias fails here rather than surfacing
        as a missing key in a browser.
        """
        client, catalog = api_client
        _seed_team(catalog, "ns-1")
        _seed_team(catalog, "ns-2")
        _seed_meta(catalog, "ns-2", name="N2")
        rows = client.get("/catalog/namespaces").json()
        assert [r["namespace"] for r in rows] == ["ns-1", "ns-2"]
        for row in rows:
            assert set(row["counts"]) == set(get_args(EntryKind))

    def test_a_namespace_holding_a_mix_of_kinds_is_tallied_per_kind(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        client, catalog = api_client
        _seed_team(catalog, "ns-mixed")
        _seed_agent(catalog, "ns-mixed", id="agent-a")
        _seed_agent(catalog, "ns-mixed", id="agent-b")
        _put_entry(catalog, "ns-mixed", "tool-a", "tool")
        _put_entry(catalog, "ns-mixed", "model-a", "model")
        _seed_meta(catalog, "ns-mixed", name="Mixed")
        rows = client.get("/catalog/namespaces").json()
        assert rows == [
            expected_summary(
                "ns-mixed",
                name="Mixed",
                meta=True,
                counts={"agent": 2, "tool": 1, "model": 1},
            )
        ]
        # Spelled out once, so the zero-filled kinds are visible in the diff
        # of a future change rather than hidden behind the helper.
        assert rows[0]["counts"] == {
            "team": {"total": 1},
            "agent": {"total": 2},
            "tool": {"total": 1},
            "model": {"total": 1},
            "prompt": {"total": 0},
            "meta": {"total": 1},
        }

    def test_flags_and_tallies_on_the_same_row_agree(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """``team`` is True iff the team tally is >= 1, on both classes of row.

        And a row whose ``shareable`` / ``public`` flags were projected
        from a meta entry necessarily counts that meta entry.
        """
        client, catalog = api_client
        _seed_team(catalog, "ns-with-team")
        catalog._repository.put(
            make_meta_entry("ns-library", name="Library", shareable=True, public=True)
        )
        rows = client.get("/catalog/namespaces").json()
        assert [r["namespace"] for r in rows] == ["ns-library", "ns-with-team"]
        for row in rows:
            assert row["team"] is (row["counts"]["team"]["total"] >= 1)
        by_ns = {r["namespace"]: r for r in rows}
        assert by_ns["ns-library"]["shareable"] is True
        assert by_ns["ns-library"]["public"] is True
        assert by_ns["ns-library"]["counts"]["meta"]["total"] >= 1


class TestNamespaceDiscoveryIsNotWidenedByTheCountQuery:
    """The widened query feeds ``counts`` — it must not add rows."""

    def test_a_namespace_holding_only_tools_is_absent_from_the_listing(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """The headline regression guard for the six-kind widening.

        A namespace is surfaced iff it has a team OR a meta entry. Six
        queries now feed the handler where two did; taking the discovery
        union over all six slices instead of the team and meta ones would
        put library-only namespaces into the team-creation picker. The
        sibling assertion shows the widened query really does see the
        tool entries — they are counted, they are simply not a reason to
        list the namespace.
        """
        client, catalog = api_client
        _seed_team(catalog, "ns-real")
        _put_entry(catalog, "ns-tools-only", "tool-a", "tool")
        _put_entry(catalog, "ns-tools-only", "tool-b", "tool")

        rows = client.get("/catalog/namespaces").json()
        assert [r["namespace"] for r in rows] == ["ns-real"]

        # The same widened query the handler issues DID see those entries.
        by_kind = {kind: catalog.list(EntryQuery(kind=kind)) for kind in _ENTRY_KINDS}
        tallies = _count_by_namespace(by_kind)
        assert tallies["ns-tools-only"]["tool"].total == 2


class TestNamespaceCountsRespectVisibility:
    """Tallies come from the visibility-filtered ``Catalog.list``, never the raw one."""

    async def test_counts_are_filtered_under_a_caller_identity(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """A caller's tallies match what that caller could actually list.

        ``ns-shared`` holds three agent entries, one of them owned by
        another user and its namespace is not public — so alice sees two.
        Counting through the unfiltered ``list_by_namespace`` would report
        three and disclose the existence of an entry alice cannot read.
        ``ns-bobs`` is another user's namespace entirely and contributes
        nothing at all.
        """
        _, catalog = api_client
        _seed_team(catalog, "ns-shared", user_id="alice")
        _put_entry(catalog, "ns-shared", "agent-a", "agent", user_id="alice")
        _put_entry(catalog, "ns-shared", "agent-b", "agent", user_id="alice")
        _put_entry(catalog, "ns-shared", "agent-c", "agent", user_id="bob")
        _seed_team(catalog, "ns-bobs", user_id="bob")

        with Catalog.as_caller("alice"):
            rows = await list_namespaces()
            visible_agents = await list_entries("agent", namespace="ns-shared")

        by_ns = {r.namespace: r for r in rows}
        assert set(by_ns) == {"ns-shared"}
        assert by_ns["ns-shared"].counts["agent"].total == 2
        # The counts agree with what GET /catalog/{kind} returns to the
        # same caller — the equivalence the frontend's numbers rely on.
        assert by_ns["ns-shared"].counts["agent"].total == len(visible_agents)

    async def test_counts_are_unfiltered_without_a_caller_identity(
        self, api_client: tuple[TestClient, Catalog]
    ) -> None:
        """Community tier — no contextvar, no filter, byte-identical to before."""
        _, catalog = api_client
        _seed_team(catalog, "ns-shared", user_id="alice")
        _put_entry(catalog, "ns-shared", "agent-a", "agent", user_id="alice")
        _put_entry(catalog, "ns-shared", "agent-b", "agent", user_id="alice")
        _put_entry(catalog, "ns-shared", "agent-c", "agent", user_id="bob")
        _seed_team(catalog, "ns-bobs", user_id="bob")

        rows = await list_namespaces()

        by_ns = {r.namespace: r for r in rows}
        assert set(by_ns) == {"ns-bobs", "ns-shared"}
        assert by_ns["ns-shared"].counts["agent"].total == 3
