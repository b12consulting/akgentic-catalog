"""``GET /catalog/namespaces`` — namespace discovery and summary projection."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from akgentic.catalog.catalog import Catalog  # noqa: E402
from akgentic.catalog.models.entry import Entry  # noqa: E402
from akgentic.catalog.serialization import dump_namespace  # noqa: E402

from ..conftest import team_payload  # noqa: E402
from .conftest import (  # noqa: E402
    _AGENT_TYPE,
    _NAMESPACE_META_TYPE,
    _TEAM_TYPE,
    _agent_payload,
    _seed_meta,
    _seed_meta_entry,
    _seed_team,
)


def expected_summary(
    namespace: str,
    *,
    name: str | None = None,
    description: str = "",
    team: bool = True,
    shareable: bool = False,
    public: bool = False,
) -> dict[str, Any]:
    """Build one expected ``NamespaceSummary`` row; ``name`` defaults to ``namespace``."""
    return {
        "namespace": namespace,
        "name": namespace if name is None else name,
        "description": description,
        "team": team,
        "shareable": shareable,
        "public": public,
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
        # Story 18.2 — six pinned fields in declaration order.
        assert set(component["properties"].keys()) == {
            "namespace",
            "name",
            "description",
            "team",
            "shareable",
            "public",
        }
        # AC5 — declaration order pinned via OpenAPI's required-list ordering
        # (FastAPI emits declaration order in ``required:`` for required
        # fields). All six fields are required (no defaults that allow
        # omission in the response shape).
        assert component["required"] == [
            "namespace",
            "name",
            "description",
            "team",
            "shareable",
            "public",
        ]


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
            expected_summary("tenant-42", name="Friendly Display", description="meta description")
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
        assert rows == [expected_summary("tenant-42", description="meta description")]


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
        )
        assert by_ns["ns-meta-only"] == expected_summary(
            "ns-meta-only",
            name="Library NS",
            description="meta-only desc",
            team=False,
            shareable=True,
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


class TestListNamespacesNoExtraRoundtrip:
    """Story 17.7 / AC20 — counting-spy regression guard against per-row catalog.get round-trips."""

    def test_at_most_two_repository_listings_independent_of_n(
        self, counting_catalog: tuple[Catalog, Any]
    ) -> None:
        """For N namespaces, exactly 2 repository ``list`` calls fire — not 2 + N.

        Wraps the underlying repository with ``CountingEntryRepository`` and
        seeds N=3 namespaces (matching AC20's "N >= 3" minimum so a per-row
        round-trip would produce >= 5 calls vs the asserted 2).
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

        # AC20 — exactly two ``list`` calls hit the repository: one for
        # kind="team", one for kind="meta". A per-row ``catalog.get(ns,
        # "_meta")`` round-trip would produce 2 + N (= 5) repository calls
        # under the previous shape; we explicitly assert 2.
        list_calls = [c for c in counting.calls if c[0] == "list"]
        assert counting.count("list") == 2, (
            f"expected exactly 2 list() calls, got {counting.count('list')}: {list_calls}"
        )
        # Verify the two queries are kind="team" and kind="meta".
        kinds = sorted(c[1][0].kind for c in list_calls)
        assert kinds == ["meta", "team"]

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
        # AC2 — six pinned fields in declaration order. The lockdown catches
        # accidental reorders that would shift the OpenAPI / wire-format
        # key order downstream.
        from akgentic.catalog.api.router import NamespaceSummary

        assert list(NamespaceSummary.model_fields.keys()) == [
            "namespace",
            "name",
            "description",
            "team",
            "shareable",
            "public",
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
