"""Tests for Story 17.3 / AC19 — ``create_app(cross_namespace_refs_allowed=...)``.

Asserts the app-factory threads the allowlist into the injected ``Catalog``.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_create_app_threads_cross_namespace_refs_allowed(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from akgentic.catalog.api import router as router_module
    from akgentic.catalog.api.app import create_app

    create_app(
        backend="yaml",
        yaml_base_path=tmp_path,
        cross_namespace_refs_allowed=frozenset({"global"}),
    )
    catalog = router_module._get_catalog()
    assert catalog._cross_namespace_refs_allowed == frozenset({"global"})


def test_create_app_default_empty_allowlist(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from akgentic.catalog.api import router as router_module
    from akgentic.catalog.api.app import create_app

    create_app(backend="yaml", yaml_base_path=tmp_path)
    catalog = router_module._get_catalog()
    assert catalog._cross_namespace_refs_allowed == frozenset()


class TestCatalogCtorAllowlistKwarg:
    """Story 17.3 / AC1 — ``Catalog.__init__`` ctor argument shape."""

    def test_default_empty_frozenset(self) -> None:
        from akgentic.catalog.catalog import Catalog

        from .conftest import FakeEntryRepository

        catalog = Catalog(FakeEntryRepository())
        assert catalog._cross_namespace_refs_allowed == frozenset()

    def test_kwarg_stored(self) -> None:
        from akgentic.catalog.catalog import Catalog

        from .conftest import FakeEntryRepository

        catalog = Catalog(
            FakeEntryRepository(),
            cross_namespace_refs_allowed=frozenset({"global"}),
        )
        assert catalog._cross_namespace_refs_allowed == frozenset({"global"})

    def test_positional_arg_rejected(self) -> None:
        from akgentic.catalog.catalog import Catalog

        from .conftest import FakeEntryRepository

        with pytest.raises(TypeError):
            Catalog(  # type: ignore[misc]
                FakeEntryRepository(), frozenset({"global"})
            )
