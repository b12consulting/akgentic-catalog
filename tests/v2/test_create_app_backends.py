"""``create_app`` factory wiring for the YAML and MongoDB backends."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402


class TestCreateApp:
    """AC4, AC5 — ``create_app`` factory wires YAML + MongoDB backends."""

    def test_yaml_backend_default_path(self, tmp_path: Any) -> None:
        from akgentic.catalog.api.app import create_app

        base = tmp_path / "custom-root"
        app = create_app(backend="yaml", yaml_base_path=base)
        assert app.title == "Akgentic Catalog"
        assert base.exists()
        client = TestClient(app)
        response = client.get("/catalog/model_types")
        assert response.status_code == 200

    def test_unknown_backend_raises(self) -> None:
        from akgentic.catalog.api.app import create_app

        with pytest.raises(ValueError, match="Unknown backend"):
            create_app(backend="sqlite")  # type: ignore[arg-type]

    def test_mongodb_missing_config_raises(self) -> None:
        from akgentic.catalog.api.app import create_app

        with pytest.raises(ValueError, match="mongo_config is required"):
            create_app(backend="mongodb")

    def test_mongodb_backend_wires_collection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("mongomock")
        import mongomock

        from akgentic.catalog.api.app import create_app
        from akgentic.catalog.repositories.mongo import MongoCatalogConfig

        config = MongoCatalogConfig(connection_string="mongodb://x", database="db_test")

        def _fake_client(self: MongoCatalogConfig) -> mongomock.MongoClient:
            return mongomock.MongoClient()

        monkeypatch.setattr(MongoCatalogConfig, "create_client", _fake_client)
        app = create_app(backend="mongodb", mongo_config=config)
        assert app.title == "Akgentic Catalog"
