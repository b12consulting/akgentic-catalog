"""Tests for ``akgentic.catalog.resolver.load_model_type`` and constants."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.catalog.resolver import REF_KEY, TYPE_KEY, load_model_type

from .conftest import register_akgentic_test_module


def _model_with_reserved_field(reserved_name: str) -> type[BaseModel]:
    """Return a throwaway BaseModel subclass with ``reserved_name`` in ``model_fields``.

    Pydantic 2 silently drops dunder-named fields through both the normal class
    body and ``create_model``. We sidestep the limitation by manually inserting
    a ``FieldInfo`` into ``model_fields`` on a plain subclass — good enough for
    the resolver's ``reserved_name in cls.model_fields`` check.
    """

    class _Host(BaseModel):
        placeholder: str = ""

    _Host.model_fields[reserved_name] = FieldInfo(annotation=str, default="")
    return _Host


class TestResolverConstants:
    """AC6 — REF_KEY / TYPE_KEY have the expected literal values."""

    def test_ref_key_value(self) -> None:
        assert REF_KEY == "__ref__"

    def test_type_key_value(self) -> None:
        assert TYPE_KEY == "__type__"


class TestLoadModelTypeAllowlist:
    """AC10 — non-allowlisted paths are rejected by ``load_model_type``."""

    def test_rejects_os_system(self) -> None:
        with pytest.raises(CatalogValidationError) as exc_info:
            load_model_type("os.system")
        assert len(exc_info.value.errors) == 1
        message = exc_info.value.errors[0]
        assert "outside allowlist" in message
        assert "os.system" in message

    def test_rejects_builtins_eval(self) -> None:
        with pytest.raises(CatalogValidationError) as exc_info:
            load_model_type("builtins.eval")
        assert "outside allowlist" in exc_info.value.errors[0]
        assert "builtins.eval" in exc_info.value.errors[0]


class TestLoadModelTypeReservedKeys:
    """AC11 — classes declaring ``__ref__`` or ``__type__`` fields are rejected."""

    def test_ref_key_collision_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        colliding_model = _model_with_reserved_field(REF_KEY)
        module_name = register_akgentic_test_module(
            monkeypatch, "tests_fixture_15_1_ref", CollidingRefModel=colliding_model
        )

        with pytest.raises(CatalogValidationError) as exc_info:
            load_model_type(f"{module_name}.CollidingRefModel")

        message = exc_info.value.errors[0]
        assert "reserved ref-sentinel fields" in message
        assert REF_KEY in message

    def test_type_key_collision_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        colliding_model = _model_with_reserved_field(TYPE_KEY)
        module_name = register_akgentic_test_module(
            monkeypatch,
            "tests_fixture_15_1_type",
            CollidingTypeModel=colliding_model,
        )

        with pytest.raises(CatalogValidationError) as exc_info:
            load_model_type(f"{module_name}.CollidingTypeModel")

        message = exc_info.value.errors[0]
        assert "reserved ref-sentinel fields" in message
        assert TYPE_KEY in message


class TestLoadModelTypeNonBaseModel:
    """AC12 — non-BaseModel classes are rejected."""

    def test_rejects_dataclass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        @dataclass
        class NotAModel:
            x: int = 0

        module_name = register_akgentic_test_module(
            monkeypatch, "tests_fixture_15_1_notmodel", NotAModel=NotAModel
        )

        with pytest.raises(CatalogValidationError) as exc_info:
            load_model_type(f"{module_name}.NotAModel")

        assert "is not a Pydantic BaseModel subclass" in exc_info.value.errors[0]

    def test_rejects_plain_function(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def some_function() -> None:  # noqa: ANN401 — irrelevant; test fixture
            return None

        module_name = register_akgentic_test_module(
            monkeypatch, "tests_fixture_15_1_fn", some_function=some_function
        )

        with pytest.raises(CatalogValidationError) as exc_info:
            load_model_type(f"{module_name}.some_function")

        assert "is not a Pydantic BaseModel subclass" in exc_info.value.errors[0]


class TestLoadModelTypeHappyPath:
    """AC13 — a real akgentic.* BaseModel class resolves by identity."""

    def test_loads_agent_card(self) -> None:
        from akgentic.core.agent_card import AgentCard

        result = load_model_type("akgentic.core.agent_card.AgentCard")
        assert result is AgentCard
