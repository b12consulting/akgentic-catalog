"""Shared fixtures and factories for the v2 test suite.

Factory functions are plain helpers (not pytest fixtures) per the project
convention for stateless constructions. The one fixture-like helper
(``register_akgentic_test_module``) is a plain function that accepts a
``monkeypatch`` argument so test cleanup is handled by pytest's built-in
fixture teardown without additional bookkeeping in the test body.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from akgentic.catalog.models.entry import Entry


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
