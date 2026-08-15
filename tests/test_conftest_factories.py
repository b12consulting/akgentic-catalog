"""Guard tests for the shared payload factories in ``tests/conftest.py``."""

from __future__ import annotations

from .conftest import team_payload


def test_team_payload_returns_independent_objects() -> None:
    """Mutating one payload must not reach the next one.

    Call sites across the suite mutate the returned dict in place — assigning
    to ``["name"]`` and, at ``tests/v2/test_api_router.py``, ``pop``-ing the
    key outright. If the factory ever returns a shared module-level literal,
    those edits leak into every test that runs afterwards in the same process,
    and the failure surfaces somewhere unrelated to its cause.

    The nested mutations are what make this non-trivial: they go red against a
    hoisted constant returned directly AND against a shallow ``.copy()`` of
    one, which top-level assertions alone would not catch.
    """
    first = team_payload()
    first["name"] = "mutated"
    first.pop("description")
    first["entry_point"]["card"]["skills"].append("leaked")
    first["entry_point"]["card"]["config"]["role"] = "mutated"

    second = team_payload()

    assert second["name"] == "team"
    assert second["description"] == ""
    assert second["entry_point"]["card"]["skills"] == []
    assert second["entry_point"]["card"]["config"]["role"] == "entry"
