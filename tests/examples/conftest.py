"""Fixtures for the examples harness.

The one thing this suite needs that the rest of the tests get elsewhere is
isolation of the ``model_type`` prefix allowlist. ``tests/v2/conftest.py`` has an
equivalent fixture, but it is scoped to ``tests/v2/`` and does not reach here.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from akgentic.catalog import ENV_VAR, reset_allowed_prefixes


@pytest.fixture(autouse=True)
def _isolate_allowlist_policy(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give every example a pristine, unconfigured ``model_type`` prefix policy.

    ``akgentic.catalog.allowlist`` caches the resolved policy in a module-level
    global. An example that widens the allowlist — teaching how a deployment
    registers its own models — would otherwise poison every test running later in
    the same process, and the damage would surface as unrelated tests going red
    rather than as a failure in the example that caused it.

    Clears the environment variable and resets the cache on both sides, so neither
    an inherited policy nor a leaked one can reach an example.
    """
    monkeypatch.delenv(ENV_VAR, raising=False)
    reset_allowed_prefixes()
    yield
    reset_allowed_prefixes()
