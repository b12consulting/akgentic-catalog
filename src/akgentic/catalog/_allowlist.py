"""Configurable allowlist of dotted-path prefixes for catalog ``model_type``.

Single source of truth shared by the two defence layers that gate which
Pydantic classes an ``Entry.model_type`` may name:

* the storage-side annotation check (``models.entry.AllowlistedPath``), and
* the runtime resolver (``resolver.load_model_type``).

Both call :func:`allowed_prefixes` so the two layers can never drift.

Security boundary: prefixes come from the **deployment environment**, never
from catalog data. The operator sets ``AKGENTIC_CATALOG_ALLOWED_PREFIXES``
(comma-separated) at process start; each consumer app thereby declares its own
trusted namespaces. ``akgentic.`` is always included so framework-owned types
resolve regardless of configuration, and it can never be removed by config.
"""

from __future__ import annotations

import os

__all__ = ["ALLOWLIST_ENV_VAR", "ALWAYS_ALLOWED_PREFIX", "allowed_prefixes"]

ALLOWLIST_ENV_VAR = "AKGENTIC_CATALOG_ALLOWED_PREFIXES"
ALWAYS_ALLOWED_PREFIX = "akgentic."


def allowed_prefixes(environ: dict[str, str] | None = None) -> tuple[str, ...]:
    """Return the effective allowlisted ``model_type`` prefixes.

    Reads ``AKGENTIC_CATALOG_ALLOWED_PREFIXES`` (comma-separated) from the
    environment and prepends the always-allowed ``akgentic.`` prefix. Read
    live (not cached) so a process that sets the variable before its first
    ``Entry`` construction gets the configured value, and tests can vary it.

    Args:
        environ: Environment mapping to read. Defaults to ``os.environ`` when
            omitted; tests pass an explicit dict to avoid touching the process
            environment.

    Returns:
        Ordered, de-duplicated prefixes with ``akgentic.`` always first.
    """
    env = environ if environ is not None else os.environ
    raw = env.get(ALLOWLIST_ENV_VAR, "")
    result: list[str] = [ALWAYS_ALLOWED_PREFIX]
    for prefix in (p.strip() for p in raw.split(",")):
        if prefix and prefix not in result:
            result.append(prefix)
    return tuple(result)
