"""The ``Entry.model_type`` runtime story: load one, or enumerate them all.

Implements ADR-019 §D4. ``allowlist.py`` already claimed to own model-type
policy, so a reader looking for it found half the answer there and the other
half — the import gate and the picker's ``sys.modules`` walk — buried in the
resolver, a module about refs. This module holds that other half.

The split with ``allowlist.py`` is deliberate and one-directional:
``allowlist`` owns the *prefix predicate* and imports nothing first-party;
this module owns everything that *acts* on it (importing a class, checking it,
walking loaded modules for candidates) and depends on ``allowlist``,
``models/errors`` and ``refs`` alone. Merging the two would drag
``import_class`` and a ``sys.modules`` walk onto ``allowlist``'s leaf.

``resolver`` re-exports both public names at module level, which is the import
path every consumer in this package and downstream uses — ``GET
/catalog/model_types`` and the ``ak-catalog model-types`` verb among them.
"""

from __future__ import annotations

import sys
from typing import Any

from pydantic import BaseModel

from akgentic.core.utils.deserializer import import_class

from .allowlist import allowed_prefixes, prefix_violation
from .models.errors import CatalogValidationError
from .refs import RESERVED_REF_KEYS

__all__ = [
    "enumerate_allowlisted_model_types",
    "load_model_type",
]


def load_model_type(path: str) -> type[BaseModel]:
    """Import and return a Pydantic ``BaseModel`` class by dotted path.

    Three checks run in order:

    1. ``path`` must start with one of the prefixes returned by
       :func:`akgentic.catalog.allowlist.allowed_prefixes`.
    2. The resolved object must be a subclass of ``pydantic.BaseModel``.
    3. The resolved class must not declare Pydantic fields named ``__ref__``
       or ``__type__``.

    Checks 2 and 3 run for every path that passes check 1 — widening the
    prefix policy widens what may be named, never what may be resolved.

    Args:
        path: Dotted class path (e.g. ``"akgentic.core.agent_card.AgentCard"``).

    Returns:
        The imported class.

    Raises:
        CatalogValidationError: If any of the three checks fails. The error
            carries a single-element ``errors`` list with a substring-stable
            message (``"outside allowlist"``, ``"is not a Pydantic BaseModel
            subclass"``, or ``"reserved ref-sentinel fields"``) so callers
            can assert on behaviour without loading the exception chain.
        ValueError: If the prefix policy itself is misconfigured — a malformed
            ``AKGENTIC_CATALOG_MODEL_TYPE_PREFIXES`` surfaces here, on the
            first read, carrying ``"invalid model_type prefix"``. Deliberately
            **not** wrapped in ``CatalogValidationError``: an operator typo in
            deployment configuration is not an invalid entry, and folding it
            into the per-entry error type would let a broken policy read as a
            catalog full of bad ``model_type`` values.
    """
    violation = prefix_violation(path)
    if violation is not None:
        raise CatalogValidationError([violation])

    cls = import_class(path)

    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise CatalogValidationError([f"model_type '{path}' is not a Pydantic BaseModel subclass"])

    collisions = sorted(RESERVED_REF_KEYS & set(cls.model_fields.keys()))
    if collisions:
        raise CatalogValidationError(
            [f"model_type '{path}' declares reserved ref-sentinel fields: {collisions}"]
        )

    return cls


def _matches_policy(dotted_name: str, prefixes: tuple[str, ...]) -> bool:
    """Return whether ``dotted_name`` sits under any allowed prefix.

    The exact-match arm is load-bearing, not cosmetic: a deployment whose
    classes live in the module ``acme.core.models`` and that configures the
    prefix ``acme.core.models.`` would otherwise have that very module
    skipped by the walk, because ``"acme.core.models".startswith(
    "acme.core.models.")`` is ``False``. Shared by the module walk and the
    class-path filter so the two cannot drift.
    """
    return any(
        dotted_name.startswith(prefix) or dotted_name == prefix.removesuffix(".")
        for prefix in prefixes
    )


def enumerate_allowlisted_model_types() -> list[str]:
    """Enumerate allowlisted ``BaseModel`` subclasses already loaded in-process.

    Walks a snapshot of ``sys.modules`` keeping every module whose name sits
    under one of the prefixes returned by
    :func:`akgentic.catalog.allowlist.allowed_prefixes` — ``akgentic.*`` by
    default, plus whatever the deployment configured. Nothing is imported:
    enumeration reports what is already in ``sys.modules`` and nothing else, so
    widening the prefix policy never triggers a module import.

    The snapshot avoids mutation-during-iteration issues. Per-module
    introspection errors are swallowed — optional dependencies may be absent or
    partially imported. ``load_model_type`` acts as the authoritative allowlist
    + ``BaseModel`` + reserved-key gate so enumeration never broadens the
    allowlist.

    Used by both the REST router (``GET /catalog/model_types``) and the
    ``ak-catalog model-types`` CLI verb.

    Returns:
        Sorted dotted class paths, deduplicated.

    Raises:
        ValueError: If the prefix policy is misconfigured. The
            :func:`akgentic.catalog.allowlist.allowed_prefixes` read happens
            before the walk and is deliberately not guarded by the per-module
            ``except`` below — a malformed
            ``AKGENTIC_CATALOG_MODEL_TYPE_PREFIXES`` is an operator error that
            must be loud (the REST route surfaces it as a 500), not a silently
            empty model-type picker.
    """
    prefixes = allowed_prefixes()
    results: set[str] = set()
    modules_snapshot = list(sys.modules.items())
    for module_name, module in modules_snapshot:
        if module is None or not _matches_policy(module_name, prefixes):
            continue
        _collect_allowlisted(module, results, prefixes)
    return sorted(results)


def _collect_allowlisted(module: Any, results: set[str], prefixes: tuple[str, ...]) -> None:
    """Add every allowlisted ``BaseModel`` subclass from ``module`` into ``results``.

    ``_matches_policy`` is a cheap pre-filter here, not the gate: its exact-match
    arm is a rule about *module* names, so against a *class* path it can admit
    one that ``load_model_type`` then rejects (that call below is authoritative
    and matches on ``startswith`` alone). Keep the ``load_model_type`` call.
    """
    try:
        items = list(vars(module).items())
    except Exception:  # noqa: BLE001 — defensive; partially imported modules
        return
    for _name, value in items:
        if not isinstance(value, type) or not issubclass(value, BaseModel):
            continue
        path = f"{value.__module__}.{value.__name__}"
        if not _matches_policy(path, prefixes) or path in results:
            continue
        try:
            load_model_type(path)
        except Exception:  # noqa: BLE001 — swallow reserved-key or import errors
            continue
        results.add(path)
