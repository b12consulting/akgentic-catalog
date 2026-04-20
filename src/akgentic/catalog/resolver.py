"""Ref-sentinel constants and the allowlist-gated Pydantic-class loader.

This module owns the v2 ref-sentinel sentinel keys (``REF_KEY``, ``TYPE_KEY``)
and the ``load_model_type(path)`` function that imports a Pydantic
``BaseModel`` class by dotted path, gated behind three defensive checks:

1. The path must start with one of ``_ALLOWED_PREFIXES`` (storage + runtime
   defence in depth — ``models.entry.AllowlistedPath`` enforces the same
   prefix at Pydantic construction time).
2. The resolved class must be a subclass of ``pydantic.BaseModel``.
3. The resolved class must not declare Pydantic fields named ``__ref__`` or
   ``__type__`` (the reserved keys used by the resolver's sentinel scheme —
   collisions would break round-tripping).

``populate_refs``, ``reconcile_refs``, ``prepare_for_write``,
``validate_delete``, and ``resolve`` are deliberately NOT defined here — they
land in Story 15.2.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel

from akgentic.core.utils.deserializer import import_class

from .models.errors import CatalogValidationError

__all__ = ["REF_KEY", "TYPE_KEY", "load_model_type"]


REF_KEY: Final[str] = "__ref__"
"""Sentinel dict key marking a ref placeholder inside a resolved payload.

A payload dict containing ``REF_KEY`` has been populated by the resolver and
must be hydrated (looked up in the repository) before use at runtime.
"""

TYPE_KEY: Final[str] = "__type__"
"""Sentinel dict key carrying the FQCN of a referenced entry's model type.

Emitted next to ``REF_KEY`` so the resolver can validate the target's type
without loading the target entry eagerly.
"""

# Runtime allowlist for ``load_model_type``. Duplicated intentionally in
# ``models.entry`` for the annotation-layer defence — two layers, two
# policies that only happen to agree today. See Story 15.1 Dev Notes.
_ALLOWED_PREFIXES: tuple[str, ...] = ("akgentic.",)

_RESERVED_KEYS: frozenset[str] = frozenset({REF_KEY, TYPE_KEY})


def load_model_type(path: str) -> type[BaseModel]:
    """Import and return a Pydantic ``BaseModel`` class by dotted path.

    Three checks run in order:

    1. ``path`` must start with one of ``_ALLOWED_PREFIXES``.
    2. The resolved object must be a subclass of ``pydantic.BaseModel``.
    3. The resolved class must not declare Pydantic fields named ``__ref__``
       or ``__type__``.

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
    """
    if not any(path.startswith(prefix) for prefix in _ALLOWED_PREFIXES):
        raise CatalogValidationError([f"model_type '{path}' outside allowlist {_ALLOWED_PREFIXES}"])

    cls = import_class(path)

    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise CatalogValidationError([f"model_type '{path}' is not a Pydantic BaseModel subclass"])

    collisions = sorted(_RESERVED_KEYS & set(cls.model_fields.keys()))
    if collisions:
        raise CatalogValidationError(
            [f"model_type '{path}' declares reserved ref-sentinel fields: {collisions}"]
        )

    return cls
