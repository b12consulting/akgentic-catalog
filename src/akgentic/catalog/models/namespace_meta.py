"""Namespace-metadata payload model — backs ``EntryKind="meta"`` entries.

This module exposes a single Pydantic ``BaseModel`` (:class:`NamespaceMeta`)
used as the payload-validation class for the ``kind="meta"`` catalog entry
introduced by ADR-008 §D1. The model carries five pinned fields — ``name``,
``description``, a free-form ``properties`` map, a typed ``shareable``
boolean, and a typed ``public`` boolean — so the namespace picker reads
tenant-level metadata from a stable, purpose-built shape instead of leaking
``TeamCard``'s schema.

Key exports:

* :class:`NamespaceMeta` — the payload model. Plain ``BaseModel`` (not
  ``SerializableBaseModel``) — catalog payload models are not actor messages.
* :data:`_NAMESPACE_META_TYPE` — the model's dotted path, written once here
  and imported by every site that stamps a ``_meta`` entry's ``model_type``.

The model is reachable via that ``model_type`` through the allowlist used by
:func:`akgentic.catalog.model_types.load_model_type`; the existing
``akgentic.*`` prefix already covers it.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, Field

from ._types import NonEmptyStr

__all__ = ["_NAMESPACE_META_TYPE", "NamespaceMeta"]

_NAMESPACE_META_TYPE: Final[str] = "akgentic.catalog.models.namespace_meta.NamespaceMeta"
"""The dotted path every ``_meta`` entry carries as its ``model_type``.

Written as a literal rather than computed from the class so that moving or
renaming the model breaks a test loudly instead of silently re-pointing every
stored entry at a path the allowlist resolver cannot load. It lives beside the
model it names so there is exactly one copy of the string in the package.
"""


class NamespaceMeta(BaseModel):
    """Payload model for the per-namespace metadata entry (``kind="meta"``).

    Five pinned fields, in declaration order:

    * ``name`` — required non-empty display name for the namespace; surfaced
      by ``GET /catalog/namespaces`` to feed the picker.
    * ``description`` — optional human-readable description; defaults to
      empty string.
    * ``properties`` — free-form ``str -> str`` annotations carrying
      tenant-level metadata (display tier, owner team, custom labels) that
      should NOT pollute :class:`~akgentic.team.models.TeamCard`. The map is
      fully free-form: there are NO catalog-reserved keys. Both keys and
      values are strings.
    * ``shareable`` — typed boolean controlling cross-namespace
      referenceability (ADR-008 §D2 as updated 2026-05-08, rev 2). When
      ``True``, the namespace is cross-namespace-referenceable as a target —
      other namespaces may carry refs into this one (subject to the existing
      ``user_id == "anonymous"`` ownership gate). When ``False`` (the default), the
      namespace is not shareable. Pydantic strict-mode rejects non-bool
      inputs so operators must opt in unambiguously with a real boolean.
    * ``public`` — typed boolean controlling namespace visibility (ADR-009
      §D2). When ``True``, non-owner users may list, read, and clone entries
      in this namespace; when ``False`` (the default), the namespace is
      tenant-private. Orthogonal to ``shareable``: their cartesian product
      yields the four named visibility states (tenant-private, forkable
      library, private shared backbone, public library). Strict-bool —
      non-bool inputs (e.g. the string ``"true"``) are rejected by Pydantic
      at construction time. Visibility filtering on ``Catalog.list`` /
      ``get`` / ``search`` / ``clone`` lands in Story 18.4; in 18.2 the
      flag is purely declarative.

    Convention id is ``"_meta"``; the route fallback in
    ``GET /catalog/namespaces`` reads ``payload["name"]`` /
    ``description`` from the meta entry and falls back to the team entry
    otherwise (so existing deployments continue to work without explicit
    backfill — see ADR-008 §D1).

    The model is plain ``BaseModel`` per architecture §D6 — catalog payload
    models are validated configuration, not actor messages, so the
    ``SerializableBaseModel`` machinery is intentionally NOT inherited.
    """

    name: NonEmptyStr = Field(
        description=(
            "Display name for the namespace; surfaced by the namespace picker "
            "(GET /catalog/namespaces)."
        ),
    )
    description: str = Field(
        default="",
        description=("Human-readable description; surfaced by the namespace picker; may be empty."),
    )
    properties: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Free-form string-to-string annotations carrying tenant-level "
            "metadata (display tier, owner team, custom labels). Both keys "
            "and values are strings. There are NO catalog-reserved keys — "
            "the map is fully operator-owned."
        ),
    )
    shareable: bool = Field(
        default=False,
        description=(
            "When True, the namespace is cross-namespace-referenceable as a "
            "target — other namespaces may carry refs into this one (subject "
            "to the existing ``user_id == 'anonymous'`` ownership gate). Strict-bool "
            '— non-bool inputs (e.g. the string ``"true"``) are rejected by '
            "Pydantic at construction time. ADR-008 §D2 as updated "
            "2026-05-08 (rev 2)."
        ),
    )
    public: bool = Field(
        default=False,
        description=(
            "When True, non-owner users may list, read, and clone entries in "
            "this namespace; when False (the default), the namespace is "
            "tenant-private. Orthogonal to ``shareable`` (which controls "
            "cross-namespace reference eligibility). Strict-bool — non-bool "
            'inputs (e.g. the string ``"true"``) are rejected by Pydantic at '
            "construction time. Visibility filtering at the Catalog "
            "list/get/search/clone boundary is introduced by Story 18.4; "
            "in Story 18.2 the flag is purely declarative. ADR-009 §D2."
        ),
    )
