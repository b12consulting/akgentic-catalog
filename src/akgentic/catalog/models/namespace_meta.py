"""Namespace-metadata payload model — backs ``EntryKind="meta"`` entries.

This module exposes a single Pydantic ``BaseModel`` (:class:`NamespaceMeta`)
used as the payload-validation class for the ``kind="meta"`` catalog entry
introduced by ADR-008 §D1. The model intentionally carries only three pinned
fields — ``name``, ``description``, and a free-form ``properties`` map — so
the namespace picker reads tenant-level metadata from a stable, purpose-built
shape instead of leaking ``TeamCard``'s schema.

Key exports:

* :class:`NamespaceMeta` — the payload model. Plain ``BaseModel`` (not
  ``SerializableBaseModel``) — catalog payload models are not actor messages.

The model is reachable via the ``model_type=
"akgentic.catalog.models.namespace_meta.NamespaceMeta"`` allowlist used by
:func:`akgentic.catalog.resolver.load_model_type`. No resolver / allowlist
change is required by this story; the existing ``akgentic.*`` prefix already
covers it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ._types import NonEmptyStr

__all__ = ["NamespaceMeta"]


class NamespaceMeta(BaseModel):
    """Payload model for the per-namespace metadata entry (``kind="meta"``).

    Three pinned fields:

    * ``name`` — required non-empty display name for the namespace; surfaced
      by ``GET /catalog/namespaces`` to feed the picker.
    * ``description`` — optional human-readable description; defaults to
      empty string.
    * ``properties`` — free-form ``str -> str`` annotations carrying
      tenant-level metadata (display tier, owner team, custom labels) that
      should NOT pollute :class:`~akgentic.team.models.TeamCard`.

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
            "and values are strings."
        ),
    )
