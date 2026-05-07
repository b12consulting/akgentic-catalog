"""Unified v2 ``Entry`` model, ``EntryKind`` literal, and allowlisted-path type.

This module is the documented import site for the catalog v2 entry type system.
It lands alongside the v1 per-kind entry models (``TemplateEntry`` etc.) without
replacing them — v1 removal is deferred to Epic 19.

Key exports:

* ``Entry`` — the single unified catalog entry model used by v2 code.
* ``EntryKind`` — ``Literal`` alias of the five supported entry kinds.
* ``AllowlistedPath`` — ``Annotated[str, AfterValidator(...)]`` enforcing the
  ``akgentic.`` prefix on class-path fields at Pydantic validation time.
* ``NonEmptyStr`` — re-exported from ``._types`` so consumers have a single
  import site for v2.

The allowlist check implemented here is the storage-side defence (catches bad
``Entry.model_type`` at construction time, before any import). The runtime
counterpart in ``akgentic.catalog.resolver.load_model_type`` repeats the prefix
check and adds the ``BaseModel`` / reserved-key checks that require the class
object in hand. The duplication is intentional (see Story 15.1 Dev Notes).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, Field, model_validator

from ._types import NonEmptyStr

__all__ = ["AllowlistedPath", "Entry", "EntryKind", "NonEmptyStr"]


EntryKind = Literal["team", "agent", "tool", "model", "prompt"]
"""The five entry kinds a v2 catalog stores.

``team``/``agent``/``tool`` mirror v1 semantics. ``model`` and ``prompt`` are
new v2 kinds — model-type configs and reusable prompt templates promoted to
first-class entries so they can be referenced via the ref-sentinel mechanism
(see ``akgentic.catalog.resolver``).
"""


# Storage-side allowlist for class paths stored in ``Entry.model_type``.
# Duplicated intentionally in ``resolver.py`` — two layers, two policies that
# only happen to agree today. See Story 15.1 Dev Notes.
_ALLOWED_PREFIXES: tuple[str, ...] = ("akgentic.",)


def _check_allowlist(v: str) -> str:
    """Reject class paths that do not start with an allowlisted prefix.

    Args:
        v: Candidate dotted class path (e.g. ``"akgentic.llm.ModelConfig"``).

    Returns:
        The input string unchanged when it passes the prefix check.

    Raises:
        ValueError: When ``v`` does not start with any allowlisted prefix.
            The error message is ``"... outside allowlist ..."`` to satisfy
            the substring assertions pinned by Story 15.1 acceptance criteria.
    """
    if not any(v.startswith(prefix) for prefix in _ALLOWED_PREFIXES):
        raise ValueError(f"model_type '{v}' outside allowlist {_ALLOWED_PREFIXES}")
    return v


AllowlistedPath = Annotated[str, AfterValidator(_check_allowlist)]
"""Dotted class path constrained to the ``akgentic.`` namespace.

Used by ``Entry.model_type``. The check fires at Pydantic validation time —
construction of an ``Entry`` with a non-allowlisted path raises
``pydantic.ValidationError`` before any import side effect runs.
"""


class Entry(BaseModel):
    """Unified catalog entry (v2).

    Replaces the four v1 per-kind entry models (``TemplateEntry``,
    ``ToolEntry``, ``AgentEntry``, ``TeamEntry``) with a single shape whose
    ``kind`` discriminator selects the semantics and whose ``payload`` carries
    the kind-specific configuration (validated against the Pydantic class named
    by ``model_type`` at resolve time — see ``akgentic.catalog.resolver``).

    Lineage fields (``parent_namespace`` / ``parent_id``) support the three
    lineage cases called out in ADR-07:

    * both ``None`` — fresh entry minted in this namespace.
    * ``parent_namespace=None`` + ``parent_id=<str>`` — same-namespace
      duplicate (same catalog, new id, e.g. a user-owned edit of a global
      entry that was cloned without changing namespace).
    * both set — cross-namespace clone.

    The validator ``_check_parent_pair`` rejects the fourth combination
    (``parent_namespace`` set, ``parent_id is None``) because a namespace
    without an id does not identify a parent entry.
    """

    id: NonEmptyStr = Field(description="Entry id, unique within (namespace, kind).")
    kind: EntryKind = Field(description="Discriminator selecting entry semantics.")
    namespace: NonEmptyStr = Field(
        description="Namespace this entry belongs to; scopes id uniqueness."
    )
    user_id: NonEmptyStr | None = Field(
        default=None,
        description=("Owner user id for user-scoped entries; None for global/enterprise entries."),
    )
    parent_namespace: NonEmptyStr | None = Field(
        default=None,
        description=(
            "Lineage pointer: namespace of the parent entry. Set together with "
            "parent_id for cross-namespace clones; leave None for same-namespace "
            "duplicates or fresh entries."
        ),
    )
    parent_id: NonEmptyStr | None = Field(
        default=None,
        description=(
            "Lineage pointer: id of the parent entry. Set for clones and "
            "same-namespace duplicates; None for fresh entries."
        ),
    )
    model_type: AllowlistedPath = Field(
        description=(
            "Dotted path to the Pydantic BaseModel class that validates payload. "
            "Restricted to the akgentic.* namespace at the annotation layer; "
            "resolver adds BaseModel and reserved-key checks at resolve time."
        ),
    )
    description: str = Field(
        default="",
        description="Human-readable description; may be empty.",
    )
    payload: dict[str, Any] = Field(
        description=(
            "Kind-specific configuration; validated against the class named by "
            "model_type during resolve. May contain ref sentinels populated by "
            "the resolver."
        ),
    )

    @model_validator(mode="after")
    def _check_parent_pair(self) -> Entry:
        """Reject lineage pairs with ``parent_namespace`` set and ``parent_id`` missing.

        Three lineage combinations are valid (see class docstring). The rejected
        case — ``parent_namespace`` set without ``parent_id`` — is rejected
        because a namespace alone cannot identify a parent entry.
        """
        if self.parent_namespace is not None and self.parent_id is None:
            raise ValueError("parent_namespace set but parent_id is None")
        return self
