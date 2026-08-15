"""Unified v2 ``Entry`` model, ``EntryKind`` literal, and allowlisted-path type.

This module is the documented import site for the catalog v2 entry type system.
It is the single per-kind-entry shape: the previous per-kind Pydantic models
were removed during Epic 19.

Key exports:

* ``Entry`` — the single unified catalog entry model used by v2 code.
* ``EntryKind`` — ``Literal`` alias of the five supported entry kinds.
* ``AllowlistedPath`` — ``Annotated[str, AfterValidator(...)]`` enforcing the
  configured class-path prefix policy at Pydantic validation time.
* ``NonEmptyStr`` — re-exported from ``._types`` so consumers have a single
  import site for v2.

There is one prefix policy, owned by ``akgentic.catalog.allowlist``, and two
enforcement points consult it. The check implemented here is the storage-side
one (catches a bad ``Entry.model_type`` at construction time, before any
import). Its runtime counterpart in
``akgentic.catalog.resolver.load_model_type`` repeats the prefix check and adds
the ``BaseModel`` / reserved-key checks that require the class object in hand.
Because both read the same policy they cannot drift apart.
"""

from __future__ import annotations

from typing import Annotated, Any, Final, Literal

from pydantic import AfterValidator, BaseModel, Field, field_validator

from ..allowlist import prefix_violation
from ._types import NonEmptyStr

__all__ = ["ANONYMOUS_USER_ID", "AllowlistedPath", "Entry", "EntryKind", "NonEmptyStr"]

ANONYMOUS_USER_ID: Final[str] = "anonymous"
"""Community-tier owner convention: the ``user_id`` of an unowned entry."""


EntryKind = Literal["team", "agent", "tool", "model", "prompt", "meta"]
"""The six entry kinds a v2 catalog stores.

``team``/``agent``/``tool`` mirror v1 semantics. ``model`` and ``prompt`` are
v2 kinds — model-type configs and reusable prompt templates promoted to
first-class entries so they can be referenced via the ref-sentinel mechanism
(see ``akgentic.catalog.resolver``). ``meta`` is the per-namespace metadata
kind introduced by ADR-008 §D1: a namespace-scoped, free-form metadata
payload addressed by the convention id ``"_meta"`` and constrained to a
singleton (zero or one ``kind=meta`` entry per namespace).
"""


def _check_allowlist(v: str) -> str:
    """Reject class paths that do not start with an allowlisted prefix.

    The predicate and the message live in
    ``akgentic.catalog.allowlist.prefix_violation``, shared with the runtime check
    in ``resolver.load_model_type``; only the exception type differs.

    Args:
        v: Candidate dotted class path (e.g. ``"akgentic.llm.ModelConfig"``).

    Returns:
        The input string unchanged when it passes the prefix check.

    Raises:
        ValueError: When ``v`` does not start with any allowlisted prefix.
            The error message is ``"... outside allowlist ..."`` to satisfy
            the substring assertions pinned by Story 15.1 acceptance criteria.
    """
    violation = prefix_violation(v)
    if violation is not None:
        raise ValueError(violation)
    return v


AllowlistedPath = Annotated[str, AfterValidator(_check_allowlist)]
"""Dotted class path constrained to the configured prefix allowlist.

Used by ``Entry.model_type``. ``akgentic.`` is always allowed; a deployment may
widen the set (see ``akgentic.catalog.allowlist``). The check fires at Pydantic
validation time — construction of an ``Entry`` with a non-allowlisted path
raises ``pydantic.ValidationError`` before any import side effect runs.
"""


class Entry(BaseModel):
    """Unified catalog entry (v2).

    A single shape whose ``kind`` discriminator selects the semantics and
    whose ``payload`` carries the kind-specific configuration (validated
    against the Pydantic class named by ``model_type`` at resolve time — see
    ``akgentic.catalog.resolver``).
    """

    id: NonEmptyStr = Field(description="Entry id, unique within (namespace, kind).")
    kind: EntryKind = Field(description="Discriminator selecting entry semantics.")
    namespace: NonEmptyStr = Field(
        description="Namespace this entry belongs to; scopes id uniqueness."
    )
    user_id: NonEmptyStr = Field(
        default=ANONYMOUS_USER_ID,
        description=(
            "Owner user identifier; defaults to 'anonymous' on community tier. "
            "Always a real string — never null, never empty. The literal "
            "'anonymous' is the community-tier convention (every entry is "
            "implicitly trusted), not a sentinel: department / enterprise "
            "deployments stamp the authenticated caller's user id instead."
        ),
    )
    model_type: AllowlistedPath = Field(
        description=(
            "Dotted path to the Pydantic BaseModel class that validates payload. "
            "Restricted at the annotation layer to the configured prefix "
            "allowlist — akgentic. is always allowed and a deployment may widen "
            "the set; resolver adds BaseModel and reserved-key checks at "
            "resolve time."
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

    @field_validator("namespace")
    @classmethod
    def _forbid_dot_in_namespace(cls, v: str) -> str:
        """Reject namespace values containing ``.`` (ADR-008 §D2).

        Dots are reserved for the cross-namespace ref shorthand
        ``{"__ref__": "<ns>.<id>"}`` parsed at the resolver layer; allowing a
        ``.`` inside a namespace identifier would make the shorthand
        ambiguous (every dot-positioned split would be a candidate). Banning
        the character at the Pydantic field-validator (Layer 1) means the
        invariant fires before any service or repository code sees the value.
        """
        if "." in v:
            raise ValueError(
                f"Entry.namespace must not contain '.': '{v}' — "
                f"dots are reserved for cross-namespace ref shorthand"
            )
        return v
