"""Query models for catalog repository search operations.

Holds both the v1 per-kind query models (``TemplateQuery``, ``ToolQuery``,
``AgentQuery``, ``TeamQuery``) and the v2 ``EntryQuery`` + ``CloneRequest``
models. v1 models are preserved unchanged; v2 additions are additive only.
v1 removal is deferred to Epic 19.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .entry import EntryKind, NonEmptyStr

__all__ = [
    "AgentQuery",
    "CloneRequest",
    "EntryQuery",
    "TeamQuery",
    "TemplateQuery",
    "ToolQuery",
]


class TemplateQuery(BaseModel):
    """Query model for filtering template entries."""

    model_config = ConfigDict(frozen=True)

    id: str | None = Field(default=None, description="Filter by exact template id")
    placeholder: str | None = Field(
        default=None, description="Filter by placeholder name present in template"
    )


class ToolQuery(BaseModel):
    """Query model for filtering tool entries."""

    model_config = ConfigDict(frozen=True)

    id: str | None = Field(default=None, description="Filter by exact tool id")
    tool_class: str | None = Field(default=None, description="Filter by exact tool class path")
    name: str | None = Field(default=None, description="Filter by tool name substring")
    description: str | None = Field(
        default=None, description="Filter by tool description substring"
    )


class AgentQuery(BaseModel):
    """Query model for filtering agent entries."""

    model_config = ConfigDict(frozen=True)

    id: str | None = Field(default=None, description="Filter by exact agent id")
    role: str | None = Field(default=None, description="Filter by exact agent role")
    skills: list[str] | None = Field(
        default=None, description="Filter by overlap with agent skill set"
    )
    description: str | None = Field(
        default=None, description="Filter by agent description substring"
    )


class TeamQuery(BaseModel):
    """Query model for filtering team entries."""

    model_config = ConfigDict(frozen=True)

    id: str | None = Field(default=None, description="Filter by exact team id")
    name: str | None = Field(default=None, description="Filter by team name substring")
    description: str | None = Field(
        default=None, description="Filter by team description substring"
    )
    agent_id: str | None = Field(
        default=None, description="Filter by agent id present in team members"
    )


class EntryQuery(BaseModel):
    """Query model for filtering v2 unified ``Entry`` rows.

    All fields are optional and default to ``None``. Fields combine with AND
    semantics — a ``None`` field means "don't filter by this attribute".

    ``user_id_set`` is a tri-state filter layered on top of ``user_id``:

    * ``None`` — ignore user_id scoping entirely.
    * ``True`` — include only entries with any non-``None`` ``user_id``
      (i.e. user-scoped entries).
    * ``False`` — include only entries with ``user_id is None``
      (i.e. global/enterprise entries).

    This is orthogonal to ``user_id``: passing ``user_id="alice"`` narrows to
    exactly that user, whereas ``user_id_set=True`` narrows to "any user".
    """

    model_config = ConfigDict(frozen=True)

    namespace: NonEmptyStr | None = Field(default=None, description="Filter by exact namespace.")
    kind: EntryKind | None = Field(
        default=None, description="Filter by entry kind (team/agent/tool/model/prompt)."
    )
    id: NonEmptyStr | None = Field(default=None, description="Filter by exact entry id.")
    user_id: NonEmptyStr | None = Field(
        default=None,
        description="Filter by exact user_id; use user_id_set for presence-only filtering.",
    )
    user_id_set: bool | None = Field(
        default=None,
        description=(
            "Tri-state user_id scope filter: None=any, True=user_id set, False=user_id is None."
        ),
    )
    parent_namespace: NonEmptyStr | None = Field(
        default=None, description="Filter by parent_namespace lineage pointer."
    )
    parent_id: NonEmptyStr | None = Field(
        default=None, description="Filter by parent_id lineage pointer."
    )
    description_contains: str | None = Field(
        default=None, description="Filter by substring match in entry description."
    )


class CloneRequest(BaseModel):
    """Request model for cloning an entry from one namespace to another.

    ``dst_namespace`` is required — cloning into an unnamed target is not
    meaningful. Creating a namespace from scratch is a ``Catalog.create``
    concern, not a clone concern.
    """

    model_config = ConfigDict(frozen=True)

    src_namespace: NonEmptyStr = Field(
        description="Source namespace containing the entry to clone."
    )
    src_id: NonEmptyStr = Field(description="Id of the source entry to clone.")
    dst_namespace: NonEmptyStr = Field(description="Destination namespace for the clone.")
    dst_user_id: NonEmptyStr | None = Field(
        default=None,
        description=("Destination user_id; None targets a global/enterprise clone."),
    )
