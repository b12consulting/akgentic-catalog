"""Query models for catalog repository search operations."""

from pydantic import BaseModel, ConfigDict

__all__ = [
    "AgentQuery",
    "TeamQuery",
    "TemplateQuery",
    "ToolQuery",
]


class TemplateQuery(BaseModel):
    """Query model for filtering template entries."""

    model_config = ConfigDict(frozen=True)

    id: str | None = None
    placeholder: str | None = None


class ToolQuery(BaseModel):
    """Query model for filtering tool entries."""

    model_config = ConfigDict(frozen=True)

    id: str | None = None
    tool_class: str | None = None
    name: str | None = None
    description: str | None = None


class AgentQuery(BaseModel):
    """Query model for filtering agent entries."""

    model_config = ConfigDict(frozen=True)

    id: str | None = None
    role: str | None = None
    skills: list[str] | None = None
    description: str | None = None


class TeamQuery(BaseModel):
    """Query model for filtering team specs."""

    model_config = ConfigDict(frozen=True)

    id: str | None = None
    name: str | None = None
    description: str | None = None
    agent_id: str | None = None
