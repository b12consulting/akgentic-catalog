"""ToolEntry model for tool configuration catalog entries."""

from typing import Annotated, Any

from pydantic import BaseModel, StringConstraints, model_validator

from akgentic.core.utils.deserializer import import_class
from akgentic.tool import ToolCard

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class ToolEntry(BaseModel):
    """A tool configuration catalog entry with dynamic class resolution."""

    id: NonEmptyStr
    tool_class: NonEmptyStr
    tool: ToolCard

    @model_validator(mode="before")
    @classmethod
    def resolve_tool(cls, data: Any) -> Any:  # noqa: ANN401
        """Resolve tool_class to concrete class and validate tool config."""
        if not isinstance(data, dict):
            return data
        if isinstance(data.get("tool"), dict):
            tool_class_path = data.get("tool_class")
            if not isinstance(tool_class_path, str):
                return data
            try:
                klass = import_class(tool_class_path)
            except (ImportError, AttributeError) as e:
                raise ValueError(
                    f"Cannot resolve tool_class '{tool_class_path}': {e}"
                ) from e
            data["tool"] = klass.model_validate(data["tool"])
        return data


__all__ = ["ToolEntry"]
