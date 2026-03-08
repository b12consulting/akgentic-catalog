"""ToolEntry model for tool configuration catalog entries."""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from akgentic.catalog.models._types import NonEmptyStr
from akgentic.core.utils.deserializer import import_class
from akgentic.tool import ToolCard


class ToolEntry(BaseModel):
    """A tool configuration catalog entry with dynamic class resolution."""

    id: NonEmptyStr = Field(description="Unique catalog identifier for this tool")
    tool_class: NonEmptyStr = Field(
        description="Fully qualified Python class path for the ToolCard subclass"
    )
    tool: ToolCard = Field(
        description="Tool configuration validated against the resolved tool_class"
    )

    # Resolves tool_class path to concrete ToolCard subclass and validates
    # the tool dict against its Pydantic schema — enables deserialization
    # of abstract ToolCard from YAML/MongoDB without knowing the concrete type.
    @model_validator(mode="before")
    @classmethod
    def resolve_tool(cls, data: Any) -> Any:  # noqa: ANN401
        """Resolve tool_class to concrete class and validate tool config.

        Args:
            data: Raw input data (typically a dict from YAML deserialization).

        Returns:
            The data dict with ``tool`` replaced by a validated concrete
            ``ToolCard`` instance.

        Raises:
            ValueError: If ``tool_class`` cannot be imported.
        """
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
