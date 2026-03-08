"""TemplateEntry model for prompt template catalog entries."""

from string import Formatter

from pydantic import BaseModel, Field, computed_field

from akgentic.catalog.models._types import NonEmptyStr


class TemplateEntry(BaseModel):
    """A prompt template catalog entry with unique id and placeholder parsing."""

    id: NonEmptyStr = Field(description="Unique catalog identifier for this template")
    template: NonEmptyStr = Field(description="Prompt template string with {placeholder} syntax")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def placeholders(self) -> list[str]:
        """Parse {placeholder} names from template string, sorted and deduplicated."""
        return sorted({name for _, name, _, _ in Formatter().parse(self.template) if name})


__all__ = ["TemplateEntry"]
