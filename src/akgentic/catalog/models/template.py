"""TemplateEntry model for prompt template catalog entries."""

from string import Formatter
from typing import Annotated

from pydantic import BaseModel, StringConstraints, computed_field

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class TemplateEntry(BaseModel):
    """A prompt template catalog entry with unique id and placeholder parsing."""

    id: NonEmptyStr
    template: NonEmptyStr

    @computed_field  # type: ignore[prop-decorator]
    @property
    def placeholders(self) -> list[str]:
        """Parse {placeholder} names from template string, sorted and deduplicated."""
        return sorted({name for _, name, _, _ in Formatter().parse(self.template) if name})


__all__ = ["TemplateEntry"]
