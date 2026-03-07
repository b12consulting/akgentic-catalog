"""Shared type aliases for catalog models."""

from typing import Annotated

from pydantic import StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]

__all__ = ["NonEmptyStr"]
