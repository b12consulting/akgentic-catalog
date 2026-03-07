"""Catalog error models.

Provides domain-specific exceptions for catalog operations, distinct from
Pydantic's ValidationError (which covers object-level validation).
"""

__all__ = [
    "CatalogValidationError",
    "EntryNotFoundError",
]


class CatalogValidationError(Exception):
    """Raised when catalog business rules are violated.

    Distinct from Pydantic ValidationError — object validation errors are
    Pydantic, repository/service validation errors are CatalogValidationError.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Catalog validation failed: {'; '.join(errors)}")


class EntryNotFoundError(Exception):
    """Raised when a catalog entry lookup by id returns no result."""
