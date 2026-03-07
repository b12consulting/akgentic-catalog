"""Catalog @-reference utilities.

Handles the catalog reference convention where ``@entry-id`` in
``PromptTemplate.template`` fields points to another catalog entry.

IMPORTANT: Two ``@`` conventions coexist — ``config.name`` uses ``@`` for
agent routing names; ``prompt.template`` uses ``@`` for catalog references.
These functions must ONLY be applied to ``PromptTemplate.template`` fields,
never to ``config.name`` or ``routes_to`` values.
"""

__all__ = [
    "_is_catalog_ref",
    "_resolve_ref",
]


def _is_catalog_ref(value: str) -> bool:
    """Check if a string is a catalog @-reference.

    ONLY call this on PromptTemplate.template fields.
    Never on config.name or routes_to values.
    """
    return value.startswith("@")


def _resolve_ref(value: str) -> str:
    """Strip @ prefix to get the catalog entry id."""
    return value[1:]
