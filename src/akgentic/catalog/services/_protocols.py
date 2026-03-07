"""Shared protocols for catalog service duck-typing (avoids circular imports)."""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from akgentic.catalog.models.agent import AgentEntry

_list = builtins.list


class _AgentRepoProtocol(Protocol):
    """Protocol for agent repository list access."""

    def list(self) -> _list[AgentEntry]: ...


class _AgentCatalogProtocol(Protocol):
    """Protocol for agent catalog lookup (duck-typed, avoids circular import)."""

    @property
    def repository(self) -> _AgentRepoProtocol: ...
