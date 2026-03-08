"""Agent catalog entry with dynamic config resolution."""

from typing import Any, Protocol, get_args

from pydantic import BaseModel, Field, model_validator

from akgentic.agent.config import AgentConfig
from akgentic.catalog.models._types import NonEmptyStr
from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.catalog.refs import _is_catalog_ref, _resolve_ref
from akgentic.core.agent import Akgent
from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_config import BaseConfig
from akgentic.core.utils.deserializer import import_class
from akgentic.llm.prompts import PromptTemplate
from akgentic.tool import ToolCard

__all__ = [
    "AgentEntry",
    "_extract_config_type",
]


def _extract_config_type(agent_cls: type) -> type[BaseConfig]:
    """Extract ConfigType from an Akgent subclass's generic parameters.

    Walks the MRO and each class's ``__orig_bases__`` to find the
    ``Akgent[ConfigType, StateType]`` generic — handles intermediate base
    classes that don't re-declare generics.

    Raises:
        ValueError: If the agent class does not parameterize
            ``Akgent[ConfigType, StateType]``.
    """
    for cls_in_mro in agent_cls.__mro__:
        for base in getattr(cls_in_mro, "__orig_bases__", ()):
            origin = getattr(base, "__origin__", None)
            if origin is not None and issubclass(origin, Akgent):
                args = get_args(base)
                if args:
                    config_type: type[BaseConfig] = args[0]
                    return config_type
    raise ValueError(
        f"{agent_cls.__name__} does not parameterize Akgent[ConfigType, StateType] "
        f"— cannot resolve config type"
    )


class _ToolCatalogProtocol(Protocol):
    """Protocol for tool catalog lookup (avoids circular import)."""

    def get(self, tool_id: str) -> Any:  # noqa: ANN401
        """Return ToolEntry or None."""
        ...


class _TemplateCatalogProtocol(Protocol):
    """Protocol for template catalog lookup (avoids circular import)."""

    def get(self, template_id: str) -> Any:  # noqa: ANN401
        """Return TemplateEntry or None."""
        ...


class AgentEntry(BaseModel):
    """An agent configuration catalog entry with dynamic config resolution."""

    id: NonEmptyStr = Field(description="Unique catalog identifier for this agent")
    tool_ids: list[str] = Field(
        default=[], description="Catalog ToolEntry ids referenced by this agent"
    )
    card: AgentCard = Field(description="Agent card with config resolved from agent_class")

    @model_validator(mode="before")
    @classmethod
    def resolve_config(cls, data: Any) -> Any:  # noqa: ANN401
        """Resolve config to the concrete type expected by agent_class.

        Args:
            data: Raw input data (typically a dict from YAML deserialization).

        Returns:
            The data dict with ``card.config`` replaced by a validated
            concrete ``BaseConfig`` instance.

        Raises:
            ValueError: If ``agent_class`` cannot be imported or does not
                parameterize ``Akgent[ConfigType, StateType]``.
        """
        if not isinstance(data, dict):
            return data  # Let Pydantic handle non-dict input
        card_data = data.get("card")
        if isinstance(card_data, dict) and isinstance(card_data.get("config"), dict):
            agent_class_path = card_data.get("agent_class")
            if not isinstance(agent_class_path, str):
                return data  # Let Pydantic handle missing/invalid agent_class
            try:
                agent_cls = import_class(agent_class_path)
                config_cls = _extract_config_type(agent_cls)
            except (ImportError, AttributeError, ValueError) as e:
                raise ValueError(
                    f"Cannot resolve agent_class '{agent_class_path}': {e}"
                ) from e
            config_data = card_data["config"]
            config_data.pop("tools", None)
            card_data["config"] = config_cls.model_validate(config_data)
        return data

    def resolve_tools(self, tool_catalog: _ToolCatalogProtocol) -> list[ToolCard]:
        """Resolve tool_ids to ToolCard instances from the catalog.

        Args:
            tool_catalog: Catalog service providing tool lookups.

        Returns:
            Resolved ToolCard instances in tool_ids order.

        Raises:
            CatalogValidationError: If any tool_id is not found.
        """
        tools: list[ToolCard] = []
        for tid in self.tool_ids:
            entry = tool_catalog.get(tid)
            if entry is None:
                raise CatalogValidationError([f"Tool '{tid}' not found in catalog"])
            tools.append(entry.tool)
        return tools

    def resolve_template(
        self, template_catalog: _TemplateCatalogProtocol
    ) -> PromptTemplate | None:
        """Resolve @-reference to PromptTemplate, or None if no prompt.

        Does NOT render templates (rendering is a runtime concern, D4).

        Args:
            template_catalog: Catalog service providing template lookups.

        Returns:
            The resolved PromptTemplate, or None if the config has no prompt.

        Raises:
            CatalogValidationError: If the @-referenced template is not found.
        """
        config = self.card.config
        if not isinstance(config, AgentConfig):
            return None
        prompt = config.prompt
        if not _is_catalog_ref(prompt.template):
            return prompt
        template_id = _resolve_ref(prompt.template)
        entry = template_catalog.get(template_id)
        if entry is None:
            raise CatalogValidationError([f"Template '@{template_id}' not found"])
        return PromptTemplate(template=entry.template, params=prompt.params)
