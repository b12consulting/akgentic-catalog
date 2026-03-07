"""Agent model utilities for catalog introspection."""

from typing import get_args

from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig

__all__ = [
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
