"""Explicit parameter-generator registration and schema metadata."""

from __future__ import annotations

from tnt.parameter_generator.base import AbstractParameterGenerator
from tnt.registry import register_typed_class

_PARAMETER_GENERATOR_REGISTRY: dict[str, type[AbstractParameterGenerator]] = {}


def register_parameter_generator(
    cls: type[AbstractParameterGenerator],
) -> type[AbstractParameterGenerator]:
    """Register one concrete parameter generator for configuration dispatch."""
    return register_typed_class(
        _PARAMETER_GENERATOR_REGISTRY,
        cls,
        family="parameter generator",
    )


def get_parameter_generator_class(
    type_name: str,
) -> type[AbstractParameterGenerator] | None:
    """Return the registered generator class for ``type_name``, if any."""
    return _PARAMETER_GENERATOR_REGISTRY.get(type_name)


def parameter_generator_type_names() -> frozenset[str]:
    """Return every explicitly registered parameter-generator type name."""
    return frozenset(_PARAMETER_GENERATOR_REGISTRY)


def parameter_generator_required_settings() -> dict[str, frozenset[str]]:
    """Return required configuration-setting names for every generator type."""
    return {
        type_name: cls._required_generator_settings
        for type_name, cls in _PARAMETER_GENERATOR_REGISTRY.items()
    }
