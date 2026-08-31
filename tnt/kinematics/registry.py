"""Explicit registry for configured kinematics runtime classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tnt.registry import register_typed_class

if TYPE_CHECKING:
    from tnt.kinematics.base import AbstractKinematics

_KINEMATICS_REGISTRY: dict[str, type[AbstractKinematics]] = {}


def register_kinematics(
    cls: type[AbstractKinematics],
) -> type[AbstractKinematics]:
    """Register one concrete kinematics class for configuration dispatch."""
    return register_typed_class(_KINEMATICS_REGISTRY, cls, family="kinematics")


def get_kinematics_class(type_name: str) -> type[AbstractKinematics] | None:
    """Return the registered class for ``type_name``, if any."""
    return _KINEMATICS_REGISTRY.get(type_name)


def kinematics_type_names() -> frozenset[str]:
    """Return every registered kinematics type name."""
    return frozenset(_KINEMATICS_REGISTRY)
