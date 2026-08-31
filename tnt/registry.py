"""Shared registration mechanics for configured runtime class families."""

from __future__ import annotations


def register_typed_class[RegisteredType](
    registry: dict[str, type[RegisteredType]],
    cls: type[RegisteredType],
    *,
    family: str,
) -> type[RegisteredType]:
    """Register ``cls`` under its own explicitly declared ``_type``.

    Args:
        registry: The class family's owned type-name-to-class mapping.
        cls: Concrete class being registered.
        family: Human-readable family name used in validation errors.

    Returns:
        ``cls`` unchanged, so this function can back a class decorator.

    Raises:
        TypeError: If ``cls`` does not declare its own non-empty string
            ``_type``. Inherited values do not count.
        ValueError: If another class is already registered under that type.
    """
    type_name = cls.__dict__.get("_type")
    if not isinstance(type_name, str) or not type_name:
        raise TypeError(
            f"{cls.__name__} must declare its own non-empty string _type."
        )
    if type_name in registry:
        existing = registry[type_name].__name__
        raise ValueError(
            f"Duplicate {family} type {type_name!r} on {existing} and "
            f"{cls.__name__}."
        )
    registry[type_name] = cls
    return cls
