"""Tests for shared configured-runtime class registration mechanics."""

from typing import ClassVar

import pytest

from tnt.registry import register_typed_class


def test_register_typed_class_uses_the_class_own_type() -> None:
    registry: dict[str, type] = {}

    class Example:
        _type: ClassVar[str] = "example"

    result = register_typed_class(registry, Example, family="example")

    assert result is Example
    assert registry == {"example": Example}


def test_register_typed_class_rejects_an_inherited_type() -> None:
    registry: dict[str, type] = {}

    class Parent:
        _type: ClassVar[str] = "parent"

    class Child(Parent):
        pass

    with pytest.raises(TypeError, match="must declare its own non-empty string _type"):
        register_typed_class(registry, Child, family="example")

    assert registry == {}


def test_register_typed_class_rejects_a_duplicate() -> None:
    registry: dict[str, type] = {}

    class First:
        _type: ClassVar[str] = "duplicate"

    class Second:
        _type: ClassVar[str] = "duplicate"

    register_typed_class(registry, First, family="example")

    with pytest.raises(ValueError, match="Duplicate example type 'duplicate'"):
        register_typed_class(registry, Second, family="example")

    assert registry == {"duplicate": First}
