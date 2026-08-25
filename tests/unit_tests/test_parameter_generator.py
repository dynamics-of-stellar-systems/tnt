"""Unit tests for `tnt.parameter_generator`."""

from __future__ import annotations

from tnt import configuration_validation
from tnt.parameter_generator import _GENERATOR_CLASSES


def test_generator_settings_keys_match_the_real_classes() -> None:
    """`configuration_validation._GENERATOR_SETTINGS_KEYS` is duplicated,
    plain-data information -- it can't import these classes directly (see
    its own comment for why: preparation-phase code shouldn't depend on
    execution-phase modules). This is the regression test that keeps that
    duplicated data in sync with each class's own
    `_required_generator_settings`.
    """
    expected = {
        generator_cls._type: generator_cls._required_generator_settings
        for generator_cls in _GENERATOR_CLASSES
    }
    assert configuration_validation._GENERATOR_SETTINGS_KEYS == expected
