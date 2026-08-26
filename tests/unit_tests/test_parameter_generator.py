"""Unit tests for `tnt.parameter_generator`."""

from __future__ import annotations

import pytest

from tnt.all_models import AllModels
from tnt.configuration import validation as configuration_validation
from tnt.parameter_generator import _GENERATOR_CLASSES, SinglePointParameterGenerator


def test_generator_settings_keys_match_the_real_classes() -> None:
    """`tnt.configuration.validation._GENERATOR_SETTINGS_KEYS` is duplicated,
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


def test_single_point_generator_proposes_quantities_in_their_declared_unit() -> None:
    # Declared units on purpose don't match any "internal" unit system --
    # SinglePointParameterGenerator never converts, it just parses what the
    # config declared (see tnt.potential's module docstring for why).
    potential_settings = {
        "bh": {
            "type": "PlummerPotential",
            "include": True,
            "parameters": {
                "m_tot": {"value": 5.0, "unit": "kg", "fixed": True},
                "r_s": {"value": 1.0, "unit": "pc", "fixed": True},
            },
        },
        "dh": {
            "type": "NFWPotential",
            "parameterization": "concentration_m200",
            "include": True,
            "parameters": {
                "c": {"value": 8.0, "fixed": True},
                "M_200": {"value": 1.0e12, "unit": "Msun", "fixed": True},
            },
        },
    }
    generator = SinglePointParameterGenerator(
        potential_settings=potential_settings, generator_settings={}
    )

    (proposed,) = generator.generate_parameters(AllModels())

    assert proposed["bh"]["m_tot"].ustrip("kg") == pytest.approx(5.0)
    assert proposed["bh"]["r_s"].ustrip("pc") == pytest.approx(1.0)
    assert proposed["dh"]["c"].ustrip("") == pytest.approx(8.0)
    assert proposed["dh"]["M_200"].ustrip("Msun") == pytest.approx(1.0e12)
