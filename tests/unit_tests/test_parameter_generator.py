"""Unit tests for `tnt.parameter_generator`."""

from __future__ import annotations

import pytest

from tnt.all_models import AllModels
from tnt.parameter_generator import (
    SinglePointParameterGenerator,
    build_parameter_generator,
    parameter_generator_required_settings,
    parameter_generator_type_names,
)


def test_parameter_generator_registry_contains_every_explicit_type() -> None:
    assert parameter_generator_type_names() == {
        "GridSearch",
        "SinglePoint",
    }


def test_parameter_generator_registry_exposes_required_settings() -> None:
    assert parameter_generator_required_settings() == {
        "GridSearch": frozenset({"delta_chi2_threshold"}),
        "SinglePoint": frozenset(),
    }


def test_build_parameter_generator_dispatches_through_the_registry() -> None:
    generator = build_parameter_generator(
        {
            "generator_type": "SinglePoint",
            "generator_settings": {},
        },
        {},
    )

    assert isinstance(generator, SinglePointParameterGenerator)


def test_build_parameter_generator_rejects_an_unregistered_type() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Unknown parameter_space_settings.generator_type: 'Unknown'; "
            "expected one of: GridSearch, SinglePoint"
        ),
    ):
        build_parameter_generator(
            {
                "generator_type": "Unknown",
                "generator_settings": {},
            },
            {},
        )


def test_single_point_generator_proposes_quantities_in_their_declared_unit() -> None:
    # Declared units on purpose don't match any "internal" unit system --
    # SinglePointParameterGenerator never converts, it just parses what the
    # config declared (see tnt.potential's module docstring for why).
    potential_settings = {
        "bh": {
            "type": "PlummerPotential",
            "parameters": {
                "m_tot": {"value": 5.0, "unit": "kg", "fixed": True},
                "r_s": {"value": 1.0, "unit": "pc", "fixed": True},
            },
        },
        "dh": {
            "type": "NFWPotential",
            "parameterization": "concentration_m200",
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
