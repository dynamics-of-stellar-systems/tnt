from copy import deepcopy

import pytest
import unxt as u

from tnt.units import (
    build_unit_systems,
    normalize_unitful_value,
    validate_configuration_quantities,
)


def _unit_settings() -> dict:
    return {
        "internal": {
            "length": "kpc",
            "time": "Myr",
            "mass": "Msun",
            "angle": "rad",
            "power": "Lsun",
        },
        "display": {
            "angle": "arcsec",
            "speed": "km / s",
        },
    }


def test_build_unit_systems_inherits_internal_display_units() -> None:
    systems = build_unit_systems(_unit_settings())

    assert systems.internal[u.dimension("length")] == u.unit("kpc")
    assert systems.display[u.dimension("length")] == u.unit("kpc")
    assert systems.display[u.dimension("angle")] == u.unit("arcsec")
    assert systems.display[u.dimension("speed")] == u.unit("km / s")


@pytest.mark.parametrize(
    ("settings_update", "error"),
    [
        ({"internal": {"length": "Myr"}}, r"units\.internal\.length"),
        ({"display": {"speed": "not_a_unit"}}, r"units\.display\.speed"),
    ],
)
def test_build_unit_systems_rejects_invalid_units(
    settings_update: dict, error: str
) -> None:
    settings = _unit_settings()
    for section, values in settings_update.items():
        settings[section].update(values)

    with pytest.raises(ValueError, match=error):
        build_unit_systems(settings)


def test_build_unit_systems_requires_every_internal_dimension() -> None:
    settings = _unit_settings()
    del settings["internal"]["power"]

    with pytest.raises(ValueError, match=r"units\.internal.*power"):
        build_unit_systems(settings)


def test_normalize_unitful_value_accepts_explicit_value_and_unit() -> None:
    systems = build_unit_systems(_unit_settings())

    assert normalize_unitful_value(
        {"value": 2.5, "unit": "Mpc"},
        "length",
        systems.internal,
        "system_attributes.distance",
    ) == pytest.approx(2500.0)


def test_normalize_unitful_value_rejects_bare_number() -> None:
    systems = build_unit_systems(_unit_settings())

    with pytest.raises(TypeError, match="must state their unit explicitly"):
        normalize_unitful_value(
            2.5,
            "length",
            systems.internal,
            "system_attributes.distance",
        )


def test_normalize_unitful_value_rejects_sequence_shorthand() -> None:
    systems = build_unit_systems(_unit_settings())

    with pytest.raises(TypeError, match="mapping containing value and unit"):
        normalize_unitful_value(
            [2.5, "Mpc"],
            "length",
            systems.internal,
            "system_attributes.distance",
        )


def test_configuration_quantity_validation_does_not_modify_declarations() -> None:
    config = {
        "cosmological_parameters": {
            "H": {"value": 70.0, "unit": "km / (s Mpc)"}
        },
        "system_attributes": {
            "distance": {"value": 2.0, "unit": "Mpc"},
        },
        "potential": {
            "stars": {
                "type": "triaxial_light_mge",
                "parameters": {
                    "ml": {
                        "value": 5.0,
                        "unit": "Msun / Lsun",
                    }
                },
            }
        },
        "kinematic_data": {},
    }
    original = deepcopy(config)

    validate_configuration_quantities(config)

    assert config == original


def test_parameter_unit_is_rejected_until_dimension_is_declared() -> None:
    config = {
        "cosmological_parameters": {},
        "system_attributes": {},
        "potential": {
            "halo": {
                "type": "not_a_registered_potential_type",
                "parameters": {
                    "c": {
                        "value": 1.0,
                        "unit": "m",
                    }
                },
            }
        },
        "kinematic_data": {},
    }

    with pytest.raises(ValueError, match=r"parameters\.c\.unit is not supported"):
        validate_configuration_quantities(config)


@pytest.mark.parametrize(
    ("potential", "error"),
    [
        (
            {
                "bh": {
                    "type": "PlummerPotential",
                    "parameters": {
                        "m_tot": {
                            "value": 10.0,
                        }
                    },
                }
            },
            r"potential\.bh\.parameters\.m_tot.*required field: unit",
        ),
        (
            {
                "stars": {
                    "type": "triaxial_light_mge",
                    "parameters": {
                        "ml": {
                            "value": 5.0,
                        }
                    },
                }
            },
            r"potential\.stars\.parameters\.ml.*required field: unit",
        ),
    ],
)
def test_unitful_parameter_requires_unit(
    potential: dict,
    error: str,
) -> None:
    config = {
        "cosmological_parameters": {},
        "system_attributes": {},
        "potential": potential,
        "kinematic_data": {},
    }

    with pytest.raises(ValueError, match=error):
        validate_configuration_quantities(config)
