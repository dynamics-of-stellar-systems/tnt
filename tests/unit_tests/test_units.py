from copy import deepcopy

import pytest
import unxt as u

from tnt.units import (
    build_unit_systems,
    declared_quantity,
    validate_configuration_quantities,
    validate_dimension,
)


def _unit_settings() -> dict:
    return {
        "internal": {
            "length": "kpc",
            "time": "Myr",
            "mass": "Msun",
            "angle": "rad",
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
    del settings["internal"]["angle"]

    with pytest.raises(ValueError, match=r"units\.internal.*angle"):
        build_unit_systems(settings)


def test_build_unit_systems_rejects_power_in_internal() -> None:
    settings = _unit_settings()
    settings["internal"]["power"] = "Lsun"

    with pytest.raises(ValueError, match=r"units\.internal.*power"):
        build_unit_systems(settings)


def test_build_unit_systems_accepts_power_display_override() -> None:
    settings = _unit_settings()
    settings["display"]["power"] = "erg / s"

    systems = build_unit_systems(settings)

    assert systems.display[u.dimension("power")] == u.unit("erg / s")


def test_declared_quantity_keeps_its_declared_unit() -> None:
    quantity = declared_quantity(
        {"value": 2.5, "unit": "Mpc"},
        "length",
        "system_attributes.distance",
    )

    assert quantity.unit == u.unit("Mpc")
    assert quantity.ustrip("Mpc") == pytest.approx(2.5)


def test_declared_quantity_rejects_bare_number() -> None:
    with pytest.raises(TypeError, match="must state their unit explicitly"):
        declared_quantity(2.5, "length", "system_attributes.distance")


def test_declared_quantity_rejects_sequence_shorthand() -> None:
    with pytest.raises(TypeError, match="mapping containing value and unit"):
        declared_quantity(
            [2.5, "Mpc"], "length", "system_attributes.distance"
        )


def test_validate_dimension_accepts_equivalent_unit() -> None:
    validate_dimension(u.unit("km / s"), "speed", "kinematic_data.x")


def test_validate_dimension_rejects_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="must describe speed"):
        validate_dimension(u.unit("kpc"), "speed", "kinematic_data.x")


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
                "type": "TriaxialLightMGEPotential",
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


def test_parameter_unit_is_rejected_on_a_dimensionless_native_parameter() -> None:
    config = {
        "cosmological_parameters": {},
        "system_attributes": {},
        "potential": {
            "halo": {
                "type": "gNFWPotential",
                "parameters": {
                    "m": {"value": 1.0e12, "unit": "Msun"},
                    "r_s": {"value": 10.0, "unit": "kpc"},
                    "gamma": {"value": 1.0, "unit": "m"},
                },
            }
        },
        "kinematic_data": {},
    }

    with pytest.raises(ValueError, match=r"parameters\.gamma\.unit is not supported"):
        validate_configuration_quantities(config)


def test_parameter_unit_check_defers_for_an_unrecognized_potential_type() -> None:
    # An unknown type has no known parameter schema, so unit validation is
    # skipped here -- the "unsupported type" error is left to resolve().
    config = {
        "cosmological_parameters": {},
        "system_attributes": {},
        "potential": {
            "halo": {
                "type": "not_a_registered_potential_type",
                "parameters": {"c": {"value": 1.0, "unit": "m"}},
            }
        },
        "kinematic_data": {},
    }

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
                    "type": "TriaxialLightMGEPotential",
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
