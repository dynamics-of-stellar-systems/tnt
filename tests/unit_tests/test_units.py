from copy import deepcopy

import pytest
import unxt as u

from tnt.units import (
    build_unit_systems,
    normalize_configuration_quantities,
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


def test_normalize_supported_configuration_quantities() -> None:
    systems = build_unit_systems(_unit_settings())
    config = {
        "cosmological_parameters": {"H0": {"value": 70.0, "unit": "km / (s Mpc)"}},
        "system_attributes": {
            "distance": {"value": 2.0, "unit": "Mpc"},
        },
        "potential": {
            "bh": {
                "type": "PlummerPotential",
                "parameters": {
                    "m_tot": {
                        "value": 10.0,
                        "unit": "kg",
                        "prior": {"distribution": "Uniform", "args": [9.0, 11.0]},
                    },
                    "r_s": {
                        "value": 500.0,
                        "unit": "pc",
                        "prior": {"distribution": "Uniform", "args": [100.0, 1000.0]},
                    },
                },
            },
            "stars": {
                "type": "triaxial_light_mge",
                "parameters": {
                    "ml": {
                        "value": 5.0,
                        "unit": "Msun / Lsun",
                    }
                },
            },
        },
        "kinematic_data": {
            "observed": {
                "type": "gauss_hermite",
                "histogram": {
                    "width": {"value": 1000.0, "unit": "km / s"},
                    "center": {"value": 10.0, "unit": "km / s"},
                },
                "observational_errors": {
                    "systematic_uncertainties": {
                        "v": {"value": 2.0, "unit": "km / s"},
                        "sigma": {"value": 3.0, "unit": "kpc / Myr"},
                        "h3": 0.0,
                    }
                },
            },
        },
    }

    normalized = normalize_configuration_quantities(config, systems)
    speed_factor = float(u.unit("km / s").to(u.unit("kpc / Myr"), 1.0))
    mass_factor = float(u.unit("kg").to(u.unit("Msun"), 1.0))

    assert normalized["system_attributes"]["distance"] == pytest.approx(2000.0)
    assert normalized["cosmological_parameters"]["H0"] == pytest.approx(
        7.158985155319864e-05
    )
    bh_parameters = normalized["potential"]["bh"]["parameters"]
    assert bh_parameters["r_s"]["value"] == pytest.approx(0.5)
    # `prior` is a search-space declaration (like `fixed`/`latex_label`),
    # not a physical quantity -- left untouched, not unit-converted.
    assert bh_parameters["r_s"]["prior"] == {
        "distribution": "Uniform",
        "args": [100.0, 1000.0],
    }
    assert bh_parameters["m_tot"]["value"] == pytest.approx(10.0 * mass_factor)
    assert bh_parameters["m_tot"]["prior"] == {
        "distribution": "Uniform",
        "args": [9.0, 11.0],
    }
    assert "unit" not in bh_parameters["m_tot"]
    histogram = normalized["kinematic_data"]["observed"]["histogram"]
    assert histogram["width"] == pytest.approx(1000.0 * speed_factor)
    assert histogram["center"] == pytest.approx(10.0 * speed_factor)
    systematics = normalized["kinematic_data"]["observed"]["observational_errors"][
        "systematic_uncertainties"
    ]
    assert systematics["v"] == pytest.approx(2.0 * speed_factor)
    assert systematics["sigma"] == 3.0
    assert normalized["potential"]["stars"]["parameters"]["ml"]["value"] == 5.0
    assert "unit" not in normalized["potential"]["stars"]["parameters"]["ml"]
    assert config["system_attributes"]["distance"]["unit"] == "Mpc"


def test_configuration_quantity_validation_does_not_modify_declarations() -> None:
    config = {
        "cosmological_parameters": {"H0": {"value": 70.0, "unit": "km / (s Mpc)"}},
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
    systems = build_unit_systems(_unit_settings())
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
        normalize_configuration_quantities(config, systems)


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
    systems = build_unit_systems(_unit_settings())
    config = {
        "cosmological_parameters": {},
        "system_attributes": {},
        "potential": potential,
        "kinematic_data": {},
    }

    with pytest.raises(ValueError, match=error):
        normalize_configuration_quantities(config, systems)
