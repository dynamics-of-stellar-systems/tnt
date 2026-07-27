import math

import pytest
import unxt as u

from tnt.units import (
    build_unit_systems,
    normalize_configuration_quantities,
    normalize_unitful_value,
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


def test_normalize_unitful_value_accepts_bare_or_explicit_values() -> None:
    systems = build_unit_systems(_unit_settings())

    assert (
        normalize_unitful_value(2.5, "length", systems, "system_attributes.distance")
        == 2.5
    )
    assert normalize_unitful_value(
        {"value": 2.5, "unit": "Mpc"},
        "length",
        systems,
        "system_attributes.distance",
    ) == pytest.approx(2500.0)


def test_normalize_unitful_value_rejects_sequence_shorthand() -> None:
    systems = build_unit_systems(_unit_settings())

    with pytest.raises(TypeError, match="number.*or a mapping"):
        normalize_unitful_value(
            [2.5, "Mpc"],
            "length",
            systems,
            "system_attributes.distance",
        )


def test_normalize_supported_configuration_quantities() -> None:
    systems = build_unit_systems(_unit_settings())
    config = {
        "cosmological_parameters": {"H0": {"value": 70.0, "unit": "km / (s Mpc)"}},
        "system_attributes": {
            "distance": {"value": 2.0, "unit": "Mpc"},
        },
        "system_components": {
            "bh": {
                "type": "plummer",
                "parameters": {
                    "m": {
                        "value": 10.0,
                        "logarithmic": True,
                        "unit": "kg",
                        "generator_settings": {
                            "lower_bound": 9.0,
                            "upper_bound": 11.0,
                            "step": 0.5,
                            "minimum_step": 0.1,
                        },
                    },
                    "a": {
                        "value": 500.0,
                        "logarithmic": False,
                        "unit": "pc",
                        "generator_settings": {
                            "lower_bound": 100.0,
                            "upper_bound": 1000.0,
                            "step": 100.0,
                            "minimum_step": 10.0,
                        },
                    },
                },
            },
            "stars": {
                "type": "triaxial_visible_component",
                "kinematics": {
                    "observed": {
                        "type": "gauss_hermite",
                        "histogram": {
                            "width": {"value": 1000.0, "unit": "km / s"},
                            "center": {"value": 10.0, "unit": "km / s"},
                        },
                        "observational_errors": {
                            "systematic_uncertainties": {
                                "v": {"value": 2.0, "unit": "km / s"},
                                "sigma": 3.0,
                                "h3": 0.0,
                            }
                        },
                    }
                },
            },
        },
        "system_parameters": {
            "ml": {
                "value": 5.0,
                "logarithmic": False,
                "unit": "Msun / Lsun",
            }
        },
    }

    normalized = normalize_configuration_quantities(config, systems)
    speed_factor = float(u.unit("km / s").to(u.unit("kpc / Myr"), 1.0))
    mass_offset = math.log10(float(u.unit("kg").to(u.unit("Msun"), 1.0)))

    assert normalized["system_attributes"]["distance"] == pytest.approx(2000.0)
    assert normalized["cosmological_parameters"]["H0"] == pytest.approx(
        7.158985155319864e-05
    )
    bh_parameters = normalized["system_components"]["bh"]["parameters"]
    assert bh_parameters["a"]["value"] == pytest.approx(0.5)
    assert bh_parameters["a"]["generator_settings"]["step"] == pytest.approx(0.1)
    assert bh_parameters["m"]["value"] == pytest.approx(10.0 + mass_offset)
    assert bh_parameters["m"]["generator_settings"]["lower_bound"] == pytest.approx(
        9.0 + mass_offset
    )
    assert bh_parameters["m"]["generator_settings"]["step"] == 0.5
    assert "unit" not in bh_parameters["m"]
    histogram = normalized["system_components"]["stars"]["kinematics"]["observed"][
        "histogram"
    ]
    assert histogram["width"] == pytest.approx(1000.0 * speed_factor)
    assert histogram["center"] == pytest.approx(10.0 * speed_factor)
    systematics = normalized["system_components"]["stars"]["kinematics"]["observed"][
        "observational_errors"
    ]["systematic_uncertainties"]
    assert systematics["v"] == pytest.approx(2.0 * speed_factor)
    assert systematics["sigma"] == 3.0
    assert normalized["system_parameters"]["ml"]["value"] == 5.0
    assert "unit" not in normalized["system_parameters"]["ml"]
    assert config["system_attributes"]["distance"]["unit"] == "Mpc"


def test_parameter_unit_is_rejected_until_dimension_is_declared() -> None:
    systems = build_unit_systems(_unit_settings())
    config = {
        "cosmological_parameters": {},
        "system_attributes": {},
        "system_components": {
            "halo": {
                "type": "nfw",
                "parameters": {
                    "c": {
                        "value": 1.0,
                        "logarithmic": False,
                        "unit": "m",
                    }
                },
            }
        },
        "system_parameters": {},
    }

    with pytest.raises(ValueError, match=r"parameters\.c\.unit is not supported"):
        normalize_configuration_quantities(config, systems)
