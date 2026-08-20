"""Unit tests for `tnt.potential`.

`Potential.generate_orbit_library` and the two MGE composite components'
`to_galax` remain `NotImplementedError` (see `tnt.potential`'s module
docstring); these tests cover what's actually implemented: dynamic
derivation from galax's own `ParameterField` metadata, type/parameterization
resolution, and the fully-working Plummer/NFW native-mode paths.
"""

from __future__ import annotations

import galax.potential as gp
import jax.numpy as jnp
import pytest
import unxt as u
from unxt import Quantity

from tnt.mge import LightMGE
from tnt.potential import (
    AbstractPotentialComponent,
    GalaxPotentialComponent,
    Potential,
    TriaxialLightMGEComponent,
    _rescale_exponent,
    build_potential,
    native_parameter_dimensions,
    raw_parameter_dimensions,
)


def _internal_unit_system() -> u.AbstractUnitSystem:
    return u.unitsystem("kpc", "Myr", "Msun", "rad", "Lsun")


def _closed_form_plummer_potential(
    m_tot: float, r_s: float, r: float, g: float
) -> float:
    """`Phi(r) = -G*M/sqrt(r**2 + r_s**2)`, computed independently of galax."""
    return -g * m_tot / (r**2 + r_s**2) ** 0.5


# ---------------------------------------------------------------------------
# Dynamic derivation from galax's own ParameterField metadata.
# ---------------------------------------------------------------------------


def test_native_parameter_dimensions_matches_known_galax_classes() -> None:
    assert native_parameter_dimensions("PlummerPotential") == {
        "m_tot": "mass",
        "r_s": "length",
    }
    assert native_parameter_dimensions("NFWPotential") == {"m": "mass", "r_s": "length"}
    assert native_parameter_dimensions("TriaxialNFWPotential") == {
        "m": "mass",
        "r_s": "length",
        "q1": "dimensionless",
        "q2": "dimensionless",
    }


def test_native_parameter_dimensions_returns_none_for_non_galax_name() -> None:
    assert native_parameter_dimensions("NotAPotential") is None


def test_native_parameter_dimensions_uses_one_name_for_an_aliased_type() -> None:
    # "speed" and "velocity" are both valid names for the same astropy
    # PhysicalType; stringifying the whole object joins every alias with
    # "/" ("speed/velocity"), which u.dimension() doesn't recognize and
    # silently treats as dimensionless instead of raising. Regression guard
    # for LogarithmicPotential's v_c, whose declared dimensions="speed"
    # comes back from galax with both aliases attached.
    dimensions = native_parameter_dimensions("LogarithmicPotential")
    assert dimensions == {"v_c": "speed", "r_s": "length"}


def test_rescale_exponent_matches_confirmed_dimensions() -> None:
    assert _rescale_exponent("mass") == pytest.approx(1.0)
    assert _rescale_exponent("length") == pytest.approx(0.0)
    assert _rescale_exponent("speed") == pytest.approx(0.5)
    assert _rescale_exponent("dimensionless") == pytest.approx(0.0)
    assert _rescale_exponent("angle") == pytest.approx(0.0)


def test_rescale_exponent_refuses_power_until_something_needs_it() -> None:
    # No native galax.potential parameter currently has dimension "power";
    # deliberately not pre-added on architectural grounds alone, following
    # the same discipline the Omega/frequency case established.
    with pytest.raises(NotImplementedError, match="'power'"):
        _rescale_exponent("power")


def test_rescale_exponent_refuses_to_guess_for_time_bearing_dimensions() -> None:
    # "frequency" has the same time-power as "speed", but a bar's pattern
    # speed (galax.potential.MonariEtAl2016BarPotential's Omega) must NOT
    # scale with mass, unlike a speed parameter that sets the potential's
    # amplitude (LogarithmicPotential's v_c). Dimension alone can't tell
    # these apart, so both "time" and "frequency" must raise rather than
    # silently pick a side.
    with pytest.raises(NotImplementedError, match="'time'"):
        _rescale_exponent("time")
    with pytest.raises(NotImplementedError, match="'frequency'"):
        _rescale_exponent("frequency")


def test_bar_pattern_speed_rescale_refuses_to_guess() -> None:
    assert native_parameter_dimensions("MonariEtAl2016BarPotential")["Omega"] == (
        "frequency"
    )
    component = GalaxPotentialComponent(
        galax_type="MonariEtAl2016BarPotential",
        parameters={"Omega": Quantity(40.0, "km/(s kpc)")},
    )
    with pytest.raises(NotImplementedError, match="Omega"):
        component.rescale(2.0)


def test_logarithmic_potential_rescale_scales_v_c_by_sqrt_mass_scale() -> None:
    # LogarithmicPotential has no mass-dimensioned native parameter at all
    # (Phi = 0.5 * v_c**2 * ln(...)); rescale must still scale v_c so that
    # Phi itself scales linearly with mass_scale, exactly as a mass
    # parameter would for Plummer/NFW.
    component = GalaxPotentialComponent(
        galax_type="LogarithmicPotential",
        parameters={"v_c": Quantity(200.0, "km/s"), "r_s": Quantity(1.0, "kpc")},
    )
    rescaled = component.rescale(4.0)
    assert rescaled.parameters["v_c"].ustrip("km/s") == pytest.approx(400.0)
    assert rescaled.parameters["r_s"].ustrip("kpc") == pytest.approx(1.0)

    unit_system = _internal_unit_system()
    xyz = Quantity(jnp.array([3.0, 0.0, 0.0]), "kpc")
    t = Quantity(0.0, "Myr")
    phi_before = component.to_galax(unit_system).potential(xyz, t).ustrip("kpc2 / Myr2")
    phi_after = rescaled.to_galax(unit_system).potential(xyz, t).ustrip("kpc2 / Myr2")
    assert float(phi_after / phi_before) == pytest.approx(4.0)


def test_multipole_rescale_finds_a_field_inherited_from_an_abstract_parent() -> None:
    # MultipolePotential doesn't redeclare m_tot/r_s itself -- they're
    # ParameterFields on its abstract parent, AbstractMultipolePotential.
    # Regression guard: an earlier version looked parameters up via
    # `galax_cls.__dict__.get(name)`, which only sees a class's own
    # attributes and silently missed anything declared on a parent class.
    assert "m_tot" in gp.AbstractMultipolePotential.__dict__
    assert "m_tot" not in gp.MultipolePotential.__dict__
    component = GalaxPotentialComponent(
        galax_type="MultipolePotential",
        parameters={"m_tot": Quantity(1e10, "Msun"), "r_s": Quantity(1.0, "kpc")},
    )
    rescaled = component.rescale(4.0)
    assert rescaled.parameters["m_tot"].ustrip("Msun") == pytest.approx(4e10)
    assert rescaled.parameters["r_s"].ustrip("kpc") == pytest.approx(1.0)


def test_rescale_raises_clearly_for_a_non_parameter_field_native_argument() -> None:
    # MultipolePotential.l_max is a plain int hyperparameter, not a
    # ParameterField/Quantity -- rescale() can't know how to scale it and
    # should say so clearly rather than a bare KeyError.
    component = GalaxPotentialComponent(
        galax_type="MultipolePotential",
        parameters={"m_tot": Quantity(1e10, "Msun"), "l_max": Quantity(2, "")},
    )
    with pytest.raises(NotImplementedError, match="l_max"):
        component.rescale(2.0)


def test_raw_parameter_dimensions_covers_all_three_sources() -> None:
    # Native galax type, no parameterization.
    assert raw_parameter_dimensions("PlummerPotential", None) == {
        "m_tot": "mass",
        "r_s": "length",
    }
    # Registered non-native parameterization.
    assert raw_parameter_dimensions("NFWPotential", "concentration_mass_ratio") == {}
    # TNT MGE composite type.
    assert raw_parameter_dimensions("triaxial_light_mge", None) == {
        "ml": "mass_to_light"
    }
    # Unrecognized (type, parameterization) pair -- defer to from_settings.
    assert raw_parameter_dimensions("not_a_type", None) == {}


# ---------------------------------------------------------------------------
# Type / parameterization resolution.
# ---------------------------------------------------------------------------


def test_from_settings_rejects_unrecognized_type() -> None:
    unit_system = _internal_unit_system()
    with pytest.raises(
        ValueError, match="Unsupported potential.dh.type 'NotAPotential'"
    ):
        AbstractPotentialComponent.from_settings(
            {"type": "NotAPotential", "include": True, "parameters": {}},
            {},
            unit_system,
            path="potential.dh",
        )


def test_from_settings_resolves_a_real_galax_class_name() -> None:
    unit_system = _internal_unit_system()
    component = AbstractPotentialComponent.from_settings(
        {
            "type": "NFWPotential",
            "include": True,
            "parameters": {"m": {"value": 1e11}, "r_s": {"value": 10.0}},
        },
        {},
        unit_system,
        path="potential.dh",
    )
    assert isinstance(component, GalaxPotentialComponent)
    assert component.galax_type == "NFWPotential"
    assert component.parameters["m"].ustrip("Msun") == pytest.approx(1e11)
    assert component.parameters["r_s"].ustrip("kpc") == pytest.approx(10.0)


def test_from_settings_rejects_unimplemented_parameterization() -> None:
    unit_system = _internal_unit_system()
    with pytest.raises(NotImplementedError, match="'bogus' is not implemented"):
        AbstractPotentialComponent.from_settings(
            {
                "type": "PlummerPotential",
                "parameterization": "bogus",
                "include": True,
                "parameters": {"m_tot": {"value": 1.0}, "r_s": {"value": 1.0}},
            },
            {},
            unit_system,
            path="potential.bh",
        )


def test_nfw_concentration_mass_ratio_parameterization_not_yet_implemented() -> None:
    unit_system = _internal_unit_system()
    with pytest.raises(NotImplementedError, match="concentration_mass_ratio"):
        AbstractPotentialComponent.from_settings(
            {
                "type": "NFWPotential",
                "parameterization": "concentration_mass_ratio",
                "include": True,
                "parameters": {"c": {"value": 3.0}, "f": {"value": 1.0}},
            },
            {},
            unit_system,
            path="potential.dh",
        )


# ---------------------------------------------------------------------------
# Plummer, end-to-end.
# ---------------------------------------------------------------------------


def test_plummer_to_galax_matches_closed_form_potential() -> None:
    unit_system = _internal_unit_system()
    m_tot, r_s = 5.0, 1e-3
    component = AbstractPotentialComponent.from_settings(
        {
            "type": "PlummerPotential",
            "include": True,
            "parameters": {"m_tot": {"value": m_tot}, "r_s": {"value": r_s}},
        },
        {},
        unit_system,
        path="potential.bh",
    )
    galax_potential = component.to_galax(unit_system)

    r = 0.01
    xyz = Quantity(jnp.array([r, 0.0, 0.0]), "kpc")
    t = Quantity(0.0, "Myr")
    value = float(galax_potential.potential(xyz, t).ustrip("kpc2 / Myr2"))

    g = float(u.Quantity(6.6743e-11, "m3 / (kg s2)").ustrip("kpc3 / (Msun Myr2)"))
    expected = _closed_form_plummer_potential(m_tot, r_s, r, g)
    assert value == pytest.approx(expected, rel=1e-5)


def test_plummer_rescale_scales_only_the_mass_parameter() -> None:
    unit_system = _internal_unit_system()
    component = AbstractPotentialComponent.from_settings(
        {
            "type": "PlummerPotential",
            "include": True,
            "parameters": {"m_tot": {"value": 5.0}, "r_s": {"value": 1e-3}},
        },
        {},
        unit_system,
        path="potential.bh",
    )
    rescaled = component.rescale(2.0)
    assert rescaled.parameters["m_tot"].ustrip("Msun") == pytest.approx(10.0)
    assert rescaled.parameters["r_s"].ustrip("kpc") == pytest.approx(1e-3)


def test_potential_composes_only_included_components() -> None:
    unit_system = _internal_unit_system()
    settings = {
        "bh": {
            "type": "PlummerPotential",
            "include": True,
            "parameters": {"m_tot": {"value": 5.0}, "r_s": {"value": 1e-3}},
        },
        "excluded": {
            "type": "PlummerPotential",
            "include": False,
            "parameters": {"m_tot": {"value": 100.0}, "r_s": {"value": 1.0}},
        },
    }
    potential = build_potential(settings, {}, unit_system)
    assert set(potential.components) == {"bh"}

    galax_potential = potential.to_galax(unit_system)
    xyz = Quantity(jnp.array([0.01, 0.0, 0.0]), "kpc")
    t = Quantity(0.0, "Myr")
    composed_value = float(galax_potential.potential(xyz, t).ustrip("kpc2 / Myr2"))
    component_value = float(
        potential.components["bh"]
        .to_galax(unit_system)
        .potential(xyz, t)
        .ustrip("kpc2 / Myr2")
    )
    assert composed_value == pytest.approx(component_value)


# ---------------------------------------------------------------------------
# NFW plumbing, independent of the parameterization gap.
# ---------------------------------------------------------------------------


def test_nfw_component_plumbing_works_without_from_settings() -> None:
    unit_system = _internal_unit_system()
    component = GalaxPotentialComponent(
        galax_type="NFWPotential",
        parameters={"m": Quantity(1e11, "Msun"), "r_s": Quantity(10.0, "kpc")},
    )

    rescaled = component.rescale(3.0)
    assert rescaled.parameters["m"].ustrip("Msun") == pytest.approx(3e11)
    assert rescaled.parameters["r_s"].ustrip("kpc") == pytest.approx(10.0)

    galax_potential = component.to_galax(unit_system)
    xyz = Quantity(jnp.array([5.0, 0.0, 0.0]), "kpc")
    t = Quantity(0.0, "Myr")
    value = galax_potential.potential(xyz, t)
    assert jnp.isfinite(value.ustrip("kpc2 / Myr2"))


# ---------------------------------------------------------------------------
# MGE composite types -- from_settings works, to_galax stays NotImplementedError.
# ---------------------------------------------------------------------------


def test_mge_component_from_settings_resolves_mge_but_to_galax_is_not_implemented() -> (
    None
):
    unit_system = _internal_unit_system()
    light_mge = LightMGE(
        I=Quantity(jnp.array([1.0]), "Lsun / rad2"),
        sigma=Quantity(jnp.array([1.0]), "rad"),
        q=Quantity(jnp.array([0.5]), ""),
        PA_twist=Quantity(jnp.array([0.0]), "rad"),
    )
    component = AbstractPotentialComponent.from_settings(
        {
            "type": "triaxial_light_mge",
            "include": True,
            "mge": "mge_lum",
            "parameters": {"ml": {"value": 5.0}},
        },
        {"mge_lum": light_mge},
        unit_system,
        path="potential.stars",
    )
    assert isinstance(component, TriaxialLightMGEComponent)
    assert component.mge is light_mge
    assert component.parameters["ml"].ustrip("Msun / Lsun") == pytest.approx(5.0)

    with pytest.raises(NotImplementedError):
        component.to_galax(unit_system)


def test_potential_generate_orbit_library_not_implemented() -> None:
    potential = Potential(components={})
    with pytest.raises(NotImplementedError):
        potential.generate_orbit_library({}, None, None)
