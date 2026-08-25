"""Unit tests for `tnt.potential`.

`Potential.generate_orbit_library` and the two MGE composite components'
`to_galax` remain `NotImplementedError` (see `tnt.potential`'s module
docstring); these tests cover what's actually implemented: dynamic
derivation from galax's own `ParameterField` metadata, type/parameterization
resolution, and the fully-working Plummer/NFW native-mode paths.
"""

from __future__ import annotations

import dataclasses

import galax.potential as gp
import jax
import jax.numpy as jnp
import pytest
import unxt as u
from galax.potential.params import ParameterField
from unxt import Quantity

from tnt.mge import LightMGE
from tnt.potential import (
    _SUPPORTED_GALAX_TYPES,
    AbstractPotentialComponent,
    GalaxPotentialComponent,
    Potential,
    TriaxialLightMGEComponent,
    _nfw_concentration_m200,
    _nfw_concentration_m200_inverse,
    _nfw_g,
    _solve_nfw_concentration,
    build_potential,
    raw_parameter_dimensions,
    raw_potential_parameters,
)


def _native_parameter_dimensions(galax_type: str) -> dict[str, str] | None:
    """Each of `galax_type`'s native constructor parameters' physical dimension.

    Derived from galax's own `ParameterField(dimensions=...)` metadata,
    independently of `tnt.potential._SUPPORTED_GALAX_TYPES` -- this is what
    `test_supported_galax_types_covers_every_curated_class_parameter`
    cross-checks the curated table against.
    """
    cls = getattr(gp, galax_type, None)
    if not (isinstance(cls, type) and issubclass(cls, gp.AbstractPotential)):
        return None
    return {
        # A PhysicalType with more than one recognized name (e.g. "speed"
        # and "velocity") stringifies as "speed/velocity", which
        # u.dimension() doesn't recognize and silently resolves to
        # "dimensionless" instead of raising -- iterate and take the first
        # canonical name instead of stringifying the whole object.
        f.name: next(iter(raw.dimensions))
        for f in dataclasses.fields(cls)
        # getattr, not cls.__dict__.get: a ParameterField declared on an
        # abstract parent (e.g. AbstractMultipolePotential's m_tot/r_s,
        # inherited by MultipolePotential) isn't in the subclass's own
        # __dict__, but getattr still finds it via the MRO.
        if isinstance(raw := getattr(cls, f.name, None), ParameterField)
    }


def _internal_unit_system() -> u.AbstractUnitSystem:
    return u.unitsystem("kpc", "Myr", "Msun", "rad", "Lsun")


# Only the concentration_m200 parameterization uses cosmological_parameters;
# every other test passes it through unused.
_NO_COSMOLOGICAL_PARAMETERS: dict[str, float] = {}


def _closed_form_plummer_potential(
    m_tot: float, r_s: float, r: float, g: float
) -> float:
    """`Phi(r) = -G*M/sqrt(r**2 + r_s**2)`, computed independently of galax."""
    return -g * m_tot / (r**2 + r_s**2) ** 0.5


# ---------------------------------------------------------------------------
# Dynamic derivation from galax's own ParameterField metadata.
# ---------------------------------------------------------------------------


def test_native_parameter_dimensions_matches_known_galax_classes() -> None:
    assert _native_parameter_dimensions("PlummerPotential") == {
        "m_tot": "mass",
        "r_s": "length",
    }
    assert _native_parameter_dimensions("NFWPotential") == {
        "m": "mass",
        "r_s": "length",
    }
    assert _native_parameter_dimensions("TriaxialNFWPotential") == {
        "m": "mass",
        "r_s": "length",
        "q1": "dimensionless",
        "q2": "dimensionless",
    }


def test_native_parameter_dimensions_returns_none_for_non_galax_name() -> None:
    assert _native_parameter_dimensions("NotAPotential") is None


def test_native_parameter_dimensions_uses_one_name_for_an_aliased_type() -> None:
    # "speed" and "velocity" are both valid names for the same astropy
    # PhysicalType; stringifying the whole object joins every alias with
    # "/" ("speed/velocity"), which u.dimension() doesn't recognize and
    # silently treats as dimensionless instead of raising. Regression guard
    # for LogarithmicPotential's v_c, whose declared dimensions="speed"
    # comes back from galax with both aliases attached.
    dimensions = _native_parameter_dimensions("LogarithmicPotential")
    assert dimensions == {"v_c": "speed", "r_s": "length"}


def test_supported_galax_types_covers_every_curated_class_parameter() -> None:
    # Every parameter in every curated class's table must actually be one of
    # that class's own native ParameterFields, with a matching dimension --
    # catches typos and stale entries if galax ever renames a parameter or
    # changes a dimension.
    for galax_type, parameters in _SUPPORTED_GALAX_TYPES.items():
        dimensions = _native_parameter_dimensions(galax_type)
        assert dimensions is not None, f"{galax_type} is not a real galax class"
        assert set(parameters) == set(dimensions), galax_type
        for name, parameter in parameters.items():
            assert parameter.dimension == dimensions[name], f"{galax_type}.{name}"


def test_bar_pattern_speed_rescale_stays_fixed() -> None:
    # MonariEtAl2016BarPotential's Omega (bar pattern speed) and v0 (sets
    # the potential's amplitude) share the same "frequency"/"speed"
    # time-power but play opposite roles under rescale -- verified against
    # galax's own prefactor formula (alpha * (v0**2/3) * (R0/Rb)**3).
    component = GalaxPotentialComponent(
        galax_type="MonariEtAl2016BarPotential",
        parameters={
            "alpha": Quantity(0.02, ""),
            "R0": Quantity(8.0, "kpc"),
            "v0": Quantity(220.0, "km/s"),
            "Rb": Quantity(3.5, "kpc"),
            "phi_b": Quantity(25.0, "deg"),
            "Omega": Quantity(40.0, "km/(s kpc)"),
        },
    )
    rescaled = component.rescale(4.0)
    assert rescaled.parameters["Omega"].ustrip("km/(s kpc)") == pytest.approx(40.0)
    assert rescaled.parameters["v0"].ustrip("km/s") == pytest.approx(440.0)
    assert rescaled.parameters["alpha"].ustrip("") == pytest.approx(0.02)


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


def test_lmj09_logarithmic_potential_rescale_scales_v_c_by_sqrt_mass_scale() -> None:
    # Same amplitude role as LogarithmicPotential's v_c
    # (Phi = 0.5 * v_c**2 * ln(r_s**2 + r2), r2 built from the shape
    # parameters q1/q2/q3/phi, which rescale holds fixed).
    component = GalaxPotentialComponent(
        galax_type="LMJ09LogarithmicPotential",
        parameters={
            "v_c": Quantity(200.0, "km/s"),
            "r_s": Quantity(1.0, "kpc"),
            "q1": Quantity(1.0, ""),
            "q2": Quantity(0.9, ""),
            "q3": Quantity(0.8, ""),
            "phi": Quantity(0.0, "rad"),
        },
    )
    rescaled = component.rescale(4.0)
    assert rescaled.parameters["v_c"].ustrip("km/s") == pytest.approx(400.0)

    unit_system = _internal_unit_system()
    xyz = Quantity(jnp.array([3.0, 0.0, 0.0]), "kpc")
    t = Quantity(0.0, "Myr")
    phi_before = component.to_galax(unit_system).potential(xyz, t).ustrip("kpc2 / Myr2")
    phi_after = rescaled.to_galax(unit_system).potential(xyz, t).ustrip("kpc2 / Myr2")
    assert float(phi_after / phi_before) == pytest.approx(4.0)


def test_harmonic_oscillator_rescale_scales_omega_by_sqrt_mass_scale() -> None:
    # No mass-dimensioned native parameter at all (Phi = 0.5*|omega*x|**2,
    # same amplitude role as LogarithmicPotential's v_c); rescale must
    # scale omega so that Phi itself scales linearly with mass_scale.
    component = GalaxPotentialComponent(
        galax_type="HarmonicOscillatorPotential",
        parameters={"omega": Quantity(0.5, "1/Myr")},
    )
    rescaled = component.rescale(4.0)
    assert rescaled.parameters["omega"].ustrip("1/Myr") == pytest.approx(1.0)

    unit_system = _internal_unit_system()
    xyz = Quantity(jnp.array([3.0, 0.0, 0.0]), "kpc")
    t = Quantity(0.0, "Myr")
    phi_before = component.to_galax(unit_system).potential(xyz, t).ustrip("kpc2 / Myr2")
    phi_after = rescaled.to_galax(unit_system).potential(xyz, t).ustrip("kpc2 / Myr2")
    assert float(phi_after / phi_before) == pytest.approx(4.0)


def test_rescale_finds_a_field_inherited_from_an_abstract_parent() -> None:
    # MN3ExponentialPotential doesn't redeclare m_tot itself -- it's a
    # ParameterField on an abstract parent. Regression guard: an earlier
    # version looked parameters up via `galax_cls.__dict__.get(name)`,
    # which only sees a class's own attributes and silently missed
    # anything declared on a parent class.
    cls = gp.MN3ExponentialPotential
    assert "m_tot" not in cls.__dict__
    assert isinstance(getattr(cls, "m_tot", None), ParameterField)
    component = GalaxPotentialComponent(
        galax_type="MN3ExponentialPotential",
        parameters={
            "m_tot": Quantity(1e10, "Msun"),
            "h_R": Quantity(3.0, "kpc"),
            "h_z": Quantity(0.3, "kpc"),
        },
    )
    rescaled = component.rescale(4.0)
    assert rescaled.parameters["m_tot"].ustrip("Msun") == pytest.approx(4e10)
    assert rescaled.parameters["h_R"].ustrip("kpc") == pytest.approx(3.0)


def test_rescale_rejects_an_unsupported_galax_type() -> None:
    # MultipolePotential is excluded from _SUPPORTED_GALAX_TYPES (its
    # required l_max: int hyperparameter isn't representable as a scalar
    # Quantity); rescale() should say so clearly, not KeyError.
    component = GalaxPotentialComponent(
        galax_type="MultipolePotential",
        parameters={"m_tot": Quantity(1e10, "Msun"), "r_s": Quantity(1.0, "kpc")},
    )
    with pytest.raises(NotImplementedError, match="MultipolePotential"):
        component.rescale(2.0)


def test_galax_type_is_a_static_field_under_direct_jit() -> None:
    # galax_type is a plain str, not an array -- without eqx.field(static=True)
    # it's a dynamic PyTree leaf, and jax.jit (unlike eqx.filter_jit, which
    # already excludes non-array leaves) fails tracing a str leaf.
    component = GalaxPotentialComponent(
        galax_type="PlummerPotential",
        parameters={"m_tot": Quantity(1e10, "Msun"), "r_s": Quantity(1.0, "kpc")},
    )

    @jax.jit
    def rescaled_mass(c: GalaxPotentialComponent) -> Quantity:
        return c.rescale(2.0).parameters["m_tot"]

    assert rescaled_mass(component).ustrip("Msun") == pytest.approx(2e10)


def test_raw_parameter_dimensions_covers_all_three_sources() -> None:
    # Native galax type, no parameterization.
    assert raw_parameter_dimensions("PlummerPotential", None) == {
        "m_tot": "mass",
        "r_s": "length",
    }
    # Registered non-native parameterization.
    assert raw_parameter_dimensions("NFWPotential", "concentration_m200") == {
        "M_200": "mass"
    }
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
            _NO_COSMOLOGICAL_PARAMETERS,
            path="potential.dh",
        )


def test_from_settings_rejects_a_real_but_uncurated_galax_class() -> None:
    # MultipolePotential is a real galax.potential class -- unlike
    # test_from_settings_rejects_unrecognized_type's made-up name -- but
    # isn't in _SUPPORTED_GALAX_TYPES (its required l_max: int
    # hyperparameter isn't representable by this module's scalar-Quantity
    # schema). "Any AbstractPotential subclass" would have wrongly
    # accepted this and failed later, confusingly, at to_galax() instead.
    unit_system = _internal_unit_system()
    with pytest.raises(
        ValueError, match="Unsupported potential.dh.type 'MultipolePotential'"
    ):
        AbstractPotentialComponent.from_settings(
            {"type": "MultipolePotential", "include": True, "parameters": {}},
            {},
            unit_system,
            _NO_COSMOLOGICAL_PARAMETERS,
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
        _NO_COSMOLOGICAL_PARAMETERS,
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
            _NO_COSMOLOGICAL_PARAMETERS,
            path="potential.bh",
        )


def test_nfw_concentration_m200_matches_galax_enclosed_mass() -> None:
    # M_200 is defined as the mass enclosed within r_200 = c * r_s; feeding
    # the derived (m, r_s) back into galax's own enclosed-mass formula at
    # that radius should reproduce the M_200 that was put in.
    from galax.potential._src.builtin.nfw.base import mass_enclosed

    unit_system = _internal_unit_system()
    c, m200 = 8.0, 1.0e12
    h0 = Quantity(
        7.158985155319864e-05, "1 / Myr"
    )  # 70 km/s/Mpc, in this unit system's 1/Myr

    component = AbstractPotentialComponent.from_settings(
        {
            "type": "NFWPotential",
            "parameterization": "concentration_m200",
            "include": True,
            "parameters": {"c": {"value": c}, "M_200": {"value": m200}},
        },
        {},
        unit_system,
        {"H0": h0},
        path="potential.dh",
    )
    m = component.parameters["m"].ustrip("Msun")
    r_s = component.parameters["r_s"].ustrip("kpc")
    r200 = r_s * c

    recovered_m200 = float(mass_enclosed({"m": m, "r_s": r_s}, r200))
    assert recovered_m200 == pytest.approx(m200, rel=1e-6)

    # And r_200 itself must enclose a mean density of exactly 200 * rho_crit.
    h0_bare = h0.ustrip("1 / Myr")
    g = float(Quantity(6.6743e-11, "m3 / (kg s2)").ustrip("kpc3 / (Msun Myr2)"))
    rho_crit = 3 * h0_bare**2 / (8 * jnp.pi * g)
    mean_density = m200 / (4 / 3 * jnp.pi * r200**3)
    assert float(mean_density / rho_crit) == pytest.approx(200.0, rel=1e-5)


def test_nfw_concentration_m200_raw_dimensions() -> None:
    assert raw_parameter_dimensions("NFWPotential", "concentration_m200") == {
        "M_200": "mass"
    }


# ---------------------------------------------------------------------------
# concentration_m200's inverse: (m, r_s) -> (c, M_200). No closed form, so
# these check the numerical root-find and the round trip directly, rather
# than against any independently derivable expected value.
# ---------------------------------------------------------------------------


def test_solve_nfw_concentration_recovers_a_known_c() -> None:
    # rel=1e-5, not tighter: this module runs in float32. _nfw_g uses
    # log1p(c), not log(1 + c), which matters down to c ~ 1e-2 -- below
    # that, ln(1+c) - c/(1+c) is a subtraction of two quantities that are
    # themselves both ~ c, so cancellation remains even with log1p (that
    # regime is far below any physically realistic halo concentration
    # anyway, so not worth a cancellation-safe series expansion here).
    for c in (0.01, 0.1, 1.0, 5.0, 8.0, 20.0, 100.0):
        target = c**3 / _nfw_g(c)
        assert float(_solve_nfw_concentration(target)) == pytest.approx(c, rel=1e-5)


def test_nfw_concentration_m200_inverse_round_trips_the_forward_conversion() -> None:
    unit_system = _internal_unit_system()
    h0 = Quantity(
        7.158985155319864e-05, "1 / Myr"
    )  # 70 km/s/Mpc, in this unit system's 1/Myr
    for c, m200 in ((3.0, 1.0e11), (8.0, 1.0e12), (20.0, 5.0e13)):
        raw = {"c": Quantity(c, ""), "M_200": Quantity(m200, "Msun")}
        native = _nfw_concentration_m200(raw, unit_system, {"H0": h0})
        recovered = _nfw_concentration_m200_inverse(native, unit_system, {"H0": h0})
        assert float(recovered["c"].ustrip("")) == pytest.approx(c, rel=1e-5)
        assert float(recovered["M_200"].ustrip("Msun")) == pytest.approx(m200, rel=1e-5)


def test_nfw_concentration_m200_inverse_is_self_consistent_after_rescale() -> None:
    # There's no closed form for (c, M_200) after a mass rescale (rescale()
    # holds r_s fixed and scales only m -- not the same as holding c fixed
    # and scaling M_200), so the only checkable invariant is that inverting
    # and then re-converting forward reproduces the same rescaled (m, r_s).
    unit_system = _internal_unit_system()
    h0 = Quantity(7.158985155319864e-05, "1 / Myr")
    raw = {"c": Quantity(8.0, ""), "M_200": Quantity(1.0e12, "Msun")}
    native = _nfw_concentration_m200(raw, unit_system, {"H0": h0})

    mass_scale = 2.5
    rescaled_native = {"m": native["m"] * mass_scale, "r_s": native["r_s"]}
    recovered_raw = _nfw_concentration_m200_inverse(
        rescaled_native, unit_system, {"H0": h0}
    )
    reconverted_native = _nfw_concentration_m200(recovered_raw, unit_system, {"H0": h0})

    assert reconverted_native["m"].ustrip("Msun") == pytest.approx(
        rescaled_native["m"].ustrip("Msun"), rel=1e-5
    )
    assert reconverted_native["r_s"].ustrip("kpc") == pytest.approx(
        rescaled_native["r_s"].ustrip("kpc"), rel=1e-5
    )
    # And the concentration genuinely changed -- holding c fixed would have
    # been wrong, per the docstring's reasoning.
    assert float(recovered_raw["c"].ustrip("")) != pytest.approx(8.0, rel=1e-3)


# ---------------------------------------------------------------------------
# raw_potential_parameters: reporting a Potential in its own configured
# parameterization, the inverse of Potential.from_settings.
# ---------------------------------------------------------------------------


def test_raw_potential_parameters_uses_each_component_own_parameterization() -> None:
    unit_system = _internal_unit_system()
    h0 = Quantity(7.158985155319864e-05, "1 / Myr")
    settings = {
        "bh": {
            "type": "PlummerPotential",
            "include": True,
            "parameters": {"m_tot": {"value": 5.0}, "r_s": {"value": 1e-3}},
        },
        "dh": {
            "type": "NFWPotential",
            "parameterization": "concentration_m200",
            "include": True,
            "parameters": {"c": {"value": 8.0}, "M_200": {"value": 1.0e12}},
        },
    }
    potential = Potential.from_settings(settings, {}, unit_system, {"H0": h0})

    raw = raw_potential_parameters(settings, potential, unit_system, {"H0": h0})
    assert set(raw["bh"]) == {"m_tot", "r_s"}
    assert set(raw["dh"]) == {"c", "M_200"}
    assert raw["dh"]["c"].ustrip("") == pytest.approx(8.0, rel=1e-5)
    assert raw["dh"]["M_200"].ustrip("Msun") == pytest.approx(1.0e12, rel=1e-5)

    rescaled = potential.rescale(2.0)
    raw_rescaled = raw_potential_parameters(settings, rescaled, unit_system, {"H0": h0})
    assert set(raw_rescaled["dh"]) == {"c", "M_200"}
    assert raw_rescaled["dh"]["M_200"].ustrip("Msun") != pytest.approx(
        raw["dh"]["M_200"].ustrip("Msun"), rel=1e-3
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
        _NO_COSMOLOGICAL_PARAMETERS,
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
        _NO_COSMOLOGICAL_PARAMETERS,
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
    potential = build_potential(settings, {}, unit_system, _NO_COSMOLOGICAL_PARAMETERS)
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
        _NO_COSMOLOGICAL_PARAMETERS,
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
