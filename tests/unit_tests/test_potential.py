"""Unit tests for `tnt.potential`.

`Potential.generate_orbit_library` remains `NotImplementedError` (see
`tnt.potential`'s module docstring); these tests cover what's actually
implemented: dynamic derivation from galax's own `ParameterField` metadata,
type/parameterization resolution, the fully-working Plummer/NFW native-mode
paths, and the two MGE composite types' `to_galax`.
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

from tnt.mge import LightMGE, MassMGE, MGEDeprojectionError
from tnt.potential import (
    _SUPPORTED_GALAX_TYPES,
    AbstractPotentialComponent,
    GalaxPotentialComponent,
    Potential,
    TriaxialLightMGEPotential,
    TriaxialMassMGEPotential,
    _nfw_concentration_m200,
    _nfw_concentration_m200_inverse,
    _nfw_g,
    _solve_nfw_concentration,
    build_potential,
    raw_parameter_dimensions,
    raw_potential_parameters,
)
from tnt.potential.nfw import _newtonian_gravitational_constant


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
        "c": "dimensionless",
        "M_200": "mass",
    }
    # TNT MGE composite type.
    assert raw_parameter_dimensions("TriaxialLightMGEPotential", None) == {
        "ml": "mass_to_light",
        "theta": "angle",
        "phi": "angle",
        "psi": "angle",
    }
    # Unrecognized (type, parameterization) pair -- defer to resolve().
    assert raw_parameter_dimensions("not_a_type", None) == {}


# ---------------------------------------------------------------------------
# Type / parameterization resolution.
# ---------------------------------------------------------------------------


def test_from_settings_rejects_unrecognized_type() -> None:
    with pytest.raises(
        ValueError, match="Unsupported potential.dh.type 'NotAPotential'"
    ):
        AbstractPotentialComponent.resolve(
            {"type": "NotAPotential", "include": True, "parameters": {}},
            {},
            path="potential.dh",
        )


def test_from_settings_rejects_a_real_but_uncurated_galax_class() -> None:
    # MultipolePotential is a real galax.potential class -- unlike
    # test_from_settings_rejects_unrecognized_type's made-up name -- but
    # isn't in _SUPPORTED_GALAX_TYPES (its required l_max: int
    # hyperparameter isn't representable by this module's scalar-Quantity
    # schema). "Any AbstractPotential subclass" would have wrongly
    # accepted this and failed later, confusingly, at to_galax() instead.
    with pytest.raises(
        ValueError, match="Unsupported potential.dh.type 'MultipolePotential'"
    ):
        AbstractPotentialComponent.resolve(
            {"type": "MultipolePotential", "include": True, "parameters": {}},
            {},
            path="potential.dh",
        )


def test_from_settings_resolves_a_real_galax_class_name() -> None:
    unit_system = _internal_unit_system()
    resolved = AbstractPotentialComponent.resolve(
        {"type": "NFWPotential", "include": True, "parameters": {}},
        {},
        path="potential.dh",
    )
    component = resolved.build(
        {"m": Quantity(1e11, "Msun"), "r_s": Quantity(10.0, "kpc")},
        unit_system,
        _NO_COSMOLOGICAL_PARAMETERS,
    )
    assert isinstance(component, GalaxPotentialComponent)
    assert component.galax_type == "NFWPotential"
    assert component.parameters["m"].ustrip("Msun") == pytest.approx(1e11)
    assert component.parameters["r_s"].ustrip("kpc") == pytest.approx(10.0)


def test_from_settings_rejects_unimplemented_parameterization() -> None:
    with pytest.raises(NotImplementedError, match="'bogus' is not implemented"):
        AbstractPotentialComponent.resolve(
            {
                "type": "PlummerPotential",
                "parameterization": "bogus",
                "include": True,
                "parameters": {},
            },
            {},
            path="potential.bh",
        )


def test_nfw_concentration_m200_matches_galax_enclosed_mass() -> None:
    # M_200 is defined as the mass enclosed within r_200 = c * r_s; feeding
    # the derived (m, r_s) back into galax's own enclosed-mass formula at
    # that radius should reproduce the M_200 that was put in.
    from galax.potential._src.builtin.nfw.base import mass_enclosed

    unit_system = _internal_unit_system()
    c, m200 = 8.0, 1.0e12
    h = Quantity(
        7.158985155319864e-05, "1 / Myr"
    )  # 70 km/s/Mpc, in this unit system's 1/Myr

    resolved = AbstractPotentialComponent.resolve(
        {
            "type": "NFWPotential",
            "parameterization": "concentration_m200",
            "include": True,
            "parameters": {},
        },
        {},
        path="potential.dh",
    )
    component = resolved.build(
        {"c": Quantity(c, ""), "M_200": Quantity(m200, "Msun")},
        unit_system,
        {"H": h},
    )
    m = component.parameters["m"].ustrip("Msun")
    r_s = component.parameters["r_s"].ustrip("kpc")
    r200 = r_s * c

    recovered_m200 = float(mass_enclosed({"m": m, "r_s": r_s}, r200))
    assert recovered_m200 == pytest.approx(m200, rel=1e-6)

    # And r_200 itself must enclose a mean density of exactly 200 * rho_crit.
    h_bare = h.ustrip("1 / Myr")
    g = float(Quantity(6.6743e-11, "m3 / (kg s2)").ustrip("kpc3 / (Msun Myr2)"))
    rho_crit = 3 * h_bare**2 / (8 * jnp.pi * g)
    mean_density = m200 / (4 / 3 * jnp.pi * r200**3)
    assert float(mean_density / rho_crit) == pytest.approx(200.0, rel=1e-5)


@pytest.mark.parametrize(
    ("x64_enabled", "expected_dtype"),
    [(False, jnp.dtype("float32")), (True, jnp.dtype("float64"))],
)
def test_nfw_gravitational_constant_follows_active_jax_precision(
    x64_enabled: bool,
    expected_dtype: jnp.dtype,
) -> None:
    with jax.enable_x64(x64_enabled):
        gravitational_constant = _newtonian_gravitational_constant()

    assert gravitational_constant.value.dtype == expected_dtype


def test_nfw_parameterization_is_invariant_to_declared_units() -> None:
    unit_system = _internal_unit_system()
    resolved = AbstractPotentialComponent.resolve(
        {
            "type": "NFWPotential",
            "parameterization": "concentration_m200",
            "include": True,
            "parameters": {},
        },
        {},
        path="potential.dh",
    )
    m200 = Quantity(1.0e12, "Msun")
    h = Quantity(70.0, "km / (s Mpc)")

    internal = resolved.build(
        {"c": Quantity(8.0, ""), "M_200": m200.to("Msun")},
        unit_system,
        {"H": h.to("1 / Myr")},
    )
    differently_declared = resolved.build(
        {
            "c": Quantity(8.0, ""),
            "M_200": Quantity(100.0, "1e10 Msun"),
        },
        unit_system,
        {"H": h},
    )

    assert differently_declared.parameters["m"].ustrip("Msun") == pytest.approx(
        internal.parameters["m"].ustrip("Msun"), rel=1e-6
    )
    assert differently_declared.parameters["r_s"].ustrip("kpc") == pytest.approx(
        internal.parameters["r_s"].ustrip("kpc"), rel=1e-6
    )


def test_nfw_concentration_m200_raw_dimensions() -> None:
    assert raw_parameter_dimensions("NFWPotential", "concentration_m200") == {
        "c": "dimensionless",
        "M_200": "mass",
    }


# ---------------------------------------------------------------------------
# concentration_m200's inverse: (m, r_s) -> (c, M_200). No closed form, so
# these check the numerical root-find and the round trip directly, rather
# than against any independently derivable expected value.
# ---------------------------------------------------------------------------


def test_solve_nfw_concentration_recovers_a_known_c() -> None:
    # _nfw_g uses log1p(c), not log(1 + c), which matters down to c ~ 1e-2.
    # Below that, ln(1+c) - c/(1+c) subtracts two quantities that are both
    # ~ c, so cancellation remains even with log1p. That regime is far below
    # any physically realistic halo concentration and does not warrant a
    # cancellation-safe series expansion here.
    for c in (0.01, 0.1, 1.0, 5.0, 8.0, 20.0, 100.0):
        target = c**3 / _nfw_g(c)
        assert float(_solve_nfw_concentration(target)) == pytest.approx(c, rel=1e-12)


def test_nfw_concentration_m200_inverse_round_trips_the_forward_conversion() -> None:
    unit_system = _internal_unit_system()
    h = Quantity(
        7.158985155319864e-05, "1 / Myr"
    )  # 70 km/s/Mpc, in this unit system's 1/Myr
    for c, m200 in ((3.0, 1.0e11), (8.0, 1.0e12), (20.0, 5.0e13)):
        raw = {"c": Quantity(c, ""), "M_200": Quantity(m200, "Msun")}
        native = _nfw_concentration_m200(raw, unit_system, {"H": h})
        recovered = _nfw_concentration_m200_inverse(native, unit_system, {"H": h})
        assert float(recovered["c"].ustrip("")) == pytest.approx(c, rel=1e-5)
        assert float(recovered["M_200"].ustrip("Msun")) == pytest.approx(m200, rel=1e-5)


def test_nfw_concentration_m200_inverse_is_self_consistent_after_rescale() -> None:
    # There's no closed form for (c, M_200) after a mass rescale (rescale()
    # holds r_s fixed and scales only m -- not the same as holding c fixed
    # and scaling M_200), so the only checkable invariant is that inverting
    # and then re-converting forward reproduces the same rescaled (m, r_s).
    unit_system = _internal_unit_system()
    h = Quantity(7.158985155319864e-05, "1 / Myr")
    raw = {"c": Quantity(8.0, ""), "M_200": Quantity(1.0e12, "Msun")}
    native = _nfw_concentration_m200(raw, unit_system, {"H": h})

    mass_scale = 2.5
    rescaled_native = {"m": native["m"] * mass_scale, "r_s": native["r_s"]}
    recovered_raw = _nfw_concentration_m200_inverse(
        rescaled_native, unit_system, {"H": h}
    )
    reconverted_native = _nfw_concentration_m200(recovered_raw, unit_system, {"H": h})

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
    h = Quantity(7.158985155319864e-05, "1 / Myr")
    settings = {
        "bh": {"type": "PlummerPotential", "include": True, "parameters": {}},
        "dh": {
            "type": "NFWPotential",
            "parameterization": "concentration_m200",
            "include": True,
            "parameters": {},
        },
    }
    parameter_values = {
        "bh": {"m_tot": Quantity(5.0, "Msun"), "r_s": Quantity(1e-3, "kpc")},
        "dh": {"c": Quantity(8.0, ""), "M_200": Quantity(1.0e12, "Msun")},
    }
    potential = Potential.from_settings(
        settings, parameter_values, {}, unit_system, {"H": h}
    )

    raw = raw_potential_parameters(settings, potential, unit_system, {"H": h})
    assert set(raw["bh"]) == {"m_tot", "r_s"}
    assert set(raw["dh"]) == {"c", "M_200"}
    assert raw["dh"]["c"].ustrip("") == pytest.approx(8.0, rel=1e-5)
    assert raw["dh"]["M_200"].ustrip("Msun") == pytest.approx(1.0e12, rel=1e-5)

    rescaled = potential.rescale(2.0)
    raw_rescaled = raw_potential_parameters(settings, rescaled, unit_system, {"H": h})
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
    resolved = AbstractPotentialComponent.resolve(
        {"type": "PlummerPotential", "include": True, "parameters": {}},
        {},
        path="potential.bh",
    )
    component = resolved.build(
        {"m_tot": Quantity(m_tot, "Msun"), "r_s": Quantity(r_s, "kpc")},
        unit_system,
        _NO_COSMOLOGICAL_PARAMETERS,
    )
    galax_potential = component.to_galax(unit_system)

    r = 0.01
    xyz = Quantity(jnp.array([r, 0.0, 0.0]), "kpc")
    t = Quantity(0.0, "Myr")
    value = float(galax_potential.potential(xyz, t).ustrip("kpc2 / Myr2"))

    g = float(u.Quantity(6.6743e-11, "m3 / (kg s2)").ustrip("kpc3 / (Msun Myr2)"))
    expected = _closed_form_plummer_potential(m_tot, r_s, r, g)
    assert value == pytest.approx(expected, rel=1e-5)


def test_plummer_potential_is_invariant_to_declared_parameter_units() -> None:
    unit_system = _internal_unit_system()
    settings = {"bh": {"type": "PlummerPotential", "include": True, "parameters": {}}}
    resolved = Potential.resolve(settings, {})
    mass = Quantity(5.0, "Msun")

    internal = Potential.build(
        resolved,
        {"bh": {"m_tot": mass, "r_s": Quantity(1.0, "kpc")}},
        unit_system,
        _NO_COSMOLOGICAL_PARAMETERS,
    ).to_galax(unit_system)
    differently_declared = Potential.build(
        resolved,
        {
            "bh": {
                "m_tot": Quantity(float(mass.ustrip("kg")), "kg"),
                "r_s": Quantity(1000.0, "pc"),
            }
        },
        unit_system,
        _NO_COSMOLOGICAL_PARAMETERS,
    ).to_galax(unit_system)

    xyz = Quantity(jnp.array([2.0, 0.0, 0.0]), "kpc")
    t = Quantity(0.0, "Myr")
    internal_value = internal.potential(xyz, t).ustrip("kpc2 / Myr2")
    differently_declared_value = differently_declared.potential(xyz, t).ustrip(
        "kpc2 / Myr2"
    )

    assert differently_declared_value == pytest.approx(internal_value, rel=1e-6)


def test_plummer_rescale_scales_only_the_mass_parameter() -> None:
    unit_system = _internal_unit_system()
    resolved = AbstractPotentialComponent.resolve(
        {"type": "PlummerPotential", "include": True, "parameters": {}},
        {},
        path="potential.bh",
    )
    component = resolved.build(
        {"m_tot": Quantity(5.0, "Msun"), "r_s": Quantity(1e-3, "kpc")},
        unit_system,
        _NO_COSMOLOGICAL_PARAMETERS,
    )
    rescaled = component.rescale(2.0)
    assert rescaled.parameters["m_tot"].ustrip("Msun") == pytest.approx(10.0)
    assert rescaled.parameters["r_s"].ustrip("kpc") == pytest.approx(1e-3)


def test_potential_composes_only_included_components() -> None:
    unit_system = _internal_unit_system()
    settings = {
        "bh": {"type": "PlummerPotential", "include": True, "parameters": {}},
        "excluded": {
            "type": "PlummerPotential",
            "include": False,
            "parameters": {},
        },
    }
    parameter_values = {
        "bh": {"m_tot": Quantity(5.0, "Msun"), "r_s": Quantity(1e-3, "kpc")},
        "excluded": {"m_tot": Quantity(100.0, "Msun"), "r_s": Quantity(1.0, "kpc")},
    }
    resolved = Potential.resolve(settings, {})
    potential = build_potential(
        resolved, parameter_values, unit_system, _NO_COSMOLOGICAL_PARAMETERS
    )
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
# MGE composite types: from_settings resolution and to_galax.
#
# The basic to_galax viewing-angle tests below deliberately use q=1
# (circular) components with a known nonsingular viewing geometry. They
# deproject to an exactly spherical Deprojected3DMGE (p = q = 1), which lets
# these tests isolate to_galax's wiring (mass-to-light/mass-scale conversion,
# deprojection, per-component TriaxialGaussianPotential construction, and
# CompositePotential summation) against galax's independent GaussianPotential.
# Invalid geometries now raise MGEDeprojectionError during component build;
# genuinely triaxial validity and axis mapping are covered separately below
# and by test_mge.py's forward-projection round-trip tests.
# ---------------------------------------------------------------------------


def _circular_light_mge(I: list[float], sigma: list[float]) -> LightMGE:
    return LightMGE(
        I=Quantity(jnp.array(I), "Lsun / rad2"),
        sigma=Quantity(jnp.array(sigma), "rad"),
        q=Quantity(jnp.ones(len(I)), ""),
        PA_twist=Quantity(jnp.zeros(len(I)), "rad"),
    )


_VIEWING_ANGLES = {
    "theta": Quantity(1.1, "rad"),
    "phi": Quantity(0.6, "rad"),
    "psi": Quantity(-0.4, "rad"),
}


def test_mge_component_resolve_and_build_stores_the_referenced_mge() -> None:
    light_mge = _circular_light_mge([1.0], [1.0]).angular_to_physical(
        Quantity(30.0, "Mpc")
    )
    resolved = AbstractPotentialComponent.resolve(
        {
            "type": "TriaxialLightMGEPotential",
            "include": True,
            "mge": "mge_lum",
            "parameters": {},
        },
        {"mge_lum": light_mge},
        path="potential.stars",
    )
    component = resolved.build(
        {"ml": Quantity(5.0, "Msun / Lsun"), **_VIEWING_ANGLES},
        _internal_unit_system(),
        _NO_COSMOLOGICAL_PARAMETERS,
    )
    assert isinstance(component, TriaxialLightMGEPotential)
    assert component.mge is light_mge
    assert component.parameters["ml"].ustrip("Msun / Lsun") == pytest.approx(5.0)


def test_mge_component_build_raises_for_invalid_geometry_not_to_galax() -> None:
    # theta=0.3, phi=0.96, psi=0.1 with q_obs=0.9 is a known finite-but-
    # convention-violating deprojection (q > p), see
    # test_deproject_triaxial_convention_violating_geometry_raises in
    # tests/unit_tests/test_mge.py.
    light_mge = LightMGE(
        I=Quantity(jnp.array([2.0]), "Lsun / rad2"),
        sigma=Quantity(jnp.array([1.5]), "rad"),
        q=Quantity(jnp.array([0.9]), ""),
        PA_twist=Quantity(jnp.array([0.0]), "rad"),
    ).angular_to_physical(Quantity(30.0, "Mpc"))
    resolved = AbstractPotentialComponent.resolve(
        {
            "type": "TriaxialLightMGEPotential",
            "include": True,
            "mge": "mge_lum",
            "parameters": {},
        },
        {"mge_lum": light_mge},
        path="potential.stars",
    )
    bad_angles = {
        "theta": Quantity(0.3, "rad"),
        "phi": Quantity(0.96, "rad"),
        "psi": Quantity(0.1, "rad"),
    }
    # Same theta/phi, but psi=-1.08 -- numerically confirmed valid (finite,
    # 0 < q <= p <= 1) for this q_obs=0.9 MGE.
    good_angles = {**bad_angles, "psi": Quantity(-1.08, "rad")}

    # Raises from build() itself, before to_galax() is ever reached.
    with pytest.raises(MGEDeprojectionError):
        resolved.build(
            {"ml": Quantity(5.0, "Msun / Lsun"), **bad_angles},
            _internal_unit_system(),
            _NO_COSMOLOGICAL_PARAMETERS,
        )

    # A component that *did* build successfully can't have to_galax() raise
    # it -- deprojection already happened, and was already validated, at
    # build time.
    component = resolved.build(
        {"ml": Quantity(5.0, "Msun / Lsun"), **good_angles},
        _internal_unit_system(),
        _NO_COSMOLOGICAL_PARAMETERS,
    )
    component.to_galax(_internal_unit_system())


def test_triaxial_light_mge_to_galax_matches_spherical_gaussian() -> None:
    unit_system = _internal_unit_system()
    distance = Quantity(30.0, "Mpc")
    light_mge = _circular_light_mge([2.0], [1.5]).angular_to_physical(distance)
    ml = Quantity(5.0, "Msun / Lsun")
    component = TriaxialLightMGEPotential._build(
        {"ml": ml, **_VIEWING_ANGLES},
        unit_system,
        _NO_COSMOLOGICAL_PARAMETERS,
        {"mge": light_mge},
    )

    potential = component.to_galax(unit_system)

    mass_mge = light_mge.to_mass(ml)
    deprojected = mass_mge.deproject_triaxial(**_VIEWING_ANGLES)
    assert deprojected.p.ustrip("") == pytest.approx(1.0)
    assert deprojected.q.ustrip("") == pytest.approx(1.0)
    m_tot = (
        deprojected.I[0]
        * deprojected.p[0]
        * deprojected.q[0]
        * (2 * jnp.pi) ** 1.5
        * deprojected.sigma[0] ** 3
    )
    reference = gp.GaussianPotential(
        m_tot=m_tot, r_s=deprojected.sigma[0], units=unit_system
    )

    speed2 = unit_system["length"] ** 2 / unit_system["time"] ** 2
    for xyz in (
        Quantity(jnp.array([3.0, 0.5, -1.0]), "kpc"),
        Quantity(jnp.array([50.0, -20.0, 8.0]), "kpc"),
    ):
        t = Quantity(0.0, "Myr")
        r = jnp.sqrt(jnp.sum(xyz.ustrip("kpc") ** 2))
        radial_xyz = Quantity(jnp.array([r, 0.0, 0.0]), "kpc")
        assert potential.potential(radial_xyz, t).ustrip(speed2) == pytest.approx(
            reference.potential(radial_xyz, t).ustrip(speed2), rel=1e-5
        )


def test_triaxial_light_mge_to_galax_sums_every_component() -> None:
    unit_system = _internal_unit_system()
    distance = Quantity(30.0, "Mpc")
    light_mge = _circular_light_mge([2.0, 0.5], [1.5, 4.0]).angular_to_physical(
        distance
    )
    ml = Quantity(5.0, "Msun / Lsun")
    component = TriaxialLightMGEPotential._build(
        {"ml": ml, **_VIEWING_ANGLES},
        unit_system,
        _NO_COSMOLOGICAL_PARAMETERS,
        {"mge": light_mge},
    )

    potential = component.to_galax(unit_system)

    deprojected = light_mge.to_mass(ml).deproject_triaxial(**_VIEWING_ANGLES)
    speed2 = unit_system["length"] ** 2 / unit_system["time"] ** 2
    xyz = Quantity(jnp.array([3.0, 0.5, -1.0]), "kpc")
    t = Quantity(0.0, "Myr")
    individual_sum = Quantity(0.0, speed2)
    for i in range(2):
        m_tot = (
            deprojected.I[i]
            * deprojected.p[i]
            * deprojected.q[i]
            * (2 * jnp.pi) ** 1.5
            * deprojected.sigma[i] ** 3
        )
        component_i = gp.TriaxialGaussianPotential(
            m_tot=m_tot,
            r_s=deprojected.sigma[i],
            q1=deprojected.p[i],
            q2=deprojected.q[i],
            units=unit_system,
        )
        individual_sum = individual_sum + component_i.potential(xyz, t)

    assert potential.potential(xyz, t).ustrip(speed2) == pytest.approx(
        individual_sum.ustrip(speed2), rel=1e-5
    )


def test_triaxial_mass_mge_to_galax_uses_mge_mass_scale() -> None:
    unit_system = _internal_unit_system()
    distance = Quantity(30.0, "Mpc")
    mass_mge = MassMGE(
        I=Quantity(jnp.array([1e2]), "Msun / rad2"),
        sigma=Quantity(jnp.array([1.5]), "rad"),
        q=Quantity(jnp.array([1.0]), ""),
        PA_twist=Quantity(jnp.array([0.0]), "rad"),
    ).angular_to_physical(distance)
    mge_mass_scale = Quantity(3.0, "")
    component = TriaxialMassMGEPotential._build(
        {"mge_mass_scale": mge_mass_scale, **_VIEWING_ANGLES},
        unit_system,
        _NO_COSMOLOGICAL_PARAMETERS,
        {"mge": mass_mge},
    )

    potential = component.to_galax(unit_system)

    scaled = mass_mge.rescaled(mge_mass_scale)
    deprojected = scaled.deproject_triaxial(**_VIEWING_ANGLES)
    m_tot = (
        deprojected.I[0]
        * deprojected.p[0]
        * deprojected.q[0]
        * (2 * jnp.pi) ** 1.5
        * deprojected.sigma[0] ** 3
    )
    reference = gp.GaussianPotential(
        m_tot=m_tot, r_s=deprojected.sigma[0], units=unit_system
    )

    speed2 = unit_system["length"] ** 2 / unit_system["time"] ** 2
    xyz = Quantity(jnp.array([4.0, 0.0, 0.0]), "kpc")
    t = Quantity(0.0, "Myr")
    assert potential.potential(xyz, t).ustrip(speed2) == pytest.approx(
        reference.potential(xyz, t).ustrip(speed2), rel=1e-5
    )


def test_potential_generate_orbit_library_not_implemented() -> None:
    potential = Potential(components={})
    with pytest.raises(NotImplementedError):
        potential.generate_orbit_library({}, None, None)


def test_triaxial_light_mge_to_galax_density_matches_analytic_along_each_axis() -> None:
    # The existing to_galax tests deliberately use circular Gaussians
    # (q_obs=1, so p=q=1 always), which can't detect a swapped p<->q1 or
    # q<->q2 mapping -- x/y/z all look the same. This uses a genuinely
    # non-spherical, non-axisymmetric deprojection (p != q != 1, both
    # numerically confirmed valid in test_mge.py) and checks density -- not
    # potential -- independently along all three intrinsic axes against the
    # closed-form Gaussian, which only matches if q1/q2 land on the right
    # axes.
    unit_system = _internal_unit_system()
    mge = LightMGE(
        I=Quantity(jnp.array([5.0]), "Lsun / kpc2"),
        sigma=Quantity(jnp.array([2.0]), "kpc"),
        q=Quantity(jnp.array([0.9]), ""),
        PA_twist=Quantity(jnp.array([-1.0]), "rad"),
    )
    angles = {
        "theta": Quantity(0.3, "rad"),
        "phi": Quantity(0.96, "rad"),
        "psi": Quantity(0.0, "rad"),
    }
    ml = Quantity(5.0, "Msun / Lsun")
    component = TriaxialLightMGEPotential._build(
        {"ml": ml, **angles}, unit_system, _NO_COSMOLOGICAL_PARAMETERS, {"mge": mge}
    )
    potential = component.to_galax(unit_system)

    p = float(component.deprojected.p.ustrip("")[0])
    q = float(component.deprojected.q.ustrip("")[0])
    assert p != pytest.approx(q)  # genuinely triaxial, not axisymmetric
    sigma = float(component.deprojected.sigma.ustrip("kpc")[0])
    rho_0 = float(component.deprojected.I.ustrip("Msun / kpc3")[0])

    t = Quantity(0.0, "Myr")
    density_unit = unit_system["mass"] / unit_system["length"] ** 3
    for axis_index, axial_ratio in ((0, 1.0), (1, p), (2, q)):
        xyz = [0.0, 0.0, 0.0]
        xyz[axis_index] = 3.0
        analytic = rho_0 * jnp.exp(-0.5 * (3.0 / (axial_ratio * sigma)) ** 2)
        density = potential.density(Quantity(jnp.array(xyz), "kpc"), t)
        assert density.ustrip(density_unit) == pytest.approx(analytic, rel=1e-5)


def test_triaxial_light_mge_rescale_scales_to_galax_without_ml_recompute() -> None:
    # rescale() should scale to_galax()'s output by mass_scale without
    # re-deriving the deprojection (p/q/sigma are mass-invariant; only
    # deprojected.I scales) -- confirms both that the potential value itself
    # scales correctly, and that shape is preserved.
    unit_system = _internal_unit_system()
    distance = Quantity(30.0, "Mpc")
    light_mge = _circular_light_mge([2.0], [1.5]).angular_to_physical(distance)
    ml = Quantity(5.0, "Msun / Lsun")
    component = TriaxialLightMGEPotential._build(
        {"ml": ml, **_VIEWING_ANGLES},
        unit_system,
        _NO_COSMOLOGICAL_PARAMETERS,
        {"mge": light_mge},
    )

    mass_scale = 2.0
    rescaled = component.rescale(mass_scale)

    assert rescaled.parameters["ml"].ustrip("Msun / Lsun") == pytest.approx(
        ml.ustrip("Msun / Lsun") * mass_scale
    )
    assert rescaled.deprojected.p.ustrip("") == pytest.approx(
        component.deprojected.p.ustrip("")
    )
    assert rescaled.deprojected.q.ustrip("") == pytest.approx(
        component.deprojected.q.ustrip("")
    )

    speed2 = unit_system["length"] ** 2 / unit_system["time"] ** 2
    xyz = Quantity(jnp.array([3.0, 0.5, -1.0]), "kpc")
    t = Quantity(0.0, "Myr")
    original_potential = component.to_galax(unit_system).potential(xyz, t)
    rescaled_potential = rescaled.to_galax(unit_system).potential(xyz, t)
    assert rescaled_potential.ustrip(speed2) == pytest.approx(
        original_potential.ustrip(speed2) * mass_scale, rel=1e-5
    )
