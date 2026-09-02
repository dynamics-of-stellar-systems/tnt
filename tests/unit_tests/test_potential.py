"""Unit tests for `tnt.potential`.

`Potential.generate_orbit_library` remains `NotImplementedError` (see
`tnt.potential`'s module docstring); these tests cover what's actually
implemented: dynamic derivation from galax's own `ParameterField` metadata,
type/parameterization resolution, the fully-working Plummer/NFW native-mode
paths, and the four MGE composite types' `to_galax`.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

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
    OblateLightMGEPotential,
    OblateMassMGEPotential,
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
from tnt.potential import registry as _registry_module
from tnt.potential.nfw import _newtonian_gravitational_constant
from tnt.potential.registry import (
    _COMPONENT_REGISTRY,
    ParameterConstraint,
    parameter_constraints,
    register_component,
)
from tnt.potential.triaxial_mge import _pqu_to_tpp, _tpp_to_pqu


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
    return u.unitsystem("kpc", "Myr", "Msun", "rad")


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
    # ParameterField on an abstract parent. Normal attribute lookup is required
    # because direct inspection of the subclass's own `__dict__` cannot see
    # fields declared on a parent class.
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
            {"type": "NotAPotential", "parameters": {}},
            {},
            path="potential.dh",
        )


def test_from_settings_rejects_a_real_but_uncurated_galax_class() -> None:
    # MultipolePotential is a real galax.potential class -- unlike
    # test_from_settings_rejects_unrecognized_type's made-up name -- but
    # isn't in _SUPPORTED_GALAX_TYPES (its required l_max: int
    # hyperparameter isn't representable by this module's scalar-Quantity
    # schema). Curating supported classes makes this fail clearly during
    # resolution instead of deferring the error to to_galax().
    with pytest.raises(
        ValueError, match="Unsupported potential.dh.type 'MultipolePotential'"
    ):
        AbstractPotentialComponent.resolve(
            {"type": "MultipolePotential", "parameters": {}},
            {},
            path="potential.dh",
        )


def test_from_settings_resolves_a_real_galax_class_name() -> None:
    resolved = AbstractPotentialComponent.resolve(
        {"type": "NFWPotential", "parameters": {}},
        {},
        path="potential.dh",
    )
    component = resolved.build(
        {"m": Quantity(1e11, "Msun"), "r_s": Quantity(10.0, "kpc")},
        _NO_COSMOLOGICAL_PARAMETERS,
    )
    assert isinstance(component, GalaxPotentialComponent)
    assert component.galax_type == "NFWPotential"
    assert component.parameters["m"].ustrip("Msun") == pytest.approx(1e11)
    assert component.parameters["r_s"].ustrip("kpc") == pytest.approx(10.0)


@pytest.mark.parametrize("invalid", [0.0, -1.0, jnp.nan, jnp.inf])
def test_native_parameter_domain_rejects_nonpositive_or_nonfinite_mass(
    invalid: float,
) -> None:
    resolved = AbstractPotentialComponent.resolve(
        {"type": "PlummerPotential", "parameters": {}},
        {},
        path="potential.bh",
    )
    message = "must be finite" if not jnp.isfinite(invalid) else "greater than"
    with pytest.raises(
        ValueError, match=rf"potential\.bh\.parameters\.m_tot.*{message}"
    ):
        resolved.build(
            {"m_tot": Quantity(invalid, "Msun"), "r_s": Quantity(1.0, "kpc")},
            _NO_COSMOLOGICAL_PARAMETERS,
        )


def test_runtime_parameter_domain_requires_scalar_quantities_and_exact_names() -> None:
    resolved = AbstractPotentialComponent.resolve(
        {"type": "PlummerPotential", "parameters": {}},
        {},
        path="potential.bh",
    )
    with pytest.raises(ValueError, match=r"expected a scalar, got shape \(1,\)"):
        resolved.build(
            {
                "m_tot": Quantity(jnp.array([1.0]), "Msun"),
                "r_s": Quantity(1.0, "kpc"),
            },
            _NO_COSMOLOGICAL_PARAMETERS,
        )
    with pytest.raises(ValueError, match=r"missing \['r_s'\].*unexpected \['x'\]"):
        resolved.build(
            {"m_tot": Quantity(1.0, "Msun"), "x": Quantity(1.0, "kpc")},
            _NO_COSMOLOGICAL_PARAMETERS,
        )
    with pytest.raises(ValueError, match=r"parameters\.r_s must describe length"):
        resolved.build(
            {"m_tot": Quantity(1.0, "Msun"), "r_s": Quantity(1.0, "s")},
            _NO_COSMOLOGICAL_PARAMETERS,
        )


@pytest.mark.parametrize(
    ("galax_type", "parameters", "parameter_name"),
    [
        (
            "PowerLawCutoffPotential",
            {
                "m_tot": Quantity(1.0, "Msun"),
                "alpha": Quantity(3.0, ""),
                "r_c": Quantity(1.0, "kpc"),
            },
            "alpha",
        ),
        (
            "gNFWPotential",
            {
                "m": Quantity(1.0, "Msun"),
                "r_s": Quantity(1.0, "kpc"),
                "gamma": Quantity(2.0, ""),
            },
            "gamma",
        ),
        (
            "Vogelsberger08TriaxialNFWPotential",
            {
                "m": Quantity(1.0, "Msun"),
                "r_s": Quantity(1.0, "kpc"),
                "q1": Quantity(3**0.5, ""),
                "a_r": Quantity(1.0, ""),
            },
            "q1",
        ),
    ],
)
def test_analytic_profile_parameter_bounds_are_enforced(
    galax_type: str,
    parameters: dict[str, Quantity],
    parameter_name: str,
) -> None:
    resolved = AbstractPotentialComponent.resolve(
        {"type": galax_type, "parameters": {}},
        {},
        path="potential.halo",
    )
    with pytest.raises(
        ValueError,
        match=rf"potential\.halo\.parameters\.{parameter_name}.*less than",
    ):
        resolved.build(parameters, _NO_COSMOLOGICAL_PARAMETERS)


def test_same_component_relationship_uses_compatible_declared_units() -> None:
    resolved = AbstractPotentialComponent.resolve(
        {"type": "StoneOstriker15Potential", "parameters": {}},
        {},
        path="potential.cluster",
    )
    with pytest.raises(ValueError, match=r"r_h.*must be >.*r_c"):
        resolved.build(
            {
                "m_tot": Quantity(1.0e6, "Msun"),
                "r_c": Quantity(1.0, "kpc"),
                "r_h": Quantity(900.0, "pc"),
            },
            _NO_COSMOLOGICAL_PARAMETERS,
        )


def test_leesuto_axis_order_is_enforced() -> None:
    resolved = AbstractPotentialComponent.resolve(
        {"type": "LeeSutoTriaxialNFWPotential", "parameters": {}},
        {},
        path="potential.halo",
    )
    with pytest.raises(ValueError, match=r"a1.*must be >=.*a2"):
        resolved.build(
            {
                "m": Quantity(1.0e11, "Msun"),
                "r_s": Quantity(10.0, "kpc"),
                "a1": Quantity(0.8, ""),
                "a2": Quantity(1.0, ""),
                "a3": Quantity(0.7, ""),
            },
            _NO_COSMOLOGICAL_PARAMETERS,
        )


def test_frequency_amplitude_domain_and_signed_pattern_speed_are_distinct() -> None:
    oscillator = AbstractPotentialComponent.resolve(
        {"type": "HarmonicOscillatorPotential", "parameters": {}},
        {},
        path="potential.core",
    )
    with pytest.raises(ValueError, match=r"omega.*greater than"):
        oscillator.build(
            {"omega": Quantity(0.0, "1 / Myr")},
            _NO_COSMOLOGICAL_PARAMETERS,
        )

    bar = AbstractPotentialComponent.resolve(
        {"type": "MonariEtAl2016BarPotential", "parameters": {}},
        {},
        path="potential.bar",
    )
    component = bar.build(
        {
            "alpha": Quantity(-0.02, ""),
            "R0": Quantity(8.0, "kpc"),
            "v0": Quantity(220.0, "km / s"),
            "Rb": Quantity(3.5, "kpc"),
            "phi_b": Quantity(25.0, "deg"),
            "Omega": Quantity(-40.0, "km / (s kpc)"),
        },
        _NO_COSMOLOGICAL_PARAMETERS,
    )
    assert component.parameters["Omega"].ustrip("km / (s kpc)") == -40.0


def test_from_settings_rejects_unimplemented_parameterization() -> None:
    with pytest.raises(NotImplementedError, match="'bogus' is not implemented"):
        AbstractPotentialComponent.resolve(
            {
                "type": "PlummerPotential",
                "parameterization": "bogus",
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

    c, m200 = 8.0, 1.0e12
    h = Quantity(
        7.158985155319864e-05, "1 / Myr"
    )  # 70 km/s/Mpc, in this unit system's 1/Myr

    resolved = AbstractPotentialComponent.resolve(
        {
            "type": "NFWPotential",
            "parameterization": "concentration_m200",
            "parameters": {},
        },
        {},
        path="potential.dh",
    )
    component = resolved.build(
        {"c": Quantity(c, ""), "M_200": Quantity(m200, "Msun")},
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


@pytest.mark.parametrize("name", ["c", "M_200"])
def test_nfw_parameterization_rejects_invalid_raw_values_before_conversion(
    name: str,
) -> None:
    resolved = AbstractPotentialComponent.resolve(
        {
            "type": "NFWPotential",
            "parameterization": "concentration_m200",
            "parameters": {},
        },
        {},
        path="potential.halo",
    )
    raw = {"c": Quantity(8.0, ""), "M_200": Quantity(1.0e12, "Msun")}
    raw[name] = Quantity(0.0, raw[name].unit)
    # Empty cosmology proves the raw-domain error occurs before the converter
    # tries to read its required H value.
    with pytest.raises(ValueError, match=rf"parameters\.{name}.*greater than"):
        resolved.build(raw, {})


def test_nfw_parameterization_validates_converted_native_values() -> None:
    resolved = AbstractPotentialComponent.resolve(
        {
            "type": "NFWPotential",
            "parameterization": "concentration_m200",
            "parameters": {},
        },
        {},
        path="potential.halo",
    )
    with pytest.raises(ValueError, match=r"converted.*parameters\.r_s.*finite"):
        resolved.build(
            {"c": Quantity(8.0, ""), "M_200": Quantity(1.0e12, "Msun")},
            {"H": Quantity(0.0, "km / (s Mpc)")},
        )


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
    resolved = AbstractPotentialComponent.resolve(
        {
            "type": "NFWPotential",
            "parameterization": "concentration_m200",
            "parameters": {},
        },
        {},
        path="potential.dh",
    )
    m200 = Quantity(1.0e12, "Msun")
    h = Quantity(70.0, "km / (s Mpc)")

    internal = resolved.build(
        {"c": Quantity(8.0, ""), "M_200": m200.to("Msun")},
        {"H": h.to("1 / Myr")},
    )
    differently_declared = resolved.build(
        {
            "c": Quantity(8.0, ""),
            "M_200": Quantity(100.0, "1e10 Msun"),
        },
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
    h = Quantity(
        7.158985155319864e-05, "1 / Myr"
    )  # 70 km/s/Mpc, in this unit system's 1/Myr
    for c, m200 in ((3.0, 1.0e11), (8.0, 1.0e12), (20.0, 5.0e13)):
        raw = {"c": Quantity(c, ""), "M_200": Quantity(m200, "Msun")}
        native = _nfw_concentration_m200(raw, {"H": h})
        recovered = _nfw_concentration_m200_inverse(native, {"M_200": "Msun"}, {"H": h})
        assert float(recovered["c"].ustrip("")) == pytest.approx(c, rel=1e-5)
        assert float(recovered["M_200"].ustrip("Msun")) == pytest.approx(m200, rel=1e-5)


def test_nfw_concentration_m200_inverse_is_self_consistent_after_rescale() -> None:
    # There's no closed form for (c, M_200) after a mass rescale (rescale()
    # holds r_s fixed and scales only m -- not the same as holding c fixed
    # and scaling M_200), so the only checkable invariant is that inverting
    # and then re-converting forward reproduces the same rescaled (m, r_s).
    h = Quantity(7.158985155319864e-05, "1 / Myr")
    raw = {"c": Quantity(8.0, ""), "M_200": Quantity(1.0e12, "Msun")}
    native = _nfw_concentration_m200(raw, {"H": h})

    mass_scale = 2.5
    rescaled_native = {"m": native["m"] * mass_scale, "r_s": native["r_s"]}
    recovered_raw = _nfw_concentration_m200_inverse(
        rescaled_native, {"M_200": "Msun"}, {"H": h}
    )
    reconverted_native = _nfw_concentration_m200(recovered_raw, {"H": h})

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
    h = Quantity(7.158985155319864e-05, "1 / Myr")
    settings = {
        "bh": {"type": "PlummerPotential", "parameters": {}},
        "dh": {
            "type": "NFWPotential",
            "parameterization": "concentration_m200",
            "parameters": {"M_200": {"unit": "Msun"}},
        },
    }
    parameter_values = {
        "bh": {"m_tot": Quantity(5.0, "Msun"), "r_s": Quantity(1e-3, "kpc")},
        "dh": {"c": Quantity(8.0, ""), "M_200": Quantity(1.0e12, "Msun")},
    }
    potential = Potential.from_settings(settings, parameter_values, {}, {"H": h})

    raw = raw_potential_parameters(settings, potential, {"H": h})
    assert set(raw["bh"]) == {"m_tot", "r_s"}
    assert set(raw["dh"]) == {"c", "M_200"}
    assert raw["dh"]["c"].ustrip("") == pytest.approx(8.0, rel=1e-5)
    assert raw["dh"]["M_200"].ustrip("Msun") == pytest.approx(1.0e12, rel=1e-5)

    rescaled = potential.rescale(2.0)
    raw_rescaled = raw_potential_parameters(settings, rescaled, {"H": h})
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
        {"type": "PlummerPotential", "parameters": {}},
        {},
        path="potential.bh",
    )
    component = resolved.build(
        {"m_tot": Quantity(m_tot, "Msun"), "r_s": Quantity(r_s, "kpc")},
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
    settings = {"bh": {"type": "PlummerPotential", "parameters": {}}}
    resolved = Potential.resolve(settings, {})
    mass = Quantity(5.0, "Msun")

    internal = Potential.build(
        resolved,
        {"bh": {"m_tot": mass, "r_s": Quantity(1.0, "kpc")}},
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
    resolved = AbstractPotentialComponent.resolve(
        {"type": "PlummerPotential", "parameters": {}},
        {},
        path="potential.bh",
    )
    component = resolved.build(
        {"m_tot": Quantity(5.0, "Msun"), "r_s": Quantity(1e-3, "kpc")},
        _NO_COSMOLOGICAL_PARAMETERS,
    )
    rescaled = component.rescale(2.0)
    assert rescaled.parameters["m_tot"].ustrip("Msun") == pytest.approx(10.0)
    assert rescaled.parameters["r_s"].ustrip("kpc") == pytest.approx(1e-3)


def test_potential_composes_every_declared_component() -> None:
    unit_system = _internal_unit_system()
    settings = {
        "bh": {"type": "PlummerPotential", "parameters": {}},
        "halo": {"type": "PlummerPotential", "parameters": {}},
    }
    parameter_values = {
        "bh": {"m_tot": Quantity(5.0, "Msun"), "r_s": Quantity(1e-3, "kpc")},
        "halo": {"m_tot": Quantity(100.0, "Msun"), "r_s": Quantity(1.0, "kpc")},
    }
    resolved = Potential.resolve(settings, {})
    potential = build_potential(resolved, parameter_values, _NO_COSMOLOGICAL_PARAMETERS)
    assert set(potential.components) == {"bh", "halo"}

    galax_potential = potential.to_galax(unit_system)
    xyz = Quantity(jnp.array([0.01, 0.0, 0.0]), "kpc")
    t = Quantity(0.0, "Myr")
    composed_value = float(galax_potential.potential(xyz, t).ustrip("kpc2 / Myr2"))
    summed_components = sum(
        float(
            potential.components[name]
            .to_galax(unit_system)
            .potential(xyz, t)
            .ustrip("kpc2 / Myr2")
        )
        for name in ("bh", "halo")
    )
    assert composed_value == pytest.approx(summed_components)


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
# Invalid geometries raise MGEDeprojectionError during component build;
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


def test_register_component_stores_the_class_under_its_own_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An isolated fake registry, swapped in for the module-level real one --
    # _COMPONENT_REGISTRY is shared, mutable, global state, so a test that
    # registers into it directly must not leak into the real registry other
    # tests (and application code) read from.
    fake_registry: dict[str, type] = {}
    monkeypatch.setattr(_registry_module, "_COMPONENT_REGISTRY", fake_registry)

    class _Test:
        _type: ClassVar[str] = "_test_potential_type"
        _raw_dimensions: ClassVar[dict[str, str]] = {"m": "mass"}

    result = register_component(_Test)

    assert result is _Test
    assert fake_registry == {"_test_potential_type": _Test}


def test_register_component_rejects_duplicate_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_registry_module, "_COMPONENT_REGISTRY", {})

    class _First:
        _type: ClassVar[str] = "_test_duplicate_potential_type"
        _raw_dimensions: ClassVar[dict[str, str]] = {}

    class _Second:
        _type: ClassVar[str] = "_test_duplicate_potential_type"
        _raw_dimensions: ClassVar[dict[str, str]] = {}

    register_component(_First)
    with pytest.raises(ValueError, match="Duplicate potential type"):
        register_component(_Second)


def test_undecorated_child_with_inherited_type_is_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subclass that inherits `_type` without its own `@register_component`
    call must not participate in dispatch under that name -- and, since
    registration is explicit, there is no ambiguity about whether the child
    is a second registered type.
    """
    monkeypatch.setattr(_registry_module, "_COMPONENT_REGISTRY", {})

    class _Parent:
        _type: ClassVar[str] = "_test_inherited_potential_type"
        _raw_dimensions: ClassVar[dict[str, str]] = {}

    class _Child(_Parent):
        pass  # inherits _type, never itself passed to register_component

    register_component(_Parent)

    assert _registry_module._COMPONENT_REGISTRY == {
        "_test_inherited_potential_type": _Parent
    }


def test_component_registry_contains_the_four_mge_types() -> None:
    assert set(_COMPONENT_REGISTRY) == {
        "TriaxialLightMGEPotential",
        "TriaxialMassMGEPotential",
        "OblateLightMGEPotential",
        "OblateMassMGEPotential",
    }
    assert _COMPONENT_REGISTRY["TriaxialLightMGEPotential"] is TriaxialLightMGEPotential
    assert _COMPONENT_REGISTRY["TriaxialMassMGEPotential"] is TriaxialMassMGEPotential
    assert _COMPONENT_REGISTRY["OblateLightMGEPotential"] is OblateLightMGEPotential
    assert _COMPONENT_REGISTRY["OblateMassMGEPotential"] is OblateMassMGEPotential


def test_nfw_concentration_m200_is_registered_with_its_converters_and_schema() -> None:
    # Registration bundles converters, schema, and constraints, so config
    # validation and runtime resolution cannot drift apart.
    spec = _registry_module.get_parameterization("NFWPotential", "concentration_m200")
    assert spec is not None
    assert spec.convert is _nfw_concentration_m200
    assert spec.invert is _nfw_concentration_m200_inverse
    assert spec.raw_dimensions == {"c": "dimensionless", "M_200": "mass"}
    assert spec.raw_constraints == {
        "c": ParameterConstraint(minimum=0.0, minimum_inclusive=False),
        "M_200": ParameterConstraint(minimum=0.0, minimum_inclusive=False),
    }
    assert _registry_module.raw_parameter_dimensions(
        "NFWPotential", "concentration_m200"
    ) == {"c": "dimensionless", "M_200": "mass"}
    assert _registry_module.parameter_schema_is_known(
        "NFWPotential", "concentration_m200"
    )
    assert not _registry_module.parameter_schema_is_known(
        "NFWPotential", "not_registered"
    )


def _identity_forward(raw: dict, cosmological_parameters: object) -> dict:
    return raw


def _identity_inverse(native: dict, declared_units: object, cosmo: object) -> dict:
    return native


def test_register_parameterization_success_and_duplicate(monkeypatch) -> None:
    monkeypatch.setattr(_registry_module, "_PARAMETERIZATION_REGISTRY", {})

    _registry_module.register_parameterization(
        type_name="PlummerPotential",
        name="scheme",
        convert=_identity_forward,
        invert=_identity_inverse,
        raw_dimensions={"a": "mass"},
        raw_constraints={},
    )
    spec = _registry_module.get_parameterization("PlummerPotential", "scheme")
    assert spec is not None
    assert spec.convert is _identity_forward
    assert spec.invert is _identity_inverse
    assert spec.raw_dimensions == {"a": "mass"}
    assert _registry_module.parameterization_names("PlummerPotential") == ["scheme"]

    before = dict(_registry_module._PARAMETERIZATION_REGISTRY)
    with pytest.raises(ValueError, match=r"Duplicate parameterization 'scheme'"):
        _registry_module.register_parameterization(
            type_name="PlummerPotential",
            name="scheme",
            convert=_identity_forward,
            invert=_identity_inverse,
            raw_dimensions={"a": "mass"},
            raw_constraints={},
        )
    assert _registry_module._PARAMETERIZATION_REGISTRY == before


def test_register_parameterization_rejects_a_fully_unknown_target_type(
    monkeypatch,
) -> None:
    # A parameterization target must be a curated native galax type OR a
    # registered TNT component type -- a typo / made-up name is neither.
    monkeypatch.setattr(_registry_module, "_PARAMETERIZATION_REGISTRY", {})
    with pytest.raises(
        ValueError, match=r"neither a curated native galax type nor a registered"
    ):
        _registry_module.register_parameterization(
            type_name="NotAPotential",
            name="shape",
            convert=_identity_forward,
            invert=_identity_inverse,
            raw_dimensions={"p": "dimensionless"},
            raw_constraints={},
        )
    assert _registry_module._PARAMETERIZATION_REGISTRY == {}


def test_register_parameterization_accepts_a_registered_tnt_component_type(
    monkeypatch,
) -> None:
    # A TNT composite type (in _COMPONENT_REGISTRY) is a valid target: its own
    # `raw_parameters` override runs the inverse converter.
    monkeypatch.setattr(_registry_module, "_PARAMETERIZATION_REGISTRY", {})
    _registry_module.register_parameterization(
        type_name="TriaxialLightMGEPotential",
        name="shape",
        convert=_identity_forward,
        invert=_identity_inverse,
        raw_dimensions={"p": "dimensionless"},
        raw_constraints={},
    )
    assert ("TriaxialLightMGEPotential", "shape") in (
        _registry_module._PARAMETERIZATION_REGISTRY
    )


def test_register_parameterization_rejects_unknown_constraint_name(monkeypatch) -> None:
    monkeypatch.setattr(_registry_module, "_PARAMETERIZATION_REGISTRY", {})
    with pytest.raises(ValueError, match=r"constraint.*not present.*missing"):
        _registry_module.register_parameterization(
            type_name="PlummerPotential",
            name="scheme",
            convert=_identity_forward,
            invert=_identity_inverse,
            raw_dimensions={"a": "mass"},
            raw_constraints={"missing": ParameterConstraint(minimum=0.0)},
        )


def test_constraint_metadata_matches_each_registered_schema() -> None:
    for galax_type, parameters in _SUPPORTED_GALAX_TYPES.items():
        constraints = parameter_constraints(galax_type, None)
        assert set(constraints) <= set(parameters), galax_type
        for name, constraint in constraints.items():
            if constraint.other_parameter is not None:
                assert constraint.other_parameter in parameters, (galax_type, name)
                assert constraint.relation is not None, (galax_type, name)
            else:
                assert constraint.relation is None, (galax_type, name)

    for component_type, component_cls in _COMPONENT_REGISTRY.items():
        assert set(component_cls._constraints) <= set(component_cls._raw_dimensions), (
            component_type
        )


def test_mge_component_resolve_and_build_stores_the_referenced_mge() -> None:
    light_mge = _circular_light_mge([1.0], [1.0]).angular_to_physical(
        Quantity(30.0, "Mpc")
    )
    resolved = AbstractPotentialComponent.resolve(
        {
            "type": "TriaxialLightMGEPotential",
            "mge": "mge_lum",
            "parameters": {},
        },
        {"mge_lum": light_mge},
        path="potential.stars",
    )
    component = resolved.build(
        {"ml": Quantity(5.0, "Msun / Lsun"), **_VIEWING_ANGLES},
        _NO_COSMOLOGICAL_PARAMETERS,
    )
    assert isinstance(component, TriaxialLightMGEPotential)
    assert component.mge is light_mge
    assert component.parameters["ml"].ustrip("Msun / Lsun") == pytest.approx(5.0)


def test_light_mge_mass_to_light_ratio_must_be_positive() -> None:
    light_mge = _circular_light_mge([1.0], [1.0]).angular_to_physical(
        Quantity(30.0, "Mpc")
    )
    resolved = AbstractPotentialComponent.resolve(
        {
            "type": "TriaxialLightMGEPotential",
            "mge": "mge_lum",
            "parameters": {},
        },
        {"mge_lum": light_mge},
        path="potential.stars",
    )
    with pytest.raises(ValueError, match=r"parameters\.ml.*greater than"):
        resolved.build(
            {"ml": Quantity(0.0, "Msun / Lsun"), **_VIEWING_ANGLES},
            _NO_COSMOLOGICAL_PARAMETERS,
        )


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
            _NO_COSMOLOGICAL_PARAMETERS,
        )

    # A component that *did* build successfully can't have to_galax() raise
    # it -- deprojection already happened, and was already validated, at
    # build time.
    component = resolved.build(
        {"ml": Quantity(5.0, "Msun / Lsun"), **good_angles},
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


# ---------------------------------------------------------------------------
# Oblate axisymmetric MGE composite types: same _build/to_galax wiring as the
# triaxial pair, under a single `inclination` instead of theta/phi/psi.
#
# `AbstractMGE.deproject_oblate` has a real solution for any component
# whose observed q_obs >= cos(inclination) -- an edge-on inclination (90 deg,
# cos == 0) is valid for any q_obs, so these tests don't need q=1 fixtures
# to route around an invalid deprojection. q=1 is still used in the spherical
# cross-checks below to get an independent reference (galax's own
# GaussianPotential), which p=q=1 makes exact; a flattened q != 1 case is
# cross-checked against galax's own TriaxialGaussianPotential(q1=1) further
# down.
# ---------------------------------------------------------------------------


_INCLINATION = {"inclination": Quantity(90.0, "deg")}


def test_oblate_mge_component_resolve_and_build_stores_the_referenced_mge() -> None:
    light_mge = _circular_light_mge([1.0], [1.0]).angular_to_physical(
        Quantity(30.0, "Mpc")
    )
    resolved = AbstractPotentialComponent.resolve(
        {
            "type": "OblateLightMGEPotential",
            "mge": "mge_lum",
            "parameters": {},
        },
        {"mge_lum": light_mge},
        path="potential.stars",
    )
    component = resolved.build(
        {"ml": Quantity(5.0, "Msun / Lsun"), **_INCLINATION},
        _NO_COSMOLOGICAL_PARAMETERS,
    )
    assert isinstance(component, OblateLightMGEPotential)
    assert component.mge is light_mge
    assert component.parameters["ml"].ustrip("Msun / Lsun") == pytest.approx(5.0)


def test_oblate_mge_inclination_domain_is_checked_before_deprojection() -> None:
    light_mge = _circular_light_mge([1.0], [1.0]).angular_to_physical(
        Quantity(30.0, "Mpc")
    )
    resolved = AbstractPotentialComponent.resolve(
        {
            "type": "OblateLightMGEPotential",
            "mge": "mge_lum",
            "parameters": {},
        },
        {"mge_lum": light_mge},
        path="potential.stars",
    )
    with pytest.raises(ValueError, match=r"inclination.*at most 90.*deg"):
        resolved.build(
            {
                "ml": Quantity(5.0, "Msun / Lsun"),
                "inclination": Quantity(jnp.pi, "rad"),
            },
            _NO_COSMOLOGICAL_PARAMETERS,
        )


def test_oblate_mge_build_raises_for_impossible_inclination_not_to_galax() -> None:
    # q_obs = 0.5 with inclination 20 deg: cos(20 deg) ~ 0.94 > 0.5, so
    # q_obs < cos(i) and the axisymmetric deprojection has no real solution.
    flattened = LightMGE(
        I=Quantity(jnp.array([2.0]), "Lsun / rad2"),
        sigma=Quantity(jnp.array([1.5]), "rad"),
        q=Quantity(jnp.array([0.5]), ""),
        PA_twist=Quantity(jnp.array([0.0]), "rad"),
    ).angular_to_physical(Quantity(30.0, "Mpc"))
    resolved = AbstractPotentialComponent.resolve(
        {
            "type": "OblateLightMGEPotential",
            "mge": "mge_lum",
            "parameters": {},
        },
        {"mge_lum": flattened},
        path="potential.stars",
    )

    with pytest.raises(MGEDeprojectionError):
        resolved.build(
            {"ml": Quantity(5.0, "Msun / Lsun"), "inclination": Quantity(20.0, "deg")},
            _NO_COSMOLOGICAL_PARAMETERS,
        )

    # An inclination that *does* deproject builds and to_galax()es fine.
    component = resolved.build(
        {"ml": Quantity(5.0, "Msun / Lsun"), **_INCLINATION},
        _NO_COSMOLOGICAL_PARAMETERS,
    )
    component.to_galax(_internal_unit_system())


def test_oblate_light_mge_to_galax_matches_spherical_gaussian() -> None:
    unit_system = _internal_unit_system()
    distance = Quantity(30.0, "Mpc")
    light_mge = _circular_light_mge([2.0], [1.5]).angular_to_physical(distance)
    ml = Quantity(5.0, "Msun / Lsun")
    component = OblateLightMGEPotential._build(
        {"ml": ml, **_INCLINATION},
        _NO_COSMOLOGICAL_PARAMETERS,
        {"mge": light_mge},
    )

    potential = component.to_galax(unit_system)

    mass_mge = light_mge.to_mass(ml)
    deprojected = mass_mge.deproject_oblate(**_INCLINATION)
    assert deprojected.p.ustrip("") == pytest.approx(1.0)
    assert deprojected.q.ustrip("") == pytest.approx(1.0)
    m_tot = (
        deprojected.I[0]
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


def test_oblate_light_mge_to_galax_sums_every_component() -> None:
    unit_system = _internal_unit_system()
    distance = Quantity(30.0, "Mpc")
    light_mge = LightMGE(
        I=Quantity(jnp.array([2.0, 0.5]), "Lsun / rad2"),
        sigma=Quantity(jnp.array([1.5, 4.0]), "rad"),
        q=Quantity(jnp.array([0.6, 0.4]), ""),
        PA_twist=Quantity(jnp.array([0.0, 0.0]), "rad"),
    ).angular_to_physical(distance)
    ml = Quantity(5.0, "Msun / Lsun")
    component = OblateLightMGEPotential._build(
        {"ml": ml, **_INCLINATION},
        _NO_COSMOLOGICAL_PARAMETERS,
        {"mge": light_mge},
    )

    potential = component.to_galax(unit_system)

    deprojected = light_mge.to_mass(ml).deproject_oblate(**_INCLINATION)
    speed2 = unit_system["length"] ** 2 / unit_system["time"] ** 2
    xyz = Quantity(jnp.array([3.0, 0.5, -1.0]), "kpc")
    t = Quantity(0.0, "Myr")
    individual_sum = Quantity(0.0, speed2)
    for i in range(2):
        m_tot = (
            deprojected.I[i]
            * deprojected.q[i]
            * (2 * jnp.pi) ** 1.5
            * deprojected.sigma[i] ** 3
        )
        component_i = gp.AxisymmetricGaussianPotential(
            m_tot=m_tot,
            r_s=deprojected.sigma[i],
            q2=deprojected.q[i],
            units=unit_system,
        )
        individual_sum = individual_sum + component_i.potential(xyz, t)

    assert potential.potential(xyz, t).ustrip(speed2) == pytest.approx(
        individual_sum.ustrip(speed2), rel=1e-5
    )


def test_oblate_mass_mge_to_galax_uses_mge_mass_scale() -> None:
    unit_system = _internal_unit_system()
    distance = Quantity(30.0, "Mpc")
    mass_mge = MassMGE(
        I=Quantity(jnp.array([1e2]), "Msun / rad2"),
        sigma=Quantity(jnp.array([1.5]), "rad"),
        q=Quantity(jnp.array([1.0]), ""),
        PA_twist=Quantity(jnp.array([0.0]), "rad"),
    ).angular_to_physical(distance)
    mge_mass_scale = Quantity(3.0, "")
    component = OblateMassMGEPotential._build(
        {"mge_mass_scale": mge_mass_scale, **_INCLINATION},
        _NO_COSMOLOGICAL_PARAMETERS,
        {"mge": mass_mge},
    )

    potential = component.to_galax(unit_system)

    scaled = mass_mge.rescaled(mge_mass_scale)
    deprojected = scaled.deproject_oblate(**_INCLINATION)
    m_tot = (
        deprojected.I[0]
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


def test_oblate_light_mge_rescale_scales_to_galax_without_recompute() -> None:
    unit_system = _internal_unit_system()
    distance = Quantity(30.0, "Mpc")
    light_mge = _circular_light_mge([2.0], [1.5]).angular_to_physical(distance)
    ml = Quantity(5.0, "Msun / Lsun")
    component = OblateLightMGEPotential._build(
        {"ml": ml, **_INCLINATION},
        _NO_COSMOLOGICAL_PARAMETERS,
        {"mge": light_mge},
    )

    mass_scale = 2.0
    rescaled = component.rescale(mass_scale)

    assert rescaled.parameters["ml"].ustrip("Msun / Lsun") == pytest.approx(
        ml.ustrip("Msun / Lsun") * mass_scale
    )
    assert rescaled.deprojected.q.ustrip("") == pytest.approx(
        component.deprojected.q.ustrip("")
    )

    speed2 = unit_system["length"] ** 2 / unit_system["time"] ** 2
    xyz = Quantity(jnp.array([3.0, 0.5, -1.0]), "kpc")
    t = Quantity(0.0, "Myr")
    original = component.to_galax(unit_system).potential(xyz, t)
    scaled = rescaled.to_galax(unit_system).potential(xyz, t)
    assert scaled.ustrip(speed2) == pytest.approx(
        original.ustrip(speed2) * mass_scale, rel=1e-5
    )


def test_oblate_light_mge_to_galax_flattened_cross_checks() -> None:
    # The spherical cross-checks above use q_obs=1 (so q_intr=1 always) and
    # can't detect a missing flattening factor or a swapped q -> q2. This uses
    # a genuinely flattened deprojection (q_obs=0.7 at inclination 70 deg gives
    # q_intr ~ 0.65) and checks to_galax() two independent ways: against
    # galax's own TriaxialGaussianPotential(q1=1) -- a different galax class
    # and code path from AxisymmetricGaussianPotential -- at off-axis points,
    # and against the closed-form Gaussian density along each intrinsic axis,
    # which only matches if q2 lands on z (not x/y, where p=1).
    unit_system = _internal_unit_system()
    light_mge = LightMGE(
        I=Quantity(jnp.array([4.0]), "Lsun / kpc2"),
        sigma=Quantity(jnp.array([2.0]), "kpc"),
        q=Quantity(jnp.array([0.7]), ""),
        PA_twist=Quantity(jnp.array([0.0]), "rad"),
    )
    ml = Quantity(5.0, "Msun / Lsun")
    component = OblateLightMGEPotential._build(
        {"ml": ml, "inclination": Quantity(70.0, "deg")},
        _NO_COSMOLOGICAL_PARAMETERS,
        {"mge": light_mge},
    )
    potential = component.to_galax(unit_system)

    deprojected = component.deprojected
    q = float(deprojected.q.ustrip("")[0])
    assert deprojected.p.ustrip("")[0] == pytest.approx(1.0)
    assert 0.0 < q < 0.99  # genuinely flattened, not a spherical special case

    m_tot = (
        deprojected.I[0]
        * deprojected.q[0]
        * (2 * jnp.pi) ** 1.5
        * deprojected.sigma[0] ** 3
    )
    reference = gp.TriaxialGaussianPotential(
        m_tot=m_tot,
        r_s=deprojected.sigma[0],
        q1=Quantity(1.0, ""),
        q2=deprojected.q[0],
        units=unit_system,
    )

    t = Quantity(0.0, "Myr")
    speed2 = unit_system["length"] ** 2 / unit_system["time"] ** 2
    for xyz in (
        Quantity(jnp.array([2.0, 3.0, 1.5]), "kpc"),
        Quantity(jnp.array([-4.0, 1.0, 6.0]), "kpc"),
        Quantity(jnp.array([0.5, -2.0, -3.0]), "kpc"),
    ):
        assert potential.potential(xyz, t).ustrip(speed2) == pytest.approx(
            reference.potential(xyz, t).ustrip(speed2), rel=1e-5
        )

    rho_0 = float(deprojected.I.ustrip("Msun / kpc3")[0])
    sigma = float(deprojected.sigma.ustrip("kpc")[0])
    density_unit = unit_system["mass"] / unit_system["length"] ** 3
    for axis_index, axial_ratio in ((0, 1.0), (1, 1.0), (2, q)):
        xyz = [0.0, 0.0, 0.0]
        xyz[axis_index] = 3.0
        analytic = rho_0 * jnp.exp(-0.5 * (3.0 / (axial_ratio * sigma)) ** 2)
        density = potential.density(Quantity(jnp.array(xyz), "kpc"), t)
        assert density.ustrip(density_unit) == pytest.approx(analytic, rel=1e-5)


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
        {"ml": ml, **angles}, _NO_COSMOLOGICAL_PARAMETERS, {"mge": mge}
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


# ---------------------------------------------------------------------------
# The `pqu` parameterization for the triaxial MGE composite types:
# intrinsic axis ratios (p, q, u) <-> viewing angles (theta, phi, psi),
# anchored at q' = min(component q), zero twist (van den Bosch et al. 2008 /
# DYNAMITE triax_pqu2tpp).
# ---------------------------------------------------------------------------


def _triaxial_light_mge() -> LightMGE:
    return LightMGE(
        I=Quantity(jnp.array([120.0, 45.0, 18.0]), "Lsun / pc2"),
        sigma=Quantity(jnp.array([0.4, 1.8, 6.0]), "kpc"),
        q=Quantity(jnp.array([0.88, 0.82, 0.76]), ""),  # min observed q' = 0.76
        PA_twist=Quantity(jnp.zeros(3), "rad"),
    )


# A (p, q, u) triple valid against _triaxial_light_mge()'s q' = 0.76:
#   0 < q <= p <= 1  and  max(q/q', p) < u <= min(p/q', 1)
_PQU = {"p": Quantity(0.85, ""), "q": Quantity(0.60, ""), "u": Quantity(0.93, "")}


def test_pqu_to_tpp_round_trips_through_tpp_to_pqu() -> None:
    mge = _triaxial_light_mge()
    raw = {"ml": Quantity(4.0, "Msun / Lsun"), **_PQU}

    native = _pqu_to_tpp(raw, _NO_COSMOLOGICAL_PARAMETERS, mge)
    assert set(native) == {"ml", "theta", "phi", "psi"}
    recovered = _tpp_to_pqu(
        native, {"ml": "Msun / Lsun"}, _NO_COSMOLOGICAL_PARAMETERS, mge
    )

    for name, value in _PQU.items():
        assert recovered[name].ustrip("") == pytest.approx(value.ustrip(""), abs=1e-10)
    assert recovered["ml"].ustrip("Msun / Lsun") == pytest.approx(4.0)


def test_pqu_to_tpp_matches_deproject_triaxial_at_the_anchor_component() -> None:
    # Deprojecting a single Gaussian whose observed q equals the anchor q'
    # at the converted angles must return exactly the input (p, q).
    mge = _triaxial_light_mge()
    native = _pqu_to_tpp(
        {"ml": Quantity(1.0, "Msun / Lsun"), **_PQU},
        _NO_COSMOLOGICAL_PARAMETERS,
        mge,
    )
    anchor = LightMGE(
        I=Quantity(jnp.array([1.0]), "Lsun / pc2"),
        sigma=Quantity(jnp.array([6.0]), "kpc"),
        q=Quantity(jnp.array([0.76]), ""),
        PA_twist=Quantity(jnp.zeros(1), "rad"),
    )
    deprojected = anchor.deproject_triaxial(
        native["theta"], native["phi"], native["psi"]
    )
    assert deprojected.p[0].ustrip("") == pytest.approx(_PQU["p"].ustrip(""), abs=1e-10)
    assert deprojected.q[0].ustrip("") == pytest.approx(_PQU["q"].ustrip(""), abs=1e-10)


def test_pqu_to_tpp_matches_dynamite_triax_pqu2tpp_at_a_known_point() -> None:
    # (p, q, u, q') = (0.85, 0.6, 0.93, 0.76) run through DYNAMITE's
    # dynamite.physical_system.TriaxialVisibleComponent.triax_pqu2tpp
    # (van den Bosch et al. 2008); it returns (theta, psi, phi) in degrees.
    mge = _triaxial_light_mge()  # min observed q' = 0.76
    native = _pqu_to_tpp(
        {"ml": Quantity(1.0, "Msun / Lsun"), **_PQU},
        _NO_COSMOLOGICAL_PARAMETERS,
        mge,
    )
    deg = 180.0 / jnp.pi
    assert float(native["theta"].ustrip("rad")) * deg == pytest.approx(
        56.5558802, abs=1e-6
    )
    assert float(native["phi"].ustrip("rad")) * deg == pytest.approx(
        42.3177283, abs=1e-6
    )
    assert float(native["psi"].ustrip("rad")) * deg == pytest.approx(
        102.3159975, abs=1e-6
    )


def test_pqu_to_tpp_accepts_u_equal_to_one() -> None:
    # u = 1 (major axis in the sky plane) is a valid limiting geometry;
    # the (1 - u^2) denominators are handled by a one-ULP nudge.
    mge = _triaxial_light_mge()
    native = _pqu_to_tpp(
        {
            "ml": Quantity(1.0, "Msun / Lsun"),
            "p": Quantity(0.85, ""),
            "q": Quantity(0.60, ""),
            "u": Quantity(1.0, ""),
        },
        _NO_COSMOLOGICAL_PARAMETERS,
        mge,
    )
    for angle in ("theta", "phi", "psi"):
        assert jnp.isfinite(native[angle].ustrip("rad"))


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        ({"p": 0.85, "q": 0.80, "u": 0.98}, "u.*min"),  # u > min(p/q', 1) = 1
        ({"p": 0.85, "q": 0.60, "u": 0.65}, "max.*u"),  # u <= max(q/q', p) = 0.85
        ({"p": 0.80, "q": 0.80, "u": 0.90}, "prolate"),  # q == p (allowed by q <= p)
    ],
)
def test_pqu_to_tpp_rejects_geometries_outside_the_mge_dependent_domain(
    bad: dict[str, float], reason: str
) -> None:
    mge = _triaxial_light_mge()
    raw = {
        "ml": Quantity(1.0, "Msun / Lsun"),
        **{k: Quantity(v, "") for k, v in bad.items()},
    }
    with pytest.raises(_registry_module.InvalidPotentialParametersError, match=reason):
        _pqu_to_tpp(raw, _NO_COSMOLOGICAL_PARAMETERS, mge)


def test_pqu_to_tpp_rejects_a_circular_mge() -> None:
    circular = LightMGE(
        I=Quantity(jnp.array([1.0]), "Lsun / pc2"),
        sigma=Quantity(jnp.array([1.0]), "kpc"),
        q=Quantity(jnp.array([1.0]), ""),
        PA_twist=Quantity(jnp.zeros(1), "rad"),
    )
    with pytest.raises(
        _registry_module.InvalidPotentialParametersError, match="genuinely flattened"
    ):
        _pqu_to_tpp(
            {"ml": Quantity(1.0, "Msun / Lsun"), **_PQU},
            _NO_COSMOLOGICAL_PARAMETERS,
            circular,
        )


def test_pqu_parameterization_is_registered_for_both_triaxial_mge_types() -> None:
    for type_name, mass_name, mass_dim in (
        ("TriaxialLightMGEPotential", "ml", "mass_to_light"),
        ("TriaxialMassMGEPotential", "mge_mass_scale", "dimensionless"),
    ):
        spec = _registry_module.get_parameterization(type_name, "pqu")
        assert spec is not None
        assert spec.convert is _pqu_to_tpp
        assert spec.invert is _tpp_to_pqu
        assert spec.raw_dimensions == {
            mass_name: mass_dim,
            "p": "dimensionless",
            "q": "dimensionless",
            "u": "dimensionless",
        }
        assert raw_parameter_dimensions(type_name, "pqu") == spec.raw_dimensions
        constraints = parameter_constraints(type_name, "pqu")
        assert set(constraints) == {mass_name, "p", "q", "u"}
        assert constraints["q"].other_parameter == "p"
        assert constraints["q"].relation == "<="
        assert constraints["u"].other_parameter == "p"
        assert constraints["u"].relation == ">"
        assert constraints["u"].maximum == 1.0


@pytest.mark.parametrize(
    "type_name", ["TriaxialLightMGEPotential", "TriaxialMassMGEPotential"]
)
def test_pqu_config_builds_the_same_deprojection_as_the_equivalent_angles(
    type_name: str,
) -> None:
    mge = _triaxial_light_mge()
    if type_name == "TriaxialMassMGEPotential":
        mge = mge.to_mass(Quantity(1.0, "Msun / Lsun"))
    mass_name = "ml" if type_name == "TriaxialLightMGEPotential" else "mge_mass_scale"
    mass_unit = "Msun / Lsun" if mass_name == "ml" else ""
    mass_value = Quantity(3.5, mass_unit)

    pqu_resolved = AbstractPotentialComponent.resolve(
        {"type": type_name, "parameterization": "pqu", "mge": "m", "parameters": {}},
        {"m": mge},
        path="potential.stars",
    )
    pqu_component = pqu_resolved.build(
        {mass_name: mass_value, **_PQU}, _NO_COSMOLOGICAL_PARAMETERS
    )

    angles = _pqu_to_tpp(
        {mass_name: mass_value, **_PQU}, _NO_COSMOLOGICAL_PARAMETERS, mge
    )
    tpp_resolved = AbstractPotentialComponent.resolve(
        {"type": type_name, "mge": "m", "parameters": {}},
        {"m": mge},
        path="potential.stars",
    )
    tpp_component = tpp_resolved.build(
        {mass_name: mass_value, **{k: angles[k] for k in ("theta", "phi", "psi")}},
        _NO_COSMOLOGICAL_PARAMETERS,
    )

    for attr in ("I", "sigma", "p", "q"):
        assert jnp.allclose(
            getattr(pqu_component.deprojected, attr).ustrip(
                getattr(pqu_component.deprojected, attr).unit
            ),
            getattr(tpp_component.deprojected, attr).ustrip(
                getattr(tpp_component.deprojected, attr).unit
            ),
        ), attr


def test_pqu_raw_potential_parameters_round_trips_and_survives_rescale() -> None:
    mge = _triaxial_light_mge()
    settings = {
        "stars": {
            "type": "TriaxialLightMGEPotential",
            "parameterization": "pqu",
            "mge": "m",
            "parameters": {"ml": {"unit": "Msun / Lsun"}},
        }
    }
    values = {"stars": {"ml": Quantity(4.0, "Msun / Lsun"), **_PQU}}
    potential = Potential.from_settings(settings, values, {"m": mge}, {})

    raw = raw_potential_parameters(settings, potential, {})["stars"]
    assert set(raw) == {"ml", "p", "q", "u"}
    for name, value in _PQU.items():
        assert raw[name].ustrip("") == pytest.approx(value.ustrip(""), abs=1e-9)
    assert raw["ml"].ustrip("Msun / Lsun") == pytest.approx(4.0)

    rescaled = raw_potential_parameters(settings, potential.rescale(3.0), {})["stars"]
    assert rescaled["ml"].ustrip("Msun / Lsun") == pytest.approx(12.0)
    for name, value in _PQU.items():
        assert rescaled[name].ustrip("") == pytest.approx(value.ustrip(""), abs=1e-9)


def test_pqu_domain_invalid_value_is_rejected_at_build_time() -> None:
    # q > p violates the data-independent ParameterConstraint, caught before
    # the converter (and before any MGE is consulted).
    mge = _triaxial_light_mge()
    resolved = AbstractPotentialComponent.resolve(
        {
            "type": "TriaxialLightMGEPotential",
            "parameterization": "pqu",
            "mge": "m",
            "parameters": {},
        },
        {"m": mge},
        path="potential.stars",
    )
    with pytest.raises(ValueError, match=r"parameters\.q"):
        resolved.build(
            {
                "ml": Quantity(1.0, "Msun / Lsun"),
                "p": Quantity(0.6, ""),
                "q": Quantity(0.8, ""),
                "u": Quantity(0.9, ""),
            },
            _NO_COSMOLOGICAL_PARAMETERS,
        )
