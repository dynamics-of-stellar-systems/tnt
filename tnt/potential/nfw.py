"""NFW's `concentration_m200` parameterization: `(c, M_200) <-> (m, r_s)`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from unxt import Quantity

from tnt.potential.registry import ParameterConstraint, register_parameterization


def _newtonian_gravitational_constant() -> Quantity:
    """Construct Newton's gravitational constant under the active JAX policy.

    Construct this at calculation time rather than module-import time so its
    dtype follows the process's configured JAX precision. The value is kept
    here instead of using galax's private ``default_constants`` internals so
    this module's physics remains self-contained and independently verifiable.
    """
    return Quantity(6.6743e-11, "m3 / (kg s2)")


def _nfw_critical_density(h: Quantity) -> Quantity:
    """Critical density in local units safe for float32 differentiation."""
    h = h.to("1 / Myr")
    # Prevent XLA from folding the unit conversion into later divisions and
    # recreating overflow/underflow in the original inverse-second units.
    h = Quantity(jax.lax.optimization_barrier(h.ustrip(h.unit)), h.unit)
    gravitational_constant = _newtonian_gravitational_constant().to(
        "kpc3 / (Msun Myr2)"
    )
    return 3 * h**2 / (8 * jnp.pi * gravitational_constant)


def _nfw_concentration_m200(
    raw: dict[str, Quantity],
    cosmological_parameters: Mapping[str, Quantity],
    mge: Any = None,
) -> dict[str, Quantity]:
    """Convert NFW's `(c, M_200)` parameterization to native `(m, r_s)`.

    `M_200` uses the critical-density convention (M_200c): the mass
    enclosed within the radius `r_200` at which the mean density equals
    `200 * rho_crit`, where `rho_crit = 3 H^2 / (8 pi G)` and `H` is the
    Hubble parameter at the halo's own epoch (not necessarily today's H0 --
    `cosmological_parameters.H` is whatever value the run declares).
    Concentration is `c = r_200 / r_s`. Both `r_s` and the native
    characteristic mass `m` follow from `galax.potential.NFWPotential`'s own
    enclosed-mass formula, `M(<r) = m * (ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s))`,
    evaluated at `r = r_200` -- verified directly against galax's own
    `mass_enclosed` at the configured precision, and that the resulting `r_200`
    truly encloses a mean density of exactly `200 * rho_crit`.
    """
    del mge  # this parameterization needs no MGE (see ForwardConverter)
    c = raw["c"]
    m200 = raw["M_200"]
    # Local astronomical units keep both values and reverse-mode intermediates
    # representable in float32, including when H is declared in inverse seconds.
    rho_crit = _nfw_critical_density(cosmological_parameters["H"])
    volume = 3 * m200.to("Msun") / (4 * jnp.pi * 200 * rho_crit)
    # A fractional power on a volume Quantity fails under batched JAX traces.
    # Cube-root the numeric value, then attach the corresponding length unit.
    r200 = Quantity(jnp.cbrt(volume.ustrip(volume.unit)), volume.unit ** (1 / 3))
    r_s = r200 / c
    concentration = c.ustrip("")
    mass_value = _nfw_characteristic_mass(m200.ustrip(m200.unit), concentration)
    g = _nfw_g(concentration)
    physical_mass = Quantity(mass_value, m200.unit).ustrip("Msun")
    # Reject conversions whose values or concentration/mass derivatives cannot
    # be represented. The traced builder keeps these outside its AD path.
    reliable = (
        jnp.isfinite(1 / g)
        & jnp.isfinite(
            physical_mass
            * ((concentration / (1 + concentration)) / g)
            / (1 + concentration)
        )
        & jnp.isfinite(r_s.ustrip(r_s.unit) / concentration)
    )
    m = Quantity(jnp.where(reliable, mass_value, jnp.nan), m200.unit)
    return {"m": m, "r_s": r_s}


@jax.custom_jvp
def _nfw_g(c: Any) -> Any:
    """`ln(1 + c) - c / (1 + c)`, NFW's enclosed-mass shape function.

    Below 0.01 use the Taylor series through c**10, whose relative
    truncation error is below 2e-18. This avoids subtracting nearly equal
    terms. Mask the polynomial input so unused large-c powers cannot overflow.
    """
    small = c < 0.01
    x = jnp.where(small, c, 0.0)
    polynomial = jnp.asarray(9 / 10, dtype=jnp.asarray(c).dtype)
    for power in range(9, 1, -1):
        polynomial = (-1) ** power * (power - 1) / power + x * polynomial
    return jnp.where(small, x * x * polynomial, jnp.log1p(c) - c / (1 + c))


@_nfw_g.defjvp
def _nfw_g_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[Any, Any]:
    """Analytic shape derivative without a potentially overflowing square."""
    (c,), (c_dot,) = primals, tangents
    return _nfw_g(c), ((c / (1 + c)) / (1 + c)) * c_dot


@jax.custom_jvp
def _nfw_characteristic_mass(m200: Any, c: Any) -> Any:
    """Characteristic mass with a derivative avoiding division by g(c)**2."""
    return m200 / _nfw_g(c)


@_nfw_characteristic_mass.defjvp
def _nfw_characteristic_mass_jvp(
    primals: tuple[Any, Any], tangents: tuple[Any, Any]
) -> tuple[Any, Any]:
    """Evaluate derivative coefficients before multiplying by cotangents."""
    m200, c = primals
    m200_dot, c_dot = tangents
    g = _nfw_g(c)
    mass = _nfw_characteristic_mass(m200, c)
    derivative = mass * ((c / (1 + c)) / g) / (1 + c)
    return mass, m200_dot / g - derivative * c_dot


@jax.custom_jvp
def _solve_nfw_concentration(target: Any) -> Any:
    """Solve `c**3 / _nfw_g(c) == target` for `c > 0`.

    No closed form. `h(c) = c**3 / _nfw_g(c)` is strictly monotonically
    increasing for `c > 0` (verified numerically across many orders of
    magnitude of `c`), so a fixed-iteration bisection over a wide,
    unit-independent bracket (`c` is dimensionless -- realistic halo
    concentrations are always well inside `[1e-6, 1e6]`) converges reliably
    for targets whose roots lie inside that bracket. The custom derivative
    comes from the root equation, not from differentiating the bisection
    comparisons. Outside the bracket, bisection returns an endpoint and the
    implicit derivative does not describe that clamped result.
    """

    def h(c: Any) -> Any:
        return c**3 / _nfw_g(c)

    lower, upper = jnp.asarray(1e-6), jnp.asarray(1e6)
    for _ in range(80):
        mid = 0.5 * (lower + upper)
        too_low = h(mid) < target
        lower = jnp.where(too_low, mid, lower)
        upper = jnp.where(too_low, upper, mid)
    return 0.5 * (lower + upper)


@_solve_nfw_concentration.defjvp
def _solve_nfw_concentration_jvp(
    primals: tuple[Any], tangents: tuple[Any]
) -> tuple[Any, Any]:
    """Differentiate the root equation instead of the bisection decisions."""
    (target,) = primals
    (target_dot,) = tangents
    c = _solve_nfw_concentration(target)
    g = _nfw_g(c)
    g_prime = c / (1 + c) ** 2
    h_prime = c**2 * (3 * g - c * g_prime) / g**2
    return c, target_dot / h_prime


def _nfw_concentration_m200_inverse(
    native: dict[str, Quantity],
    declared_units: Mapping[str, str],
    cosmological_parameters: Mapping[str, Quantity],
    mge: Any = None,
) -> dict[str, Quantity]:
    """Convert NFW's native `(m, r_s)` back to `(c, M_200)`.

    `M_200` is returned in `declared_units["M_200"]` -- the unit the
    configuration declares for it -- so `AllModels` reports it exactly as
    configured. `c` is dimensionless.

    The inverse of `_nfw_concentration_m200`. Substituting
    `r_200 = c * r_s` into that function's `r_200`/`m` relations leaves one
    equation in `c` alone, `c**3 / _nfw_g(c) = m / ((4 pi 200 rho_crit / 3) * r_s**3)`,
    solved numerically by `_solve_nfw_concentration` since it has no closed
    form. `M_200` then follows directly from `c` via the forward relation
    `m = M_200 / _nfw_g(c)`.

    This matters after `GalaxPotentialComponent.rescale()`, which scales
    `m` while holding `r_s` fixed (see `tnt.potential.registry._SUPPORTED_GALAX_TYPES`):
    that is *not* the same as holding `c` fixed and scaling `M_200`, so the
    rescaled `(c, M_200)` genuinely differs from the original and must be
    recomputed here, not just carried through unchanged.
    """
    del mge  # this parameterization needs no MGE (see ForwardConverter)
    m = native["m"]
    r_s = native["r_s"]
    rho_crit = _nfw_critical_density(cosmological_parameters["H"])
    target = (
        m.to("Msun") / (4 * jnp.pi * 200 * rho_crit / 3 * r_s.to("kpc") ** 3)
    ).ustrip("")
    c = _solve_nfw_concentration(target)
    m200 = m * _nfw_g(c)
    return {"c": Quantity(c, ""), "M_200": m200.to(declared_units["M_200"])}


register_parameterization(
    type_name="NFWPotential",
    name="concentration_m200",
    convert=_nfw_concentration_m200,
    invert=_nfw_concentration_m200_inverse,
    raw_dimensions={"c": "dimensionless", "M_200": "mass"},
    raw_constraints={
        "c": ParameterConstraint(minimum=0.0, minimum_inclusive=False),
        "M_200": ParameterConstraint(minimum=0.0, minimum_inclusive=False),
    },
)
