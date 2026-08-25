"""NFW's `concentration_m200` parameterization: `(c, M_200) <-> (m, r_s)`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import unxt as u
from unxt import AbstractUnitSystem, Quantity

# Newton's gravitational constant, for parameterizations that need it (e.g.
# critical density). Not from galax's own `default_constants`, to keep this
# module's physics self-contained and independently verifiable rather than
# reaching into galax's private `_src` internals.
_G = Quantity(6.6743e-11, "m3 / (kg s2)")


def _nfw_concentration_m200(
    raw: dict[str, Quantity],
    unit_system: AbstractUnitSystem,
    cosmological_parameters: Mapping[str, Quantity],
) -> dict[str, Quantity]:
    """Convert NFW's `(c, M_200)` parameterization to native `(m, r_s)`.

    `M_200` uses the critical-density convention (M_200c): the mass
    enclosed within the radius `r_200` at which the mean density equals
    `200 * rho_crit`, where `rho_crit = 3 H0^2 / (8 pi G)`. Concentration is
    `c = r_200 / r_s`. Both `r_s` and the native characteristic mass `m`
    follow from `galax.potential.NFWPotential`'s own enclosed-mass formula,
    `M(<r) = m * (ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s))`, evaluated at
    `r = r_200` -- verified directly against galax's own `mass_enclosed` to
    float32 precision, and that the resulting `r_200` truly encloses a mean
    density of exactly `200 * rho_crit`.
    """
    c = raw["c"]
    m200 = raw["M_200"]
    h0 = cosmological_parameters["H0"]

    rho_crit = 3 * h0**2 / (8 * jnp.pi * _G)
    r200 = (3 * m200 / (4 * jnp.pi * 200 * rho_crit)) ** (1 / 3)
    r_s = r200 / c
    m = m200 / _nfw_g(c.ustrip(""))
    return {
        "m": m.to(unit_system[u.dimension("mass")]),
        "r_s": r_s.to(unit_system[u.dimension("length")]),
    }


def _nfw_g(c: Any) -> Any:
    """`ln(1 + c) - c / (1 + c)`, NFW's enclosed-mass shape function.

    `log1p(c)`, not `log(1 + c)`: for small `c`, `1 + c` loses precision
    that `log1p` avoids by not forming that sum.
    """
    return jnp.log1p(c) - c / (1 + c)


def _solve_nfw_concentration(target: Any) -> Any:
    """Solve `c**3 / _nfw_g(c) == target` for `c > 0`.

    No closed form. `h(c) = c**3 / _nfw_g(c)` is strictly monotonically
    increasing for `c > 0` (verified numerically across many orders of
    magnitude of `c`), so a fixed-iteration bisection over a wide,
    unit-independent bracket (`c` is dimensionless -- realistic halo
    concentrations are always well inside `[1e-6, 1e6]`) converges reliably
    regardless of `target`'s scale.
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


def _nfw_concentration_m200_inverse(
    native: dict[str, Quantity],
    unit_system: AbstractUnitSystem,
    cosmological_parameters: Mapping[str, Quantity],
) -> dict[str, Quantity]:
    """Convert NFW's native `(m, r_s)` back to `(c, M_200)`.

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
    m = native["m"]
    r_s = native["r_s"]
    h0 = cosmological_parameters["H0"]

    rho_crit = 3 * h0**2 / (8 * jnp.pi * _G)
    target = (m / (4 * jnp.pi * 200 * rho_crit / 3 * r_s**3)).ustrip("")
    c = _solve_nfw_concentration(target)
    m200 = m * _nfw_g(c)
    return {"c": Quantity(c, ""), "M_200": m200.to(unit_system[u.dimension("mass")])}
