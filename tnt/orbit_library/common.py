"""Numerics shared by every orbit-sampler variant.

Equipotential search and launch-point formulas for the stationary start
space (box orbits, `stationary_grid.py`) and every `(x, z)`-plane start
space (`xz_grid_from_origin.py`, `xz_grid_from_boundary.py`) -- nothing here
is specific to any one of them.
"""

from __future__ import annotations

from typing import Any

import galax.potential as gp
import jax
import jax.numpy as jnp
import optimistix as optx
from unxt import Quantity


class EquipotentialSearchError(Exception):
    """`_equipotential_radius` failed to locate a genuine equipotential.

    Raised eagerly by a sampler's own `generate_ics`, after aggregating
    `_equipotential_radius`'s `valid` flag across its whole grid -- orbit
    sampling is not part of the differentiable `Potential.build_with_validity`
    contract, so there is no traced flag to thread through here; a failed
    search is a configuration/potential-shape problem to fix, not a
    proposal to reject and move on from.
    """


# `flip=False`, not `optimistix`'s default `flip="detect"`: `detect`'s
# runtime bracket check doesn't batch correctly under `jax.vmap` (as of
# optimistix 0.1.0). Expected to be correct, not just assumed: `r_lo` is
# normally deep in the potential well (negative `potential - energy`),
# `r_hi` normally far out (positive) -- `_equipotential_radius` checks this
# explicitly rather than trusting it blindly, since it can fail for an
# extreme enough direction/shape (see that function's own docstring).
_EQUIPOTENTIAL_SOLVER = optx.Bisection(rtol=1e-10, atol=1e-12, flip=False)
_EQUIPOTENTIAL_MAX_STEPS = 200

# `findReq`'s own bracket (`orbitstart_f.f90:560-561`, `rmx = 1.1*Req`,
# `rmn = 0.01*Req`) -- scaled per energy shell from that shell's own x-axis
# equipotential radius (`Req`/`Rcirc` there), not one fixed bracket shared
# across every shell regardless of its own natural scale. `_R_FLOOR_FACTOR`
# is a separate, physical near-origin floor for the `(x, z)`-plane samplers'
# own sampled population (not an equipotential-search bracket bound) --
# unrelated to either of these factors.
_EQUIPOTENTIAL_SEARCH_INNER_FACTOR = 0.01
_EQUIPOTENTIAL_SEARCH_OUTER_FACTOR = 1.1
# `findReq`'s own convergence tolerance (`orbitstart_f.f90:572`,
# `abs((E - pot)/E) < 1.0e-7`).
_EQUIPOTENTIAL_RESIDUAL_RTOL = 1e-7

_R_FLOOR_FACTOR = 1e-3


def _energy_unit(potential: gp.AbstractPotential) -> Any:
    return potential.units["length"] ** 2 / potential.units["time"] ** 2


def _potential_at(
    potential: gp.AbstractPotential, position: jnp.ndarray, t0: Quantity
) -> jnp.ndarray:
    """`potential.potential(position, t0)`, as a bare float in the potential's
    own energy unit.

    `position` is a plain, dimensionless `(3,)` array already expressed in
    the potential's own length unit.
    """
    q = Quantity(position, potential.units["length"])
    return potential.potential(q, t0).ustrip(_energy_unit(potential))


def _equipotential_radius(
    potential: gp.AbstractPotential,
    direction: jnp.ndarray,
    energy: jnp.ndarray,
    r_ref: jnp.ndarray,
    t0: Quantity,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The radius `r` along `direction` where `potential(r * direction) == energy`.

    `direction` is a unit vector; the equipotential radius varies with it in
    a non-spherical potential -- this is what makes the stationary start
    space's box orbits actually sample the ellipsoidal equipotential
    surface, not a sphere.

    Bisected within `[r_ref * _EQUIPOTENTIAL_SEARCH_INNER_FACTOR, r_ref *
    _EQUIPOTENTIAL_SEARCH_OUTER_FACTOR]` -- DYNAMITE's own `findReq`
    bracket, scaled from `r_ref`, the same energy shell's own x-axis
    equipotential radius (`potential(r_ref, 0, 0) == energy` by
    construction), rather than one fixed bracket shared across every shell
    regardless of its own natural scale. A direction far enough from the
    x-axis in a sufficiently flattened potential can still place the true
    root outside even this bracket.

    Returns:
        `(r0, valid)`: `valid` requires the bracket to actually contain a
        sign change (`potential(r_lo) <= energy <= potential(r_hi)`), the
        solver's own result to report success, and the residual at `r0` to
        be within `_EQUIPOTENTIAL_RESIDUAL_RTOL` -- DYNAMITE's own `findReq`
        convergence tolerance. A caller must not use `r0` where `valid` is
        `False`; it is not a located equipotential.
    """
    r_lo = r_ref * _EQUIPOTENTIAL_SEARCH_INNER_FACTOR
    r_hi = r_ref * _EQUIPOTENTIAL_SEARCH_OUTER_FACTOR

    def objective(r: jnp.ndarray, _args: None) -> jnp.ndarray:
        return _potential_at(potential, r * direction, t0) - energy

    bracket_valid = (objective(r_lo, None) <= 0) & (objective(r_hi, None) >= 0)

    solution = optx.root_find(
        objective,
        _EQUIPOTENTIAL_SOLVER,
        y0=0.5 * (r_lo + r_hi),
        options={"lower": r_lo, "upper": r_hi},
        max_steps=_EQUIPOTENTIAL_MAX_STEPS,
        throw=False,
    )
    converged = solution.result == optx.RESULTS.successful
    residual = jnp.abs(objective(solution.value, None) / energy)
    valid = bracket_valid & converged & (residual < _EQUIPOTENTIAL_RESIDUAL_RTOL)
    return solution.value, valid


def _spherical_direction(theta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """Unit vector at spherical angles `(theta, phi)` in the intrinsic frame.

    `theta` from the `z`-axis (short axis), `phi` from the `x`-axis (long
    axis) in the `x`-`y` plane -- restricted to one octant `[0, pi/2]^2`
    (van den Bosch et al. 2008 sec. 4.5's eightfold symmetry).
    """
    sin_theta = jnp.sin(theta)
    return jnp.array(
        [sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), jnp.cos(theta)]
    )


def _xz_direction(theta: jnp.ndarray) -> jnp.ndarray:
    """Unit vector in the `(x, z)` plane at polar angle `theta` from the `z`-axis."""
    return jnp.array([jnp.sin(theta), 0.0, jnp.cos(theta)])


def _box_orbit_ic(
    potential: gp.AbstractPotential,
    energy: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
    r_ref: jnp.ndarray,
    t0: Quantity,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One stationary-start-space box orbit's `(x, y, z, vx, vy, vz)`.

    Launched from rest exactly on the equipotential at `(energy, theta,
    phi)` -- the defining condition for box-orbit support (Schwarzschild
    1979): zero velocity everywhere, the point itself found by
    `_equipotential_radius`. `r_ref` is this energy shell's own x-axis
    equipotential radius; see that function for the search bracket it sets
    and the `valid` flag's meaning.
    """
    direction = _spherical_direction(theta, phi)
    r0, valid = _equipotential_radius(potential, direction, energy, r_ref, t0)
    return jnp.concatenate([r0 * direction, jnp.zeros(3)]), valid


def _xz_orbit_ic(
    potential: gp.AbstractPotential,
    energy: jnp.ndarray,
    theta: jnp.ndarray,
    r: jnp.ndarray,
    t0: Quantity,
) -> jnp.ndarray:
    """One `(x, z)`-start-space orbit's `(x, y, z, vx, vy, vz)` at a given `r`.

    Launched from `r * _xz_direction(theta)` with velocity purely along
    `y`, `v_y**2 = 2 * (energy - V(x0, 0, z0))` (van den Bosch et al. 2008
    sec. 4.3) -- the remaining energy at that point, converted entirely to
    tangential motion.
    """
    direction = _xz_direction(theta)
    position = r * direction
    v_y_sq = jnp.clip(2.0 * (energy - _potential_at(potential, position, t0)), min=0.0)
    velocity = jnp.array([0.0, jnp.sqrt(v_y_sq), 0.0])
    return jnp.concatenate([position, velocity])


def _v_circ(
    potential: gp.AbstractPotential, r: jnp.ndarray, t0: Quantity
) -> jnp.ndarray:
    """The local circular velocity `sqrt(r * |dV/dr(r)|)` along the `x`-axis
    (via autodiff). Deliberately not referenced to the potential's central
    value (`V(0, 0, 0)`), which is singular for a cusped or point-mass
    profile.
    """

    def v_of_r(radius: jnp.ndarray) -> jnp.ndarray:
        return _potential_at(potential, radius * jnp.array([1.0, 0.0, 0.0]), t0)

    dv_dr = jax.grad(v_of_r)(r)
    return jnp.sqrt(r * jnp.abs(dv_dr))


def _circular_period(
    potential: gp.AbstractPotential, r: jnp.ndarray, t0: Quantity
) -> jnp.ndarray:
    """`2 * pi * r / v_circ(r)` -- the circular-orbit period at `r`, same
    formula as `orbitstart_f.f90`'s own `tcirc`.
    """
    return 2.0 * jnp.pi * r / _v_circ(potential, r, t0)


def _phase_space_derivs(potential: gp.AbstractPotential, t0: Quantity) -> Any:
    """`diffrax.ODETerm`-compatible `(t, y, args) -> dy/dt`, `y = (x, y, z,
    vx, vy, vz)`, for one `galax` potential.

    Ignores `t`: every potential this sampler builds is static (no
    explicit time dependence), so `potential.acceleration(q, t0)` at the
    fixed reference `t0` already equals `potential.acceleration(q, t)` for
    any `t` -- exact, not an approximation, for an autonomous system.
    """
    length_unit = potential.units["length"]
    time_unit = potential.units["time"]
    accel_unit = length_unit / time_unit**2

    def derivs(_t: jnp.ndarray, y: jnp.ndarray, _args: None) -> jnp.ndarray:
        position, velocity = y[:3], y[3:]
        acceleration = potential.acceleration(
            Quantity(position, length_unit), t0
        ).ustrip(accel_unit)
        return jnp.concatenate([velocity, acceleration])

    return derivs
