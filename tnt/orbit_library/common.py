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

# `flip=False`, not `optimistix`'s default `flip="detect"`: `detect`'s
# runtime bracket check doesn't batch correctly under `jax.vmap` (as of
# optimistix 0.1.0). Safe here because the bracket order is known
# analytically -- `r_lo` is always deep in the potential well (very negative
# `potential - energy`), `r_hi` always far out (positive) -- so `flip=False`
# is simply correct, not a workaround for the vmap issue.
_EQUIPOTENTIAL_SOLVER = optx.Bisection(rtol=1e-10, atol=1e-12, flip=False)
_EQUIPOTENTIAL_MAX_STEPS = 200

# A fixed, direction- and energy-independent bracket for every equipotential
# search: wide enough that the true root -- which varies with direction in a
# non-spherical potential -- stays safely inside it for every (E, theta, phi)
# this sampler ever requests, given a potential that increases monotonically
# outward along every ray from the centre.
_R_FLOOR_FACTOR = 1e-3
_R_CEILING_FACTOR = 1e2


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
    r_lo: jnp.ndarray,
    r_hi: jnp.ndarray,
    t0: Quantity,
) -> jnp.ndarray:
    """The radius `r` along `direction` where `potential(r * direction) == energy`.

    Bisected between `r_lo` and `r_hi`. `direction` is a unit vector; the
    equipotential radius varies with it in a non-spherical potential -- this
    is what makes the stationary start space's box orbits actually sample
    the ellipsoidal equipotential surface, not a sphere.
    """

    def objective(r: jnp.ndarray, _args: None) -> jnp.ndarray:
        return _potential_at(potential, r * direction, t0) - energy

    solution = optx.root_find(
        objective,
        _EQUIPOTENTIAL_SOLVER,
        y0=0.5 * (r_lo + r_hi),
        options={"lower": r_lo, "upper": r_hi},
        max_steps=_EQUIPOTENTIAL_MAX_STEPS,
        throw=False,
    )
    return solution.value


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
    r_lo: jnp.ndarray,
    r_hi: jnp.ndarray,
    t0: Quantity,
) -> jnp.ndarray:
    """One stationary-start-space box orbit's `(x, y, z, vx, vy, vz)`.

    Launched from rest exactly on the equipotential at `(energy, theta,
    phi)` -- the defining condition for box-orbit support (Schwarzschild
    1979): zero velocity everywhere, the point itself found by
    `_equipotential_radius`.
    """
    direction = _spherical_direction(theta, phi)
    r0 = _equipotential_radius(potential, direction, energy, r_lo, r_hi, t0)
    return jnp.concatenate([r0 * direction, jnp.zeros(3)])


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
