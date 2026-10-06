"""`StationaryGridOrbitSampler`: a pure stationary-start-space (box orbit) sampler.

Box orbits only, launched from rest exactly on the equipotential surface
(Schwarzschild 1979) -- no tube-orbit start space at all, unlike a combined
sampler that would also need a third, radius-related grid dimension. Hence
`nE`/`nI1`/`nI2` rather than `nE`/`nI2`/`nI3`: there's no "I3" dimension here
to reserve the name for.
"""

from __future__ import annotations

from typing import Any, ClassVar

import galax.potential as gp
import jax
import jax.numpy as jnp
import optimistix as optx
from unxt import Quantity

from tnt.orbit_library.base import AbstractOrbitSampler

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


class StationaryGridOrbitSampler(AbstractOrbitSampler):
    """A stationary start space only: `nE * nI1 * nI2` box orbits.

    `nE` energy shells between `rmin`/`rmax` (converted to the resolved
    potential's own length unit), each paired with an open, bin-centred
    `nI1 * nI2` grid of `theta, phi in (0, pi/2)` -- never exactly `0` or
    `pi/2`, which would respectively collapse every `phi` to one duplicate
    point on the z-axis, or launch a box orbit exactly in the equatorial
    plane (able to pass through the centre of a cuspy potential). Every
    orbit starts from rest exactly on the equipotential surface at its
    `(energy, theta, phi)`.
    """

    _type: ClassVar[str] = "StationaryGrid"
    rmin: Quantity
    rmax: Quantity
    nE: int
    nI1: int
    nI2: int

    def n_bundles(self) -> int:
        return self.nE * self.nI1 * self.nI2

    def generate_ics(self, potential: gp.AbstractPotential) -> jnp.ndarray:
        t0 = Quantity(0.0, potential.units["time"])
        length_unit = potential.units["length"]
        rmin_val = self.rmin.ustrip(length_unit)
        rmax_val = self.rmax.ustrip(length_unit)
        r_floor = rmin_val * _R_FLOOR_FACTOR
        r_ceiling = rmax_val * _R_CEILING_FACTOR

        r_grid = 10.0 ** jnp.linspace(jnp.log10(rmin_val), jnp.log10(rmax_val), self.nE)
        x_axis_points = jnp.stack(
            [r_grid, jnp.zeros_like(r_grid), jnp.zeros_like(r_grid)], axis=-1
        )
        energies = jax.vmap(lambda pos: _potential_at(potential, pos, t0))(
            x_axis_points
        )

        theta_grid = (jnp.arange(self.nI1) + 0.5) * (jnp.pi / 2) / self.nI1
        phi_grid = (jnp.arange(self.nI2) + 0.5) * (jnp.pi / 2) / self.nI2

        e_grid, theta_full, phi_full = (
            a.reshape(-1)
            for a in jnp.meshgrid(energies, theta_grid, phi_grid, indexing="ij")
        )
        return jax.vmap(
            lambda e, th, ph: _box_orbit_ic(
                potential, e, th, ph, r_floor, r_ceiling, t0
            )
        )(e_grid, theta_full, phi_full)
