"""`StationaryGridOrbitSampler`: a pure stationary-start-space (box orbit) sampler.

Box orbits only, launched from rest exactly on the equipotential surface
(Schwarzschild 1979) -- no tube-orbit start space at all, unlike a combined
sampler that would also need a radial grid dimension in addition to its two
angles. `nE`/`nI2`/`nI3` match DYNAMITE's own field names and roles
(`orbitstart_f.f90`'s `nEner`/`nI2`/`nI3`) across all three samplers --
`nI2` always the (first, and here only) angular grid count, `nI3` always
the second non-energy dimension, `phi` here rather than a radial count.
"""

from __future__ import annotations

from typing import ClassVar

import galax.potential as gp
import jax
import jax.numpy as jnp
from unxt import Quantity

from tnt.orbit_library.base import AbstractOrbitSampler
from tnt.orbit_library.common import (
    _R_CEILING_FACTOR,
    _R_FLOOR_FACTOR,
    _box_orbit_ic,
    _potential_at,
)


class StationaryGridOrbitSampler(AbstractOrbitSampler):
    """A stationary start space only: `nE * nI2 * nI3` box orbits.

    `nE` energy shells between `rmin`/`rmax` (converted to the resolved
    potential's own length unit), each paired with an open, bin-centred
    `nI2 * nI3` grid of `theta, phi in (0, pi/2)` -- never exactly `0` or
    `pi/2`, which would respectively collapse every `phi` to one duplicate
    point on the z-axis, or launch a box orbit exactly in the equatorial
    plane (able to pass through the centre of a cuspy potential). Every
    orbit starts from rest exactly on the equipotential surface at its
    `(energy, theta, phi)`.
    """

    _type: ClassVar[str] = "StationaryGrid"
    # Box orbits have no definite sense of circulation to begin with --
    # nothing to mirror.
    _add_reverse_copies: ClassVar[bool] = False
    rmin: Quantity
    rmax: Quantity
    nE: int
    nI2: int
    nI3: int

    def n_bundles(self) -> int:
        return self.nE * self.nI2 * self.nI3

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

        theta_grid = (jnp.arange(self.nI2) + 0.5) * (jnp.pi / 2) / self.nI2
        phi_grid = (jnp.arange(self.nI3) + 0.5) * (jnp.pi / 2) / self.nI3

        e_grid, theta_full, phi_full = (
            a.reshape(-1)
            for a in jnp.meshgrid(energies, theta_grid, phi_grid, indexing="ij")
        )
        return jax.vmap(
            lambda e, th, ph: _box_orbit_ic(
                potential, e, th, ph, r_floor, r_ceiling, t0
            )
        )(e_grid, theta_full, phi_full)
