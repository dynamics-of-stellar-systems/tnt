"""`XZGridFromOriginOrbitSampler`: a regular `(x, z)`-plane grid from near the
origin out to the equipotential.

At each `(energy, theta)`, samples radii directly over `[r_floor,
r_outer(theta)]` -- no search for where tube-orbit support actually begins
or ends, the kind a boundary-search sampler would do. This is exactly what
an irregular energy shell already falls back to in that scheme (see
`docs/source/orbit_sampling.md`'s "irregular shell" discussion): a uniform
grid reaching to the centre, with no boundary search at all, made the rule
here instead of the fallback for some shells.

Produces only the `+v_y` population -- the counter-rotating (`-v_y`) mirror
is a later, post-integration concern (reversing an integrated trajectory
needs no re-integration), not something this initial-condition generator
computes.
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
    _equipotential_radius,
    _potential_at,
    _xz_direction,
    _xz_orbit_ic,
)


class XZGridFromOriginOrbitSampler(AbstractOrbitSampler):
    """A regular `(x, z)`-plane start space only: `nE * nI1 * nI2` orbits.

    `nE` energy shells between `rmin`/`rmax` (converted to the resolved
    potential's own length unit), each paired with an open, bin-centred
    `nI1`-point grid of `theta in (0, pi/2)`. At each `(energy, theta)`,
    `nI2` orbits are launched from `r * (sin(theta), 0, cos(theta))` with
    velocity purely along `y`, from energy conservation
    (`common._xz_orbit_ic`), at radii spanning `[r_floor, r_outer(theta)]`
    -- `r_floor` a fixed near-origin floor (not zero: a cuspy or
    BH-dominated potential has `v_y -> infinity` as `r -> 0`), `r_outer`
    the equipotential radius in that direction. The radial sub-grid uses
    `(k - 0.9) / (nI2 - 0.8)`, not `linspace(0, 1, nI2)`: landing exactly on
    the equipotential is itself degenerate (`v_y = 0` there).
    """

    _type: ClassVar[str] = "XZGridFromOrigin"
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

        e_grid, theta_full = (
            a.reshape(-1) for a in jnp.meshgrid(energies, theta_grid, indexing="ij")
        )
        r_outer = jax.vmap(
            lambda e, th: _equipotential_radius(
                potential, _xz_direction(th), e, r_floor, r_ceiling, t0
            )
        )(e_grid, theta_full).reshape(self.nE, self.nI1)

        frac = (jnp.arange(1, self.nI2 + 1) - 0.9) / (self.nI2 - 0.8)
        r_grid_xz = r_floor + frac[None, None, :] * (r_outer[:, :, None] - r_floor)

        shape = (self.nE, self.nI1, self.nI2)
        e_full = jnp.broadcast_to(energies[:, None, None], shape).reshape(-1)
        theta_full_xz = jnp.broadcast_to(theta_grid[None, :, None], shape).reshape(-1)
        r_full = r_grid_xz.reshape(-1)

        return jax.vmap(lambda e, th, r: _xz_orbit_ic(potential, e, th, r, t0))(
            e_full, theta_full_xz, r_full
        )
