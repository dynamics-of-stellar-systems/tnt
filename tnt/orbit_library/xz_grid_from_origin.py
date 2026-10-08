"""`XZGridFromOriginOrbitSampler`: a regular `(x, z)`-plane grid from near the
origin out to the equipotential.

At each `(energy, theta)`, samples radii directly over `[r_floor,
r_outer(theta)]` -- no search for where tube-orbit support actually begins
or ends, the kind `xz_grid_from_boundary.py`'s boundary search does. This is
exactly what an irregular energy shell there already falls back to (see
`docs/source/orbit_sampling.md`'s "irregular shell" discussion): a uniform
grid reaching to the centre, with no boundary search at all, made the rule
here instead of the fallback for some shells -- `_single_shell_ics` below is
that same fallback, reused directly rather than reimplemented.

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
    _R_FLOOR_FACTOR,
    EquipotentialSearchError,
    _equipotential_radius,
    _potential_at,
    _xz_direction,
    _xz_orbit_ic,
)


def _single_shell_ics(
    potential: gp.AbstractPotential,
    energy: jnp.ndarray,
    theta_grid: jnp.ndarray,
    r_floor: jnp.ndarray,
    r_ref: jnp.ndarray,
    t0: Quantity,
    nI3: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One energy shell's `(x, z)`-plane grid: `len(theta_grid) * nI3` orbits.

    At each `theta` in `theta_grid`, `nI3` orbits span `[r_floor,
    r_outer(theta)]` via the "nearly closed" fractional spacing `(k - 0.9) /
    (nI3 - 0.8)` -- landing exactly on `r_floor` has no special meaning, but
    landing exactly on the equipotential is degenerate (`v_y = 0` there), so
    neither endpoint is ever sampled exactly. `r_ref` is this shell's own
    x-axis equipotential radius, setting `_equipotential_radius`'s search
    bracket for `r_outer` -- unrelated to `r_floor`, the fixed physical
    near-origin floor.

    Returns:
        `(ics, valid)`: `valid` is `False` if `_equipotential_radius`
        failed for any `theta` in this shell.
    """
    r_outer, valid = jax.vmap(
        lambda th: _equipotential_radius(
            potential, _xz_direction(th), energy, r_ref, t0
        )
    )(theta_grid)

    frac = (jnp.arange(1, nI3 + 1) - 0.9) / (nI3 - 0.8)
    r_grid = r_floor + frac[None, :] * (r_outer[:, None] - r_floor)

    n_theta = theta_grid.shape[0]
    shape = (n_theta, nI3)
    theta_full = jnp.broadcast_to(theta_grid[:, None], shape).reshape(-1)
    r_full = r_grid.reshape(-1)

    ics = jax.vmap(lambda th, r: _xz_orbit_ic(potential, energy, th, r, t0))(
        theta_full, r_full
    )
    return ics, jnp.all(valid)


class XZGridFromOriginOrbitSampler(AbstractOrbitSampler):
    """A regular `(x, z)`-plane start space only: `nE * nI2 * nI3` orbits.

    `nE` energy shells between `rmin`/`rmax` (converted to the resolved
    potential's own length unit), each paired with an open, bin-centred
    `nI2`-point grid of `theta in (0, pi/2)`. At each `(energy, theta)`,
    `nI3` orbits are launched from `r * (sin(theta), 0, cos(theta))` with
    velocity purely along `y`, from energy conservation
    (`common._xz_orbit_ic`), at radii spanning `[r_floor, r_outer(theta)]`
    -- `r_floor` a fixed near-origin floor (not zero: a cuspy or
    BH-dominated potential has `v_y -> infinity` as `r -> 0`), `r_outer`
    the equipotential radius in that direction. See `_single_shell_ics` for
    the radial sub-grid formula.
    """

    _type: ClassVar[str] = "XZGridFromOrigin"
    # A static-potential orbit's counter-rotating partner is free (no
    # re-integration): negate velocity at each recorded point, keep
    # position -- valid because `F(x, t1-s) = F(x, s)` holds unconditionally
    # for a time-independent potential (not generally true for a rotating
    # pattern speed, not yet supported here).
    _add_reverse_copies: ClassVar[bool] = True
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

        r_grid = 10.0 ** jnp.linspace(jnp.log10(rmin_val), jnp.log10(rmax_val), self.nE)
        x_axis_points = jnp.stack(
            [r_grid, jnp.zeros_like(r_grid), jnp.zeros_like(r_grid)], axis=-1
        )
        energies = jax.vmap(lambda pos: _potential_at(potential, pos, t0))(
            x_axis_points
        )

        theta_grid = (jnp.arange(self.nI2) + 0.5) * (jnp.pi / 2) / self.nI2

        ics, valid = jax.vmap(
            lambda e, r_ref: _single_shell_ics(
                potential, e, theta_grid, r_floor, r_ref, t0, self.nI3
            )
        )(energies, r_grid)
        if not bool(jnp.all(valid)):
            raise EquipotentialSearchError(
                "Could not locate the equipotential for every (energy, theta) "
                "shell -- see `_equipotential_radius` for what this requires."
            )
        return ics.reshape(-1, 6)
