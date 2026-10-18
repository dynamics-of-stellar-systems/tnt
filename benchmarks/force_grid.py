"""Benchmark-only logarithmic spherical force interpolation.

This is deliberately outside TNT's runtime potential registry. Only static,
coaxial Gaussian/Plummer/NFW models are accepted, with positive restoring
coefficients at every node. There is no extrapolation or direct-force fallback.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import equinox as eqx
import galax.potential as gp
import jax
import jax.numpy as jnp
import numpy as np
from unxt import Quantity

from tnt.orbit_library.integration import (
    OrbitIntegrationSettings,
    _solve,
    require_static_potential,
)


@dataclass(frozen=True)
class GridSpec:
    """Explicit grid geometry; radii use the source potential's length unit."""

    shape: tuple[int, int, int]
    r_min: float
    r_max: float
    axis_floor: float = 1e-7

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 2
            for n in self.shape
        ):
            raise ValueError("Grid shape must contain three integers >= 2.")
        if not 0 < self.r_min < self.r_max < np.inf:
            raise ValueError("Require finite 0 < r_min < r_max.")
        if not 0 < self.axis_floor <= 1e-4:
            raise ValueError("Require 0 < axis_floor <= 1e-4.")


class ForceGrid(eqx.Module):
    """Trilinear log coefficients with a direct physical-energy evaluator."""

    direct: gp.AbstractPotential
    log_coefficients: jax.Array
    spec: GridSpec = eqx.field(static=True)

    @property
    def units(self):
        return self.direct.units

    def potential(self, position: Quantity, time: Quantity) -> Quantity:
        """Use the physical potential, not a surrogate interpolation energy."""
        return self.direct.potential(position, time)

    def acceleration(self, position: Quantity, time: Quantity) -> Quantity:
        """Evaluate finite in-domain coordinates, including symmetry planes."""
        xyz = position.ustrip(self.units["length"])
        if xyz.shape[-1:] != (3,):
            raise ValueError("Expected Cartesian coordinates with trailing axis 3.")
        flat = xyz.reshape((-1, 3))
        result = jax.vmap(self._acceleration)(flat).reshape(xyz.shape)
        return Quantity(result, self.units["length"] / self.units["time"]**2)

    def _acceleration(self, xyz: jax.Array) -> jax.Array:
        radius = jnp.linalg.norm(xyz)
        xyz = eqx.error_if(
            xyz,
            ~jnp.all(jnp.isfinite(xyz))
            | (radius < self.spec.r_min) | (radius > self.spec.r_max),
            "Force-grid query is non-finite or outside its declared radius range.",
        )
        absolute = jnp.abs(xyz)
        coordinates = jnp.array([
            (jnp.log(radius) - jnp.log(self.spec.r_min))
            / jnp.log(self.spec.r_max / self.spec.r_min),
            jnp.arctan2(jnp.linalg.norm(absolute[:2]), absolute[2]) / (jnp.pi / 2),
            jnp.arctan2(absolute[1], absolute[0]) / (jnp.pi / 2),
        ]) * (jnp.asarray(self.spec.shape) - 1)
        # Clipping selects the final valid cell at an inclusive upper boundary;
        # the radius check above forbids extrapolation. Angular folding is exact.
        lower = jnp.clip(
            jnp.floor(coordinates).astype(jnp.int32),
            0, jnp.asarray(self.spec.shape) - 2,
        )
        fraction = coordinates - lower
        offsets = jnp.array([
            [i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)
        ])
        indices = lower + offsets
        weights = jnp.prod(jnp.where(offsets, fraction, 1 - fraction), axis=1)
        values = self.log_coefficients[
            indices[:, 0], indices[:, 1], indices[:, 2]
        ]
        return -xyz * jnp.exp(jnp.sum(weights[:, None] * values, axis=0))


def _require_model(potential: gp.AbstractPotential) -> None:
    require_static_potential(potential)
    if isinstance(potential, Mapping):
        for component in potential.values():
            _require_model(component)
    elif type(potential) not in {
        gp.TriaxialGaussianPotential, gp.AxisymmetricGaussianPotential,
        gp.PlummerPotential, gp.NFWPotential,
    }:
        raise NotImplementedError("Grid supports only coaxial Gaussian/Plummer/NFW.")


@eqx.filter_jit
def build_grid(potential: gp.AbstractPotential, spec: GridSpec) -> ForceGrid:
    """Tabulate direct restoring coefficients in bounded batches of 128 nodes.

    Setup includes host/device node creation and direct force evaluation. The
    caller synchronizes the returned arrays before stopping a setup timer.
    """
    _require_model(potential)
    radii = jnp.geomspace(spec.r_min, spec.r_max, spec.shape[0])
    theta = jnp.linspace(0, jnp.pi / 2, spec.shape[1])
    phi = jnp.linspace(0, jnp.pi / 2, spec.shape[2])
    r, t, p = jnp.meshgrid(radii, theta, phi, indexing="ij")
    direction = jnp.stack(
        [jnp.sin(t) * jnp.cos(p), jnp.sin(t) * jnp.sin(p), jnp.cos(t)], axis=-1
    )
    # A positive floor computes the limiting a_i/x_i without dividing by zero.
    points = (r[..., None] * jnp.maximum(direction, spec.axis_floor)).reshape(-1, 3)
    length, time = potential.units["length"], potential.units["time"]

    def coefficient(point: jax.Array) -> jax.Array:
        force = potential.acceleration(
            Quantity(point, length), Quantity(0.0, time)
        ).ustrip(length / time**2)
        return -force / point

    values = jax.lax.map(coefficient, points, batch_size=128)
    values = eqx.error_if(
        values, ~jnp.all(jnp.isfinite(values) & (values > 0)),
        "Grid coefficients must be finite and strictly restoring.",
    )
    return ForceGrid(potential, jnp.log(values).reshape((*spec.shape, 3)), spec)


@eqx.filter_jit
def integrate_grid(
    grid: ForceGrid,
    position: Quantity,
    velocity: Quantity,
    reference_period: Quantity,
    key: jax.Array,
    settings: OrbitIntegrationSettings,
):
    """Use the unchanged production solve with this explicit experimental force.

    The public production API still rejects custom models. This benchmark-only
    adapter calls its internal solver after validating the concrete inputs.
    """
    length, time = grid.units["length"], grid.units["time"]
    q, v = position.ustrip(length), velocity.ustrip(length / time)
    period = reference_period.ustrip(time)
    if q.shape != (3,) or v.shape != (3,) or period.shape != ():
        raise ValueError("Expected position/velocity (3,) and a scalar period.")
    initial = jnp.concatenate([q, v])
    initial = eqx.error_if(
        initial, ~jnp.all(jnp.isfinite(initial)), "Initial coordinates must be finite."
    )
    period = eqx.error_if(
        period, ~jnp.isfinite(period) | (period <= 0), "Period must be positive."
    )
    return _solve(grid, initial, period, key, settings)
