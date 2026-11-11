"""Benchmark-only logarithmic spherical force interpolation.

This is deliberately outside TNT's runtime potential registry. Only static,
coaxial Gaussian/Plummer/NFW models are accepted. Force-coefficient methods
require positive restoring coefficients at every node. Extrapolation is never
used; an explicit policy either rejects out-of-domain queries or uses direct forces.
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
    interpolation: str = "linear"
    outside_policy: str = "reject"

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 2
            for n in self.shape
        ):
            raise ValueError("Grid shape must contain three integers >= 2.")
        if self.interpolation not in {"linear", "cubic", "hermite"}:
            raise ValueError("Interpolation must be linear or cubic or hermite.")
        if self.interpolation == "cubic" and min(self.shape) < 4:
            raise ValueError("Cubic interpolation needs four nodes per axis.")
        if self.outside_policy not in {"reject", "direct"}:
            raise ValueError("Outside policy must be reject or direct.")
        if not 0 < self.r_min < self.r_max < np.inf:
            raise ValueError("Require finite 0 < r_min < r_max.")
        if not 0 < self.axis_floor <= 1e-4:
            raise ValueError("Require 0 < axis_floor <= 1e-4.")


class ForceGrid(eqx.Module):
    """Linear or cubic log-force coefficients with direct physical energy.

    Hermite grids store potential derivative jets in log_coefficients (a
    historical field name), with shape (*spec.shape, 2, 2, 2). Their forces
    derive from one C1 potential. potential() remains the direct energy
    evaluator; interpolated_potential() exposes the approximate Hamiltonian.

    For cubic grids, log_coefficients stores interpolating B-spline
    coefficients, not node values. The radial spline has natural endpoint
    conditions; angular splines have zero endpoint slopes to preserve smooth
    reflection symmetry. This does not enforce a conservative force field.
    """

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
        if xyz.ndim == 1:
            result = self._acceleration(xyz)
        elif self.spec.outside_policy == "direct":
            # Preserve scalar branch selection; vmap(cond) can evaluate both.
            result = jax.lax.map(
                lambda point: self._acceleration(point), flat
            ).reshape(xyz.shape)
        else:
            result = jax.vmap(self._acceleration)(flat).reshape(xyz.shape)
        return Quantity(result, self.units["length"] / self.units["time"]**2)

    def interpolated_potential(self, position: Quantity, time: Quantity) -> Quantity:
        """Hermite energy, with direct values outside if fallback is enabled.

        A hard boundary switch need not be globally potential-consistent.
        Ordinary potential() always retains direct physical diagnostics.
        """
        if self.spec.interpolation != "hermite":
            raise ValueError("Only Hermite grids interpolate a scalar potential.")
        xyz = position.ustrip(self.units["length"])
        if xyz.shape[-1:] != (3,):
            raise ValueError("Expected Cartesian coordinates with trailing axis 3.")

        def value(point):
            radius = jnp.linalg.norm(point)
            point = eqx.error_if(
                point, ~jnp.all(jnp.isfinite(point))
                | (radius < self.spec.r_min) | (radius > self.spec.r_max),
                "Force-grid query is non-finite or outside its declared radius range.",
            )
            x, y, z = jnp.abs(point)
            coordinates = jnp.array([
                jnp.log(radius / self.spec.r_min),
                jnp.arctan2(jnp.hypot(x, y), z), jnp.arctan2(y, x),
            ]) / _grid_spacing(self.spec)
            return _hermite_value(self.log_coefficients, coordinates)

        def selected_value(point):
            point = eqx.error_if(point, ~jnp.all(jnp.isfinite(point)),
                                 "Force-grid query must be finite.")
            radius = jnp.linalg.norm(point)
            return jax.lax.cond(
                (radius < self.spec.r_min) | (radius > self.spec.r_max),
                lambda q: self.direct.potential(
                    Quantity(q, self.units["length"]), time
                ).ustrip(self.units["length"]**2 / self.units["time"]**2),
                value, point,
            )

        if self.spec.outside_policy == "direct":
            if xyz.ndim == 1:
                values = selected_value(xyz)
            else:
                values = jax.lax.map(selected_value, xyz.reshape(-1, 3)).reshape(
                    xyz.shape[:-1])
        else:
            values = jax.vmap(value)(xyz.reshape(-1, 3)).reshape(xyz.shape[:-1])
        return Quantity(values, self.units["length"]**2 / self.units["time"]**2)

    def _acceleration(self, xyz: jax.Array) -> jax.Array:
        if self.spec.outside_policy == "reject":
            return self._in_domain_acceleration(xyz)
        xyz = eqx.error_if(xyz, ~jnp.all(jnp.isfinite(xyz)),
                           "Force-grid query must be finite.")
        radius = jnp.linalg.norm(xyz)
        return jax.lax.cond(
            (radius < self.spec.r_min) | (radius > self.spec.r_max),
            lambda point: self.direct.acceleration(
                Quantity(point, self.units["length"]), Quantity(0., self.units["time"])
            ).ustrip(self.units["length"] / self.units["time"]**2),
            lambda point: self._in_domain_acceleration(point), xyz,
        )

    def _in_domain_acceleration(self, xyz: jax.Array) -> jax.Array:
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
        if self.spec.interpolation == "hermite":
            gradient = jax.grad(_hermite_value, argnums=1)(
                self.log_coefficients, coordinates
            ) / _grid_spacing(self.spec)
            rho = jnp.linalg.norm(absolute[:2])
            pole = rho == 0
            safe_rho = jnp.where(pole, 1.0, rho)
            cp, sp = absolute[0] / safe_rho, absolute[1] / safe_rho
            st, ct = rho / radius, absolute[2] / radius
            radial = absolute / radius
            polar = jnp.array([ct * cp, ct * sp, -st])
            azimuthal = jnp.array([-sp, cp, 0.0])
            force = -(gradient[0] * radial + gradient[1] * polar) / radius
            force -= gradient[2] * azimuthal / safe_rho
            return force * jnp.sign(xyz)
        if self.spec.interpolation == "cubic":
            return -xyz * jnp.exp(_cubic_value(self.log_coefficients, coordinates))
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



def _spline_coefficients(values: jax.Array) -> jax.Array:
    """Prefilter node values into tensor-product cardinal cubic coefficients."""
    for axis in range(3):
        n = values.shape[axis]
        matrix = (4 * jnp.eye(n) + jnp.eye(n, k=1) + jnp.eye(n, k=-1)) / 6
        if axis == 0:
            # Linear ghost extension makes the radial second derivative zero.
            matrix = matrix.at[0].set(jnp.eye(n)[0])
            matrix = matrix.at[-1].set(jnp.eye(n)[-1])
        else:
            # Even ghost extension imposes zero angular derivative.
            matrix = matrix.at[0, 1].set(2 / 6)
            matrix = matrix.at[-1, -2].set(2 / 6)
        moved = jnp.moveaxis(values, axis, 0)
        solved = jnp.linalg.solve(matrix, moved.reshape(n, -1))
        values = jnp.moveaxis(solved.reshape(moved.shape), 0, axis)
    return values


def _cubic_value(coefficients: jax.Array, coordinates: jax.Array) -> jax.Array:
    """Evaluate a local 4x4x4 B-spline stencil, including endpoint ghosts."""
    shape = jnp.asarray(coefficients.shape[:3])
    lower = jnp.clip(jnp.floor(coordinates).astype(jnp.int32), 0, shape - 2)
    t = coordinates - lower
    weights = jnp.stack([
        (1 - t)**3, 3*t**3 - 6*t**2 + 4,
        -3*t**3 + 3*t**2 + 3*t + 1, t**3,
    ], axis=1) / 6
    # Pad only the selected stencil logically: radial ghosts are linearly
    # extended, angular ghosts reflected. No full padded table per query.
    indices = lower[:, None] + jnp.arange(-1, 3)
    radial = indices[0]
    clipped = jnp.clip(radial, 0, shape[0] - 1)
    neighbor = jnp.where(radial < 0, 1, shape[0] - 2)
    angular = []
    for axis in (1, 2):
        index = indices[axis]
        angular.append(jnp.where(index < 0, -index,
                                jnp.where(index >= shape[axis],
                                          2 * (shape[axis] - 1) - index, index)))
    j, k = angular
    stencil = coefficients[clipped[:, None, None], j[None, :, None], k[None, None, :]]
    adjacent = coefficients[neighbor[:, None, None], j[None, :, None], k[None, None, :]]
    outside = (radial < 0) | (radial >= shape[0])
    stencil = jnp.where(outside[:, None, None, None], 2 * stencil - adjacent, stencil)
    return jnp.einsum("i,j,k,ijkc->c", *weights, stencil)



def _grid_spacing(spec: GridSpec) -> jax.Array:
    return jnp.array([jnp.log(spec.r_max / spec.r_min), jnp.pi / 2, jnp.pi / 2
                      ]) / (jnp.asarray(spec.shape) - 1)


def _hermite_value(table: jax.Array, coordinates: jax.Array) -> jax.Array:
    """Tensor Hermite basis; last axes select derivative order 0 or 1."""
    lower = jnp.clip(jnp.floor(coordinates).astype(jnp.int32),
                     0, jnp.asarray(table.shape[:3]) - 2)
    t = coordinates - lower
    # basis[axis, endpoint, derivative_order]
    basis = jnp.stack([
        jnp.stack([2*t**3 - 3*t**2 + 1, t**3 - 2*t**2 + t], axis=-1),
        jnp.stack([-2*t**3 + 3*t**2, t**3 - t**2], axis=-1),
    ], axis=1)
    i, j, k = (lower[a] + jnp.arange(2) for a in range(3))
    cell = table[i[:, None, None], j[None, :, None], k[None, None, :]]
    return jnp.einsum("ia,jb,kc,ijkabc->", *basis, cell)


def _hermite_table(potential: gp.AbstractPotential, spec: GridSpec) -> jax.Array:
    """Autodiff all eight derivative combinations; no finite differences."""
    spacing = _grid_spacing(spec)
    axes = [jnp.linspace(0, n - 1, n) for n in spec.shape]
    nodes = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    length, time = potential.units["length"], potential.units["time"]

    def scalar(index):
        u, theta, phi = index * spacing
        r = spec.r_min * jnp.exp(u)
        xyz = r * jnp.array([jnp.sin(theta) * jnp.cos(phi),
                             jnp.sin(theta) * jnp.sin(phi), jnp.cos(theta)])
        return potential.potential(Quantity(xyz, length), Quantity(0., time)
                                   ).ustrip(length**2 / time**2)

    def derivative(function, axis):
        def evaluate(index):
            return jax.jvp(function, (index,), (jnp.eye(3)[axis],))[1]
        return evaluate

    functions = []
    for a in range(2):
        for b in range(2):
            for c in range(2):
                function = scalar
                for axis, order in enumerate((a, b, c)):
                    if order:
                        function = derivative(function, axis)
                functions.append(function)

    def jet(index):
        return jnp.stack([function(index) for function in functions]).reshape(2, 2, 2)

    table = jax.lax.map(jet, nodes, batch_size=16).reshape((*spec.shape, 2, 2, 2))
    table = eqx.error_if(table, ~jnp.all(jnp.isfinite(table)),
                         "Hermite potential derivatives must be finite.")
    # Exact symmetry identities, removing trigonometric roundoff at boundaries.
    table = table.at[:, 0, :, :, 1, :].set(0)
    table = table.at[:, -1, :, :, 1, :].set(0)
    table = table.at[:, :, 0, :, :, 1].set(0)
    table = table.at[:, :, -1, :, :, 1].set(0)
    table = table.at[:, 0, :, :, 0, 0].set(table[:, 0, :1, :, 0, 0])
    table = table.at[:, 0, :, :, :, 1].set(0)
    return table


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
    if spec.interpolation == "hermite":
        return ForceGrid(potential, _hermite_table(potential, spec), spec)
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
    table = jnp.log(values).reshape((*spec.shape, 3))
    if spec.interpolation == "cubic":
        table = _spline_coefficients(table)
    return ForceGrid(potential, table, spec)


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
