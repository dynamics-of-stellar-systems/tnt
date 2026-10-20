"""Audit Galax Gaussian forces against independent adaptive scalar integrals.

Run from the repository root with ``python benchmarks/gaussian_forces.py``.
The pinned JAX scientific environment supplies SciPy. This is an offline
benchmark, not a runtime evaluator, fallback, or DYNAMITE dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import time
import warnings
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import galax.potential as gp
import jax.numpy as jnp
import numpy as np
from scipy.integrate import IntegrationWarning, quad
from unxt import Quantity

SHAPES = {
    "spherical": (1.0, 1.0),
    "oblate-0.8": (1.0, 0.8),
    "oblate-0.3": (1.0, 0.3),
    "oblate-0.1": (1.0, 0.1),
    "oblate-0.03": (1.0, 0.03),
    "oblate-0.01": (1.0, 0.01),
    "triaxial-0.8-0.6": (0.8, 0.6),
    "triaxial-0.7-0.3": (0.7, 0.3),
    "triaxial-0.3-0.1": (0.3, 0.1),
    "triaxial-0.1-0.03": (0.1, 0.03),
    "triaxial-0.1-0.01": (0.1, 0.01),
}
RADII = np.array([1e-4, 0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000, 10000])
DIRECTIONS = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
        [1, 0.01, 0.001],
        [0.01, 1, 0.001],
        [-1, 0.2, -0.5],
    ],
    dtype=float,
)
DIRECTIONS /= np.linalg.norm(DIRECTIONS, axis=1)[:, None]


def reference_force(point: np.ndarray, p: float, q: float) -> np.ndarray:
    """Return acceleration for G=M=sigma=1 via scalar adaptive quadrature.

    Differentiate the ellipsoidal Gaussian potential analytically, then
    integrate each force component independently (not Galax autodiff).
    u=s*max(1,r) resolves the narrow s~1/r contribution at large radii.
    Beyond u=12 the exponential is bounded by exp(-72) for q<=p<=1;
    that tail is negligible at the audit's precision. Near s=1, adaptive
    subdivision resolves the boundary layer of flattened Gaussians.
    """
    scale = max(1.0, float(np.linalg.norm(point)))
    upper = min(scale, 12.0)
    result = []
    for axis in range(3):
        if point[axis] == 0:
            result.append(0.0)
            continue

        def integrand(u: float, axis: int = axis) -> float:
            s2 = (u / scale) ** 2
            denominators = np.array([1.0, 1 - s2 + p * p * s2, 1 - s2 + q * q * s2])
            exponent = -0.5 * s2 * np.sum(point**2 / denominators)
            return float(
                u
                * u
                * np.exp(exponent)
                / (denominators[axis] * np.sqrt(np.prod(denominators)))
            )

        value, error = quad(
            integrand,
            0,
            upper,
            epsabs=1e-13,
            epsrel=2e-12,
            limit=300,
        )
        if error > 1e-10 * max(abs(value), 1.0):
            raise RuntimeError(f"Unresolved reference integral at {point}, {p=}, {q=}")
        result.append(-np.sqrt(2 / np.pi) * point[axis] * value / scale**3)
    return np.asarray(result)


@eqx.filter_jit
def batch_force(potential: gp.AbstractPotential, points: jax.Array) -> jax.Array:
    """Evaluate a synchronized batch, matching the force used by the solver."""
    return potential.acceleration(Quantity(points, "kpc"), Quantity(0.0, "Myr")).ustrip(
        "kpc/Myr2"
    )


@eqx.filter_jit
def sequential_force(potential: gp.AbstractPotential, points: jax.Array) -> jax.Array:
    """Evaluate serial force requests inside a compiled scan, excluding dispatch."""

    def step(carry: None, point: jax.Array) -> tuple[None, jax.Array]:
        force = potential.acceleration(
            Quantity(point, "kpc"), Quantity(0.0, "Myr")
        ).ustrip("kpc/Myr2")
        return carry, force

    return jax.lax.scan(step, None, points)[1]


def measure(
    function: Callable, potential: gp.AbstractPotential, points: jax.Array, repeats: int
) -> tuple[np.ndarray, dict]:
    """Separate first-call compilation from repeated synchronized execution."""
    started = time.perf_counter()
    values = function(potential, points).block_until_ready()
    first = time.perf_counter() - started
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        function(potential, points).block_until_ready()
        timings.append(time.perf_counter() - started)
    return np.asarray(values), {
        "first_call_seconds": first,
        "warm_seconds": timings,
        "warm_median_seconds": float(np.median(timings)),
    }


def run(orders: list[int], repeats: int) -> dict:
    """Audit both Gaussian classes and preserve probes, errors, and provenance."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError("The Gaussian force audit requires float64.")
    warnings.simplefilter("error", IntegrationWarning)
    points = np.concatenate(
        (
            np.zeros((1, 3)),
            (RADII[:, None, None] * DIRECTIONS[None, :, :]).reshape(-1, 3),
        )
    )
    point_radii = np.linalg.norm(points, axis=1)
    # Longer batches/scans reduce dispatch and timer noise in warm timings.
    timing_points = jnp.tile(jnp.asarray(points), (32, 1))
    rows = []
    for shape, (p, q) in SHAPES.items():
        reference = np.stack([reference_force(point, p, q) for point in points])
        # Independent analytic shell-theorem check on the spherical reference.
        if p == q == 1:
            from math import erf

            radius = point_radii[1:]
            enclosed = np.array([erf(r / np.sqrt(2)) for r in radius])
            enclosed -= np.sqrt(2 / np.pi) * radius * np.exp(-(radius**2) / 2)
            analytic = -points[1:] * (enclosed / radius**3)[:, None]
            np.testing.assert_allclose(reference[1:], analytic, rtol=1e-7, atol=2e-13)
        kinds = [gp.TriaxialGaussianPotential]
        if p == 1:
            kinds.append(gp.AxisymmetricGaussianPotential)
        for kind in kinds:
            for order in orders:
                potential = kind(
                    m_tot=Quantity(1.0, "Msun"),
                    r_s=Quantity(1.0, "kpc"),
                    q2=q,
                    **({"q1": p} if kind is gp.TriaxialGaussianPotential else {}),
                    units="galactic",
                    constants={"G": Quantity(1.0, "kpc3/(Msun Myr2)")},
                    integration_order=order,
                )
                force, batch_timing = measure(
                    batch_force, potential, timing_points, repeats
                )
                serial, sequential_timing = measure(
                    sequential_force, potential, timing_points, repeats
                )
                np.testing.assert_allclose(serial, force, rtol=1e-12, atol=1e-13)
                force = force[: len(points)]
                if not np.all(np.isfinite(force)):
                    raise RuntimeError(f"Non-finite forces: {shape}, {order}")
                absolute = np.linalg.norm(force - reference, axis=1)
                reference_norm = np.linalg.norm(reference, axis=1)
                relative = np.divide(
                    absolute,
                    reference_norm,
                    out=np.zeros_like(absolute),
                    where=reference_norm > 0,
                )
                rows.append(
                    {
                        "shape": shape,
                        "p": p,
                        "q": q,
                        "class": kind.__name__,
                        "order": order,
                        "forces": force.tolist(),
                        "reference_forces": reference.tolist(),
                        "relative_errors": relative.tolist(),
                        "absolute_errors": absolute.tolist(),
                        "max_relative_error": float(relative.max()),
                        "max_relative_error_r_le_100": float(
                            relative[point_radii <= 100].max()
                        ),
                        "max_relative_error_r_le_1000": float(
                            relative[point_radii <= 1000].max()
                        ),
                        "batch_timing": batch_timing,
                        "sequential_timing": sequential_timing,
                    }
                )
                print(
                    kind.__name__,
                    shape,
                    order,
                    f"max rel={relative.max():.3g}",
                    flush=True,
                )
    return report(points, point_radii, rows, len(timing_points))


def report(
    points: np.ndarray, point_radii: np.ndarray, rows: list[dict], timing_requests: int
) -> dict:
    """Attach reproducible environment and source identities to measured results."""
    return {
        "schema_version": 1,
        "tnt_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in ["jax", "jaxlib", "galax", "numpy", "scipy"]
            },
            "devices": [str(device) for device in jax.devices()],
            "float64": True,
        },
        "potential_source_sha256": {
            kind.__name__: hashlib.sha256(inspect.getsource(kind).encode()).hexdigest()
            for kind in [gp.TriaxialGaussianPotential, gp.AxisymmetricGaussianPotential]
        },
        "normalization": "G=M=sigma=1; positions in sigma, acceleration in GM/sigma^2",
        "reference": (
            "Independent analytic force integrands; scipy.integrate.quad, "
            "epsabs=1e-13, epsrel=2e-12; u=s*max(1,r), upper=min(max(1,r),12)"
        ),
        "points": points.tolist(),
        "radii": point_radii.tolist(),
        "rows": rows,
        "force_requests_per_timing": timing_requests,
        "timing_scope": (
            "Same Linux CPU; repeated synchronized force evaluations, batch and "
            "compiled serial scan; not full orbit timings or a direct-versus-grid "
            "benchmark."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orders", type=int, nargs="+", default=[50, 100, 200, 400, 800]
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/gaussian_forces.json")
    )
    args = parser.parse_args()
    if args.repeats < 1 or any(order < 1 for order in args.orders):
        parser.error("Orders and repeats must be positive.")
    result = run(args.orders, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
