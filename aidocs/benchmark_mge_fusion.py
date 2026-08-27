"""Benchmark: composing vs. fusing the triaxial MGE potential quadrature.

Reproduces the methodology from GitHub discussion #34 (dynamics-of-stellar-
systems/tnt): TNT's shipped MGE composite types sum one
`galax.potential.TriaxialGaussianPotential` per Gaussian component via
`CompositePotential` ("sum of integrals" -- each component runs its own
independent Gauss-Legendre quadrature). The alternative, implemented as
`tnt.potential.fused_triaxial_gaussian_composite.FusedTriaxialGaussianCompositePotential`,
fuses all N components into one shared quadrature via `jax.lax.scan` over
the nodes, summing every component's integrand at each node before
integrating once ("integral of sum") -- mathematically identical, since
integration is linear and every component uses the same fixed quadrature
order.

Benchmarks the two implementations directly against each other -- not a
re-derived copy of the fused math, the actual class that would ship -- so
what's measured here is exactly what a real switch-over would run.
Consequently this script now needs the TNT environment itself (not just
bare galax/jax/unxt): run it via `uv run python aidocs/benchmark_mge_fusion.py`
from a checkout of this repo, on whatever CPU/GPU hardware you want numbers
for -- JAX picks up whatever backend is installed/visible, nothing else
about the script changes.

Component parameters and query points are drawn from `numpy.random.default_rng`
with a fixed seed -- a pure-software PRNG with no dependency on JAX or the
accelerator, so the exact same numbers are generated on CPU and GPU runs
alike. Each run also prints a short fingerprint hash per N as a cheap
sanity check that two runs (e.g. one per machine) really did use identical
inputs before comparing their timings.

Results are written to a CSV (default: auto-named from the JAX backend and
a timestamp) as well as printed to stdout, so runs from different machines
can be collated later (e.g. concatenated and loaded with pandas).

Usage:
    python benchmark_mge_fusion.py [--n 5 25 125] \\
        [--batch 1 10 100 1000 10000 100000] [--repeats 5] [--output PATH]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import galax.potential as gp
import jax
import jax.numpy as jnp
import numpy as np
import unxt as u

from tnt.potential.fused_triaxial_gaussian_composite import (
    FusedTriaxialGaussianCompositePotential,
)

UNITS = "galactic"
INTEGRATION_ORDER = 50


# ============================================================================
# Component generation


def _random_components(
    n: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """N triaxial Gaussian components with plausible MGE-like parameters.

    `numpy.random.default_rng` is a deterministic, pure-software PRNG --
    the same seed produces the same arrays regardless of CPU/GPU backend.
    """
    rng = np.random.default_rng(seed)
    m_tot = rng.uniform(1e6, 1e9, n)
    r_s = rng.uniform(0.05, 5.0, n)
    q1 = rng.uniform(0.5, 1.0, n)
    q2 = rng.uniform(0.4, q1)
    return m_tot, r_s, q1, q2


def _component_fingerprint(
    m_tot: np.ndarray, r_s: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> str:
    """Short hash of the generated components, to sanity-check reproducibility."""
    payload = np.concatenate([m_tot, r_s, q1, q2]).tobytes()
    return hashlib.sha256(payload).hexdigest()[:12]


def _build_children(
    m_tot: np.ndarray, r_s: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> dict[str, gp.TriaxialGaussianPotential]:
    return {
        str(i): gp.TriaxialGaussianPotential(
            m_tot=u.Quantity(float(m_tot[i]), "Msun"),
            r_s=u.Quantity(float(r_s[i]), "kpc"),
            q1=float(q1[i]),
            q2=float(q2[i]),
            units=UNITS,
            integration_order=INTEGRATION_ORDER,
        )
        for i in range(len(m_tot))
    }


# ============================================================================
# Two implementations, each exposed as potential_fn(xyz) -> scalar, so
# potential/gradient timing uses one identical vmap/grad harness for both.


def _composite_potential_fn(children: dict[str, gp.TriaxialGaussianPotential]):
    pot = gp.CompositePotential(children, units=UNITS)

    def potential_fn(xyz: jnp.ndarray) -> jnp.ndarray:
        return pot._potential(xyz, 0.0)

    return potential_fn


def _fused_potential_fn(children: dict[str, gp.TriaxialGaussianPotential]):
    pot = FusedTriaxialGaussianCompositePotential(children)

    def potential_fn(xyz: jnp.ndarray) -> jnp.ndarray:
        return pot._potential(xyz, 0.0)

    return potential_fn


# ============================================================================
# Timing harness


@dataclass
class Timing:
    potential_ms: float
    gradient_ms: float


def _time_call(fn, *args, repeats: int) -> float:
    fn(*args).block_until_ready()  # warmup / compile, not timed
    start = time.perf_counter()
    for _ in range(repeats):
        result = fn(*args)
    result.block_until_ready()
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / repeats


def _benchmark_one(potential_fn, xyz: jnp.ndarray, *, repeats: int) -> Timing:
    potential_batched = jax.jit(jax.vmap(potential_fn))
    gradient_batched = jax.jit(jax.vmap(jax.grad(potential_fn)))
    potential_ms = _time_call(potential_batched, xyz, repeats=repeats)
    gradient_ms = _time_call(gradient_batched, xyz, repeats=repeats)
    return Timing(potential_ms, gradient_ms)


def _default_output_path(backend: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    host = platform.node().split(".")[0] or "unknown-host"
    return f"benchmark_mge_fusion_{backend}_{host}_{timestamp}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[5, 25, 125])
    parser.add_argument(
        "--batch", type=int, nargs="+", default=[1, 10, 100, 1_000, 10_000, 100_000]
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    backend = jax.default_backend()
    print(f"jax.devices(): {jax.devices()}")
    print(f"jax default backend: {backend}")
    print()

    output_path = args.output or _default_output_path(backend)
    device_name = str(jax.devices()[0])

    rng = np.random.default_rng(42)
    header = (
        f"{'N':>3} {'B':>7} {'impl':<10} "
        f"{'potential (ms)':>15} {'gradient (ms)':>15}"
    )
    print(header)
    print("-" * len(header))

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "timestamp",
                "device",
                "backend",
                "n",
                "component_fingerprint",
                "batch",
                "impl",
                "potential_ms",
                "gradient_ms",
            ]
        )

        for n in args.n:
            m_tot, r_s, q1, q2 = _random_components(n, seed=0)
            fingerprint = _component_fingerprint(m_tot, r_s, q1, q2)
            print(f"# N={n} component fingerprint: {fingerprint}")
            children = _build_children(m_tot, r_s, q1, q2)
            implementations = {
                "composite": _composite_potential_fn(children),
                "fused": _fused_potential_fn(children),
            }
            for batch_size in args.batch:
                xyz = jnp.asarray(rng.uniform(-10.0, 10.0, size=(batch_size, 3)))
                for name, potential_fn in implementations.items():
                    timing = _benchmark_one(potential_fn, xyz, repeats=args.repeats)
                    print(
                        f"{n:>3} {batch_size:>7} {name:<10} "
                        f"{timing.potential_ms:>15.3f} {timing.gradient_ms:>15.3f}"
                    )
                    writer.writerow(
                        [
                            datetime.now(UTC).isoformat(),
                            device_name,
                            backend,
                            n,
                            fingerprint,
                            batch_size,
                            name,
                            f"{timing.potential_ms:.6f}",
                            f"{timing.gradient_ms:.6f}",
                        ]
                    )
                    csv_file.flush()

    print()
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
