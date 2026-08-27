"""Benchmark: composing vs. fusing the triaxial MGE potential quadrature.

Reproduces the methodology from GitHub discussion #34 (dynamics-of-stellar-
systems/tnt): TNT's shipped MGE composite types sum one
`galax.potential.TriaxialGaussianPotential` per Gaussian component via
`CompositePotential` ("sum of integrals" -- each component runs its own
independent Gauss-Legendre quadrature). The alternative fuses all N
components into one shared quadrature, summing their integrands at each
node before integrating once ("integral of sum") -- mathematically
identical, since integration is linear and every component uses the same
fixed quadrature order. Two ways to do the node reduction are compared:
plain vectorized array ops ("array-fused") and `jax.lax.scan` over nodes
("scan-fused").

Self-contained and hardware-agnostic: run this unmodified on CPU or GPU
(JAX picks up whatever backend is installed/visible) to get directly
comparable numbers. No TNT imports -- only galax/jax/unxt -- so it can run
anywhere those are installed.

Usage:
    python benchmark_mge_fusion.py [--n 6 15] [--batch 1 100 10000] [--repeats 5]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import galax.potential as gp
import jax
import jax.numpy as jnp
import numpy as np
import unxt as u

UNITS = "galactic"
INTEGRATION_ORDER = 50


# ============================================================================
# Component generation


def _random_components(
    n: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """N triaxial Gaussian components with plausible MGE-like parameters."""
    rng = np.random.default_rng(seed)
    m_tot = rng.uniform(1e6, 1e9, n)
    r_s = rng.uniform(0.05, 5.0, n)
    q1 = rng.uniform(0.5, 1.0, n)
    q2 = rng.uniform(0.4, q1)
    return m_tot, r_s, q1, q2


def _gauss_legendre_01(order: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gauss-Legendre nodes/weights on [0, 1] (galax's own convention)."""
    x, w = np.polynomial.legendre.leggauss(order)
    return jnp.asarray(0.5 * (x + 1)), jnp.asarray(0.5 * w)


# ============================================================================
# Three implementations, each exposed as potential_fn(xyz) -> scalar,
# so potential/gradient timing uses one identical vmap/grad harness for all.


def _composite_potential_fn(
    m_tot: np.ndarray, r_s: np.ndarray, q1: np.ndarray, q2: np.ndarray
):
    components = {
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
    pot = gp.CompositePotential(components, units=UNITS)

    def potential_fn(xyz: jnp.ndarray) -> jnp.ndarray:
        return pot._potential(xyz, 0.0)

    return potential_fn


def _fused_prefactors(
    m_tot: np.ndarray, r_s: np.ndarray, q1: np.ndarray, q2: np.ndarray, G: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-component (prefactor, r_s, q1sq, q2sq), shared by both fused variants."""
    m_tot_, r_s_, q1_, q2_ = (jnp.asarray(a) for a in (m_tot, r_s, q1, q2))
    rho0 = m_tot_ / (q1_ * q2_ * (2 * jnp.pi) ** 1.5 * r_s_**3)
    prefactor = -2 * jnp.pi * G * rho0 * r_s_**2 * q1_ * q2_
    return prefactor, r_s_, q1_**2, q2_**2


def _array_fused_potential_fn(
    m_tot: np.ndarray, r_s: np.ndarray, q1: np.ndarray, q2: np.ndarray, G: float
):
    prefactor, r_s_, q1sq, q2sq = _fused_prefactors(m_tot, r_s, q1, q2, G)
    x_nodes, w_nodes = _gauss_legendre_01(INTEGRATION_ORDER)
    s2 = x_nodes**2  # (O,)

    def potential_fn(xyz: jnp.ndarray) -> jnp.ndarray:
        x, y, z = xyz[0], xyz[1], xyz[2]
        # Broadcast: nodes (O, 1), components (1, N).
        s2_o = s2[:, None]
        xi2 = (
            s2_o
            * (
                x**2
                + y**2 / (1 + (q1sq[None, :] - 1) * s2_o)
                + z**2 / (1 + (q2sq[None, :] - 1) * s2_o)
            )
            / r_s_**2
        )  # (O, N)
        denom = jnp.sqrt(
            ((q1sq[None, :] - 1) * s2_o + 1) * ((q2sq[None, :] - 1) * s2_o + 1)
        )  # (O, N)
        integrand = 2.0 * jnp.exp(-xi2 / 2) / denom  # (O, N)
        per_node = jnp.sum(prefactor[None, :] * integrand, axis=1)  # (O,)
        return jnp.sum(w_nodes * per_node)

    return potential_fn


def _scan_fused_potential_fn(
    m_tot: np.ndarray, r_s: np.ndarray, q1: np.ndarray, q2: np.ndarray, G: float
):
    prefactor, r_s_, q1sq, q2sq = _fused_prefactors(m_tot, r_s, q1, q2, G)
    x_nodes, w_nodes = _gauss_legendre_01(INTEGRATION_ORDER)

    def potential_fn(xyz: jnp.ndarray) -> jnp.ndarray:
        x, y, z = xyz[0], xyz[1], xyz[2]

        def body(carry: jnp.ndarray, node: tuple[jnp.ndarray, jnp.ndarray]):
            s, w = node
            s2 = s**2
            xi2 = (
                s2
                * (
                    x**2
                    + y**2 / (1 + (q1sq - 1) * s2)
                    + z**2 / (1 + (q2sq - 1) * s2)
                )
                / r_s_**2
            )  # (N,)
            denom = jnp.sqrt(((q1sq - 1) * s2 + 1) * ((q2sq - 1) * s2 + 1))  # (N,)
            integrand = 2.0 * jnp.exp(-xi2 / 2) / denom  # (N,)
            contribution = jnp.sum(prefactor * integrand)
            return carry + w * contribution, None

        total, _ = jax.lax.scan(body, jnp.asarray(0.0), (x_nodes, w_nodes))
        return total

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[6, 15])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 100, 10_000])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    print(f"jax.devices(): {jax.devices()}")
    print(f"jax default backend: {jax.default_backend()}")
    print()

    # Pull G in the target unit system the same way galax itself does.
    probe = gp.TriaxialGaussianPotential(
        m_tot=u.Quantity(1.0, "Msun"), r_s=u.Quantity(1.0, "kpc"), units=UNITS
    )
    G = float(probe.constants["G"].decompose(probe.units).value)

    rng = np.random.default_rng(42)
    header = (
        f"{'N':>3} {'B':>7} {'impl':<14} "
        f"{'potential (ms)':>15} {'gradient (ms)':>15}"
    )
    print(header)
    print("-" * len(header))

    for n in args.n:
        m_tot, r_s, q1, q2 = _random_components(n, seed=0)
        implementations = {
            "composite": _composite_potential_fn(m_tot, r_s, q1, q2),
            "array-fused": _array_fused_potential_fn(m_tot, r_s, q1, q2, G),
            "scan-fused": _scan_fused_potential_fn(m_tot, r_s, q1, q2, G),
        }
        for batch_size in args.batch:
            xyz = jnp.asarray(rng.uniform(-10.0, 10.0, size=(batch_size, 3)))
            for name, potential_fn in implementations.items():
                timing = _benchmark_one(potential_fn, xyz, repeats=args.repeats)
                print(
                    f"{n:>3} {batch_size:>7} {name:<14} "
                    f"{timing.potential_ms:>15.3f} {timing.gradient_ms:>15.3f}"
                )


if __name__ == "__main__":
    main()
