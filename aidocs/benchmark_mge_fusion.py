"""Benchmark: composing vs. fusing the triaxial MGE potential quadrature.

Reproduces (and extends) the methodology from GitHub discussion #34
(dynamics-of-stellar-systems/tnt): TNT's shipped MGE composite types sum one
`galax.potential.TriaxialGaussianPotential` per Gaussian component via
`CompositePotential` ("sum of integrals" -- each component runs its own
independent Gauss-Legendre quadrature). Two fused alternatives sum every
component's integrand at each of the 50 shared nodes before integrating once
("integral of sum") -- mathematically identical, since integration is linear
and every component uses the same fixed quadrature order -- differing only
in how the reduction over nodes is done:

- "array-fused": a plain vectorized `jnp.sum` over one
  `(nodes, components, batch)` tensor, built here as a standalone function
  (never shipped -- discussion #34 found it never won anywhere it was
  tested, laptop or cluster CPU).
- "scan-fused": `jax.lax.scan` over the nodes, summing components inside
  the scan body. This one *is* shipped, as
  `tnt.potential.fused_composite.FusedCompositePotential`
  -- benchmarked here via the actual class, not a reimplementation, so
  what's measured is exactly what a real switch-over would run.

Consequently this script needs the TNT environment itself (not just bare
galax/jax/unxt): run it via `uv run python aidocs/benchmark_mge_fusion.py`
from a checkout of this repo, on whatever CPU/GPU hardware you want numbers
for -- JAX picks up whatever backend is installed/visible, nothing else
about the script changes. See `aidocs/slurm_benchmark_mge_fusion_cpu.sbatch`/
`_gpu.sbatch` for submitting this as a batch job on VSC-5 rather than
running it in an interactive session.

Component parameters for N in {5, 25, 125} are loaded from the committed
`benchmark_mge_fusion_components.json` rather than generated live via
`numpy.random.default_rng` at each run -- empirically, `default_rng`'s
output for the same seed was NOT reproducible between a laptop and the
VSC-5 cluster (root cause not identified; `numpy.__version__` matched
exactly, so it isn't simply a version difference), even though it's
documented to be a deterministic, platform-independent PRNG. Loading fixed
values sidesteps that entirely: every machine reads the same floats from
the same file, so the MGE components being timed are guaranteed identical
regardless of platform RNG quirks. A `--n` value with no entry in that file
falls back to live `default_rng` generation (with a warning printed) --
fine for a quick local check, just not guaranteed reproducible across
machines. Query points (`xyz`) are still drawn live via `default_rng` --
their exact values don't need to match across machines, only within one
run (every implementation is always timed against the same batch), so
that's not a concern the same way component parameters are. Each run
prints a short fingerprint hash per N as a sanity check that different
runs (e.g. one per machine) really did use identical components before
comparing their timings.

"array-fused" materializes a full `(nodes=50, N, batch)` tensor -- unlike
"composite"/"scan-fused", which never hold the batch and node dimensions in
memory at once -- so it can run out of memory well before the other two do,
especially at large N and batch together (e.g. N=125, batch=1e6, float64 is
already ~50GB just for that one tensor, before any of its intermediates).
Each timed call is wrapped to catch that and record "OOM" in the results
rather than crashing the whole sweep -- this only catches JAX/XLA's own
resource-exhaustion errors; if the OS OOM-killer gets there first (more
likely on a memory-constrained laptop than on a cluster node), the process
just dies, same as any other OOM.

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
import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import galax.potential as gp
import jax
import jax.numpy as jnp
import numpy as np
import unxt as u

from tnt.potential.fused_composite import (
    FusedCompositePotential,
)

UNITS = "galactic"
INTEGRATION_ORDER = 50
COMPONENTS_FILE = Path(__file__).with_name("benchmark_mge_fusion_components.json")
OOM_MARKER = "OOM"


# ============================================================================
# Component generation


def _random_components(
    n: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """N triaxial Gaussian components with plausible MGE-like parameters.

    Not guaranteed reproducible across machines in practice -- see
    `_fixed_components`, which is what `main` actually uses for the
    default `--n` values.
    """
    rng = np.random.default_rng(seed)
    m_tot = rng.uniform(1e6, 1e9, n)
    r_s = rng.uniform(0.05, 5.0, n)
    q1 = rng.uniform(0.5, 1.0, n)
    q2 = rng.uniform(0.4, q1)
    return m_tot, r_s, q1, q2


def _fixed_components(
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Pre-generated component values for `n`, loaded from `COMPONENTS_FILE`.

    `None` if `n` has no entry -- the caller should fall back to
    `_random_components` in that case.
    """
    if not COMPONENTS_FILE.exists():
        return None
    entry = json.loads(COMPONENTS_FILE.read_text()).get(str(n))
    if entry is None:
        return None
    return (
        np.asarray(entry["m_tot"]),
        np.asarray(entry["r_s"]),
        np.asarray(entry["q1"]),
        np.asarray(entry["q2"]),
    )


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
# Three implementations, each exposed as potential_fn(xyz) -> scalar, so
# potential/gradient timing uses one identical vmap/grad harness for all.


def _composite_potential_fn(children: dict[str, gp.TriaxialGaussianPotential]):
    pot = gp.CompositePotential(children, units=UNITS)

    def potential_fn(xyz: jnp.ndarray) -> jnp.ndarray:
        return pot._potential(xyz, 0.0)

    return potential_fn


def _scan_fused_potential_fn(children: dict[str, gp.TriaxialGaussianPotential]):
    pot = FusedCompositePotential(children)

    def potential_fn(xyz: jnp.ndarray) -> jnp.ndarray:
        return pot._potential(xyz, 0.0)

    return potential_fn


def _array_fused_potential_fn(
    m_tot: np.ndarray, r_s: np.ndarray, q1: np.ndarray, q2: np.ndarray, G: float
):
    """Never shipped -- standalone, for comparison only (see module docstring)."""
    m_tot_, r_s_, q1_, q2_ = (jnp.asarray(a) for a in (m_tot, r_s, q1, q2))
    rho0 = m_tot_ / (q1_ * q2_ * (2 * jnp.pi) ** 1.5 * r_s_**3)
    prefactor = -2 * jnp.pi * G * rho0 * r_s_**2 * q1_ * q2_
    q1sq, q2sq = q1_**2, q2_**2

    x_, w_ = np.polynomial.legendre.leggauss(INTEGRATION_ORDER)
    x_nodes, w_nodes = jnp.asarray(0.5 * (x_ + 1)), jnp.asarray(0.5 * w_)
    s2 = x_nodes**2  # (O,)

    def potential_fn(xyz: jnp.ndarray) -> jnp.ndarray:
        x, y, z = xyz[0], xyz[1], xyz[2]
        s2_o = s2[:, None]  # (O, 1) broadcasts against components (1, N)
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


# ============================================================================
# Timing harness


@dataclass
class Timing:
    potential_ms: float | None  # None means OOM
    gradient_ms: float | None


def _is_oom_error(error: Exception) -> bool:
    if isinstance(error, MemoryError):
        return True
    message = str(error)
    return "RESOURCE_EXHAUSTED" in message or "out of memory" in message.lower()


def _time_call(fn, *args, repeats: int) -> float | None:
    try:
        fn(*args).block_until_ready()  # warmup / compile, not timed
        start = time.perf_counter()
        for _ in range(repeats):
            result = fn(*args)
        result.block_until_ready()
        elapsed = time.perf_counter() - start
        return 1000.0 * elapsed / repeats
    except (RuntimeError, MemoryError) as error:
        if _is_oom_error(error):
            return None
        raise


def _benchmark_one(potential_fn, xyz: jnp.ndarray, *, repeats: int) -> Timing:
    potential_batched = jax.jit(jax.vmap(potential_fn))
    gradient_batched = jax.jit(jax.vmap(jax.grad(potential_fn)))
    potential_ms = _time_call(potential_batched, xyz, repeats=repeats)
    # Skip gradient timing after an OOM on the (cheaper) potential call --
    # gradient is strictly more memory-hungry (holds a backward pass too),
    # so it would just OOM again.
    gradient_ms = (
        None
        if potential_ms is None
        else _time_call(gradient_batched, xyz, repeats=repeats)
    )
    return Timing(potential_ms, gradient_ms)


def _default_output_path(backend: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    host = platform.node().split(".")[0] or "unknown-host"
    return f"benchmark_mge_fusion_{backend}_{host}_{timestamp}.csv"


def _fmt(value: float | None) -> str:
    return OOM_MARKER if value is None else f"{value:.6f}"


def _fmt_print(value: float | None) -> str:
    return f"{OOM_MARKER:>15}" if value is None else f"{value:>15.3f}"


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

    # Pull G in the target unit system the same way galax itself does.
    probe = gp.TriaxialGaussianPotential(
        m_tot=u.Quantity(1.0, "Msun"), r_s=u.Quantity(1.0, "kpc"), units=UNITS
    )
    G = float(probe.constants["G"].decompose(probe.units).value)

    rng = np.random.default_rng(42)
    header = (
        f"{'N':>3} {'B':>8} {'impl':<12} "
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
            fixed = _fixed_components(n)
            if fixed is not None:
                m_tot, r_s, q1, q2 = fixed
            else:
                print(
                    f"# N={n}: no entry in {COMPONENTS_FILE.name}, generating "
                    "live (not guaranteed reproducible across machines)"
                )
                m_tot, r_s, q1, q2 = _random_components(n, seed=0)
            fingerprint = _component_fingerprint(m_tot, r_s, q1, q2)
            print(f"# N={n} component fingerprint: {fingerprint}")
            children = _build_children(m_tot, r_s, q1, q2)
            implementations = {
                "composite": _composite_potential_fn(children),
                "array-fused": _array_fused_potential_fn(m_tot, r_s, q1, q2, G),
                "scan-fused": _scan_fused_potential_fn(children),
            }
            for batch_size in args.batch:
                xyz = jnp.asarray(rng.uniform(-10.0, 10.0, size=(batch_size, 3)))
                for name, potential_fn in implementations.items():
                    timing = _benchmark_one(potential_fn, xyz, repeats=args.repeats)
                    print(
                        f"{n:>3} {batch_size:>8} {name:<12} "
                        f"{_fmt_print(timing.potential_ms)} "
                        f"{_fmt_print(timing.gradient_ms)}"
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
                            _fmt(timing.potential_ms),
                            _fmt(timing.gradient_ms),
                        ]
                    )
                    csv_file.flush()

    print()
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
