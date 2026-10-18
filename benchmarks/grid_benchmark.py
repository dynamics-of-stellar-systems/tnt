"""Measure an experimental grid against the same direct physical model.

Run as ``python -m benchmarks.grid_benchmark`` in the Linux environment.
Use explicit new work/output paths. No DYNAMITE executable is run, no prior
timings are reused, and no production evaluator is changed.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from benchmarks.force_grid import GridSpec, build_grid
from benchmarks.gaussian_forces import batch_force, measure, sequential_force
from benchmarks.orbit_checks import (
    CASE_IDS,
    ROOT,
    build_models,
    load_cases,
    load_comparison,
    make_report,
    require_working_defaults,
    sha256,
    write_json,
)
from tnt.orbit_library.library import _potential_metadata

RESOLUTIONS = {
    "coarse": (80, 12, 12),
    "medium": (160, 24, 24),
    "fine": (320, 48, 48),
}


def probes(spec: GridSpec, seed: int = 731) -> np.ndarray:
    """Fixed off-node log-radius/angular probes plus axes and symmetry planes."""
    rng = np.random.default_rng(seed)
    radius = np.exp(rng.uniform(np.log(spec.r_min), np.log(spec.r_max), 4096))
    theta, phi = rng.uniform(0, np.pi / 2, (2, len(radius)))
    directions = np.stack([
        np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
    ], axis=1)
    random = radius[:, None] * directions * rng.choice([-1, 1], (len(radius), 3))
    directions = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
        [0, 1, 1], [1, 1, 1], [1, 1e-10, 1e-10], [1e-10, 1, 1e-10],
    ], dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    # Keep force probes inside inclusive radial boundaries despite roundoff;
    # exact boundaries are covered by the independent unit tests.
    radii = np.geomspace(spec.r_min * (1 + 1e-12), spec.r_max * (1 - 1e-12), 61)
    structured = (radii[:, None, None] * directions[None]).reshape(-1, 3)
    return np.concatenate([random, structured])


def error_metrics(actual: np.ndarray, direct: np.ndarray) -> dict:
    """Use vector relative errors, avoiding component division on symmetry axes."""
    absolute = np.linalg.norm(actual - direct, axis=1)
    norm = np.linalg.norm(direct, axis=1)
    if not np.all(np.isfinite(actual)) or np.any(norm <= 0):
        raise RuntimeError("Invalid force comparison or zero reference force.")
    relative = absolute / norm
    return {
        "relative_errors": relative.tolist(),
        "absolute_errors_native": absolute.tolist(),
        "max_relative": float(relative.max()),
        "p99_relative": float(np.quantile(relative, 0.99)),
        "median_relative": float(np.median(relative)),
        "rms_relative": float(np.sqrt(np.mean(relative**2))),
    }


def construct(potential, spec: GridSpec, path: Path) -> tuple:
    """Separate lowering/compilation, first execution, and one warm rebuild."""
    started = time.perf_counter()
    compiled = build_grid.lower(potential, spec).compile()
    compilation = time.perf_counter() - started
    timings = []
    for _ in range(2):
        started = time.perf_counter()
        grid = compiled(potential, spec)
        jax.block_until_ready(eqx.filter(grid, eqx.is_array))
        timings.append(time.perf_counter() - started)
    table = np.asarray(grid.log_coefficients)
    with path.open("xb") as stream:
        np.savez_compressed(
            stream, log_coefficients=table,
            spec=np.asarray(json.dumps(asdict(spec), sort_keys=True)),
        )
    return grid, {
        "spec": asdict(spec),
        "lowering_and_compilation_seconds": compilation,
        "first_construction_seconds": timings[0],
        "warm_reconstruction_seconds": timings[1],
        "table_bytes": table.nbytes,
        "table_file": path.name,
        "table_sha256": sha256(path),
    }


def force_matrix(potential, geometry: str, args: argparse.Namespace) -> list[dict]:
    """Compare identical probes and scalar/batch request patterns for every grid."""
    spec = GridSpec(RESOLUTIONS[args.resolution[0]], args.r_min, args.r_max)
    points = probes(spec)
    # Timings use a separate, fixed subset for all methods and resolutions.
    timing = jnp.asarray(points[:2048])
    print(f"Measuring {geometry} direct forces on {len(points)} probes", flush=True)
    direct, batch_timing = measure(batch_force, potential, timing, args.warm_repeats)
    serial, serial_timing = measure(
        sequential_force, potential, timing, args.warm_repeats
    )
    pattern_difference = error_metrics(serial, direct)
    # NFW central cancellation makes the scalar and vectorized CPU expression
    # differ slightly. Record it and enforce an explicit protocol check, rather
    # than mistaking this arithmetic floor for force-grid interpolation error.
    if pattern_difference["max_relative"] > 1e-9:
        raise RuntimeError("Direct scalar/batch relative discrepancy exceeds 1e-9.")
    reference = np.asarray(batch_force(potential, jnp.asarray(points)))
    with (args.work_dir / f"{geometry}-force-probes.npz").open("xb") as stream:
        np.savez_compressed(stream, points=points, direct=reference)
    rows = [{
        "geometry": geometry, "variant": "direct", "probe_count": len(points),
        "timing_probe_count": len(timing), "batch_timing": batch_timing,
        "serial_timing": serial_timing,
        "scalar_vs_batch_max_relative": pattern_difference["max_relative"],
        "physical_model": _potential_metadata(potential),
        "points_native": points.tolist(), "forces_native": reference.tolist(),
        "probe_file": f"{geometry}-force-probes.npz",
        "probe_sha256": sha256(args.work_dir / f"{geometry}-force-probes.npz"),
    }]
    write_json(args.work_dir / f"{geometry}-direct-forces.json", rows[0])
    for name in args.resolution:
        print(f"Building {geometry} {name} {RESOLUTIONS[name]}", flush=True)
        grid, setup = construct(
            potential, GridSpec(RESOLUTIONS[name], args.r_min, args.r_max),
            args.work_dir / f"{geometry}-{name}-grid.npz",
        )
        values = np.asarray(batch_force(grid, jnp.asarray(points)))
        batch, batch_timing = measure(batch_force, grid, timing, args.warm_repeats)
        serial, serial_timing = measure(
            sequential_force, grid, timing, args.warm_repeats
        )
        np.testing.assert_allclose(batch, serial, rtol=2e-12, atol=1e-15)
        record = {
            "geometry": geometry, "variant": name,
            "probe_count": len(points), "timing_probe_count": len(timing),
            "setup": setup, "force_error": error_metrics(values, reference),
            "batch_timing": batch_timing, "serial_timing": serial_timing,
            "forces_native": values.tolist(),
        }
        write_json(args.work_dir / f"{geometry}-{name}-forces.json", record)
        rows.append(record)
        print(
            f"  Max relative force error={record['force_error']['max_relative']:.3g}, "
            f"warm build={setup['warm_reconstruction_seconds']:.3f}s, "
            f"serial={serial_timing['warm_median_seconds']:.4f}s/2048", flush=True,
        )
    return rows


def run(args: argparse.Namespace) -> dict:
    """Run models sequentially without overwriting or silently altering settings."""
    comparison = args.comparison_checkout.resolve()
    reference = load_comparison(comparison)
    cases = load_cases(comparison, args.case)
    require_working_defaults()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    potentials, model_build_seconds = build_models(reference, comparison, cases)
    force_rows = []
    for geometry, potential in potentials.items():
        force_rows.extend(force_matrix(potential, geometry, args))
    args.variant = ["direct", *args.resolution]
    result = make_report(
        args, comparison, cases, [], [], model_build_seconds,
    )
    result.update({
        "schema_version": 1,
        "scope": "Experimental direct versus log-spherical trilinear force grid.",
        "harness_sha256": sha256(Path(__file__)),
        "force_rows": force_rows,
        "probe_seed": 731,
        "force_probe_scope": (
            "4096 uniform log-radius/polar-angle/azimuth draws in all octants, "
            "plus 61 radii on nine axes/planes/diagonal/near-axis directions. "
            "The deterministic probes are shared across all resolutions."
        ),
        "timing_scope": (
            "Synchronized force batch and sequential scan timings for 2048 "
            "identical off-node requests; first calls include any compilation. "
            "Grid lowering/compilation, first construction and one warm rebuild "
            "are separate; table I/O and statistics are excluded."
        ),
        "statistics_scope": "Force comparisons only in this phase.",
        "boundary_policy": "Hard radius-domain rejection; no fallback/extrapolation.",
        "physical_energy": "Unchanged direct potential, not an interpolated scalar.",
        "prototype_sources_sha256": {
            name: sha256(ROOT / name) for name in (
                "benchmarks/force_grid.py", "benchmarks/grid_benchmark.py",
                "benchmarks/gaussian_forces.py", "benchmarks/orbit_checks.py",
                "uv.lock",
            )
        },
    })
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-checkout", type=Path, required=True)
    parser.add_argument("--case", nargs="+", choices=CASE_IDS, default=list(CASE_IDS))
    parser.add_argument(
        "--resolution", nargs="+", choices=RESOLUTIONS, default=list(RESOLUTIONS)
    )
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--r-min", type=float, default=1e-5)
    parser.add_argument("--r-max", type=float, default=200.0)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Use the Linux development environment.")
    if args.warm_repeats < 1:
        parser.error("Warm repeats must be positive.")
    if len(set(args.case)) != len(args.case) or len(set(args.resolution)) != len(
        args.resolution
    ):
        parser.error("Duplicate cases/resolutions are not allowed.")
    if args.output.exists():
        parser.error("Output exists; choose a new path.")
    args.work_dir = args.work_dir.resolve()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
