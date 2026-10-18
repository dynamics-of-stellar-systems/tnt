"""Compare matched production-duration orbits with premeasured force grids.

Run after ``benchmarks.grid_benchmark``. Grid tables are explicit, verified
inputs: construction timings come from the force experiment; loading,
compilation and new whole-orbit timings are measured separately here.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from unxt import Quantity

from benchmarks.force_grid import ForceGrid, GridSpec, integrate_grid
from benchmarks.gaussian_forces import batch_force
from benchmarks.grid_benchmark import error_metrics
from benchmarks.orbit_checks import (
    CASE_IDS,
    ROOT,
    build_models,
    compare_controls,
    evaluate,
    load_cases,
    load_comparison,
    make_report,
    require_working_defaults,
    sha256,
    write_json,
)
from tnt.orbit_library.library import _potential_metadata


def load_grid(potential, setup: dict, directory: Path) -> tuple[ForceGrid, float]:
    """Verify the stored table and measure explicit loading/device transfer."""
    path = directory / setup["table_file"]
    if sha256(path) != setup["table_sha256"]:
        raise RuntimeError(f"Grid artifact hash mismatch: {path}")
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as data:
        declaration = json.loads(str(data["spec"]))
        declaration["shape"] = tuple(declaration["shape"])
        spec = GridSpec(**declaration)
        table = data["log_coefficients"]
    expected = dict(setup["spec"])
    expected["shape"] = tuple(expected["shape"])
    if declaration != expected or table.shape != (*spec.shape, 3):
        raise RuntimeError("Grid geometry does not match its recorded specification.")
    if table.dtype != np.float64 or not np.all(np.isfinite(table)):
        raise RuntimeError("Grid table must contain finite float64 log coefficients.")
    grid = ForceGrid(potential, jnp.asarray(table), spec)
    grid.log_coefficients.block_until_ready()
    return grid, time.perf_counter() - started


def check_provenance(force: dict, comparison_report: dict) -> None:
    """Require identical physical/numerical sources and exported inputs."""
    for key in ("comparison_input_sha256", "tnt_source_sha256"):
        if force[key] != comparison_report[key]:
            raise RuntimeError(f"Force experiment no longer matches current {key}.")
    for name, digest in force["prototype_sources_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"Force experiment source changed: {name}")
    if force["environment"]["packages"] != comparison_report["environment"]["packages"]:
        raise RuntimeError("Numerical library versions changed after force experiment.")


def run(args: argparse.Namespace) -> dict:
    """Measure all selected launches using the same tables and physical model."""
    force = json.loads(args.force_result.read_text())
    comparison = args.comparison_checkout.resolve()
    reference = load_comparison(comparison)
    cases = load_cases(comparison, args.case)
    settings = require_working_defaults()
    variants = [row["variant"] for row in force["force_rows"]]
    args.variant = list(dict.fromkeys(variants))
    if args.variant[0] != "direct":
        raise ValueError("Force artifact must begin with a direct reference row.")
    potentials, model_build_seconds = build_models(reference, comparison, cases)
    report = make_report(args, comparison, cases, [], [], model_build_seconds)
    # Input hashes include the geometry set; request the same case set as phase 1.
    if args.case != force["case_ids"]:
        raise ValueError("Use the same cases as the force experiment.")
    check_provenance(force, report)
    args.work_dir.mkdir(parents=True, exist_ok=False)
    rows, loads, orbit_probes = [], [], []
    for geometry, potential in potentials.items():
        physical = next(row["physical_model"] for row in force["force_rows"]
                        if row["geometry"] == geometry and row["variant"] == "direct")
        if _potential_metadata(potential) != physical:
            raise RuntimeError("Physical model differs from the force experiment.")
        selected = [case for case in cases if case["geometry"] == geometry]
        grids = {}
        for row in force["force_rows"]:
            if row["geometry"] != geometry or row["variant"] == "direct":
                continue
            grid, seconds = load_grid(potential, row["setup"], args.force_work_dir)
            grids[row["variant"]] = grid
            loads.append({"geometry": geometry, "variant": row["variant"],
                          "load_and_transfer_seconds": seconds})
        for case in selected:
            direct = evaluate(
                reference, case, "direct", potential, settings,
                args.warm_repeats, args.work_dir, integration_order=800,
                separate_compilation=True,
            )
            rows.append(direct)
            with np.load(args.work_dir / direct["trajectory_file"],
                         allow_pickle=False) as data:
                # Uniform output indices: includes all orbit phases, never just
                # the launch or a small selection of visually convenient points.
                points = data["state_km_kms"][::25, :3]
            points = Quantity(points, "km").ustrip("kpc")
            direct_force = np.asarray(batch_force(potential, points))
            probe = {"case_id": case["id"], "stride": 25,
                     "points_kpc": np.asarray(points).tolist(),
                     "direct_forces_kpc_Myr2": direct_force.tolist(), "grids": []}
            for name, grid in grids.items():
                values = np.asarray(batch_force(grid, points))
                probe["grids"].append({
                    "variant": name, "error": error_metrics(values, direct_force),
                })
                rows.append(evaluate(
                    reference, case, name, potential, settings,
                    args.warm_repeats, args.work_dir, force_model=grid,
                    integrator=integrate_grid, integration_order=800,
                    separate_compilation=True,
                ))
            orbit_probes.append(probe)
    controls = compare_controls(
        reference, rows, cases, args.work_dir, baseline_variant="direct"
    )
    for control in controls:
        case = next(c for c in cases if c["id"] == control["case_id"])
        setup = next(
            row["setup"] for row in force["force_rows"]
            if row["geometry"] == case["geometry"]
            and row["variant"] == control["working_vs"]
        )
        direct = next(r for r in rows if r["case_id"] == case["id"]
                      and r["variant"] == "direct")
        grid = next(r for r in rows if r["case_id"] == case["id"]
                    and r["variant"] == control["working_vs"])
        saved = direct["warm_median_seconds"] - grid["warm_median_seconds"]
        control["build_only_break_even_identical_orbits"] = (
            setup["warm_reconstruction_seconds"] / saved if saved > 0 else None
        )
    result = make_report(
        args, comparison, cases, rows, controls, model_build_seconds,
    )
    result.update({
        "scope": "Matched direct versus experimental force-grid orbit comparison.",
        "harness_sha256": sha256(Path(__file__)),
        "force_result_file": str(args.force_result),
        "force_result_sha256": sha256(args.force_result),
        "force_environment": force["environment"],
        "grid_loads": loads,
        "orbit_force_probes": orbit_probes,
        "prototype_sources_sha256": {
            **force["prototype_sources_sha256"],
            "benchmarks/grid_orbits.py": sha256(Path(__file__)),
        },
        "timing_scope": (
            "Fresh synchronized whole solves, including 50000 saved samples and "
            "batched direct physical-energy diagnostics. Lowering/compilation, "
            "first compiled execution and warm repeats are separate. Grid build "
            "costs come from the identified force experiment; table loading and "
            "transfer are measured here. Statistics and I/O are excluded from "
            "solve timings. Break-even is warm construction cost divided by "
            "warm time saved for identical repeated launches, excluding compiler "
            "cost and statistical acceptance; it is not library throughput."
        ),
        "boundary_policy": force["boundary_policy"],
        "physical_energy": force["physical_energy"],
    })
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-checkout", type=Path, required=True)
    parser.add_argument("--force-result", type=Path, required=True)
    parser.add_argument("--force-work-dir", type=Path, required=True)
    parser.add_argument("--case", nargs="+", choices=CASE_IDS, default=list(CASE_IDS))
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Use the Linux development environment.")
    if args.warm_repeats < 1 or len(set(args.case)) != len(args.case):
        parser.error("Use positive repeats and distinct cases.")
    if args.output.exists():
        parser.error("Output exists; choose a new result path.")
    args.work_dir = args.work_dir.resolve()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
