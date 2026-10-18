"""Interleave timings for one orbit after a noisy full-matrix observation."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from unxt import Quantity

from benchmarks.force_grid import integrate_grid
from benchmarks.grid_orbits import check_provenance, load_grid
from benchmarks.orbit_checks import (
    ROOT,
    build_models,
    load_cases,
    load_comparison,
    make_report,
    require_working_defaults,
    sha256,
    write_json,
)
from tnt.orbit_library import integrate_orbit


def run(args: argparse.Namespace) -> dict:
    """Preserve every timing and require unchanged trajectories/diagnostics."""
    force = json.loads(args.force_result.read_text())
    parent = json.loads(args.orbit_result.read_text())
    if parent["force_result_sha256"] != sha256(args.force_result):
        raise RuntimeError("Parent orbit/force artifact mismatch.")
    for name, digest in parent["prototype_sources_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"Parent benchmark source changed: {name}")
    reference = load_comparison(args.comparison_checkout)
    cases = load_cases(args.comparison_checkout, parent["case_ids"])
    settings = require_working_defaults()
    args.case, args.variant = parent["case_ids"], parent["variants"]
    # make_report needs a directory label; this control writes only portable JSON.
    args.work_dir = args.orbit_work_dir
    potentials, build_seconds = build_models(reference, args.comparison_checkout, cases)
    report = make_report(args, args.comparison_checkout, cases, [], [], build_seconds)
    check_provenance(force, report)
    case = next(c for c in cases if c["id"] == args.case_id)
    potential = potentials[case["geometry"]]
    state0 = np.asarray(case["state_km_kms"])
    shared = (
        Quantity(state0[:3], "km"), Quantity(state0[3:], "km/s"),
        Quantity(case["period_s"], "s"), jax.random.key(case["key_seed"]), settings,
    )
    runners, inputs, diagnostics, records = {}, {}, {}, {}

    def timed(variant: str) -> float:
        started = time.perf_counter()
        trajectory = runners[variant](*inputs[variant])
        reference.ready(trajectory)
        seconds = time.perf_counter() - started
        actual = {
            name: np.asarray(getattr(trajectory.diagnostics, name)).item()
            for name in diagnostics[variant]
        }
        if actual != diagnostics[variant]:
            raise RuntimeError("Timing control changed numerical diagnostics.")
        original = next(r for r in parent["rows"] if r["case_id"] == case["id"]
                        and r["variant"] == variant)
        path = args.orbit_work_dir / original["trajectory_file"]
        if sha256(path) != original["trajectory_sha256"]:
            raise RuntimeError("Parent trajectory hash mismatch.")
        with np.load(path, allow_pickle=False) as data:
            state = np.concatenate([
                np.asarray(trajectory.positions.ustrip("km")),
                np.asarray(trajectory.velocities.ustrip("km/s")),
            ], axis=1)
            np.testing.assert_array_equal(state, data["state_km_kms"])
            np.testing.assert_array_equal(trajectory.times.ustrip("s"), data["times_s"])
        print(f"{variant}: {seconds:.3f}s", flush=True)
        return seconds

    for variant in args.variant:
        if variant == "direct":
            model, integrator = potential, integrate_orbit
        else:
            setup = next(r["setup"] for r in force["force_rows"]
                         if r["geometry"] == case["geometry"]
                         and r["variant"] == variant)
            model, _ = load_grid(potential, setup, args.force_work_dir)
            integrator = integrate_grid
        inputs[variant] = (model, *shared)
        started = time.perf_counter()
        runners[variant] = integrator.lower(*inputs[variant]).compile()
        compilation = time.perf_counter() - started
        diagnostics[variant] = next(r["diagnostics"] for r in parent["rows"]
                                    if r["case_id"] == case["id"]
                                    and r["variant"] == variant)
        records[variant] = {"lowering_and_compilation_seconds": compilation,
                            "first_execution_seconds": timed(variant),
                            "warm_seconds": []}
    chronological = []
    for cycle in range(args.warm_repeats):
        order = args.variant[cycle % 4:] + args.variant[:cycle % 4]
        for variant in order:
            seconds = timed(variant)
            records[variant]["warm_seconds"].append(seconds)
            chronological.append({
                "cycle": cycle, "variant": variant, "seconds": seconds,
            })
    for record in records.values():
        record["warm_median_seconds"] = float(np.median(record["warm_seconds"]))
    report = make_report(args, args.comparison_checkout, cases, [], [], build_seconds)
    report.update({
        "scope": "Interleaved timing control requiring equivalence to parent orbits.",
        "case_id": case["id"], "records": records, "chronological": chronological,
        "parent_orbit_sha256": sha256(args.orbit_result),
        "force_result_sha256": sha256(args.force_result),
        "harness_sha256": sha256(Path(__file__)),
        "prototype_sources_sha256": {
            **parent["prototype_sources_sha256"],
            "benchmarks/grid_timing.py": sha256(Path(__file__)),
        },
        "timing_scope": (
            "One compiled warmup per method, then three rotated-order cycles. "
            "Whole-solve synchronization includes direct energy diagnostics; "
            "all artifact/trajectory verification is outside the timer. "
            "No original observation is overwritten or excluded."
        ),
    })
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-checkout", type=Path, required=True)
    parser.add_argument("--force-result", type=Path, required=True)
    parser.add_argument("--force-work-dir", type=Path, required=True)
    parser.add_argument("--orbit-result", type=Path, required=True)
    parser.add_argument("--orbit-work-dir", type=Path, required=True)
    parser.add_argument("--case-id", default="strong_triaxial-box-E5")
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or args.warm_repeats != 3:
        parser.error("Use Linux and the fixed three-cycle timing-control protocol.")
    if args.output.exists():
        parser.error("Output exists; choose a new result path.")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
