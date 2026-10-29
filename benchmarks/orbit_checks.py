"""Check representative 200-period orbits using TNT's direct Gaussian evaluator.

Reuse the existing comparison's physical-model builder, exported launches,
clocks, and diagnostic bins. No DYNAMITE executable or interpolation is run.
Full trajectories remain in an explicitly selected local work directory;
the portable JSON result retains diagnostics, statistics, and provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from types import ModuleType

import jax

jax.config.update("jax_enable_x64", True)

import galax.potential as gp
import numpy as np
import yaml
from unxt import Quantity

from tnt.numerics import DEFAULT_GAUSSIAN_POTENTIAL_QUAD_ORDER
from tnt.orbit_library import OrbitIntegrationSettings, integrate_orbit
from tnt.orbit_library.library import _potential_metadata

ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = (
    "strong_triaxial-tube-E3",
    "strong_triaxial-box-E3",
    "oblate-box-E5",
    "strong_triaxial-box-E5",
)
VARIANTS = {
    "working": (800, 1e-9),
    "higher-order": (1600, 1e-9),
    "tighter-solver": (800, 1e-11),
}


def sha256(path: Path) -> str:
    """Fingerprint a concrete source or retained artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_comparison(checkout: Path) -> ModuleType:
    """Load the established helpers without modifying the external checkout."""
    directory = checkout / "orbit-integration"
    source = directory / "compare.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing comparison source: {source}")
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("orbit_check_reference", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def with_order(potential: gp.AbstractPotential, order: int) -> gp.AbstractPotential:
    """Change only the Gaussian numerical order in an explicit control model."""
    if isinstance(potential, Mapping):
        return gp.CompositePotential(
            {name: with_order(value, order) for name, value in potential.items()},
            units=potential.units,
            constants=potential.constants,
        )
    if isinstance(
        potential, (gp.TriaxialGaussianPotential, gp.AxisymmetricGaussianPotential)
    ):
        return replace(potential, integration_order=order)
    return potential


def write_json(path: Path, value: dict) -> None:
    """Preserve a new result without overwriting any earlier run."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def retained_statistics(reference: ModuleType, state: np.ndarray, case: dict) -> dict:
    """Use the same normalized diagnostic bins as the established comparison."""
    length, speed = reference.scales(case)
    values = reference.distributions(state, length, speed)
    for name in ("radial", "velocity", "spatial"):
        array = values[name]
        assert np.all(np.isfinite(array)) and np.all(array >= 0)
        totals = array.sum(axis=-1) if name == "velocity" else array.sum()
        np.testing.assert_allclose(totals, 1.0, rtol=0, atol=1e-12)
    if not all(np.all(np.isfinite(value)) for value in values.values()):
        raise RuntimeError("Non-finite retained orbit statistics.")
    return {name: np.asarray(value).tolist() for name, value in values.items()}


def evaluate(
    reference: ModuleType,
    case: dict,
    variant: str,
    potential: gp.AbstractPotential,
    settings: OrbitIntegrationSettings,
    repeats: int,
    work: Path,
) -> dict:
    """Measure the production solve, then retain all-sample statistics outside it."""
    state0 = np.asarray(case["state_km_kms"])
    args = (
        potential,
        Quantity(state0[:3], "km"),
        Quantity(state0[3:], "km/s"),
        Quantity(case["period_s"], "s"),
        jax.random.key(case["key_seed"]),
        settings,
    )
    print(f"Running {case['id']} {variant}: 200 periods, 50000 samples", flush=True)
    started = time.perf_counter()
    trajectory = integrate_orbit(*args)
    reference.ready(trajectory)
    first = time.perf_counter() - started
    if not bool(trajectory.diagnostics.successful):
        raise RuntimeError(f"First solve failed: {case['id']} {variant}")
    print(
        f"  First call: {first:.3f}s, "
        f"{int(trajectory.diagnostics.num_steps)} steps",
        flush=True,
    )
    del trajectory
    timings, repeat_diagnostics = [], []
    for index in range(repeats):
        started = time.perf_counter()
        trajectory = integrate_orbit(*args)
        reference.ready(trajectory)
        timings.append(time.perf_counter() - started)
        diagnostics = {
            name: np.asarray(value).item()
            for name, value in asdict(trajectory.diagnostics).items()
        }
        if not diagnostics["successful"]:
            raise RuntimeError(f"Warm solve failed: {case['id']} {variant}")
        print(f"  Warm {index + 1}/{repeats}: {timings[-1]:.3f}s", flush=True)
        repeat_diagnostics.append(diagnostics)
        if index != repeats - 1:
            del trajectory
    times = np.asarray(trajectory.times.ustrip("s"))
    state = np.concatenate(
        (
            np.asarray(trajectory.positions.ustrip("km")),
            np.asarray(trajectory.velocities.ustrip("km/s")),
        ),
        axis=1,
    )
    if len(times) != settings.n_stored_timesteps or not np.all(np.isfinite(state)):
        raise RuntimeError("Incomplete or non-finite trajectory.")
    if any(record != repeat_diagnostics[0] for record in repeat_diagnostics):
        raise RuntimeError("Identical warm solves produced different diagnostics.")
    physics = _potential_metadata(potential)
    stats = retained_statistics(reference, state, case)
    halves = [
        retained_statistics(reference, part, case) for part in np.array_split(state, 2)
    ]
    split = reference.difference(
        *[{name: np.asarray(value) for name, value in half.items()} for half in halves]
    )
    record = {
        "case_id": case["id"],
        "variant": variant,
        "integration_order": VARIANTS[variant][0],
        "settings": asdict(settings),
        "diagnostics": diagnostics,
        "repeat_diagnostics": repeat_diagnostics,
        "first_call_seconds": first,
        "warm_seconds": timings,
        "warm_median_seconds": statistics.median(timings),
        "retained_statistics": stats,
        "split_half_variation": split,
        "minimum_sampled_radius_kpc": float(
            np.linalg.norm(state[:, :3], axis=1).min()
            / Quantity(1.0, "kpc").ustrip("km")
        ),
        "maximum_sampled_radius_kpc": float(
            np.linalg.norm(state[:, :3], axis=1).max()
            / Quantity(1.0, "kpc").ustrip("km")
        ),
        "potential": physics,
    }
    path = work / f"{case['id']}-{variant}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            times_s=times,
            state_km_kms=state,
            metadata=np.asarray(json.dumps({"case": case, **record}, sort_keys=True)),
        )
    record["trajectory_file"] = path.name
    record["trajectory_sha256"] = sha256(path)
    write_json(path.with_suffix(".json"), record)
    print(
        f"{case['id']} {variant}: energy={diagnostics['max_energy_error']:.3g}, "
        f"steps={diagnostics['num_steps']}, warm={statistics.median(timings):.3f}s",
        flush=True,
    )
    return record


def compare_controls(
    reference: ModuleType, rows: list[dict], cases: list[dict], work: Path
) -> list[dict]:
    """Check matched output times and compare controls against each working run."""
    result = []
    for case in cases:
        selected = [row for row in rows if row["case_id"] == case["id"]]
        working = next(row for row in selected if row["variant"] == "working")
        with np.load(work / working["trajectory_file"], allow_pickle=False) as data:
            baseline_times, baseline_state = data["times_s"], data["state_km_kms"]
        baseline_stats = {
            name: np.asarray(value)
            for name, value in working["retained_statistics"].items()
        }
        length, speed = reference.scales(case)
        short = baseline_times <= case["period_s"]
        for row in selected:
            if row is working:
                continue
            with np.load(work / row["trajectory_file"], allow_pickle=False) as data:
                np.testing.assert_array_equal(data["times_s"], baseline_times)
                state = data["state_km_kms"]
            result.append(
                {
                    "case_id": case["id"],
                    "working_vs": row["variant"],
                    "distribution_difference": reference.difference(
                        baseline_stats,
                        {
                            name: np.asarray(value)
                            for name, value in row["retained_statistics"].items()
                        },
                    ),
                    "first_period_position_rms_over_L": float(
                        np.sqrt(
                            np.mean(
                                (
                                    (state[short, :3] - baseline_state[short, :3])
                                    / length
                                )
                                ** 2
                            )
                        )
                    ),
                    "first_period_velocity_rms_over_V": float(
                        np.sqrt(
                            np.mean(
                                ((state[short, 3:] - baseline_state[short, 3:]) / speed)
                                ** 2
                            )
                        )
                    ),
                    "warm_time_ratio_to_working": row["warm_median_seconds"]
                    / working["warm_median_seconds"],
                    "step_ratio_to_working": row["diagnostics"]["num_steps"]
                    / working["diagnostics"]["num_steps"],
                }
            )
    return result


def run(args: argparse.Namespace) -> dict:
    """Run the explicitly selected matrix; never substitute settings or orbits."""
    comparison = args.comparison_checkout.resolve()
    reference = load_comparison(comparison)
    prior = json.loads(
        (comparison / "orbit-integration/results/summary.json").read_text()
    )
    cases = [case for case in prior["provenance"]["cases"] if case["id"] in args.case]
    if {case["id"] for case in cases} != set(args.case):
        raise ValueError("Missing requested launch records in the comparison results.")
    config_path = ROOT / "tnt/defaults/default_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    settings = OrbitIntegrationSettings()
    if (
        not config["numerics_settings"]["jax_enable_x64"]
        or config["numerics_settings"]["gaussian_potential_quad_order"] != 800
        or DEFAULT_GAUSSIAN_POTENTIAL_QUAD_ORDER != 800
        or settings.orbital_periods != 200
        or settings.n_stored_timesteps != 50000
        or settings.accuracy != 1e-9
        or any(
            config["orbit_library_settings"][key] != value
            for key, value in {
                "orbital_periods": 200,
                "n_stored_timesteps": 50000,
                "accuracy": 1e-9,
            }.items()
        )
    ):
        raise RuntimeError(
            "This protocol requires the agreed float64/order-800 defaults."
        )
    args.work_dir.mkdir(parents=True, exist_ok=False)
    rows, potentials, model_build_seconds = [], {}, {}
    for case in cases:
        geometry = case["geometry"]
        if geometry not in potentials:
            started = time.perf_counter()
            parameters = reference.sampling._parse_parameters_pot(
                comparison / "orbit-start-spaces/input" / geometry / "parameters_pot.in"
            )
            potentials[geometry] = reference.sampling.build_potential(parameters)[0]
            model_build_seconds[geometry] = time.perf_counter() - started
            if any(
                g.integration_order != 800
                for g in potentials[geometry]["stars"].values()
            ):
                raise RuntimeError(
                    "The TNT builder did not apply quadrature order 800."
                )
        for variant in args.variant:
            order, accuracy = VARIANTS[variant]
            started = time.perf_counter()
            potential = (
                potentials[geometry]
                if order == 800
                else with_order(potentials[geometry], order)
            )
            settings = OrbitIntegrationSettings(accuracy=accuracy)
            control_setup_seconds = time.perf_counter() - started
            record = evaluate(
                reference,
                case,
                variant,
                potential,
                settings,
                args.warm_repeats,
                args.work_dir,
            )
            record["control_setup_seconds"] = control_setup_seconds
            rows.append(record)
    return make_report(
        args,
        comparison,
        cases,
        rows,
        compare_controls(reference, rows, cases, args.work_dir),
        model_build_seconds,
    )


def make_report(
    args: argparse.Namespace,
    comparison: Path,
    cases: list[dict],
    rows: list[dict],
    comparisons: list[dict],
    model_build_seconds: dict[str, float],
) -> dict:
    """Capture numerical inputs, environment, source identities, and comparisons."""
    paths = [
        "orbit-start-spaces/run_comparison.py",
        "orbit-start-spaces/mge_lum.ecsv",
        "orbit-integration/compare.py",
        "orbit-integration/prepare_cases.py",
        "orbit-integration/results/summary.json",
        *[
            f"orbit-start-spaces/input/{case['geometry']}/parameters_pot.in"
            for case in cases
        ],
    ]
    return {
        "schema_version": 1,
        "scope": (
            "Direct order-800 representative orbit checks; "
            "no interpolation or new DYNAMITE runs."
        ),
        "tnt_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "harness_sha256": sha256(Path(__file__)),
        "tnt_tracked_changes": subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD"], cwd=ROOT, text=True
        ).splitlines(),
        "comparison_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=comparison, text=True
        ).strip(),
        "comparison_input_sha256": {
            name: sha256(comparison / name) for name in sorted(set(paths))
        },
        "tnt_source_sha256": {
            name: sha256(ROOT / name)
            for name in [
                "tnt/orbit_library/integration.py",
                "tnt/potential/components.py",
                "tnt/potential/triaxial_mge.py",
                "tnt/potential/core.py",
                "tnt/numerics.py",
                "tnt/defaults/default_config.yaml",
            ]
        },
        "potential_source_sha256": hashlib.sha256(
            inspect.getsource(gp.TriaxialGaussianPotential).encode()
        ).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "linux_memory_total_kib": int(
                next(
                    line.split()[1]
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemTotal:")
                )
            ),
            "packages": {
                name: version(name)
                for name in [
                    "jax",
                    "jaxlib",
                    "galax",
                    "diffrax",
                    "equinox",
                    "unxt",
                    "numpy",
                ]
            },
            "devices": [str(device) for device in jax.devices()],
            "float64": bool(jax.config.jax_enable_x64),
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "case_ids": args.case,
        "variants": args.variant,
        "warm_repeats": args.warm_repeats,
        "cases": cases,
        "model_build_seconds": model_build_seconds,
        "rows": rows,
        "control_comparisons": comparisons,
        "work_directory": str(args.work_dir.relative_to(ROOT))
        if args.work_dir.is_relative_to(ROOT)
        else str(args.work_dir),
        "statistics_scope": (
            "All 50,000 samples; raw intrinsic normalized histograms/moments, "
            "before spatial symmetries, counter-rotation, or observational projection."
        ),
        "timing_scope": (
            "Synchronized whole production solve including sampling and energy "
            "diagnostics; excludes statistics and artifact I/O. First calls may "
            "reuse earlier compiled kernels; warm repeats are separate. Process "
            "peak RSS is cumulative, not per-orbit or device memory."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-checkout", type=Path, required=True)
    parser.add_argument("--case", nargs="+", choices=CASE_IDS, default=list(CASE_IDS))
    parser.add_argument(
        "--variant", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warm_repeats < 1 or "working" not in args.variant:
        parser.error("Use positive warm repeats and include the working variant.")
    if len(set(args.case)) != len(args.case) or len(set(args.variant)) != len(
        args.variant
    ):
        parser.error("Duplicate cases or variants are not allowed.")
    if args.output.exists():
        parser.error("Output already exists; choose a new result path.")
    if platform.system() != "Linux":
        parser.error(
            "This benchmark protocol requires the Linux development environment."
        )
    args.work_dir = args.work_dir.resolve()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
