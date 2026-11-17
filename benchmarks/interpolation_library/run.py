"""Native TNT small-library interpolation experiment; run from repository root."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import unxt as u
import yaml

from benchmarks.force_grid import ForceGrid, GridSpec, build_grid, integrate_grid
from tnt.mge import LightMGE
from tnt.orbit_library import (
    OrbitIntegrationSettings,
    StationaryGridOrbitSampler,
    XZGridFromBoundaryOrbitSampler,
    integrate_orbit,
)
from tnt.potential import Potential

ROOT = Path(__file__).resolve().parent


def ready(value):
    jax.block_until_ready(eqx.filter(value, eqx.is_array))
    return value


def dump(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def build_model(config):
    m = config["mge"]
    mge = LightMGE(
        I=u.Quantity(jnp.array(m["I_Lsun_pc2"]), "Lsun/pc2"),
        sigma=u.Quantity(jnp.array(m["sigma_arcsec"]), "arcsec"),
        q=u.Quantity(jnp.array(m["q"]), ""),
        PA_twist=u.Quantity(jnp.zeros(len(m["q"])), "deg"),
        major_axis_pa=u.Quantity(0.0, "deg"),
    ).angular_to_physical(u.Quantity(m["distance_Mpc"], "Mpc"))
    resolved = Potential.resolve(
        config["potential"],
        {"light": mge},
        gaussian_potential_quad_order=config["numerics_settings"][
            "gaussian_potential_quad_order"
        ],
    )
    proposal = {
        name: {
            key: u.Quantity(p["value"], p["unit"])
            for key, p in spec["parameters"].items()
        }
        for name, spec in config["potential"].items()
    }
    model, valid = Potential.build_with_validity(resolved, proposal, {})
    if not bool(valid):
        raise ValueError("Invalid physical model")
    return model.to_galax(u.unitsystem("kpc", "Myr", "Msun", "rad"))


def starts(model, config, work):
    path = work / "launches.npz"
    if path.exists():
        with np.load(path) as saved:
            return {k: saved[k] for k in saved.files}
    settings = dict(config["initial_conditions"])
    for key in ("rmin", "rmax"):
        settings[key] = u.Quantity(**settings[key])
    start = time.perf_counter()
    boundary = XZGridFromBoundaryOrbitSampler(**settings).generate_initial_conditions(
        model
    )
    periods = boundary.reference_periods[:: settings["nI2"] * settings["nI3"], 0]
    stationary = StationaryGridOrbitSampler(**settings).generate_initial_conditions(
        model, reference_periods=periods
    )
    ready((boundary, stationary))
    launches = {
        "positions": np.concatenate(
            [
                np.asarray(s.positions[:, 0].ustrip("kpc"))
                for s in (boundary, stationary)
            ]
        ),
        "velocities": np.concatenate(
            [
                np.asarray(s.velocities[:, 0].ustrip("kpc/Myr"))
                for s in (boundary, stationary)
            ]
        ),
        "periods": np.concatenate(
            [
                np.asarray(s.reference_periods[:, 0].ustrip("Myr"))
                for s in (boundary, stationary)
            ]
        ),
        "sampler": np.repeat(
            ["XZGridFromBoundary", "StationaryGrid"], boundary.bundle_ids.size
        ),
    }
    np.savez_compressed(path, **launches)
    dump(
        work / "launches.json",
        {"seconds": time.perf_counter() - start, "count": len(launches["periods"])},
    )
    return launches


def grid(model, config, method, work):
    path = work / f"grid-{method}.npz"
    spec = GridSpec(
        tuple(config["grid"]["shape"]),
        config["grid"]["r_min"],
        config["grid"]["r_max"],
        interpolation=method,
        outside_policy=config["grid"].get("outside_policy", "reject"),
    )
    if path.exists():
        with np.load(path) as saved:
            return ForceGrid(model, jnp.asarray(saved["table"]), spec)
    start = time.perf_counter()
    compiled = build_grid.lower(model, spec).compile()
    compilation = time.perf_counter() - start
    start = time.perf_counter()
    result = ready(compiled(model, spec))
    construction = time.perf_counter() - start
    np.savez_compressed(path, table=np.asarray(result.log_coefficients))
    dump(
        work / f"grid-{method}.json",
        {
            "spec": asdict(spec),
            "compilation_seconds": compilation,
            "construction_seconds": construction,
            "bytes": result.log_coefficients.nbytes,
        },
    )
    print(
        f"Grid {method}: compile {compilation:.2f}s, build {construction:.2f}s",
        flush=True,
    )
    return result


def energy_history(model, trajectory, q, v, period, hermite=False):
    states = jnp.concatenate(
        [trajectory.positions.ustrip("kpc"), trajectory.velocities.ustrip("kpc/Myr")],
        axis=1,
    )
    initial = jnp.concatenate([q, v])
    potential = model.interpolated_potential if hermite else model.potential

    def energy(state):
        return potential(u.Quantity(state[:3], "kpc"), u.Quantity(0.0, "Myr")).ustrip(
            "kpc2/Myr2"
        ) + 0.5 * jnp.sum(state[3:] ** 2)

    values = jax.lax.map(
        energy, jnp.concatenate([initial[None], states]), batch_size=128
    )
    length = jnp.maximum(jnp.linalg.norm(q), jnp.linalg.norm(v) * period / (2 * jnp.pi))
    return np.asarray((values[1:] - values[0]) / (2 * jnp.pi * length / period) ** 2)


def solve(model, evaluator, method, config, launches, index, tolerance, work):
    stem = f"orbit-{index:03d}-{method}-{tolerance:.0e}"
    path = work / f"{stem}.json"
    if path.exists():
        return json.loads(path.read_text())
    settings = OrbitIntegrationSettings(accuracy=tolerance, **config["integration"])
    q, v, period = (
        launches[name][index] for name in ("positions", "velocities", "periods")
    )
    args = (
        evaluator,
        u.Quantity(q, "kpc"),
        u.Quantity(v, "kpc/Myr"),
        u.Quantity(period, "Myr"),
        jax.random.fold_in(jax.random.key(config["seed"]), index),
        settings,
    )
    runner = integrate_orbit if method == "direct" else integrate_grid
    record = {
        "index": index,
        "sampler": str(launches["sampler"][index]),
        "method": method,
        "tolerance": tolerance,
        "successful": False,
    }
    start = time.perf_counter()
    try:
        compiled = runner.lower(*args).compile()
        record["compilation_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        trajectory = ready(compiled(*args))
        record["integration_seconds"] = time.perf_counter() - start
        record["diagnostics"] = {
            k: np.asarray(v).item() for k, v in asdict(trajectory.diagnostics).items()
        }
        record["successful"] = bool(trajectory.diagnostics.successful)
        if record["successful"]:
            payload = {
                "times_Myr": np.asarray(trajectory.times.ustrip("Myr")),
                "positions_kpc": np.asarray(trajectory.positions.ustrip("kpc")),
                "velocities_kpc_Myr": np.asarray(
                    trajectory.velocities.ustrip("kpc/Myr")
                ),
                "physical_energy_drift": energy_history(
                    model, trajectory, q, v, period
                ),
            }
            if method != "direct":
                radii = np.linalg.norm(payload["positions_kpc"], axis=1)
                record["saved_samples_outside_grid"] = int(np.sum(
                    (radii < evaluator.spec.r_min) | (radii > evaluator.spec.r_max)))
                # This counts saved samples, not internal adaptive-stage calls.
            if method in {"hermite", "quintic"}:
                payload["hermite_energy_drift"] = energy_history(
                    evaluator, trajectory, q, v, period, hermite=True
                )
            np.savez_compressed(work / f"{stem}.npz", **payload)
    except (RuntimeError, ValueError, FloatingPointError) as error:
        record["successful"] = False
        record["error"] = str(error)
        record["failed_attempt_seconds"] = time.perf_counter() - start
    dump(path, record)
    print(f"{stem}: {record}", flush=True)
    return record


def run(args):
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    args.work.mkdir(parents=True, exist_ok=True)
    provenance = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path("benchmarks/force_grid.py"))
        },
        "jax": jax.__version__,
        "platform": platform.platform(),
    }
    marker = args.work / "provenance.json"
    if marker.exists():
        if json.loads(marker.read_text()) != provenance:
            raise ValueError("Resume provenance changed; choose a fresh work directory")
    else:
        dump(marker, provenance)
    if args.reuse_setup:
        parent = json.loads((args.reuse_setup / "provenance.json").read_text())
        if parent["config_sha256"] != provenance["config_sha256"]:
            raise ValueError("Setup configuration differs")
        source = "benchmarks/force_grid.py"
        if parent["source_sha256"][source] != provenance["source_sha256"][source]:
            raise ValueError("Setup force evaluator differs")
        for source_path in [
            *args.reuse_setup.glob("grid-*"),
            *args.reuse_setup.glob("launches.*"),
        ]:
            target = args.work / source_path.name
            if not target.exists():
                shutil.copyfile(source_path, target)
        origin = args.work / "setup-origin.json"
        if not origin.exists():
            dump(origin, {"directory": str(args.reuse_setup), "provenance": parent})
    snapshot = args.work / "runner-source.py"
    if not snapshot.exists():
        shutil.copyfile(__file__, snapshot)
    model = ready(build_model(config))
    print("Generating/loading shared launch set", flush=True)
    launches = starts(model, config, args.work)
    indices = [0] if args.phase == "pilot" else range(len(launches["periods"]))
    tolerances = [1e-9] if args.phase == "pilot" else config["tolerances"]
    methods = config.get("methods", ["direct", "linear", "cubic", "hermite"])
    if not methods or len(set(methods)) != len(methods) or any(
        method not in {"direct", "linear", "cubic", "hermite", "quintic"}
        for method in methods
    ):
        raise ValueError("Choose distinct supported interpolation methods.")
    records = []
    # Keep only one grid in memory; no concurrent scientific solves.
    for method in methods:
        evaluator = (
            model if method == "direct" else grid(model, config, method, args.work)
        )
        for tolerance in tolerances:
            for index in indices:
                records.append(
                    solve(
                        model,
                        evaluator,
                        method,
                        config,
                        launches,
                        index,
                        tolerance,
                        args.work,
                    )
                )
    summary = {
        "phase": args.phase,
        "records": records,
        "integration_totals_seconds": {
            method: sum(
                row.get("integration_seconds", 0)
                for row in records
                if row["method"] == method
            )
            for method in methods
        },
    }
    summary["library_integration_seconds"] = {
        method: {
            str(tolerance): sum(
                row.get("integration_seconds", 0)
                for row in records
                if row["method"] == method and row["tolerance"] == tolerance
            )
            for tolerance in tolerances
        }
        for method in methods
    }
    summary["failed_solves"] = sum(not row["successful"] for row in records)
    dump(args.work / f"{args.phase}-summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "input/config.yaml")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--reuse-setup", type=Path)
    parser.add_argument("--phase", choices=("pilot", "suite"), required=True)
    run(parser.parse_args())
