"""Targeted quintic comparison using preserved round-two tables and launches."""

import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import yaml
from unxt import Quantity

from benchmarks.force_grid import ForceGrid, GridSpec, _quintic_table
from benchmarks.interpolation_library.run import build_model, ready, solve


def main():
    root = Path("benchmarks/interpolation_library")
    old = root / "work/round2-20261111"
    work = root / "work/quintic-probe-20261117"
    work.mkdir(exist_ok=True)
    config = yaml.safe_load((old / "config.yaml").read_text())
    model = ready(build_model(config))
    with np.load(old / "grid-hermite.npz") as f:
        jets = jnp.asarray(f["table"])
    t = time.perf_counter()
    table = ready(jax.jit(_quintic_table)(jets))
    print("quintic conversion seconds", time.perf_counter() - t, flush=True)
    spec = GridSpec(
        tuple(config["grid"]["shape"]),
        config["grid"]["r_min"],
        config["grid"]["r_max"],
        interpolation="quintic",
        outside_policy="direct",
    )
    quintic = ForceGrid(model, table, spec)
    rng = np.random.default_rng(20261117)
    points = rng.normal(size=(2048, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    points *= np.exp(rng.uniform(np.log(0.001), np.log(100), size=(2048, 1)))

    def forces(g):
        return np.asarray(
            ready(
                jax.jit(
                    lambda q: g.acceleration(
                        Quantity(q, "kpc"), Quantity(0.0, "Myr")
                    ).ustrip("kpc/Myr2")
                )(jnp.asarray(points))
            )
        )

    reference = forces(model)
    report = {}
    for method in ["linear", "cubic", "hermite", "quintic"]:
        if method == "quintic":
            g = quintic
        else:
            with np.load(old / f"grid-{method}.npz") as f:
                a = jnp.asarray(f["table"])
            g = ForceGrid(
                model,
                a,
                GridSpec(
                    spec.shape,
                    spec.r_min,
                    spec.r_max,
                    interpolation=method,
                    outside_policy="direct",
                ),
            )
        err = np.linalg.norm(forces(g) - reference, axis=1) / np.linalg.norm(
            reference, axis=1
        )
        report[method] = dict(
            zip(
                ["median", "p95", "max"],
                map(float, [np.median(err), np.quantile(err, 0.95), np.max(err)]),
            )
        )
        print(method, report[method], flush=True)
    (work / "force-errors.json").write_text(json.dumps(report, indent=2) + "\n")
    with np.load(old / "launches.npz") as f:
        launches = {k: f[k] for k in f.files}
    for index in [0, 16]:
        record = solve(model, quintic, "quintic", config, launches, index, 1e-9, work)
        print("orbit", index, record, flush=True)
        if record["successful"]:
            with np.load(work / f"orbit-{index:03d}-quintic-1e-09.npz") as f:
                print(
                    "energies",
                    index,
                    {
                        k: float(np.max(np.abs(f[k])))
                        for k in ["physical_energy_drift", "hermite_energy_drift"]
                    },
                    flush=True,
                )


if __name__ == "__main__":
    main()
