"""Export standalone scientific figures from the retained grid experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot(force_path: Path, orbit_path: Path, timing_path: Path, output: Path) -> None:
    """Show force convergence, request costs and full-orbit accuracy/runtime."""
    forces = json.loads(force_path.read_text())
    orbits = json.loads(orbit_path.read_text())
    timing = json.loads(timing_path.read_text())
    force_hash = hashlib.sha256(force_path.read_bytes()).hexdigest()
    if orbits["force_result_sha256"] != force_hash:
        raise ValueError("Orbit results reference a different force experiment.")
    if not all(row["diagnostics"]["successful"] for row in orbits["rows"]):
        raise ValueError("The summary figure requires successful orbit results.")
    orbit_hash = hashlib.sha256(orbit_path.read_bytes()).hexdigest()
    if timing["parent_orbit_sha256"] != orbit_hash:
        raise ValueError("Timing control references a different orbit experiment.")
    variants = ["direct", "coarse", "medium", "fine"]
    colors = ["#234E70", "#D98B28", "#438964", "#AD4B54"]
    labels = ["Direct", "Coarse grid", "Medium grid", "Fine grid"]
    cases = orbits["cases"]
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for geometry, color, marker in [
        ("oblate", "#234E70", "o"), ("strong_triaxial", "#AD4B54", "s")
    ]:
        rows = [r for r in forces["force_rows"] if r["geometry"] == geometry]
        grid_rows = [r for r in rows if r["variant"] != "direct"]
        axes[0, 0].plot(
            [np.prod(r["setup"]["spec"]["shape"]) for r in grid_rows],
            [100 * r["force_error"]["max_relative"] for r in grid_rows],
            marker=marker, color=color, label=geometry.replace("_", " "),
        )
        axes[0, 1].plot(
            np.arange(4),
            [1000 * r["serial_timing"]["warm_median_seconds"] for r in rows],
            marker=marker, color=color,
        )
    axes[0, 0].set(xscale="log", yscale="log", xlabel="Number of grid nodes",
                   ylabel="Maximum relative force error (%)",
                   title="A   Force convergence on 4,645 probes per model")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set(yscale="log", xticks=np.arange(4), xticklabels=labels,
                   ylabel="Warm median (ms / 2,048 requests)",
                   title="B   Sequential compiled force requests")
    x = np.arange(len(cases))
    for i, (variant, color, label) in enumerate(zip(variants, colors, labels)):
        rows = [next(r for r in orbits["rows"]
                     if r["case_id"] == case["id"] and r["variant"] == variant)
                for case in cases]
        for axis, values in [
            (axes[1, 0], [r["diagnostics"]["max_energy_error"] for r in rows]),
            (axes[1, 1], [
                timing["records"][variant]["warm_median_seconds"]
                if r["case_id"] == timing["case_id"] else r["warm_median_seconds"]
                for r in rows
            ]),
        ]:
            axis.bar(x + (i - 1.5) * 0.19, values, width=0.18,
                     color=color, label=label)
    names = [c["id"].replace("strong_triaxial", "Triaxial")
             .replace("oblate", "Oblate").replace("-", "\n", 1)
             + (" *" if c["id"] == timing["case_id"] else "") for c in cases]
    for axis in axes[1]:
        axis.set_xticks(x, names)
    axes[1, 0].set(yscale="log", ylabel="Maximum |E - E launch| / V²",
                   title="C   Energy measured in the direct physical potential")
    axes[1, 0].legend(frameon=False, fontsize=9)
    axes[1, 1].set(ylabel="Warm whole-solve median (s)",
                   title="D   200 periods · 50,000 samples · three repeats")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Direct versus experimental force grids\n"
                 "CPU · float64 · Gaussian order 800 · Dopri8 accuracy 1e-9",
                 fontsize=15)
    fig.supxlabel("* Runtime from the keep-awake, rotated-order timing control. "
                  "All original observations remain in the report.", fontsize=9)
    provenance = "; ".join(
        f"{p.name} SHA256 {hashlib.sha256(p.read_bytes()).hexdigest()}"
        for p in (force_path, orbit_path, timing_path)
    )
    output.mkdir(parents=True, exist_ok=False)
    fig.savefig(output / "comparison.png", dpi=180,
                metadata={"Description": provenance})
    fig.savefig(output / "comparison.pdf", metadata={"Subject": provenance})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forces", type=Path, required=True)
    parser.add_argument("--orbits", type=Path, required=True)
    parser.add_argument("--timing-control", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.forces, args.orbits, args.timing_control, args.output_dir)
