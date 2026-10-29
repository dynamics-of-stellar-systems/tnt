"""Prior, support-point, and actual optimizer-trace diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from jax import Array
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from tnt.all_models import AllModels
from tnt.parameter_generator import ParameterSet, PriorSupportPoints
from tnt.priors_diagnostic_plot import _build_diagnostics_figure


def _add_support_points(
    fig: Figure,
    names: tuple[str, ...],
    new: dict[str, np.ndarray],
    fixed: dict[str, np.ndarray] | None,
) -> None:
    """Add overlays to existing corner axes; leave prior traces untouched."""
    for ax in fig.axes:
        spec = ax.get_subplotspec()
        if spec is None or spec.rowspan.stop - spec.rowspan.start != 1:
            continue
        i, j = spec.rowspan.start, spec.colspan.start
        if (
            spec.colspan.stop - spec.colspan.start != 1
            or i >= len(names)
            or j >= len(names)
            or i < j
        ):
            continue
        for values, is_fixed in ((new, False), (fixed, True)):
            if not values:
                continue
            gid = "fixed-support-points" if is_fixed else "support-points"
            if i == j:
                ax.plot(
                    values[names[j]],
                    np.full(len(values[names[j]]), 0.15 if is_fixed else 0.06),
                    linestyle="",
                    marker="|",
                    markersize=13,
                    color="black" if is_fixed else "darkorange",
                    transform=ax.get_xaxis_transform(),
                    zorder=5,
                    gid=gid + "-rug",
                )
            else:
                ax.scatter(
                    values[names[j]],
                    values[names[i]],
                    marker="o" if is_fixed else "*",
                    s=38 if is_fixed else 95,
                    facecolors="none" if is_fixed else "darkorange",
                    edgecolors="black",
                    linewidths=0.7,
                    zorder=5,
                    gid=gid + "-scatter",
                )
    if names:
        old = fig.legends[0]
        handles = list(old.legend_handles)
        old.remove()
        handles.append(
            Line2D(
                [],
                [],
                marker="*",
                linestyle="",
                color="darkorange",
                markeredgecolor="black",
                markersize=10,
                label=f"new support points (m={len(new[names[0]])})",
            )
        )
        if fixed:
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    color="black",
                    markerfacecolor="none",
                    label=f"fixed support points (k={len(fixed[names[0]])})",
                )
            )
        fig.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            fontsize=9,
            frameon=False,
        )


def _add_optimization_panels(
    fig: Figure, info: dict[str, Any], tolerance: float
) -> None:
    """Reserve space below the original figure without shrinking its panels."""
    diagnostics = info["optimization"]
    history = diagnostics.history
    if history is None:
        raise ValueError(
            "Support-point plotting requires recorded optimization history"
        )
    width, height = fig.get_size_inches()
    extra = 3.1
    # The extra overlay legend entry needs clearance above the corner titles.
    new_height = height + extra + 0.45
    fig.set_size_inches(max(width, 10.0), new_height)
    for ax in fig.axes:
        position = ax.get_position()
        ax.set_position(
            [
                position.x0,
                (extra + position.y0 * height) / new_height,
                position.width,
                position.height * height / new_height,
            ]
        )
    for text in fig.texts:
        if text is not fig._suptitle:
            x, y = text.get_position()
            text.set_position((x, (extra + y * height) / new_height))
    bottom, panel_height = 0.58 / new_height, 1.95 / new_height
    objective = fig.add_axes([0.08, bottom, 0.39, panel_height])
    objective.plot(np.arange(len(history.objective)), history.objective, color="C0")
    objective.set(
        xlabel="optimizer iteration",
        ylabel="combined energy objective",
        title=f"Objective: {info['existing_support_points']} fixed + "
        f"{info['new_support_points']} new",
    )
    objective.set_gid("support-objective-history")
    objective.ticklabel_format(axis="y", style="plain", useOffset=False)
    objective.grid(alpha=0.2)
    movement = fig.add_axes([0.57, bottom, 0.39, panel_height])
    steps = np.arange(1, len(history.maximum_displacement) + 1)
    movement.plot(steps, history.maximum_displacement, label="maximum displacement")
    movement.plot(steps, history.rms_displacement, label="RMS displacement")
    movement.axhline(
        tolerance, color="black", linestyle=":", label="stopping tolerance"
    )
    movement.set(
        xlabel="optimizer iteration",
        ylabel="scaled coordinate distance",
        title="New-point movement",
        yscale="symlog",
    )
    movement.set_yscale("symlog", linthresh=tolerance)
    movement.set_ylim(bottom=0)
    movement.set_gid("support-displacement-history")
    movement.grid(alpha=0.2)
    movement.legend(fontsize=8)
    fig.suptitle(f"Prior support-point diagnostics — {diagnostics.status}", fontsize=13)


def _build_support_points_diagnostics_figure(
    generator: PriorSupportPoints,
    key: Array,
    all_models: AllModels,
) -> tuple[Figure, list[ParameterSet], dict[str, Any]]:
    """Generate once with tracing, then reuse the ordinary prior plot layout."""
    candidates, info = generator.generate_with_diagnostics(
        all_models, record_history=True
    )
    names = info["site_names"]
    overlay = {}
    for site in names:
        component, parameter = site.split(".")
        unit = generator.potential_settings[component]["parameters"][parameter].get(
            "unit", ""
        )
        overlay[site] = np.array(
            [
                float(candidate[component][parameter].ustrip(unit))
                for candidate in candidates
            ]
        )
    existing = generator._existing_locations(all_models, names)
    fixed = (
        {site: existing[:, index] for index, site in enumerate(names)}
        if len(existing)
        else None
    )
    fig = _build_diagnostics_figure(
        generator.prior,
        key,
        generator.num_samples,
        generator.num_warmup,
        generator.target_accept_prob,
    )
    _add_support_points(fig, names, overlay, fixed)
    _add_optimization_panels(
        fig, info, generator.generator_settings.get("tolerance", 1e-6)
    )
    return fig, candidates, info


def plot_prior_support_points_diagnostics(
    generator: PriorSupportPoints,
    path: str | Path,
    key: Array,
    all_models: AllModels | None = None,
) -> tuple[list[ParameterSet], dict[str, Any]]:
    """Generate a design and save prior/support/optimization diagnostics.

    Args:
        generator: The configured support-point generator and its run prior.
        path: Destination image path; its parent must already exist.
        key: Independently chosen key for the plotted prior background.
        all_models: Existing locations to hold fixed, or None for the first batch.

    Returns:
        New candidates and the exact generation diagnostics, including history.
        The caller may use these candidates without running generation again.

    Orange stars show new points and open black circles show existing fixed
    locations. Original units and the prior plot's automatic display scales
    are retained. The objective includes existing points; movement measures
    only the new ones. Iteration zero is initialization. Trace collection uses
    ordered JAX diagnostic callbacks from the actual built-in Optimistix solve,
    not a separate reconstruction or repeated optimization. No point trajectory
    is retained. Recording adds objective evaluations and host synchronization;
    ordinary generation does not incur that cost. No expensive model is fitted.
    """
    history = AllModels() if all_models is None else all_models
    fig, candidates, info = _build_support_points_diagnostics_figure(
        generator, key, history
    )
    fig.savefig(path, dpi=150)
    return candidates, info
