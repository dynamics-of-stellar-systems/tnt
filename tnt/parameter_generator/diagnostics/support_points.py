"""Prior, support-point, and actual optimizer-trace diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from tnt.parameter_generator.base import ParameterSet
from tnt.parameter_generator.diagnostics.overlays import add_proposal_overlay
from tnt.parameter_generator.diagnostics.prior import build_prior_diagnostics_figure
from tnt.parameter_generator.diagnostics.style import (
    SECTION_FONTSIZE,
    TITLE_FONTSIZE,
    place_header,
    style_axes,
)


def _add_optimization_panels(fig: Figure, info: dict[str, Any]) -> None:
    """Reserve space below the original figure without shrinking its panels."""
    diagnostics = info["optimization"]
    history = diagnostics.history
    if history is None:
        raise ValueError(
            "Support-point plotting requires recorded optimization history"
        )
    width, height = fig.get_size_inches()
    extra = 6.7
    new_height = height + extra
    fig.set_size_inches(width, new_height)
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
    corners = [
        ax for ax in fig.axes if (ax.get_gid() or "").startswith("prior-corner-")
    ]
    left = min(ax.get_position().x0 for ax in corners) if corners else 0.15
    traces = [ax for ax in fig.axes if (ax.get_gid() or "").startswith("prior-trace-")]
    right = max((ax.get_position().x1 for ax in traces + corners), default=0.95)
    span = right - left
    steps = diagnostics.iterations
    status = {
        "converged": f"Converged after {steps:,} steps",
        "iteration_limit": f"Iteration limit reached ({steps:,} steps)",
        "nonfinite": f"Non-finite result after {steps:,} steps",
        "solver_failure": f"Solver failed after {steps:,} steps",
    }.get(diagnostics.status, f"{diagnostics.status} ({steps:,} steps)")
    heading = "Support Point Optimization\n" + status
    fig.text(
        (left + right) / 2,
        6.3 / new_height,
        heading,
        ha="center",
        va="top",
        fontsize=SECTION_FONTSIZE,
    )
    objective = fig.add_axes([left, 3.25 / new_height, span, 2.15 / new_height])
    values = np.asarray(history.objective)
    objective.plot(np.arange(len(values)), values, color="C0")
    objective.set(ylabel="Energy objective")
    objective.tick_params(labelbottom=False)
    objective.set_gid("support-objective-history")
    objective.ticklabel_format(axis="y", style="plain", useOffset=False)
    objective.grid(alpha=0.2)
    spacing = fig.add_axes(
        [left, 0.7 / new_height, span, 2.15 / new_height], sharex=objective
    )
    spacing.plot(
        np.arange(len(history.minimum_pairwise_distance)),
        history.minimum_pairwise_distance,
        color="C1",
    )
    spacing.set(xlabel="optimizer iteration", ylabel="Minimum pairwise\ndistance")
    spacing.set_ylim(bottom=0)
    spacing.set_gid("support-spacing-history")
    spacing.grid(alpha=0.2)
    fig.suptitle(
        f"PriorSupportPoints — iteration {info['iteration']}", fontsize=TITLE_FONTSIZE
    )


def build_support_points_diagnostics_figure(
    candidates: list[ParameterSet], info: dict[str, Any]
) -> Figure:
    """Render returned candidates, fixed points and recorded generation data."""
    if "sample_batch" not in info or info["optimization"].history is None:
        raise ValueError(
            "Support-point rendering requires record_history=True during generation "
            "or plot_diagnostics enabled."
        )
    names = info["site_names"]
    overlay = {}
    for site in names:
        component, parameter = site.split(".")
        unit = info["parameter_units"][site]
        overlay[site] = np.array(
            [float(point[component][parameter].ustrip(unit)) for point in candidates]
        )
    fig = build_prior_diagnostics_figure(
        info["sample_batch"], iteration=info["iteration"]
    )
    fixed = info["fixed_points"] if info["existing_support_points"] else None
    add_proposal_overlay(
        fig,
        names,
        overlay,
        fixed,
        {
            name: np.asarray(values)
            for name, values in info["sample_batch"].samples.items()
        },
    )
    _add_optimization_panels(fig, info)
    style_axes(fig)
    place_header(fig)
    return fig


def plot_prior_support_points_diagnostics(
    candidates: list[ParameterSet], info: dict[str, Any], path: str | Path
) -> None:
    """Save diagnostic data from one generation call without resampling.

    Obtain a batch from `generate_parameters(record_history=True)` and pass
    its `parameters` and `diagnostics`, or enable `plot_diagnostics` to render
    automatically. Overlays show only returned points; objective and
    spacing histories describe the optimized design before duplicate filtering.
    Rendering performs no prior sampling, optimization or model evaluation.
    """
    fig = build_support_points_diagnostics_figure(candidates, info)
    fig.savefig(path, dpi=150)
