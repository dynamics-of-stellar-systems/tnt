"""Shared overlays for returned parameter proposals and fixed locations."""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from tnt.parameter_generator.diagnostics.marginals import (
    histogram_edges,
    marginal_histogram,
)
from tnt.parameter_generator.diagnostics.style import (
    FIXED_POINT_COLOR,
    LEGEND_FONTSIZE,
    NEW_POINT_COLOR,
    PRIOR_COLOR,
)


def add_proposal_overlay(
    fig: Figure,
    names: tuple[str, ...],
    new: dict[str, np.ndarray],
    fixed: dict[str, np.ndarray] | None,
    prior_samples: dict[str, np.ndarray],
    *,
    new_label: str = "new support points",
    new_gid: str = "support",
) -> None:
    """Add overlays to existing corner axes; leave prior traces untouched."""
    for ax in fig.axes:
        if not (ax.get_gid() or "").startswith("prior-corner-"):
            continue
        i, j = map(int, ax.get_gid().rsplit("-", 2)[-2:])
        if i == j:
            groups = [prior_samples[names[j]], new[names[j]]]
            if fixed:
                groups.append(fixed[names[j]])
            edges = histogram_edges(np.concatenate(groups), ax.get_xscale() == "log")
            for artist in list(ax.patches):
                if artist.get_gid() == "prior-marginal":
                    artist.remove()
            marginal_histogram(
                ax,
                prior_samples[names[j]],
                edges,
                color=PRIOR_COLOR,
                gid="prior-marginal",
                filled=True,
            )
            for values, is_fixed in ((fixed, True), (new, False)):
                if values and len(values[names[j]]):
                    marginal_histogram(
                        ax,
                        values[names[j]],
                        edges,
                        color=FIXED_POINT_COLOR if is_fixed else NEW_POINT_COLOR,
                        gid="fixed-support-marginal"
                        if is_fixed
                        else new_gid + "-marginal",
                        linestyle="--" if is_fixed else "-",
                    )
            ax.set_xlim(edges[0], edges[-1])
            ax.relim()
            ax.autoscale_view(scalex=False)
        else:
            for values, is_fixed in ((fixed, True), (new, False)):
                if not values:
                    continue
                gid = "fixed-support-points" if is_fixed else new_gid + "-points"
                ax.scatter(
                    values[names[j]],
                    values[names[i]],
                    marker="o" if is_fixed else "D",
                    s=10 if is_fixed else 16,
                    facecolors="none" if is_fixed else NEW_POINT_COLOR,
                    edgecolors=FIXED_POINT_COLOR if is_fixed else "white",
                    linewidths=0.65 if is_fixed else 0.4,
                    zorder=6 if is_fixed else 7,
                    gid=gid + "-scatter",
                )
                # Corner sets explicit limits; include optimized locations that
                # extend beyond the empirical sample range.
                for axis, site in (("x", names[j]), ("y", names[i])):
                    points = values[site]
                    if len(points):
                        limits = ax.get_xlim() if axis == "x" else ax.get_ylim()
                        limits = (
                            min(limits[0], points.min()),
                            max(limits[1], points.max()),
                        )
                        (ax.set_xlim if axis == "x" else ax.set_ylim)(limits)
    if names:
        old = fig.legends[0]
        handles = list(old.legend_handles)
        old.remove()
        handles.append(
            Line2D(
                [],
                [],
                marker="D",
                linestyle="-",
                color=NEW_POINT_COLOR,
                markeredgecolor="white",
                markersize=4,
                label=f"{new_label} (m={len(new[names[0]])})",
            )
        )
        if fixed:
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="--",
                    color=FIXED_POINT_COLOR,
                    markerfacecolor="none",
                    markersize=4,
                    label=f"fixed support points (k={len(fixed[names[0]])})",
                )
            )
        fig.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            fontsize=LEGEND_FONTSIZE,
            frameon=True,
        )
