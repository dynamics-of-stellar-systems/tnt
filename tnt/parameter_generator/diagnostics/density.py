"""Probability-mass contours on existing axes, without creating pyplot figures."""

import corner
import numpy as np
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet

from tnt.parameter_generator.diagnostics.style import (
    CONTOUR_MASSES,
    DIVERGENCE_ALPHA,
    DIVERGENCE_COLOR,
    DIVERGENCE_LINEWIDTH,
    DIVERGENCE_SIZE,
    PRIOR_COLOR,
)


def plot_density(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    diverging: np.ndarray,
    scales: tuple[str, str],
) -> None:
    """Estimate enclosed mass with corner's smoothed display-coordinate histogram.

    Prior sample scatter is suppressed, including outside the outer contour.
    Divergences remain visible in a separate purple overlay. Degenerate
    coordinates use a point projection rather than manufactured contours.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    if scales[0] == "log":
        valid &= x > 0
    if scales[1] == "log":
        valid &= y > 0
    x, y, diverging = x[valid], y[valid], diverging[valid]
    if not len(x):
        return
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        ax.plot(
            x,
            y,
            ".",
            color=PRIOR_COLOR,
            markersize=2,
            alpha=0.5,
            gid="prior-degenerate-samples",
            zorder=1,
        )
    else:
        before = len(ax.collections)
        corner.hist2d(
            x,
            y,
            ax=ax,
            bins=32,
            smooth=1.0,
            axes_scale=scales,
            levels=CONTOUR_MASSES,
            color=PRIOR_COLOR,
            quiet=True,
            plot_density=False,
            fill_contours=True,
            plot_datapoints=False,
            contour_kwargs={"linewidths": 0.8},
        )
        for artist in ax.collections[before:]:
            if isinstance(artist, QuadContourSet):
                artist.set_gid(
                    "prior-density-fill" if artist.filled else "prior-density-contours"
                )
    ax.scatter(
        x[diverging],
        y[diverging],
        s=DIVERGENCE_SIZE,
        marker="x",
        color=DIVERGENCE_COLOR,
        alpha=DIVERGENCE_ALPHA,
        linewidths=DIVERGENCE_LINEWIDTH,
        zorder=4,
        gid="prior-divergences",
    )
