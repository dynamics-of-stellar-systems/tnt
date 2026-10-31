"""Comparable probability-per-bin marginal histograms in display coordinates."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import StepPatch


def histogram_edges(
    values: np.ndarray, logarithmic: bool, bins: int = 28
) -> np.ndarray:
    """Use equally spaced display-coordinate bins, including constant samples."""
    values = np.asarray(values)
    valid = values[np.isfinite(values) & ((values > 0) if logarithmic else True)]
    if not len(valid):
        raise ValueError("No finite samples available for marginal histogram")
    low, high = float(valid.min()), float(valid.max())
    if logarithmic:
        low, high = np.log10([low, high])
    if low == high:
        pad = 0.025 if logarithmic else max(abs(low) * 0.025, 0.025)
        low, high = low - pad, high + pad
    edges = np.linspace(low, high, bins + 1)
    if logarithmic:
        edges = 10**edges
    # Log/power round-trips must not exclude an observed extreme by one ULP.
    edges[0] = min(edges[0], valid.min())
    edges[-1] = max(edges[-1], valid.max())
    return edges


def marginal_histogram(
    ax: Axes,
    values: np.ndarray,
    edges: np.ndarray,
    *,
    color: str,
    gid: str,
    filled: bool = False,
    linestyle: str = "-",
) -> StepPatch:
    """Normalize each group independently so unequal sample counts compare."""
    finite = np.asarray(values)[np.isfinite(values)]
    counts, _ = np.histogram(finite, bins=edges)
    fraction = counts / len(finite) if len(finite) else counts.astype(float)
    return ax.stairs(
        fraction,
        edges,
        color=color,
        fill=filled,
        alpha=0.25 if filled else 0.95,
        linewidth=0.8 if filled else 1.25,
        linestyle=linestyle,
        zorder=1 if filled else 5,
        gid=gid,
    )
