"""Corner, trace and sampling diagnostics for an actual generation batch.

Uses Figure/CanvasAgg directly without changing Matplotlib global state.
Rendering does not sample, optimize, or evaluate models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

from tnt.parameter_generator.diagnostics.density import plot_density
from tnt.parameter_generator.diagnostics.marginals import (
    histogram_edges,
    marginal_histogram,
)
from tnt.parameter_generator.diagnostics.style import (
    DIVERGENCE_ALPHA,
    DIVERGENCE_COLOR,
    DIVERGENCE_LINEWIDTH,
    DIVERGENCE_SIZE,
    LEGEND_FONTSIZE,
    PRIOR_COLOR,
    place_header,
    style_axes,
)
from tnt.parameter_generator.diagnostics.style import (
    LABEL_FONTSIZE as _LABEL_FONTSIZE,
)
from tnt.parameter_generator.diagnostics.style import (
    PANEL_SIZE as _PANEL_SIZE,
)
from tnt.parameter_generator.diagnostics.style import (
    TITLE_FONTSIZE as _TITLE_FONTSIZE,
)
from tnt.parameter_generator.samples import PriorSampleBatch

# wspace/hspace between corner-plot panels; large enough that a diagonal
# panel's title (drawn just above its axes) doesn't collide with the row above.
_PANEL_GAP = 0.15


def _use_log_scale(distribution: str | None) -> bool:
    """Use logarithmic display only for actual LogUniform/LogNormal sites."""
    return distribution in {"LogUniform", "LogNormal"}


def _set_log_axis(ax: Axes, which: str) -> None:
    """Keep logarithmic tick labels sparse, including on shared corner axes."""
    if which == "x":
        ax.set_xscale("log")
        axis = ax.xaxis
    else:
        ax.set_yscale("log")
        axis = ax.yaxis
    axis.set_major_locator(LogLocator(base=10, numticks=4))
    axis.set_major_formatter(LogFormatterMathtext(base=10))
    axis.set_minor_formatter(NullFormatter())


def _ess_color(ess: float) -> str:
    """Green/amber/red for ESS `> 100` / `30-100` / `< 30`."""
    if ess > 100:
        return "#DCE9DF"
    if ess >= 30:
        return "#F3E7C9"
    return "#EED9D8"


def _split_rhat_color(rhat: float) -> str:
    """Green/amber/red for split-R-hat `< 1.01` / `1.01-1.05` / `> 1.05`
    (Vehtari et al. 2021's recommended convergence threshold).
    """
    if rhat < 1.01:
        return "#DCE9DF"
    if rhat <= 1.05:
        return "#F3E7C9"
    return "#EED9D8"


def _divergence_color(fraction: float) -> str:
    """Green/amber/red for a divergence fraction of `< 1%` / `1-10%` / `>= 10%`."""
    if fraction < 0.01:
        return "#DCE9DF"
    if fraction < 0.1:
        return "#F3E7C9"
    return "#EED9D8"


def plot_prior_diagnostics(
    batch: PriorSampleBatch,
    path: str | Path,
    *,
    iteration: int = 0,
    proposed_samples: dict[str, np.ndarray] | None = None,
) -> None:
    """Render a generation batch without sampling or modifying its data."""
    fig = build_prior_diagnostics_figure(
        batch, iteration=iteration, proposed_samples=proposed_samples
    )
    fig.savefig(path, dpi=150)


def build_prior_diagnostics_figure(
    batch: PriorSampleBatch,
    *,
    iteration: int = 0,
    proposed_samples: dict[str, np.ndarray] | None = None,
) -> Figure:
    """Build a corner grid, horizontal traces and a sampling-summary block.

    Rendering uses actual generation data and never samples or evaluates models.
    Log display follows sampling distributions without changing design distances.
    """
    samples, diagnostics = batch
    names = sorted(samples)
    n = len(names)
    title = f"PriorSampler — iteration {iteration}"
    if n == 0:
        fig = Figure(figsize=(10.0, 2.0), dpi=150)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No non-fixed parameters to sample.",
            ha="center",
            va="center",
            fontsize=_LABEL_FONTSIZE,
        )
        fig.suptitle(title, fontsize=_TITLE_FONTSIZE)
        return fig

    values = {name: np.asarray(samples[name]) for name in names}
    distributions = diagnostics.sample_distributions or {}
    log_scale = {name: _use_log_scale(distributions.get(name)) for name in names}
    diverging = np.asarray(diagnostics.diverging)
    num_samples = len(diverging)
    n_divergent = int(np.sum(diverging))
    # A minimum two-slot grid also leaves a summary area for one parameter.
    grid_n = max(2, n)
    row_offset = grid_n - n
    corner_size = _PANEL_SIZE * grid_n + 0.3 * (grid_n - 1)
    left, bottom, trace_width, trace_gap = 0.95, 0.8, 2 * _PANEL_SIZE, 0.9
    width = left + corner_size + trace_gap + trace_width + 0.35
    corner_bottom = bottom
    height = corner_bottom + corner_size + 0.85
    fig = Figure(figsize=(width, height), dpi=150)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(
        grid_n,
        grid_n,
        left=left / width,
        right=(left + corner_size) / width,
        bottom=corner_bottom / height,
        top=(corner_bottom + corner_size) / height,
        wspace=_PANEL_GAP,
        hspace=_PANEL_GAP,
    )
    col_anchor: dict[int, Axes] = {}
    row_anchor: dict[int, Axes] = {}
    for i in range(n):
        for j in range(i + 1):
            row_name, col_name = names[i], names[j]
            diagonal = i == j
            ax = fig.add_subplot(
                gs[i + row_offset, j],
                sharex=col_anchor.get(j),
                sharey=None if diagonal else row_anchor.get(i),
            )
            ax.set_gid(f"prior-corner-{i}-{j}")
            col_anchor.setdefault(j, ax)
            if not diagonal:
                row_anchor.setdefault(i, ax)
            if diagonal:
                bins = histogram_edges(values[row_name], log_scale[row_name])
                marginal_histogram(
                    ax,
                    values[row_name],
                    bins,
                    color=PRIOR_COLOR,
                    gid="prior-marginal",
                    filled=True,
                )
                ax.set_title(row_name, fontsize=_LABEL_FONTSIZE, loc="left")
                ax.set_yticks([])
            else:
                plot_density(
                    ax,
                    values[col_name],
                    values[row_name],
                    diverging,
                    (
                        "log" if log_scale[col_name] else "linear",
                        "log" if log_scale[row_name] else "linear",
                    ),
                )
                if log_scale[row_name]:
                    _set_log_axis(ax, "y")
            if log_scale[col_name]:
                _set_log_axis(ax, "x")
            if i == n - 1:
                ax.set_xlabel(col_name)
            else:
                ax.tick_params(which="both", labelbottom=False)
            if j == 0 and i != 0:
                ax.set_ylabel(row_name)
            elif not diagonal:
                ax.tick_params(which="both", labelleft=False)

    traces = fig.add_gridspec(
        grid_n,
        1,
        left=(left + corner_size + trace_gap) / width,
        right=(left + corner_size + trace_gap + trace_width) / width,
        bottom=corner_bottom / height,
        top=(corner_bottom + corner_size) / height,
        hspace=_PANEL_GAP,
    )
    trace_axes = []
    draws = np.arange(num_samples)
    for j, name in enumerate(names):
        ax = fig.add_subplot(
            traces[j + row_offset, 0],
            sharex=trace_axes[0] if trace_axes else None,
            sharey=row_anchor.get(j),
        )
        ax.set_gid(f"prior-trace-{j}")
        trace_axes.append(ax)
        ax.plot(draws, values[name], linewidth=0.55, color=PRIOR_COLOR)
        ax.scatter(
            draws[diverging],
            values[name][diverging],
            s=DIVERGENCE_SIZE,
            marker="x",
            color=DIVERGENCE_COLOR,
            alpha=DIVERGENCE_ALPHA,
            linewidths=DIVERGENCE_LINEWIDTH,
            zorder=3,
            gid="trace-divergences",
        )
        ax.set_ylabel(name)
        if log_scale[name]:
            _set_log_axis(ax, "y")
        ax.set_ylim(col_anchor[j].get_xlim())
        if j == n - 1:
            ax.set_xlabel("draw")
        else:
            ax.tick_params(labelbottom=False)
        if j == 0:
            ax.set_title("Prior Sample Traces", loc="left", fontsize=_LABEL_FONTSIZE)

    table_rows = grid_n // 2
    ax_table = fig.add_subplot(gs[:table_rows, table_rows:grid_n])
    ax_table.set_gid("prior-sampling-table")
    ax_table.axis("off")
    col_labels = ["parameter", "ESS", "split-R̂"]
    cell_text = []
    cell_colors = []
    for name in names:
        ess = diagnostics.ess[name]
        rhat = diagnostics.split_rhat[name]
        cell_text.append([name, f"{ess:.0f}", f"{rhat:.3f}"])
        cell_colors.append(["white", _ess_color(ess), _split_rhat_color(rhat)])
    divergence_fraction = n_divergent / num_samples
    divergence_label = (
        f"divergent: {n_divergent}/{num_samples} ({divergence_fraction:.1%})"
    )

    # Two separate tables, not one merged one: `ax.table` has no
    # native colspan, and `Table._do_cell_alignment` sizes a whole
    # *column* to its widest cell across every row, so widening one
    # row's cell to "merge" columns drags every other row along with
    # it. A standalone single-column table for the divergence summary
    # avoids that -- with one column, it already spans the full width.
    param_rows = len(names) + 1  # header + one row per parameter
    n_table_rows = param_rows + 1  # + 1 for the divergence summary
    param_height = param_rows / n_table_rows
    divergence_height = 1.0 - param_height

    param_table = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        bbox=(0.0, divergence_height, 1.0, param_height),
    )
    divergence_table = ax_table.table(
        cellText=[[divergence_label]],
        cellColours=[[_divergence_color(divergence_fraction)]],
        cellLoc="center",
        bbox=(0.0, 0.0, 1.0, divergence_height),
    )

    for table in (param_table, divergence_table):
        table.auto_set_font_size(False)
        table.set_fontsize(_LABEL_FONTSIZE)

    # Size each column to its own widest cell, once -- NOT via
    # `auto_set_column_width`, which only *registers* the column for
    # continuous resizing on every future draw, overwriting this at
    # the next redraw (e.g. `savefig` itself).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    required_widths = {
        col: max(
            cell.get_required_width(renderer)
            for (_r, c), cell in param_table.get_celld().items()
            if c == col
        )
        for col in range(3)
    }
    # Required widths rarely sum to 1.0 -- normalize rather than leave
    # the rest as dead space.
    total_required = sum(required_widths.values())
    width = max(total_required, divergence_table[0, 0].get_required_width(renderer))
    text_height = max(
        cell.get_text().get_window_extent(renderer).height
        for table in (param_table, divergence_table)
        for cell in table.get_celld().values()
    )
    row_height = 1.8 * text_height / ax_table.get_window_extent(renderer).height
    # Keep the summary block's top-right anchor; fit around measured text.
    param_table._bbox = (
        1 - width,
        1 - param_rows * row_height,
        width,
        param_rows * row_height,
    )
    divergence_table._bbox = (
        1 - width,
        1 - n_table_rows * row_height,
        width,
        row_height,
    )
    for col, required_width in required_widths.items():
        cells = [cell for (_r, c), cell in param_table.get_celld().items() if c == col]
        for cell in cells:
            cell.set_width(required_width / total_required)

    legend_handles = [
        Patch(
            facecolor=PRIOR_COLOR,
            edgecolor=PRIOR_COLOR,
            alpha=0.6,
            label="prior: 68 / 95 / 99.7%",
        ),
        Line2D(
            [],
            [],
            marker="x",
            linestyle="",
            color=DIVERGENCE_COLOR,
            label="divergent draws",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
    )

    fig.suptitle(
        title,
        fontsize=_TITLE_FONTSIZE,
    )

    if proposed_samples is not None:
        from tnt.parameter_generator.diagnostics.overlays import add_proposal_overlay

        add_proposal_overlay(
            fig,
            tuple(names),
            proposed_samples,
            None,
            values,
            new_label="prior samples",
            new_gid="prior-proposal",
        )

    style_axes(fig)
    place_header(fig)
    return fig
