"""A human-reviewed diagnostic plot for a `Prior`'s sampled draw.

Not an automated pass/fail gate -- a threshold can't tell a genuinely
broken chain from a prior that's simply as wide as the science requires.
Draws once with the real run's `num_samples`/`num_warmup` and saves a
figure to review before committing to the full search.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from jax import Array

from tnt.priors import Prior

_PANEL_SIZE = 2.6  # inches, one corner-plot row/column
_TRACE_WIDTH_RATIO = 2  # the trace-plot column is this many panel-widths wide
# wspace/hspace between corner-plot panels; large enough that a diagonal
# panel's title (drawn just above its axes) doesn't collide with the row above.
_PANEL_GAP = 0.15

# Font sizes scale with `_PANEL_SIZE` rather than being fixed constants, so
# the whole figure rescales consistently if `_PANEL_SIZE` changes.
_LABEL_FONTSIZE = 9.0 / 2.6 * _PANEL_SIZE
_TITLE_FONTSIZE = 13.0 / 2.6 * _PANEL_SIZE
_TABLE_FONTSIZE_MAX = _LABEL_FONTSIZE * 1.8


def _use_log_scale(values: np.ndarray) -> bool:
    """True if positive and spanning orders of magnitude (reads better logged)."""
    positive = values[values > 0]
    if len(positive) < 2:
        return False
    return bool(np.max(positive) / np.min(positive) >= 100)


def _ess_color(ess: float) -> str:
    """Green/amber/red for ESS `> 100` / `30-100` / `< 30`."""
    if ess > 100:
        return "lightgreen"
    if ess >= 30:
        return "moccasin"
    return "lightcoral"


def _split_rhat_color(rhat: float) -> str:
    """Green/amber/red for split-R-hat `< 1.01` / `1.01-1.05` / `> 1.05`
    (Vehtari et al. 2021's recommended convergence threshold).
    """
    if rhat < 1.01:
        return "lightgreen"
    if rhat <= 1.05:
        return "moccasin"
    return "lightcoral"


def _divergence_color(fraction: float) -> str:
    """Green/amber/red for a divergence fraction of `< 1%` / `1-10%` / `>= 10%`."""
    if fraction < 0.01:
        return "lightgreen"
    if fraction < 0.1:
        return "moccasin"
    return "lightcoral"


def plot_prior_diagnostics(
    prior: Prior,
    path: str | Path,
    key: Array,
    num_samples: int,
    num_warmup: int | None = None,
) -> None:
    """Draw once from `prior` and save a diagnostic figure to `path`.

    A corner plot over every non-fixed, prior-bearing parameter: scatter
    panels (divergent draws in red) below the diagonal, marginal
    histograms on it, log axes where a parameter spans orders of
    magnitude. An ESS/split-R-hat table (`_ess_color`/`_split_rhat_color`,
    plus a divergence-fraction row via `_divergence_color`) fills the
    blank block above the diagonal; a per-parameter trace plot runs down
    an extra column on the right.

    When `prior.has_factors()` is `False`, this is an unconditioned
    `numpyro.infer.Predictive` draw, not MCMC -- `diverging` is trivially
    all `False`, and ESS/split-R-hat are a sanity check on independence,
    not a convergence diagnostic (see `PriorDiagnostics`).
    """
    samples, diagnostics = prior.sample_with_diagnostics(key, num_samples, num_warmup)

    names = sorted(samples)
    values = {name: np.asarray(samples[name]) for name in names}
    log_scale = {name: _use_log_scale(values[name]) for name in names}
    diverging = np.asarray(diagnostics.diverging)
    n_divergent = int(np.sum(diverging))

    n = len(names)
    # A narrow spacer column (index `n`) separates the corner grid from the
    # trace column (index `n + 1`) by more than the usual `_PANEL_GAP`.
    spacer_ratio = 0.3
    fig_width = _PANEL_SIZE * (n + spacer_ratio + _TRACE_WIDTH_RATIO)
    fig = plt.figure(
        figsize=(fig_width, _PANEL_SIZE * n),
        # Match `savefig`'s dpi: the table's column widths are measured via
        # a draw at the figure's dpi, and font hinting rounds glyph widths
        # differently at different dpis.
        dpi=150,
    )
    top, bottom = 0.84, 0.10
    gs = fig.add_gridspec(
        n,
        n + 2,
        width_ratios=[1.0] * n + [spacer_ratio, _TRACE_WIDTH_RATIO],
        left=0.07,
        right=0.98,
        bottom=bottom,
        top=top,
        wspace=_PANEL_GAP,
        hspace=_PANEL_GAP,
    )
    trace_col = n + 1

    # The largest axis-aligned block strictly above the diagonal (`i < j`
    # throughout) is rows `[0, a)` x cols `[a, n)`, `a = n // 2` -- the
    # ESS/split-R-hat table goes there. `n == 1` has no such block, so the
    # table is skipped rather than contrived for a one-parameter prior.
    table_rows = n // 2

    def in_table_block(i: int, j: int) -> bool:
        return table_rows > 0 and i < table_rows and j >= table_rows

    # Shared axes (same x-range per column, same y-range per scatter row),
    # so only the edge panel needs its tick *labels* (every panel keeps its
    # own tick marks, just not the numbers).
    col_anchor: dict[int, plt.Axes] = {}
    row_anchor: dict[int, plt.Axes] = {}

    for i in range(n):
        for j in range(n):
            if in_table_block(i, j):
                continue
            row_name, col_name = names[i], names[j]
            is_diagonal = i == j
            ax = fig.add_subplot(
                gs[i, j],
                sharex=col_anchor.get(j),
                sharey=None if is_diagonal else row_anchor.get(i),
            )
            col_anchor.setdefault(j, ax)
            if not is_diagonal:
                row_anchor.setdefault(i, ax)
            if i < j:
                ax.axis("off")
                continue
            if is_diagonal:
                x = values[row_name]
                bins = (
                    np.logspace(np.log10(x.min()), np.log10(x.max()), 28)
                    if log_scale[row_name]
                    else 28
                )
                ax.hist(x, bins=bins)
                ax.set_title(row_name, fontsize=_LABEL_FONTSIZE, loc="left")
                ax.set_yticks([])  # a marginal's count scale isn't worth an axis
                if log_scale[row_name]:
                    ax.set_xscale("log")
            else:
                x, y = values[col_name], values[row_name]
                ax.scatter(x[~diverging], y[~diverging], s=10, alpha=0.5)
                ax.scatter(x[diverging], y[diverging], s=16, color="red")
                if log_scale[col_name]:
                    ax.set_xscale("log")
                if log_scale[row_name]:
                    ax.set_yscale("log")
            if i == n - 1:
                ax.set_xlabel(col_name, fontsize=_LABEL_FONTSIZE)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i != 0:
                ax.set_ylabel(row_name, fontsize=_LABEL_FONTSIZE)
            elif not is_diagonal:
                ax.tick_params(labelleft=False)

    # Per-parameter trace plot, one per row, in the extra right-hand
    # column -- shared x-axis, tick labels only on the bottom-most panel.
    trace_axes: list[plt.Axes] = []
    for i, row_name in enumerate(names):
        ax = fig.add_subplot(
            gs[i, trace_col], sharex=trace_axes[0] if trace_axes else None
        )
        trace_axes.append(ax)
        x = values[row_name]
        ax.plot(x, linewidth=0.6)
        ax.scatter(
            np.flatnonzero(diverging), x[diverging], s=10, color="red", zorder=3
        )
        ax.set_ylabel(row_name, fontsize=_LABEL_FONTSIZE)
        if log_scale[row_name]:
            ax.set_yscale("log")
        if i == n - 1:
            ax.set_xlabel("draw", fontsize=_LABEL_FONTSIZE)
        else:
            ax.tick_params(labelbottom=False)
        if i == 0:
            ax.set_title("trace", fontsize=_LABEL_FONTSIZE, loc="left")

    if table_rows > 0:
        ax_table = fig.add_subplot(gs[0:table_rows, table_rows:n])
        ax_table.axis("off")
        # Small bottom-left margin; top and right stay flush with the
        # corner grid's own top-right corner.
        table_margin = 0.015
        pos = ax_table.get_position()
        ax_table.set_position(
            [
                pos.x0 + table_margin,
                pos.y0 + table_margin,
                pos.width - table_margin,
                pos.height - table_margin,
            ]
        )
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

        # Scale the font to whichever of the block's two dimensions is
        # tighter. Reads `ax_table`'s own just-shrunk position rather than
        # a nominal `_PANEL_SIZE`-based formula -- `_PANEL_GAP`/
        # `table_margin` both make the real block smaller than that.
        fig_width_inches, fig_height_inches = fig.get_size_inches()
        table_position = ax_table.get_position()
        block_height_inches = table_position.height * fig_height_inches
        block_width_inches = table_position.width * fig_width_inches
        height_bound = block_height_inches / n_table_rows * 72 * 0.45
        longest_text = max(
            len(s)
            for row in (col_labels, *cell_text, [divergence_label])
            for s in row
        )
        width_bound = block_width_inches * 72 / (longest_text * 0.62)
        fontsize = np.clip(min(height_bound, width_bound), 7, _TABLE_FONTSIZE_MAX)
        for table in (param_table, divergence_table):
            table.auto_set_font_size(False)
            table.set_fontsize(fontsize)

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
        for col, required_width in required_widths.items():
            cells = [
                cell
                for (_r, c), cell in param_table.get_celld().items()
                if c == col
            ]
            for cell in cells:
                cell.set_width(required_width / total_required)

    legend_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="C0", label="non-divergent"),
        plt.Line2D([], [], marker="o", linestyle="", color="red", label="divergent"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=_LABEL_FONTSIZE,
        frameon=False,
    )

    fig.suptitle("Diagnostic Plot for Prior Sampling", fontsize=_TITLE_FONTSIZE)

    fig.savefig(path, dpi=150)
    plt.close(fig)
