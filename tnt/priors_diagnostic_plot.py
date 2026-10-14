"""A human-reviewed diagnostic plot for a `Prior`'s sampled draw.

Not an automated pass/fail gate -- a threshold can't tell a genuinely
broken chain from a prior that's simply as wide as the science requires.
Draws once with the real run's `num_samples`/`num_warmup` and saves a
figure to review before committing to the full search.

Builds its own `matplotlib.figure.Figure` with an explicit `FigureCanvasAgg`
attached directly to it, rather than `matplotlib.pyplot`'s `figure()`/
`subplots()`/`close()` (backed by its global current-figure-manager state)
or `matplotlib.use("Agg")` (backed by the library's global current-backend
state). Importing this module -- which `tnt.model_iterator` does
unconditionally, whether or not a run ever uses `PriorSampler` -- must not
silently change which backend an importing application's own, unrelated
plotting ends up using.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from jax import Array
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

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
    """True if every value is positive, finite, and spans orders of magnitude.

    Checking only the positive subset (rather than every value) let a
    signed parameter -- e.g. a bar orientation angle, `Uniform(-1, 1)`
    radians -- get log-scaled whenever its positive half alone happened to
    span two-plus orders of magnitude: `np.log10` of the (negative)
    overall minimum is then `NaN`, so every histogram/axis bin edge built
    from it is `NaN` too, silently dropping every draw from the rendered
    figure rather than raising. Requiring the *whole* array positive and
    finite first means a log axis is never chosen for data it can't
    actually represent.
    """
    if len(values) < 2 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        return False
    return bool(np.max(values) / np.min(values) >= 100)


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
    target_accept_prob: float = 0.8,
) -> None:
    """Draw a pilot batch from `prior` and save a diagnostic figure to `path`.

    This is a pilot: `key` is independent of whatever key the real search
    actually draws its own batches with (`PriorSampler` folds in each
    iteration's cumulative count, `jax.random.fold_in(PRNGKey(seed),
    n_iterations())`, which `fold_in`'s own definition guarantees never
    equals `PRNGKey(seed)` -- not even for iteration `0`). A useful
    pre-flight sanity check on the composed model as a whole, but not a
    stand-in for any specific real batch's own diagnostics -- those are
    logged by `PriorSampler` itself, per iteration, at `INFO` level.

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
    not a convergence diagnostic (see `PriorDiagnostics`). `target_accept_prob`
    is then unused -- it only affects `NUTS`'s step-size adaptation.
    """
    fig = _build_diagnostics_figure(
        prior, key, num_samples, num_warmup, target_accept_prob
    )
    fig.savefig(path, dpi=150)


def _build_diagnostics_figure(
    prior: Prior,
    key: Array,
    num_samples: int,
    num_warmup: int | None,
    target_accept_prob: float,
) -> Figure:
    """The actual figure-building logic behind `plot_prior_diagnostics`.

    Split out so a test can inspect the built `Figure` directly (e.g. via
    `get_window_extent` on its text artists, to check a layout fix actually
    keeps text on-canvas and non-overlapping) without round-tripping
    through a saved file.
    """
    samples, diagnostics = prior.sample_with_diagnostics(
        key, num_samples, num_warmup, target_accept_prob
    )

    names = sorted(samples)
    n = len(names)
    if n == 0:
        # Every potential parameter is fixed (or covered by a default
        # prior's own `known`-resolved siblings) -- a supported,
        # degenerate `PriorSampler` setup with nothing left to sample, not
        # an error. `fig.add_gridspec(0, ...)` rejects zero rows, so there
        # is no corner plot to build; return a minimal figure saying so
        # instead, keeping this function's "produces a figure" contract
        # even here.
        fig = Figure(figsize=(4.0, 2.0), dpi=150)
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
            wrap=True,
        )
        return fig

    values = {name: np.asarray(samples[name]) for name in names}
    log_scale = {name: _use_log_scale(values[name]) for name in names}
    diverging = np.asarray(diagnostics.diverging)
    n_divergent = int(np.sum(diverging))
    # A narrow spacer column (index `n`) separates the corner grid from the
    # trace column (index `n + 1`) by more than the usual `_PANEL_GAP`.
    spacer_ratio = 0.3
    grid_width = _PANEL_SIZE * (n + spacer_ratio + _TRACE_WIDTH_RATIO)
    # A single parameter has no off-diagonal table block (see `table_rows`
    # below) to hold its ESS/split-R-hat/divergence summary, so that text
    # has to live in the header instead -- reserve genuinely extra height
    # for it there (not just the sliver the title alone needs), or it
    # overflows the canvas and collides with the title. The narrower
    # one-panel canvas also leaves the centered title too little clearance
    # from the fixed-position legend at its default size, so reserve extra
    # width too -- as blank left margin, not by stretching the one panel
    # wider than `_PANEL_SIZE`. (Re-review R3 follow-up: reproduced both
    # the summary clipping and this title/legend collision in the saved
    # figure before this.)
    grid_height = _PANEL_SIZE * n
    summary_height = _PANEL_SIZE * 0.6 if n == 1 else 0.0
    legend_clearance = _PANEL_SIZE * 1.3 if n == 1 else 0.0
    fig_height = grid_height + summary_height
    fig_width = grid_width + legend_clearance
    fig = Figure(
        figsize=(fig_width, fig_height),
        # Match `savefig`'s dpi: the table's column widths are measured via
        # a draw at the figure's dpi, and font hinting rounds glyph widths
        # differently at different dpis.
        dpi=150,
    )
    FigureCanvasAgg(fig)
    # `top`/`bottom`/`left`/`right` are fractions of `grid_height`/
    # `grid_width` -- rescale to the same fractions of the (possibly
    # taller/wider) full figure, so the grid itself keeps the same
    # absolute size and position relative to the figure's *right* edge,
    # with only a reserved header above it and a blank margin to its left
    # (where the legend already sits) growing.
    top = 0.84 * grid_height / fig_height
    bottom = 0.10 * grid_height / fig_height
    left = (legend_clearance + 0.07 * grid_width) / fig_width
    right = (legend_clearance + 0.98 * grid_width) / fig_width
    gs = fig.add_gridspec(
        n,
        n + 2,
        width_ratios=[1.0] * n + [spacer_ratio, _TRACE_WIDTH_RATIO],
        left=left,
        right=right,
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
    col_anchor: dict[int, Axes] = {}
    row_anchor: dict[int, Axes] = {}

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
    trace_axes: list[Axes] = []
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

    elif n == 1:
        # No axis-aligned off-diagonal block exists for a single parameter
        # (`table_rows == n // 2 == 0`) to put the usual table in -- a
        # plain text summary centered over the trace column, in the extra
        # header space `summary_height` reserved above the grid (between
        # it and the title) specifically for this case, instead of no
        # numerical summary at all.
        name = names[0]
        divergence_fraction = n_divergent / num_samples
        trace_position = trace_axes[0].get_position()
        header_top = 1.0 - 0.3 * (1.0 - top)  # leaves room below the title
        fig.text(
            trace_position.x0 + trace_position.width / 2,
            (top + header_top) / 2,
            f"{name}\nESS: {diagnostics.ess[name]:.0f}\n"
            f"split-R̂: {diagnostics.split_rhat[name]:.3f}\n"
            f"divergent: {n_divergent}/{num_samples} ({divergence_fraction:.1%})",
            ha="center",
            va="center",
            fontsize=_LABEL_FONTSIZE,
        )

    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color="C0", label="non-divergent"),
        Line2D([], [], marker="o", linestyle="", color="red", label="divergent"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=_LABEL_FONTSIZE,
        frameon=False,
    )

    fig.suptitle(
        "Prior Sampling Pilot Diagnostics (independent of real search batches)",
        fontsize=_TITLE_FONTSIZE,
    )

    return fig
