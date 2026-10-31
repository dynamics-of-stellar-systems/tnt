"""Shared diagnostic colors and probability-mass contour levels."""

from matplotlib.figure import Figure

PANEL_SIZE = 3.0  # inches; fonts scale with panels, not parameter count
FONT_SCALE = PANEL_SIZE / 2.6
LABEL_FONTSIZE = 11 * FONT_SCALE
TICK_FONTSIZE = 10 * FONT_SCALE
LEGEND_FONTSIZE = 10 * FONT_SCALE
TITLE_FONTSIZE = 18 * FONT_SCALE
SECTION_FONTSIZE = 13 * FONT_SCALE


def place_header(fig: Figure) -> None:
    """Anchor the legend immediately below the fitted sampling table."""
    height = fig.get_figheight()
    if fig._suptitle is not None:
        fig._suptitle.set_y(1 - 0.12 / height)
    table_axes = next(
        (ax for ax in fig.axes if ax.get_gid() == "prior-sampling-table"), None
    )
    if table_axes is None:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = [
        table.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        for table in table_axes.tables
    ]
    lower_edge = min(box.y0 for box in bounds)
    center = (min(box.x0 for box in bounds) + max(box.x1 for box in bounds)) / 2
    for legend in fig.legends:
        legend.set_loc("upper center")
        legend.set_frame_on(True)
        legend.get_frame().set_edgecolor("#AAB2B9")
        legend.get_frame().set_linewidth(0.7)
        legend.get_frame().set_alpha(1.0)
        legend.set_bbox_to_anchor((center, lower_edge - 0.12 / height))


def style_axes(fig: Figure) -> None:
    """Size axis text locally, without changing Matplotlib global settings."""
    for ax in fig.axes:
        ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE)
        ax.xaxis.label.set_fontsize(LABEL_FONTSIZE)
        ax.yaxis.label.set_fontsize(LABEL_FONTSIZE)
        ax.xaxis.get_offset_text().set_fontsize(TICK_FONTSIZE)
        ax.yaxis.get_offset_text().set_fontsize(TICK_FONTSIZE)
        ax.title.set_fontsize(LABEL_FONTSIZE)


PRIOR_COLOR = "#74879B"
DIVERGENCE_COLOR = "#7350A2"
DIVERGENCE_SIZE = 16
DIVERGENCE_LINEWIDTH = 0.9
DIVERGENCE_ALPHA = 0.9
NEW_POINT_COLOR = "#D55E00"
FIXED_POINT_COLOR = "#343C46"
CONTOUR_MASSES = (0.68, 0.95, 0.997)
