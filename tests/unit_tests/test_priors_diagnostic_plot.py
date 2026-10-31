"""Unit tests for `tnt.parameter_generator.diagnostics.prior`."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from unxt import Quantity

from tnt.mge import LightMGE
from tnt.parameter_generator.diagnostics.prior import (
    _use_log_scale,
    build_prior_diagnostics_figure,
    plot_prior_diagnostics,
)
from tnt.parameter_generator.samples import PriorSampleBatch
from tnt.priors import Prior, PriorDiagnostics

_POTENTIAL_SETTINGS = {
    "dh": {
        "type": "NFWPotential",
        "parameters": {
            "m": {
                "unit": "Msun",
                "fixed": False,
                "prior": {"distribution": "LogUniform", "args": [1.0e10, 1.0e13]},
            },
        },
    },
    "stars": {
        "type": "triaxial_light_mge",
        "parameters": {
            "ml": {
                "unit": "Msun / Lsun",
                "fixed": False,
                "prior": {"distribution": "Uniform", "args": [1.0, 9.0]},
            },
        },
    },
}
_STARS_MGE = LightMGE(
    I=Quantity(jnp.array([1.0e10 / (2 * jnp.pi)]), "Lsun / rad2"),
    sigma=Quantity(jnp.array([1.0]), "rad"),
    q=Quantity(jnp.array([1.0]), ""),
    PA_twist=Quantity(jnp.array([0.0]), "rad"),
    major_axis_pa=Quantity(0.0, "deg"),
)
_MGES = {"stars": _STARS_MGE}


def _mass_fraction_plugin(context) -> None:
    stars = context.mges["stars"]
    candidate = context.candidate
    total_light = jnp.sum(
        (2 * jnp.pi * stars.I * stars.sigma**2 * stars.q).ustrip("Lsun")
    )
    f = candidate["dh"]["m"] / (total_light * candidate["stars"]["ml"])
    numpyro.factor("dh_mass_fraction", dist.Normal(0.5, 0.05).log_prob(f))


def _batch(prior, num_samples=50, num_warmup=None):
    return PriorSampleBatch(
        *prior.sample_with_diagnostics(jax.random.PRNGKey(0), num_samples, num_warmup)
    )


def test_plot_prior_diagnostics_writes_a_figure_without_factors(tmp_path: Path) -> None:
    """A pure prior-predictive draw still gets a diagnostic figure -- ESS/
    split-R̂ there are a sanity check on independence, not a convergence
    check, but still worth drawing (see `PriorDiagnostics`).
    """
    prior = Prior(_POTENTIAL_SETTINGS, {}, _MGES)
    path = tmp_path / "prior_diagnostics.png"

    plot_prior_diagnostics(_batch(prior), path, iteration=0)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_prior_diagnostics_writes_a_figure_with_factors(tmp_path: Path) -> None:
    prior = Prior(
        _POTENTIAL_SETTINGS, {"dh_mass_fraction": _mass_fraction_plugin}, _MGES
    )
    path = tmp_path / "prior_diagnostics.png"

    plot_prior_diagnostics(_batch(prior, num_warmup=50), path, iteration=0)

    assert path.exists()
    assert path.stat().st_size > 0


@pytest.mark.parametrize(
    ("distribution", "args", "scale"),
    [
        ("LogUniform", [1.0, 1.001], "log"),
        ("LogNormal", [0.0, 0.001], "log"),
        ("Uniform", [1e-8, 1e8], "linear"),
        ("Normal", [0.0, 1.0], "linear"),
    ],
)
def test_axis_scale_follows_sampling_distribution_not_range(distribution, args, scale):
    prior = Prior(
        {
            "halo": {
                "type": "NFWPotential",
                "parameters": {
                    "m": {
                        "fixed": False,
                        "unit": "Msun",
                        "prior": {"distribution": distribution, "args": args},
                    },
                },
            }
        },
        {},
        {},
    )
    batch = _batch(prior, 100)
    assert batch.diagnostics.sample_distributions == {"halo.m": distribution}
    figure = build_prior_diagnostics_figure(batch)
    marginal = next(ax for ax in figure.axes if ax.get_subplotspec().colspan.start == 0)
    assert marginal.get_xscale() == scale
    assert _use_log_scale(distribution) == (scale == "log")


def test_contours_enclose_requested_probability_mass_on_independent_samples():
    rng = np.random.default_rng(72)
    reference = rng.normal(size=(30000, 2))
    reference[:, 1] += 0.6 * reference[:, 0]
    batch = PriorSampleBatch(
        {"x": reference[:, 0], "y": reference[:, 1]},
        PriorDiagnostics(
            np.zeros(len(reference), dtype=bool),
            {"x": 30000.0, "y": 30000.0},
            {"x": 1.0, "y": 1.0},
        ),
    )
    figure = build_prior_diagnostics_figure(batch)
    contour = next(
        artist
        for ax in figure.axes
        for artist in ax.collections
        if artist.get_gid() == "prior-density-contours"
    )
    held_out = rng.normal(size=(20000, 2))
    held_out[:, 1] += 0.6 * held_out[:, 0]
    fractions = [
        np.mean(path.contains_points(held_out)) for path in contour.get_paths()
    ]
    np.testing.assert_allclose(fractions, [0.997, 0.95, 0.68], atol=0.03)
    assert len(contour.levels) == 3


def test_divergences_use_a_muted_overlay_without_dense_prior_scatter():
    from matplotlib.colors import to_rgba

    from tnt.parameter_generator.diagnostics.style import DIVERGENCE_COLOR

    rng = np.random.default_rng(9)
    values = rng.normal(size=(300, 2))
    divergent = np.arange(300) % 50 == 0
    batch = PriorSampleBatch(
        {"p": values[:, 0], "q": values[:, 1]},
        PriorDiagnostics(divergent, {"p": 300.0, "q": 300.0}, {"p": 1.0, "q": 1.0}),
    )
    figure = build_prior_diagnostics_figure(batch)
    overlay = next(
        artist
        for ax in figure.axes
        for artist in ax.collections
        if artist.get_gid() == "prior-divergences"
    )
    np.testing.assert_array_equal(overlay.get_offsets(), values[divergent])
    np.testing.assert_allclose(
        overlay.get_edgecolors()[0][:3], to_rgba(DIVERGENCE_COLOR)[:3]
    )
    assert 10 <= overlay.get_sizes()[0] <= 20
    assert overlay.get_alpha() >= 0.8
    assert not overlay.axes.lines  # contour backgrounds have no sample scatter
    assert all("outside" not in text.get_text() for text in figure.legends[0].texts)


def test_plot_prior_diagnostics_handles_zero_free_parameters(tmp_path: Path) -> None:
    # Re-review R3: every parameter fixed is a supported, degenerate
    # PriorSampler setup -- `Prior` correctly returns no stochastic sites,
    # but the plot used to assume at least one and crash building a
    # zero-row gridspec.
    settings = {
        "dh": {
            "type": "NFWPotential",
            "parameters": {
                "m": {"unit": "Msun", "fixed": True, "value": 1.0e12},
            },
        },
    }
    prior = Prior(settings, {}, {})
    path = tmp_path / "prior_diagnostics.png"

    plot_prior_diagnostics(_batch(prior), path, iteration=0)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_prior_diagnostics_includes_a_summary_for_one_free_parameter(
    tmp_path: Path,
) -> None:
    # A single free parameter still needs sampling statistics in the summary area.
    settings = {
        "dh": {
            "type": "NFWPotential",
            "parameters": {
                "m": {
                    "unit": "Msun",
                    "fixed": False,
                    "prior": {"distribution": "LogUniform", "args": [1.0e10, 1.0e13]},
                },
                "r_s": {"unit": "kpc", "fixed": True, "value": 15.0},
            },
        },
    }
    prior = Prior(settings, {}, {})
    path = tmp_path / "prior_diagnostics.png"

    plot_prior_diagnostics(_batch(prior), path, iteration=0)

    assert path.exists()
    assert path.stat().st_size > 0


def test_one_parameter_summary_is_fully_visible_and_does_not_overlap() -> None:
    # Check rendered geometry: a nonempty PNG alone cannot detect clipping.
    settings = {
        "bar": {
            "type": "LongMuraliBarPotential",
            "parameters": {
                "m_tot": {"unit": "Msun", "fixed": True, "value": 1.0e10},
                "a": {"unit": "kpc", "fixed": True, "value": 5.0},
                "b": {"unit": "kpc", "fixed": True, "value": 1.0},
                "c": {"unit": "kpc", "fixed": True, "value": 0.5},
                "alpha": {
                    "unit": "rad",
                    "fixed": False,
                    "prior": {"distribution": "Uniform", "args": [-1.0, 1.0]},
                },
            },
        },
    }
    prior = Prior(settings, {}, {})

    fig = build_prior_diagnostics_figure(_batch(prior, 1000), iteration=3)
    fig.canvas.draw()

    from matplotlib.transforms import Bbox

    table_axes = next(ax for ax in fig.axes if ax.get_gid() == "prior-sampling-table")
    boxes = {
        "summary": Bbox.union(
            [table.get_window_extent() for table in table_axes.tables]
        ),
        "title": fig._suptitle.get_window_extent(),
        "legend": fig.legends[0].get_window_extent(),
    }

    for label, bbox in boxes.items():
        assert fig.bbox.contains(bbox.x0, bbox.y0), f"{label} clipped (lower-left)"
        assert fig.bbox.contains(bbox.x1, bbox.y1), f"{label} clipped (upper-right)"

    assert not boxes["summary"].overlaps(boxes["title"])
    assert not boxes["title"].overlaps(boxes["legend"])
    assert not boxes["summary"].overlaps(boxes["legend"])


def test_prior_proposals_overlay_actual_coordinates_and_normalized_marginals():
    prior = Prior(_POTENTIAL_SETTINGS, {}, _MGES)
    batch = _batch(prior, num_samples=120)
    proposed = {
        name: np.asarray(values)[[2, 8, 17]] for name, values in batch.samples.items()
    }
    fig = build_prior_diagnostics_figure(batch, proposed_samples=proposed)
    markers = [
        artist
        for ax in fig.axes
        for artist in ax.collections
        if artist.get_gid() == "prior-proposal-points-scatter"
    ]
    assert len(markers) == 1
    names = sorted(proposed)
    np.testing.assert_allclose(
        markers[0].get_offsets(), np.column_stack([proposed[name] for name in names])
    )
    histograms = [
        artist
        for ax in fig.axes
        for artist in ax.patches
        if artist.get_gid() == "prior-proposal-marginal"
    ]
    assert len(histograms) == 2
    for histogram in histograms:
        assert histogram.get_data().values.sum() == pytest.approx(1.0)
    assert any(
        "prior samples (m=3)" == text.get_text()
        for text in fig.legends[0].get_texts()
    )
    from matplotlib.patches import Patch

    prior_key = fig.legends[0].legend_handles[0]
    assert isinstance(prior_key, Patch)
    assert prior_key.get_fill()
    assert prior_key.get_facecolor()[-1] > 0


def test_sampling_table_fits_text_and_stays_anchored_at_top_right():
    fig = build_prior_diagnostics_figure(_batch(Prior(_POTENTIAL_SETTINGS, {}, _MGES)))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tables = [table for ax in fig.axes for table in ax.tables]
    assert len(tables) == 2
    parameter_table = tables[0]
    ax = parameter_table.axes
    bounds = parameter_table.get_window_extent(renderer)
    assert bounds.x1 == pytest.approx(ax.get_window_extent(renderer).x1)
    assert bounds.y1 == pytest.approx(ax.get_window_extent(renderer).y1)
    assert bounds.x0 >= 0 and bounds.x1 <= fig.bbox.width
    for table in tables:
        for cell in table.get_celld().values():
            cell_bounds = cell.get_window_extent(renderer)
            text_bounds = cell.get_text().get_window_extent(renderer)
            assert cell_bounds.width >= text_bounds.width
            assert cell_bounds.height >= text_bounds.height
    assert fig._suptitle.get_text().startswith("PriorSampler")
    assert not fig._suptitle.get_window_extent(renderer).overlaps(
        fig.legends[0].get_window_extent(renderer)
    )
    from tnt.parameter_generator.diagnostics.style import TITLE_FONTSIZE

    assert fig._suptitle.get_fontsize() == TITLE_FONTSIZE


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_horizontal_traces_and_boxed_summary_in_upper_triangle(n):
    rng = np.random.default_rng(41)
    samples = {f"p{i}": rng.normal(size=120) for i in range(n)}
    batch = PriorSampleBatch(
        samples,
        PriorDiagnostics(
            np.arange(120) % 40 == 0,
            dict.fromkeys(samples, 120.0),
            dict.fromkeys(samples, 1.0),
        ),
    )
    fig = build_prior_diagnostics_figure(batch)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    corners = [
        ax for ax in fig.axes if (ax.get_gid() or "").startswith("prior-corner-")
    ]
    traces = [ax for ax in fig.axes if (ax.get_gid() or "").startswith("prior-trace-")]
    assert len(traces) == n
    for i, trace in enumerate(traces):
        np.testing.assert_array_equal(trace.lines[0].get_ydata(), samples[f"p{i}"])
        np.testing.assert_array_equal(trace.lines[0].get_xdata(), np.arange(120))
        assert not trace.xaxis_inverted()
        assert trace.get_position().x0 > max(ax.get_position().x1 for ax in corners)
    table_ax = next(ax for ax in fig.axes if ax.get_gid() == "prior-sampling-table")
    corner_right = max(ax.get_position().x1 for ax in corners)
    assert table_ax.get_position().x1 >= corner_right - 1e-12
    assert table_ax.get_subplotspec().rowspan.start == 0
    assert table_ax.get_subplotspec().colspan.start > 0
    assert fig.legends[0].get_frame_on()
    bottom = min(table.get_window_extent(renderer).y0 for table in table_ax.tables)
    legend = fig.legends[0].get_window_extent(renderer)
    assert legend.y1 < bottom
    table_bounds = [table.get_window_extent(renderer) for table in table_ax.tables]
    table_center = (
        min(box.x0 for box in table_bounds) + max(box.x1 for box in table_bounds)
    ) / 2
    assert (legend.x0 + legend.x1) / 2 == pytest.approx(table_center)
    assert legend.x0 >= 0 and legend.x1 <= fig.bbox.width
    assert legend.y0 > min(ax.get_window_extent(renderer).y0 for ax in corners)
