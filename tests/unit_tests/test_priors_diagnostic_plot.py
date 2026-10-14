"""Unit tests for `tnt.priors_diagnostic_plot`."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from unxt import Quantity

from tnt.mge import LightMGE
from tnt.priors import Prior
from tnt.priors_diagnostic_plot import (
    _build_diagnostics_figure,
    _use_log_scale,
    plot_prior_diagnostics,
)

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


def test_plot_prior_diagnostics_writes_a_figure_without_factors(tmp_path: Path) -> None:
    """A pure prior-predictive draw still gets a diagnostic figure -- ESS/
    split-R̂ there are a sanity check on independence, not a convergence
    check, but still worth drawing (see `PriorDiagnostics`).
    """
    prior = Prior(_POTENTIAL_SETTINGS, {}, _MGES)
    path = tmp_path / "prior_diagnostics.png"

    plot_prior_diagnostics(prior, path, jax.random.PRNGKey(0), num_samples=50)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_prior_diagnostics_writes_a_figure_with_factors(tmp_path: Path) -> None:
    prior = Prior(
        _POTENTIAL_SETTINGS, {"dh_mass_fraction": _mass_fraction_plugin}, _MGES
    )
    path = tmp_path / "prior_diagnostics.png"

    plot_prior_diagnostics(
        prior, path, jax.random.PRNGKey(0), num_samples=50, num_warmup=50
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_use_log_scale_rejects_signed_data_even_with_a_wide_positive_subset() -> None:
    # Re-review R3: checking only the positive subset let a signed
    # parameter (e.g. a bar orientation angle, Uniform(-1, 1) radians) get
    # log-scaled whenever that subset alone spanned >= 2 orders of
    # magnitude -- np.log10 of the (negative) true minimum is then NaN,
    # silently dropping every value from the rendered figure.
    values = np.concatenate([np.array([-0.9, -0.5]), np.linspace(0.001, 1.0, 50)])
    assert not _use_log_scale(values)


def test_use_log_scale_accepts_a_wide_strictly_positive_range() -> None:
    values = np.array([1.0e-3, 1.0, 1.0e2])
    assert _use_log_scale(values)


def test_use_log_scale_rejects_a_narrow_positive_range() -> None:
    values = np.array([1.0, 2.0, 5.0])
    assert not _use_log_scale(values)


def test_use_log_scale_rejects_nonfinite_values() -> None:
    values = np.array([1.0e-3, 1.0, 1.0e2, np.nan])
    assert not _use_log_scale(values)


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

    plot_prior_diagnostics(prior, path, jax.random.PRNGKey(0), num_samples=50)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_prior_diagnostics_includes_a_summary_for_one_free_parameter(
    tmp_path: Path,
) -> None:
    # Re-review R2/R3: a single free parameter has no off-diagonal grid
    # block (table_rows = n // 2 == 0) for the usual ESS/split-R-hat table,
    # so it was skipped entirely rather than shown some other way.
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

    plot_prior_diagnostics(prior, path, jax.random.PRNGKey(0), num_samples=50)

    assert path.exists()
    assert path.stat().st_size > 0


def test_one_parameter_summary_is_fully_visible_and_does_not_overlap() -> None:
    # Re-review R3 follow-up: the one-parameter summary text previously
    # overflowed the canvas (clipped at the top) and collided with the
    # figure title, which itself collided with the legend -- reproduced
    # directly (and visually confirmed in the saved PNG) before this fix.
    # Checking only that a PNG exists and has nonzero size (the preceding
    # test, and the original regression this finding was raised against)
    # does not catch either problem -- inspect the actual drawn text
    # geometry via matplotlib's own `get_window_extent` instead.
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

    fig = _build_diagnostics_figure(prior, jax.random.PRNGKey(0), 1000, None, 0.8)
    fig.canvas.draw()

    summary = next(t for t in fig.texts if "ESS" in t.get_text())
    boxes = {
        "summary": summary.get_window_extent(),
        "title": fig._suptitle.get_window_extent(),
        "legend": fig.legends[0].get_window_extent(),
    }

    for label, bbox in boxes.items():
        assert fig.bbox.contains(bbox.x0, bbox.y0), f"{label} clipped (lower-left)"
        assert fig.bbox.contains(bbox.x1, bbox.y1), f"{label} clipped (upper-right)"

    assert not boxes["summary"].overlaps(boxes["title"])
    assert not boxes["title"].overlaps(boxes["legend"])
    assert not boxes["summary"].overlaps(boxes["legend"])
