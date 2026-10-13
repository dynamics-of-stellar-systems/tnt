"""Unit tests for `tnt.priors_diagnostic_plot`."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from unxt import Quantity

from tnt.mge import LightMGE
from tnt.priors import Prior
from tnt.priors_diagnostic_plot import plot_prior_diagnostics

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
