"""Unit tests for `tnt.priors`."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from unxt import Quantity

from tnt.mge import LightMGE
from tnt.priors import Prior, load_prior_plugin, load_prior_plugins

_POTENTIAL_SETTINGS = {
    "dh": {
        "type": "NFWPotential",
        "parameterization": "concentration_m200",
        "include": True,
        "parameters": {
            "c": {
                "fixed": False,
                "prior": {"distribution": "Uniform", "args": [0.5, 5.0]},
            },
            "M_200": {
                "unit": "Msun",
                "fixed": False,
                "prior": {"distribution": "LogUniform", "args": [1.0e9, 1.0e11]},
            },
        },
    },
    "stars": {
        "type": "triaxial_light_mge",
        "include": True,
        "parameters": {
            "ml": {
                "unit": "Msun / Lsun",
                "fixed": False,
                "prior": {"distribution": "Uniform", "args": [1.0, 10.0]},
            },
        },
    },
}


# A single circular (sigma=1 rad, q=1) Gaussian component, sized so its
# closed-form total light -- 2*pi*I*sigma**2*q -- comes out to exactly
# 1e10, matching this test file's previous hardcoded total_light value.
_STARS_MGE = LightMGE(
    I=Quantity(jnp.array([1.0e10 / (2 * jnp.pi)]), "Lsun / rad2"),
    sigma=Quantity(jnp.array([1.0]), "rad"),
    q=Quantity(jnp.array([1.0]), ""),
    PA_twist=Quantity(jnp.array([0.0]), "rad"),
)
_MGES = {"stars": _STARS_MGE}


def _mass_fraction_plugin(mges, candidate) -> None:
    import numpyro

    stars = mges["stars"]
    # `candidate`'s values are bare (unit-stripped) numbers -- see
    # `Prior.sample`'s docstring -- so `total_light` must be stripped to
    # match, in the same unit its `ml` prior's args are implicitly in
    # (Lsun, paired with `ml`'s Msun / Lsun). Stripping the per-component
    # term before summing (rather than summing the `Quantity` and stripping
    # after) also sidesteps a `jax.numpy.sum` incompatibility with
    # `unxt.Quantity`'s multi-alias physical types (Lsun's `PhysicalType`
    # has two names, `{'power', 'radiant flux'}`).
    total_light = jnp.sum(
        (2 * jnp.pi * stars.I * stars.sigma**2 * stars.q).ustrip("Lsun")
    )
    f = candidate["dh"]["M_200"] / (total_light * candidate["stars"]["ml"])
    numpyro.factor("dh_mass_fraction", dist.Normal(0.5, 0.05).log_prob(f))


def test_has_factors_is_false_for_a_pure_sample_model() -> None:
    prior = Prior(_POTENTIAL_SETTINGS, {}, {})

    assert prior.has_factors() is False


def test_has_factors_is_true_when_a_plugin_adds_one() -> None:
    prior = Prior(
        _POTENTIAL_SETTINGS, {"dh_mass_fraction": _mass_fraction_plugin}, _MGES
    )

    assert prior.has_factors() is True


def test_sample_without_factors_uses_predictive_and_respects_support() -> None:
    prior = Prior(_POTENTIAL_SETTINGS, {}, {})

    samples = prior.sample(jax.random.PRNGKey(0), num_samples=50)

    assert set(samples) == {"dh.M_200", "dh.c", "stars.ml"}
    assert samples["dh.M_200"].shape == (50,)
    assert bool(jnp.all(samples["dh.M_200"] >= 1.0e9))
    assert bool(jnp.all(samples["dh.M_200"] <= 1.0e11))
    assert bool(jnp.all(samples["dh.c"] >= 0.5))
    assert bool(jnp.all(samples["dh.c"] <= 5.0))
    assert bool(jnp.all(samples["stars.ml"] >= 1.0))
    assert bool(jnp.all(samples["stars.ml"] <= 10.0))


def test_sample_with_factors_requires_num_warmup() -> None:
    prior = Prior(
        _POTENTIAL_SETTINGS, {"dh_mass_fraction": _mass_fraction_plugin}, _MGES
    )

    with pytest.raises(ValueError, match="num_warmup is required"):
        prior.sample(jax.random.PRNGKey(0), num_samples=10)


def test_sample_with_factors_uses_mcmc_and_concentrates_near_the_factor() -> None:
    prior = Prior(
        _POTENTIAL_SETTINGS, {"dh_mass_fraction": _mass_fraction_plugin}, _MGES
    )

    samples = prior.sample(jax.random.PRNGKey(0), num_samples=200, num_warmup=500)

    f = samples["dh.M_200"] / (1.0e10 * samples["stars.ml"])
    # A smooth (Normal) factor gives NUTS real gradient signal, unlike a hard
    # Uniform factor (verified separately not to work well with NUTS: flat
    # interior gradient, discontinuous boundary) -- this is why the canonical
    # mass-fraction example uses Normal, not Uniform, for its factor.
    assert float(jnp.mean(f)) == pytest.approx(0.5, abs=0.05)


def test_load_prior_plugin_loads_the_named_function(tmp_path: Path) -> None:
    plugin_file = tmp_path / "mass_fraction.py"
    plugin_file.write_text(
        "def mass_fraction(mges, candidate):\n    pass\n", encoding="utf-8"
    )

    function = load_prior_plugin("mass_fraction.py:mass_fraction", tmp_path)

    assert function.__name__ == "mass_fraction"


def test_load_prior_plugin_rejects_a_missing_function(tmp_path: Path) -> None:
    plugin_file = tmp_path / "mass_fraction.py"
    plugin_file.write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(TypeError, match="does not name a callable function"):
        load_prior_plugin("mass_fraction.py:mass_fraction", tmp_path)


def test_load_prior_plugins_loads_every_configured_entry(tmp_path: Path) -> None:
    plugin_file = tmp_path / "mass_fraction.py"
    plugin_file.write_text(
        "def mass_fraction(mges, candidate):\n    pass\n", encoding="utf-8"
    )

    plugins = load_prior_plugins(
        {"dh_mass_fraction": {"plugin": "mass_fraction.py:mass_fraction"}}, tmp_path
    )

    assert set(plugins) == {"dh_mass_fraction"}
    assert plugins["dh_mass_fraction"].__name__ == "mass_fraction"
