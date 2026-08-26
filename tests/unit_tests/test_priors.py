"""Unit tests for `tnt.priors`."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from tnt.priors import Prior, load_prior_plugin, load_prior_plugins

_POTENTIAL_SETTINGS = {
    "dh": {
        "type": "NFWPotential",
        "parameterization": "concentration_m200",
        "include": True,
        "parameters": {
            "c": {"fixed": True, "value": 8.0},
            "M_200": {
                "unit": "Msun",
                "fixed": False,
                "prior": {"distribution": "LogUniform", "args": [1.0e10, 1.0e14]},
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


def _mass_fraction_plugin(mges, candidate) -> None:
    import numpyro

    total_light = 1.0e10
    f = candidate["dh"]["M_200"] / (total_light * candidate["stars"]["ml"])
    numpyro.factor("dh_mass_fraction", dist.Normal(0.2, 0.1).log_prob(f))


def test_has_factors_is_false_for_a_pure_sample_model() -> None:
    prior = Prior(_POTENTIAL_SETTINGS, {}, {})

    assert prior.has_factors() is False


def test_has_factors_is_true_when_a_plugin_adds_one() -> None:
    prior = Prior(_POTENTIAL_SETTINGS, {"dh_mass_fraction": _mass_fraction_plugin}, {})

    assert prior.has_factors() is True


def test_sample_without_factors_uses_predictive_and_respects_support() -> None:
    prior = Prior(_POTENTIAL_SETTINGS, {}, {})

    samples = prior.sample(jax.random.PRNGKey(0), num_samples=50)

    assert set(samples) == {"dh.M_200", "stars.ml"}
    assert samples["dh.M_200"].shape == (50,)
    assert bool(jnp.all(samples["dh.M_200"] >= 1.0e10))
    assert bool(jnp.all(samples["dh.M_200"] <= 1.0e14))
    assert bool(jnp.all(samples["stars.ml"] >= 1.0))
    assert bool(jnp.all(samples["stars.ml"] <= 10.0))
    # `dh.c` is fixed -- never a sample site, so never in Prior.sample's output.
    assert "dh.c" not in samples


def test_sample_with_factors_requires_num_warmup() -> None:
    prior = Prior(_POTENTIAL_SETTINGS, {"dh_mass_fraction": _mass_fraction_plugin}, {})

    with pytest.raises(ValueError, match="num_warmup is required"):
        prior.sample(jax.random.PRNGKey(0), num_samples=10)


def test_sample_with_factors_uses_mcmc_and_concentrates_near_the_factor() -> None:
    prior = Prior(_POTENTIAL_SETTINGS, {"dh_mass_fraction": _mass_fraction_plugin}, {})

    samples = prior.sample(jax.random.PRNGKey(0), num_samples=200, num_warmup=500)

    f = samples["dh.M_200"] / (1.0e10 * samples["stars.ml"])
    # A smooth (Normal) factor gives NUTS real gradient signal, unlike a hard
    # Uniform factor (verified separately not to work well with NUTS: flat
    # interior gradient, discontinuous boundary) -- this is why the canonical
    # mass-fraction example uses Normal, not Uniform, for its factor.
    assert float(jnp.mean(f)) == pytest.approx(0.2, abs=0.05)


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
