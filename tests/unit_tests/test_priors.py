"""Unit tests for `tnt.priors`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import galax.potential as gp
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
import unxt as u
from unxt import Quantity

from tnt.mge import LightMGE
from tnt.potential import Potential
from tnt.priors import Prior, PriorContext, load_prior_plugin, load_prior_plugins

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
    major_axis_pa=Quantity(0.0, "deg"),
)
_MGES = {"stars": _STARS_MGE}


def _mass_fraction_plugin(context) -> None:
    import numpyro

    stars = context.mges["stars"]
    candidate = context.candidate
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
    plugin_file.write_text("def mass_fraction(context):\n    pass\n", encoding="utf-8")

    function = load_prior_plugin("mass_fraction.py:mass_fraction", tmp_path)

    assert function.__name__ == "mass_fraction"


def test_load_prior_plugin_rejects_a_missing_function(tmp_path: Path) -> None:
    plugin_file = tmp_path / "mass_fraction.py"
    plugin_file.write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(TypeError, match="does not name a callable function"):
        load_prior_plugin("mass_fraction.py:mass_fraction", tmp_path)


def test_load_prior_plugin_rejects_a_wrong_arity_signature(tmp_path: Path) -> None:
    plugin_file = tmp_path / "mass_fraction.py"
    plugin_file.write_text(
        "def mass_fraction(mges, candidate):\n    pass\n", encoding="utf-8"
    )

    with pytest.raises(TypeError, match="must be callable as"):
        load_prior_plugin("mass_fraction.py:mass_fraction", tmp_path)


def test_load_prior_plugin_allows_a_trailing_default_argument(tmp_path: Path) -> None:
    plugin_file = tmp_path / "mass_fraction.py"
    plugin_file.write_text(
        "def mass_fraction(context, _scratch=None):\n    pass\n", encoding="utf-8"
    )

    function = load_prior_plugin("mass_fraction.py:mass_fraction", tmp_path)

    assert function.__name__ == "mass_fraction"


def test_load_prior_plugins_loads_every_configured_entry(tmp_path: Path) -> None:
    plugin_file = tmp_path / "mass_fraction.py"
    plugin_file.write_text("def mass_fraction(context):\n    pass\n", encoding="utf-8")

    plugins = load_prior_plugins(
        {"dh_mass_fraction": {"plugin": "mass_fraction.py:mass_fraction"}}, tmp_path
    )

    assert set(plugins) == {"dh_mass_fraction"}
    assert plugins["dh_mass_fraction"].__name__ == "mass_fraction"


# ---------------------------------------------------------------------------
# context.build_potential(): a plugin factor over a derived quantity of the
# whole (traced) potential, here the enclosed mass M(< 10 kpc).
# ---------------------------------------------------------------------------

_UNITS = u.unitsystem("kpc", "Myr", "Msun", "rad")
# A genuinely triaxial 3-Gaussian light MGE, physical sigma, plus the viewing
# angles it deprojects validly at (0 < q <= p <= 1 for every component).
_TRIAXIAL_MGE = LightMGE(
    I=Quantity(jnp.array([1.0e3, 2.0e2, 3.0e1]), "Lsun / pc2"),
    sigma=Quantity(jnp.array([0.3, 1.2, 4.0]), "kpc"),
    q=Quantity(jnp.array([0.85, 0.78, 0.90]), ""),
    PA_twist=Quantity(jnp.zeros(3), "rad"),
    major_axis_pa=Quantity(0.0, "deg"),
)
_ANGLES = {
    "theta": 0.4280191007651427,
    "phi": 1.0363519257903688,
    "psi": 2.033205815042,
}
_ENCLOSED_TARGET, _ENCLOSED_WIDTH = 4.0e10, 5.0e9
_R_10_KPC = Quantity(jnp.array([10.0, 0.0, 0.0]), "kpc")
_T0 = Quantity(0.0, "Myr")

_ENCLOSED_MASS_POTENTIAL = {
    "dh": {
        "type": "NFWPotential",
        "parameters": {
            "m": {
                "unit": "Msun",
                "fixed": False,
                "value": 1.0e12,
                "prior": {"distribution": "LogUniform", "args": [1.0e10, 1.0e13]},
            },
            "r_s": {"unit": "kpc", "fixed": True, "value": 15.0},
        },
    },
    "stars": {
        "type": "TriaxialLightMGEPotential",
        "mge": "stars_mge",
        "parameters": {
            "ml": {
                "unit": "Msun / Lsun",
                "fixed": False,
                "value": 5.0,
                "prior": {"distribution": "Uniform", "args": [1.0, 9.0]},
            },
            **{
                name: {"unit": "rad", "fixed": True, "value": value}
                for name, value in _ANGLES.items()
            },
        },
    },
}


def _m_within_10kpc_plugin(context: PriorContext) -> None:
    import numpyro

    galax_potential = context.build_potential().to_galax(context.unit_system)
    enclosed = gp.spherical_mass_enclosed(
        galax_potential, _R_10_KPC, _T0
    ).ustrip("Msun")
    numpyro.factor(
        "m_within_10kpc",
        dist.Normal(_ENCLOSED_TARGET, _ENCLOSED_WIDTH).log_prob(jnp.asarray(enclosed)),
    )


def _enclosed_mass_of(resolved, ml: jnp.ndarray, m_halo: jnp.ndarray) -> jnp.ndarray:
    parameters = {
        "dh": {"m": Quantity(m_halo, "Msun"), "r_s": Quantity(15.0, "kpc")},
        "stars": {
            "ml": Quantity(ml, "Msun / Lsun"),
            **{name: Quantity(value, "rad") for name, value in _ANGLES.items()},
        },
    }
    # Called under `jax.vmap` below, so the validity flag can't be asserted
    # here (a traced boolean); every drawn `ml`/`m_halo` in this test is
    # physically valid by construction.
    built, _valid = Potential.build_with_validity(resolved, parameters, {})
    galax_potential = built.to_galax(_UNITS)
    return gp.spherical_mass_enclosed(galax_potential, _R_10_KPC, _T0).ustrip("Msun")


def test_build_potential_is_unavailable_without_potential_context() -> None:
    context = PriorContext(candidate={"stars": {"ml": 5.0}}, mges={})

    with pytest.raises(RuntimeError, match="build_potential.*needs the run's"):
        context.build_potential()


def test_plugin_factor_on_enclosed_mass_concentrates_the_prior() -> None:
    mges = {"stars_mge": _TRIAXIAL_MGE}
    resolved = Potential.resolve(_ENCLOSED_MASS_POTENTIAL, mges)

    prior_only = Prior(_ENCLOSED_MASS_POTENTIAL, {}, mges)
    assert prior_only.has_factors() is False
    draws = prior_only.sample(jax.random.PRNGKey(1), num_samples=400)
    prior_only_enclosed = jax.vmap(lambda a, b: _enclosed_mass_of(resolved, a, b))(
        draws["stars.ml"], draws["dh.m"]
    )

    prior = Prior(
        _ENCLOSED_MASS_POTENTIAL,
        {"m_within_10kpc": _m_within_10kpc_plugin},
        mges,
        resolved_potential=resolved,
        cosmological_parameters={"H": Quantity(70.0, "km / (s Mpc)")},
        unit_system=_UNITS,
    )
    assert prior.has_factors() is True

    samples = prior.sample(
        jax.random.PRNGKey(0), num_samples=200, num_warmup=400
    )
    assert set(samples) == {"dh.m", "stars.ml"}
    enclosed = jax.vmap(lambda a, b: _enclosed_mass_of(resolved, a, b))(
        samples["stars.ml"], samples["dh.m"]
    )

    # The factor pulls M(< 10 kpc) to its target and sharply narrows it
    # relative to the unconstrained prior.
    assert float(jnp.mean(enclosed)) == pytest.approx(
        _ENCLOSED_TARGET, abs=2 * _ENCLOSED_WIDTH
    )
    assert float(jnp.std(enclosed)) < 0.3 * float(jnp.std(prior_only_enclosed))


# ---------------------------------------------------------------------------
# The same build_potential() validity factor, now actually exercised: an
# independent U(0, 1) prior on each of (p, q, u) routinely violates TNT's
# 0 < q <= p <= 1, p < u <= 1 convention, unlike every other potential in
# this file, which keeps its shape fixed at an already-valid point.
# ---------------------------------------------------------------------------

_POTENTIAL_PQU = {
    "dh": {
        "type": "NFWPotential",
        "parameters": {
            "m": {
                "unit": "Msun",
                "fixed": False,
                "value": 1.0e12,
                "prior": {"distribution": "LogUniform", "args": [1.0e10, 1.0e13]},
            },
            "r_s": {"unit": "kpc", "fixed": True, "value": 15.0},
        },
    },
    "stars": {
        "type": "TriaxialLightMGEPotential",
        "parameterization": "pqu",
        "mge": "stars_mge",
        "parameters": {
            "ml": {"unit": "Msun / Lsun", "fixed": True, "value": 5.0},
            **{
                name: {
                    "fixed": False,
                    "value": value,
                    "prior": {"distribution": "Uniform", "args": [0.0, 1.0]},
                }
                for name, value in (("p", 0.85), ("q", 0.6), ("u", 0.93))
            },
        },
    },
}


def _pqu_validity_and_enclosed_mass(resolved, p, q, u, m_halo):
    parameters = {
        "dh": {"m": Quantity(m_halo, "Msun"), "r_s": Quantity(15.0, "kpc")},
        "stars": {
            "ml": Quantity(5.0, "Msun / Lsun"),
            "p": Quantity(p, ""),
            "q": Quantity(q, ""),
            "u": Quantity(u, ""),
        },
    }
    built, valid = Potential.build_with_validity(resolved, parameters, {})
    enclosed = gp.spherical_mass_enclosed(
        built.to_galax(_UNITS), _R_10_KPC, _T0
    ).ustrip("Msun")
    return valid, enclosed


def test_plugin_factor_on_enclosed_mass_rejects_invalid_pqu_shape_draws() -> None:
    mges = {"stars_mge": _TRIAXIAL_MGE}
    resolved = Potential.resolve(_POTENTIAL_PQU, mges)

    prior_only = Prior(_POTENTIAL_PQU, {}, mges)
    draws = prior_only.sample(jax.random.PRNGKey(1), num_samples=4000)
    valid_prior, _ = jax.vmap(
        lambda p, q, u, m: _pqu_validity_and_enclosed_mass(resolved, p, q, u, m)
    )(draws["stars.p"], draws["stars.q"], draws["stars.u"], draws["dh.m"])
    # Independent U(0, 1) draws land in the valid region only a small
    # fraction of the time -- most of the unit cube violates the shape
    # convention.
    assert float(jnp.mean(valid_prior)) < 0.2

    prior = Prior(
        _POTENTIAL_PQU,
        {"m_within_10kpc": _m_within_10kpc_plugin},
        mges,
        resolved_potential=resolved,
        cosmological_parameters={"H": Quantity(70.0, "km / (s Mpc)")},
        unit_system=_UNITS,
    )
    assert prior.has_factors() is True

    samples = prior.sample(jax.random.PRNGKey(0), num_samples=1000, num_warmup=1000)
    assert set(samples) == {"dh.m", "stars.p", "stars.q", "stars.u"}

    valid, enclosed = jax.vmap(
        lambda p, q, u, m: _pqu_validity_and_enclosed_mass(resolved, p, q, u, m)
    )(samples["stars.p"], samples["stars.q"], samples["stars.u"], samples["dh.m"])

    # The -inf validity factor keeps NUTS inside the valid region almost
    # every draw, despite that region being a small, oddly-shaped slice of
    # the (p, q, u) prior's own unit cube -- not just the trivial
    # always-valid case every other potential in this file uses.
    assert float(jnp.mean(valid)) > 0.95
    # The plugin's own Normal factor still pulls M(< 10 kpc) to its target
    # on top of that, exactly as it does when the shape is fixed.
    assert float(jnp.mean(enclosed)) == pytest.approx(
        _ENCLOSED_TARGET, abs=2 * _ENCLOSED_WIDTH
    )


# ---------------------------------------------------------------------------
# A prior's distribution is resolved and validated once, at Prior
# construction time -- nothing upstream (config preparation deliberately
# never imports numpyro; `distribution_cls(*args)` without
# `validate_args=True`) previously caught a reversed/degenerate Uniform
# range, a non-positive LogUniform range, a non-positive Normal scale, or a
# `distribution` name that isn't a real numpyro.distributions class.
# ---------------------------------------------------------------------------


def _single_parameter_potential_settings(
    distribution: str, args: list[float]
) -> dict[str, dict[str, Any]]:
    return {
        "dh": {
            "parameters": {
                "m": {
                    "unit": "Msun",
                    "fixed": False,
                    "prior": {"distribution": distribution, "args": args},
                },
            },
        },
    }


@pytest.mark.parametrize(
    ("distribution", "args"),
    [
        ("Uniform", [9.0, 1.0]),  # reversed bounds
        ("Uniform", [1.0, 1.0]),  # degenerate (equal) bounds
        ("LogUniform", [-1.0, 9.0]),  # non-positive low
        ("Normal", [5.0, -1.0]),  # non-positive scale
    ],
)
def test_prior_rejects_invalid_distribution_arguments(
    distribution: str, args: list[float]
) -> None:
    settings = _single_parameter_potential_settings(distribution, args)
    with pytest.raises(ValueError, match=r"dh\.m"):
        Prior(settings, {}, {})


def test_prior_rejects_a_distribution_name_that_is_not_a_numpyro_distribution() -> None:
    settings = _single_parameter_potential_settings("not_a_distribution", [1.0])
    with pytest.raises(TypeError, match="not a numpyro.distributions class"):
        Prior(settings, {}, {})


def test_prior_accepts_a_well_formed_distribution() -> None:
    settings = _single_parameter_potential_settings("Uniform", [1.0, 9.0])
    prior = Prior(settings, {}, {})
    samples = prior.sample(jax.random.PRNGKey(0), num_samples=4)
    assert samples["dh.m"].shape == (4,)
