"""Focused numerical and prior integration checks for support-point designs."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import optimistix as optx
import pytest

from tnt.all_models import AllModels
from tnt.parameter_generator import (
    AbstractParameterGenerator,
    PriorSampler,
    PriorSupportPoints,
    build_parameter_generator,
)
from tnt.priors import Prior
from tnt.support_points import (
    energy_objective,
    energy_objective_jax,
    optimize_support_points,
    optimize_support_points_jax,
)


def test_support_generator_depends_directly_on_abstract_contract() -> None:
    assert PriorSupportPoints.__bases__ == (AbstractParameterGenerator,)
    prior, generator = make_generator()
    assert generator.prior is prior
    assert not isinstance(generator, PriorSampler)


def test_compiled_kernel_matches_dense_numpy_iterations() -> None:
    rng = np.random.default_rng(13)
    reference = rng.normal(size=(47, 3))
    initial = rng.normal(size=(5, 3))
    expected = initial.copy()
    for _ in range(20):
        weights = 1 / np.linalg.norm(expected[:, None] - reference[None], axis=-1)
        delta = expected[:, None] - expected[None]
        norm = np.linalg.norm(delta, axis=-1)
        unit = np.divide(
            delta, norm[..., None], out=np.zeros_like(delta), where=norm[..., None] > 0
        )
        attraction = (weights[..., None] * reference[None]).sum(1)
        expected = (
            attraction + len(reference) / len(expected) * unit.sum(1)
        ) / weights.sum(1)[:, None]
    compiled = jax.jit(
        lambda x, z: optimize_support_points_jax(
            x, z, max_iterations=20, tolerance=1e-12, chunk_size=13
        )
    )
    result = compiled(jnp.asarray(reference), jnp.asarray(initial))
    np.testing.assert_allclose(result.value, expected, rtol=1e-10, atol=1e-10)
    assert int(result.stats["num_steps"]) == 20
    dense_value = (
        2 * np.linalg.norm(expected[:, None] - reference[None], axis=-1).mean()
        - np.linalg.norm(expected[:, None] - expected[None], axis=-1).mean()
    )
    assert float(energy_objective_jax(result.value, reference)) == pytest.approx(
        dense_value, abs=1e-12
    )


def test_kernel_jit_vmap_and_dynamic_tolerance() -> None:
    reference = jnp.array([[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    initial = jnp.array([[[-0.3, 0.2], [0.4, 0.1]], [[-0.2, 0.3], [0.1, 0.5]]])
    run = jax.jit(
        jax.vmap(
            lambda z: optimize_support_points_jax(
                reference,
                z,
                max_iterations=10,
                tolerance=jnp.asarray(1e3),
                chunk_size=2,
            )
        )
    )
    result = run(initial)
    assert bool(jnp.all(result.result == optx.RESULTS.successful))
    np.testing.assert_array_equal(result.stats["num_steps"], [1, 1])
    assert np.isfinite(result.aux).all()


def test_fixed_support_update_matches_combined_energy() -> None:
    rng = np.random.default_rng(24)
    reference = rng.normal(size=(79, 2))
    initial = rng.normal(size=(4, 2))
    fixed = rng.normal(size=(3, 2))
    original_fixed = fixed.copy()
    expected = initial.copy()
    for _ in range(20):
        weights = 1 / np.linalg.norm(expected[:, None] - reference[None], axis=-1)
        combined = np.concatenate([expected, fixed])
        delta = expected[:, None] - combined[None]
        distance = np.linalg.norm(delta, axis=-1)
        unit = np.divide(
            delta,
            distance[..., None],
            out=np.zeros_like(delta),
            where=distance[..., None] > 0,
        )
        expected = (
            (weights[..., None] * reference[None]).sum(1)
            + len(reference) / len(combined) * unit.sum(1)
        ) / weights.sum(1)[:, None]
    result = jax.jit(
        lambda old: optimize_support_points_jax(
            reference, initial, fixed_points=old, max_iterations=20, tolerance=1e-12
        )
    )(fixed)
    np.testing.assert_allclose(result.value, expected, atol=1e-10)
    np.testing.assert_array_equal(fixed, original_fixed)
    actual, diagnostics = optimize_support_points(
        reference, initial, fixed_points=fixed, max_iterations=20, tolerance=1e-12
    )
    assert diagnostics.objective == pytest.approx(
        energy_objective(np.concatenate([actual, fixed]), reference)
    )
    assert diagnostics.initial_objective == pytest.approx(
        energy_objective(np.concatenate([initial, fixed]), reference)
    )


def test_empty_fixed_points_match_first_iteration() -> None:
    rng = np.random.default_rng(25)
    reference, initial = rng.normal(size=(101, 2)), rng.normal(size=(5, 2))
    first, first_info = optimize_support_points(reference, initial, max_iterations=20)
    empty, empty_info = optimize_support_points(
        reference, initial, fixed_points=np.empty((0, 2)), max_iterations=20
    )
    np.testing.assert_array_equal(first, empty)
    assert first_info == empty_info


def test_kernel_preserves_explicit_float32_precision() -> None:
    reference = jnp.array([[-1.0], [0.0], [1.0]], dtype=jnp.float32)
    initial = jnp.array([[-0.3], [0.4]], dtype=jnp.float32)
    result = optimize_support_points_jax(reference, initial, max_iterations=5)
    assert result.value.dtype == jnp.float32
    assert result.aux.dtype == jnp.float32


def test_objective_compiles_and_eager_result_is_a_python_scalar() -> None:
    reference = jnp.array([[-1.0], [0.0], [1.0]])
    points = jnp.array([[-0.3], [0.4]])
    compiled = jax.jit(lambda z: energy_objective_jax(z, reference, chunk_size=2))
    assert float(compiled(points)) == pytest.approx(energy_objective(points, reference))
    assert isinstance(energy_objective(points, reference), float)


def test_kernel_promotes_integer_coordinates_to_floating_point() -> None:
    reference = jnp.array([[-2], [0], [2]])
    initial = jnp.array([[-1], [1]])
    result = optimize_support_points_jax(reference, initial, max_iterations=5)
    assert jnp.issubdtype(result.value.dtype, jnp.floating)
    assert np.isfinite(result.value).all()
    assert np.isfinite(energy_objective_jax(initial, reference))


def test_uniform_quantiles_and_independent_discrepancy():
    rng = np.random.default_rng(42)
    reference = rng.uniform(size=(4000, 1))
    initial = rng.uniform(size=(8, 1))
    points, diag = optimize_support_points(reference, initial, max_iterations=1000)
    repeated, repeated_diag = optimize_support_points(
        reference, initial, max_iterations=1000
    )
    np.testing.assert_array_equal(points, repeated)
    assert diag == repeated_diag
    np.testing.assert_allclose(
        np.sort(points[:, 0]), (np.arange(8) + 0.5) / 8, atol=0.025
    )
    assert diag.objective < diag.initial_objective
    held_out = rng.uniform(size=(5000, 1))
    assert energy_objective(points, held_out) < energy_objective(initial, held_out)
    assert diag.minimum_spacing > 0


def test_iteration_limit_is_not_convergence():
    _, diag = optimize_support_points(
        np.array([[-1.0], [0.0], [1.0]]),
        np.array([[-0.3], [0.4]]),
        max_iterations=1,
        tolerance=1e-12,
    )
    assert diag.status == "iteration_limit"
    assert not diag.converged
    assert diag.iterations == 1


def test_chunked_objective_and_updates_agree():
    rng = np.random.default_rng(5)
    x, initial = rng.normal(size=(103, 2)), rng.normal(size=(7, 2))
    a, da = optimize_support_points(x, initial, chunk_size=13, max_iterations=30)
    b, db = optimize_support_points(x, initial, chunk_size=1000, max_iterations=30)
    np.testing.assert_allclose(a, b, atol=1e-10)
    assert da.objective == pytest.approx(db.objective)
    assert da.status == "iteration_limit"


SETTINGS = {
    "dh": {
        "type": "NFWPotential",
        "parameters": {
            "m": {
                "fixed": False,
                "unit": "Msun",
                "prior": {"distribution": "LogUniform", "args": [1e10, 1e13]},
            },
            "r_s": {"fixed": True, "unit": "kpc", "value": 15.0},
        },
    }
}


def make_generator(settings=SETTINGS, plugins=None):
    prior = Prior(settings, plugins or {}, {})
    generator = build_parameter_generator(
        {
            "generator_type": "PriorSupportPoints",
            "generator_settings": {
                "seed": 5,
                "num_samples": 200,
                "num_warmup": 120,
                "target_accept_prob": 0.8,
                "max_iterations": 60,
            },
            "stopping_criteria": {"max_new_mods_per_iter": 5},
        },
        settings,
        prior,
    )
    return prior, generator


def _append_history(history, points, iteration):
    from tnt.model import Model
    from tnt.potential import Potential

    for point in points:
        history = history.append(
            Model(
                potential=Potential(components={}),
                valid_potential=True,
                raw_parameters=point,
                orblib_done=False,
                weights_done=False,
                weights=None,
                chi2=None,
                iteration=iteration,
            )
        )
    return history


def test_successive_iterations_reuse_no_sample_state(monkeypatch) -> None:
    prior, generator = make_generator()
    original_sample = prior.sample_with_diagnostics
    calls = []

    def sample(*args, **kwargs):
        calls.append(kwargs["num_samples"])
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(prior, "sample_with_diagnostics", sample)
    first, first_info = generator.generate_with_diagnostics(AllModels())
    history = _append_history(AllModels(), first, 0)
    snapshot = history.table.copy()
    second, second_info = generator.generate_with_diagnostics(history)
    repeated, repeated_info = generator.generate_with_diagnostics(history)
    assert calls == [205, 205, 205]
    assert len(first) == len(second) == generator.max_new_mods_per_iter
    assert second_info["existing_support_points"] == 5
    np.testing.assert_array_equal(first_info["center"], second_info["center"])
    np.testing.assert_array_equal(first_info["scale"], second_info["scale"])
    assert repeated_info["optimization"] == second_info["optimization"]
    first_values = [float(point["dh"]["m"].ustrip("Msun")) for point in first]
    second_values = [float(point["dh"]["m"].ustrip("Msun")) for point in second]
    np.testing.assert_array_equal(
        second_values, [float(point["dh"]["m"].ustrip("Msun")) for point in repeated]
    )
    assert not set(first_values).intersection(second_values)
    for name in snapshot.colnames:
        np.testing.assert_array_equal(history.table[name], snapshot[name])
    history = _append_history(history, second, 1)
    third, info = generator.generate_with_diagnostics(history)
    assert len(third) == 5
    assert info["existing_support_points"] == 10


def test_existing_points_convert_units_and_reject_duplicates() -> None:
    from astropy import units as au

    _, generator = make_generator()
    points, _ = generator.generate_with_diagnostics(AllModels())
    history = _append_history(AllModels(), points, 0)
    expected = generator._existing_locations(history, ["dh.m"])
    converted = history.table.copy()
    converted["dh.m"] = converted["dh.m"].to(au.Unit("1000 Msun"))
    np.testing.assert_allclose(
        generator._existing_locations(AllModels(converted), ["dh.m"]),
        expected,
        rtol=1e-15,
    )
    duplicate = _append_history(history, [points[0]], 1)
    with pytest.raises(ValueError, match="duplicate existing parameter locations"):
        generator.generate_with_diagnostics(duplicate)


def test_existing_points_must_match_prior_columns_and_bounds() -> None:
    from unxt import Quantity

    _, generator = make_generator()
    missing = _append_history(AllModels(), [{"dh": {"r_s": Quantity(15.0, "kpc")}}], 0)
    with pytest.raises(ValueError, match="missing prior parameter column"):
        generator.generate_with_diagnostics(missing)
    invalid = _append_history(AllModels(), [{"dh": {"m": Quantity(1.0, "Msun")}}], 0)
    with pytest.raises(ValueError, match="outside the current prior"):
        generator.generate_with_diagnostics(invalid)


def test_existing_prior_integration_units_bounds_reproducibility():
    prior, generator = make_generator()
    points, diag = generator.generate_with_diagnostics(AllModels())
    again, other = generator.generate_with_diagnostics(AllModels())
    assert len(points) == 5
    assert diag["site_names"] == ("dh.m",)
    assert diag["optimization"] == other["optimization"]
    for point, repeated in zip(points, again, strict=True):
        mass = float(point["dh"]["m"].ustrip("Msun"))
        assert 1e10 <= mass <= 1e13
        assert mass == float(repeated["dh"]["m"].ustrip("Msun"))
        assert float(point["dh"]["r_s"].ustrip("kpc")) == 15.0
        assert prior.is_feasible({"dh.m": mass})
    with pytest.warns(UserWarning, match="Out-of-support"):
        assert not prior.is_feasible({"dh.m": 1.0})


def test_joint_plugin_hole_rejects_interpolated_valid_draws():
    def hole(context):
        mass = context.candidate["dh"]["m"]
        numpyro.factor("hole", jnp.where((mass < 3e11) | (mass > 8e11), 0.0, -jnp.inf))

    prior = Prior(SETTINGS, {"hole": hole}, {})
    assert prior.is_feasible({"dh.m": 2e11})
    assert prior.is_feasible({"dh.m": 9e11})
    assert not prior.is_feasible({"dh.m": 5e11})


def test_constants_and_single_point():
    points, diag = optimize_support_points(np.zeros((20, 0)), np.zeros((1, 0)))
    assert points.shape == (1, 0)
    assert diag.converged
    assert diag.objective == 0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        optimize_support_points([[np.nan]], [[0.0]])
    with pytest.raises(ValueError):
        optimize_support_points([[0.0]], [[1.0]], chunk_size=0)


def test_correlated_distribution():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(900, 2))
    x[:, 1] = 0.8 * x[:, 0] + 0.2 * x[:, 1]
    initial = x[-10:] + 1e-5
    points, diag = optimize_support_points(x[:-10], initial)
    assert diag.objective <= diag.initial_objective
    held_out = rng.normal(size=(2000, 2))
    held_out[:, 1] = 0.8 * held_out[:, 0] + 0.2 * held_out[:, 1]
    assert energy_objective(points, held_out) < energy_objective(initial, held_out)


def test_finite_design_can_underrepresent_narrow_distribution_width() -> None:
    from scipy.special import ndtri

    quantiles = ndtri((np.arange(31) + 0.5) / 31)
    x, y = np.meshgrid(quantiles, quantiles)
    reference = np.column_stack([x.ravel(), 0.08 * y.ravel()])
    rng = np.random.default_rng(15)
    held_out = rng.normal(size=(2000, 2)) * [1.0, 0.08]
    width_ratios = []
    for count in (4, 64):
        initial = rng.normal(size=(count, 2)) * [1.0, 0.08]
        points, diagnostics = optimize_support_points(
            reference, initial, max_iterations=500
        )
        assert diagnostics.objective < diagnostics.initial_objective
        assert energy_objective(points, held_out) < energy_objective(initial, held_out)
        width_ratios.append(np.std(points[:, 1]) / np.std(reference[:, 1]))
    # A finite energy design need not preserve variance. Broad bounds capture
    # this example without requiring a universal variance or spacing ordering.
    assert width_ratios[0] < 0.2
    assert width_ratios[1] > 0.5


@pytest.mark.parametrize("free_ml", [False, True])
def test_default_conditional_prior_and_geometry(free_ml):
    import jax
    from unxt import Quantity, unitsystem

    from tnt.mge import LightMGE
    from tnt.potential import Potential

    mge = LightMGE(
        I=Quantity(jnp.array([1e3, 2e2, 3e1]), "Lsun / pc2"),
        sigma=Quantity(jnp.array([0.3, 1.2, 4.0]), "kpc"),
        q=Quantity(jnp.array([0.85, 0.78, 0.90]), ""),
        PA_twist=Quantity(jnp.zeros(3), "rad"),
        major_axis_pa=Quantity(0.0, "deg"),
    )
    settings = {
        "stars": {
            "type": "TriaxialLightMGEPotential",
            "parameterization": "pqu",
            "mge": "mge",
            "parameters": {
                "ml": {
                    "fixed": not free_ml,
                    "value": 5.0,
                    "unit": "Msun / Lsun",
                    "prior": {"distribution": "Uniform", "args": [1.0, 9.0]},
                },
                **{name: {"fixed": False} for name in ("p", "q", "u")},
            },
        }
    }
    mges = {"mge": mge}
    prior = Prior(
        settings,
        {},
        mges,
        resolved_potential=Potential.resolve(settings, mges),
        cosmological_parameters={},
        unit_system=unitsystem("galactic"),
    )
    draws = prior.sample(jax.random.PRNGKey(3), 10)
    assert any(
        prior.is_feasible({name: values[i] for name, values in draws.items()})
        for i in range(10)
    )
    extra = {"stars.ml": 5.0} if free_ml else {}
    constrained = {name: values[0] for name, values in draws.items()}
    latent = prior.unconstrain(constrained)
    round_trip = prior.constrain(latent)
    for name, value in constrained.items():
        np.testing.assert_allclose(round_trip[name], value, rtol=2e-5, atol=2e-6)
    assert not prior.is_feasible(
        {"stars.p": 0.4, "stars.q": 0.8, "stars.u": 0.9, **extra}
    )
    assert not prior.is_feasible(
        {"stars.p": 0.8, "stars.q": 0.5, "stars.u": 0.6, **extra}
    )

    generator = build_parameter_generator(
        {
            "generator_type": "PriorSupportPoints",
            "generator_settings": {
                "seed": 3,
                "num_samples": 120,
                "num_warmup": 120,
                "target_accept_prob": 0.8,
                "max_iterations": 10,
            },
            "stopping_criteria": {"max_new_mods_per_iter": 4},
        },
        settings,
        prior,
    )
    points, info = generator.generate_with_diagnostics(AllModels())
    assert len(info["site_names"]) == (4 if free_ml else 3)
    assert len(points) == 4
    assert info["optimization"].objective <= info["optimization"].initial_objective
    for point in points:
        assert prior.is_feasible(
            {
                site: point["stars"][site.split(".")[1]].ustrip(
                    "Msun / Lsun" if site == "stars.ml" else ""
                )
                for site in info["site_names"]
            }
        )


def test_constant_sample_sites():
    settings = {
        "dh": {
            "type": "NFWPotential",
            "parameters": {
                "m": {
                    "fixed": False,
                    "unit": "Msun",
                    "prior": {"distribution": "Delta", "args": [1e10]},
                },
                "r_s": {"fixed": True, "unit": "kpc", "value": 15.0},
            },
        }
    }
    _, generator = make_generator(settings)
    points, diag = generator.generate_with_diagnostics(AllModels())
    assert len(points) == 1
    assert diag["constant_dimensions"] == ("dh.m",)
    assert diag["optimization"].objective == 0
    assert diag["optimization"].minimum_spacing == 0


def test_generator_metric_is_invariant_to_declared_units():
    from copy import deepcopy

    _, generator = make_generator()
    points, _ = generator.generate_with_diagnostics(AllModels())
    converted = deepcopy(SETTINGS)
    parameter = converted["dh"]["parameters"]["m"]
    # Use a representable unit conversion at the configured JAX precision.
    parameter["unit"] = "1000 Msun"
    parameter["prior"]["args"] = [1e7, 1e10]
    _, other_generator = make_generator(converted)
    other, _ = other_generator.generate_with_diagnostics(AllModels())
    np.testing.assert_allclose(
        [float(point["dh"]["m"].ustrip("Msun")) for point in points],
        [float(point["dh"]["m"].ustrip("Msun")) for point in other],
        rtol=2e-4,
    )


def test_all_fixed_generator():
    from copy import deepcopy

    settings = deepcopy(SETTINGS)
    settings["dh"]["parameters"]["m"].update(fixed=True, value=1e12)
    _, generator = make_generator(settings)
    points, info = generator.generate_with_diagnostics(AllModels())
    assert len(points) == 1
    assert info["optimization"].objective == 0
    assert info["site_names"] == ()
    assert all(float(point["dh"]["m"].ustrip("Msun")) == 1e12 for point in points)
    assert info["new_support_points"] == 5
    assert info["returned_support_points"] == 1
    assert info["proposals"]["duplicates_removed"] == 4
    history = _append_history(AllModels(), points, 0)
    again, later = generator.generate_with_diagnostics(history)
    assert again == []
    assert later["proposals"]["returned"] == 0
    assert generator.generate_parameters(history) == []


def test_unconstrained_rejects_discrete_sites():
    from copy import deepcopy

    settings = deepcopy(SETTINGS)
    settings["dh"]["parameters"]["m"]["prior"] = {
        "distribution": "BernoulliProbs",
        "args": [0.5],
    }
    _, generator = make_generator(settings)
    with pytest.raises(ValueError, match="require continuous sites: dh.m"):
        generator.generate_with_diagnostics(AllModels())


def test_unconstrained_rejects_vector_sites():
    from copy import deepcopy

    settings = deepcopy(SETTINGS)
    settings["dh"]["parameters"]["m"]["prior"] = {
        "distribution": "Normal",
        "args": [jnp.full(2, 1e12), jnp.full(2, 1e10)],
    }
    _, generator = make_generator(settings)
    with pytest.raises(ValueError, match="requires scalar parameter sites"):
        generator.generate_with_diagnostics(AllModels())


def test_nonfinite_boundary_transform_is_not_clipped(monkeypatch):
    prior, generator = make_generator()
    _, diagnostics = prior.sample_with_diagnostics(
        jax.random.PRNGKey(1),
        num_samples=205,
    )
    monkeypatch.setattr(
        prior,
        "sample_with_diagnostics",
        lambda *args, **kwargs: ({"dh.m": jnp.full(205, 1e10)}, diagnostics),
    )
    with pytest.raises(ValueError, match="non-finite unconstrained"):
        generator.generate_with_diagnostics(AllModels())


def test_invalid_final_design_returns_no_candidates(monkeypatch):
    from copy import deepcopy

    import jax

    settings = deepcopy(SETTINGS)
    for name in ("m", "r_s"):
        settings["dh"]["parameters"][name].update(
            fixed=False, prior={"distribution": "Normal", "args": [0.0, 1.0]}
        )

    def ring(context):
        candidate = context.candidate["dh"]
        numpyro.factor(
            "outside_circle",
            jnp.where(
                jnp.abs(candidate["m"] ** 2 + candidate["r_s"] ** 2 - 4.0) < 1e-4,
                0.0,
                -jnp.inf,
            ),
        )

    prior, generator = make_generator(settings, {"ring": ring})
    # Continuous support-point optimization can put points inside this thin
    # ring even though every reference draw lies on it. No model is evaluated.
    angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    reference = np.column_stack([2 * np.cos(angles), 2 * np.sin(angles)])
    initial_angles = 0.02 + np.arange(5) * 0.1
    initial = np.column_stack([2 * np.cos(initial_angles), 2 * np.sin(initial_angles)])
    matrix = np.concatenate([reference, initial])
    from tnt.priors import PriorDiagnostics

    diag = PriorDiagnostics(
        jnp.zeros(205, dtype=bool),
        {"dh.m": 200.0, "dh.r_s": 200.0},
        {"dh.m": 1.0, "dh.r_s": 1.0},
    )
    monkeypatch.setattr(
        prior,
        "sample_with_diagnostics",
        lambda *args, **kwargs: (
            {"dh.m": jnp.asarray(matrix[:, 0]), "dh.r_s": jnp.asarray(matrix[:, 1])},
            diag,
        ),
    )
    # All original draws satisfy the plugin; the unconstrained optimum need not.
    assert np.asarray(
        jax.vmap(prior.feasibility)({"dh.m": matrix[:, 0], "dh.r_s": matrix[:, 1]})
    ).all()
    with pytest.raises(ValueError, match="No candidates returned"):
        generator.generate_with_diagnostics(AllModels())


def test_unconstrained_mixed_constant_and_variable_sites():
    from copy import deepcopy

    settings = deepcopy(SETTINGS)
    parameters = settings["dh"]["parameters"]
    parameters["m"]["prior"] = {"distribution": "Delta", "args": [1e10]}
    parameters["r_s"].update(
        fixed=False, prior={"distribution": "Uniform", "args": [5.0, 25.0]}
    )
    _, generator = make_generator(settings)
    points, info = generator.generate_with_diagnostics(AllModels())
    assert info["constant_dimensions"] == ("dh.m",)
    assert info["optimization"].objective < info["optimization"].initial_objective
    for point in points:
        assert float(point["dh"]["m"].ustrip("Msun")) == 1e10
        assert 5.0 <= float(point["dh"]["r_s"].ustrip("kpc")) <= 25.0


def test_conditional_transforms_round_trip_with_soft_factor():
    from numpyro import distributions as dist

    def preference(context):
        numpyro.factor(
            "preference", dist.Normal(1e12, 1e11).log_prob(context.candidate["dh"]["m"])
        )

    prior, generator = make_generator(plugins={"preference": preference})
    points, _ = generator.generate_with_diagnostics(AllModels())
    for point in points:
        physical = {"dh.m": point["dh"]["m"].ustrip("Msun")}
        reconstructed = prior.constrain(prior.unconstrain(physical))
        np.testing.assert_allclose(reconstructed["dh.m"], physical["dh.m"], rtol=1e-6)
        assert prior.is_feasible(reconstructed)
